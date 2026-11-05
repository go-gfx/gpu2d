// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package metal draws a gpu2d list on a Mac's GPU with a compute kernel,
// through go-macos/metal (pure Go, no cgo). Instances are binned into tiles
// on the CPU; one GPU thread per pixel walks its tile's instances in draw
// order, computes gpu2d.Coverage, and blends in integers exactly as the CPU
// reference does. On unified memory the frame is a Go slice the GPU writes
// directly: nothing is uploaded or downloaded.
//
// On a platform that is not macOS, New returns go-macos/metal's
// ErrUnsupported, so a program cross-compiles without build tags.
package metal

import (
	_ "embed"
	"errors"

	"github.com/go-gfx/gpu2d"
	mtl "github.com/go-macos/metal"
)

// Kernel is the Metal Shading Language source; its entry point is "raster".
//
//go:embed kernel.metal
var Kernel string

// ErrSize reports a list whose surface is not the renderer's.
var ErrSize = errors.New("gpu2d/metal: list size differs from the renderer's")

// frame is the kernel's Frame, field for field.
type frame struct{ W, H, Across, Tile uint32 }

// ErrBusy reports a Submit with every slot in flight, or a Draw while work
// is in flight: Wait first.
var ErrBusy = errors.New("gpu2d/metal: frames in flight; Wait first")

// ErrNothingInFlight reports a Wait with no Submit to wait for.
var ErrNothingInFlight = errors.New("gpu2d/metal: Wait without a Submit")

// slots is how many frames may be in flight: the GPU draws one while the CPU
// bins and packs the next.
const slots = 2

// slot is the per-frame input memory. The GPU reads it until the slot's
// Pending is waited on, so each in-flight frame owns its own.
type slot struct {
	inst, spans, list, data *mtl.Buffer
	bytes                   []byte
	spanIdx, listIdx        []uint32
	pend                    *mtl.Pending
}

// Renderer draws lists into a w × h RGBA frame in GPU-visible memory.
//
// Submit and Wait are the asynchronous pair; Draw is both. A Renderer is
// for one goroutine at a time.
type Renderer struct {
	d      *mtl.Device
	lib    *mtl.Library
	pipe   *mtl.Pipeline
	show   *mtl.Pipeline // copies the frame into a drawable
	w, h   int
	target *mtl.Buffer
	slot   [slots]slot
	head   int // the oldest frame in flight
	n      int // frames in flight
}

// New compiles the kernel on the default GPU and allocates a w × h frame.
func New(w, h int) (*Renderer, error) {
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return nil, gpu2d.ErrSize
	}
	d, err := mtl.Default()
	if err != nil {
		return nil, err
	}
	r := &Renderer{d: d, w: w, h: h}
	if r.lib, err = d.Compile(Kernel); err != nil {
		r.Close()
		return nil, err
	}
	if r.pipe, err = r.lib.Pipeline("raster"); err != nil {
		r.Close()
		return nil, err
	}
	if r.show, err = r.lib.Pipeline("present"); err != nil {
		r.Close()
		return nil, err
	}
	if r.target, err = d.NewBuffer(4 * w * h); err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

// Name is the GPU's own name, such as "Apple M4 Max".
func (r *Renderer) Name() string { return r.d.Name() }

// Pixels is the frame, 4·w·h bytes of RGBA rows top to bottom. On unified
// memory it is the memory the GPU draws into: write the ground here before
// Draw and read the result here after it.
func (r *Renderer) Pixels() []byte { return r.target.Bytes() }

// fit returns b if it holds n bytes, else a new buffer of at least n
// (doubling), closing b.
func (r *Renderer) fit(b *mtl.Buffer, n int) (*mtl.Buffer, error) {
	n = max(n, 4)
	if b != nil && len(b.Bytes()) >= n {
		return b, nil
	}
	c := n
	if b != nil {
		c = max(n, 2*len(b.Bytes()))
		b.Close()
	}
	return r.d.NewBuffer(c)
}

// Draw composites the list over the frame's current content and waits for
// the GPU to finish: Submit followed by Wait. It is ErrBusy while a frame
// is in flight.
func (r *Renderer) Draw(l *gpu2d.List) error {
	if r.n > 0 {
		return ErrBusy
	}
	if err := r.Submit(l); err != nil {
		return err
	}
	return r.Wait()
}

// Submit bins and packs the list into the next free slot, sends it to the
// GPU and returns without waiting. The list is copied: the caller may Reset
// and record the next frame at once, which is the overlap. Frames run in
// submission order, each compositing over the previous one's result. At most
// two frames are in flight; a third Submit is ErrBusy.
//
// The frame's memory (Pixels) belongs to the GPU until Wait has returned for
// every frame in flight: write the ground before the first Submit, read the
// result after the last Wait.
func (r *Renderer) Submit(l *gpu2d.List) error {
	s, f, err := r.pack(l)
	if err != nil {
		return err
	}
	p, err := r.d.Submit(func(e *mtl.Encoder) { r.raster(e, s, &f) })
	return r.sent(s, p, err)
}

// Present is Submit, and then the frame goes on screen: a second kernel
// copies it into the layer's next drawable, which is shown once the GPU is
// done. The layer must be the renderer's size (NewLayer makes one; Resize
// both together). Wait for it as for Submit; Pixels still holds the frame.
//
// It waits for a free drawable first. A layer has three, so a caller that
// presents faster than the display shows them is held back here, up to the
// one second CoreAnimation allows before it gives up (mtl.ErrNoDrawable).
func (r *Renderer) Present(l *gpu2d.List, layer *mtl.Layer) error {
	if w, h := layer.Size(); w != r.w || h != r.h {
		return ErrSize
	}
	s, f, err := r.pack(l)
	if err != nil {
		return err
	}
	p, err := r.d.Present(layer, func(e *mtl.Encoder, t *mtl.Texture) {
		r.raster(e, s, &f)
		e.Use(r.show)
		e.Buffer(3, r.target)
		mtl.Constant(e, 4, &f)
		e.Texture(0, t)
		e.Dispatch(r.w, r.h)
		if afterPresent != nil {
			afterPresent(e, t)
		}
	})
	return r.sent(s, p, err)
}

// afterPresent lets a test read the drawable back in the same submission,
// the only moment its texture exists.
var afterPresent func(*mtl.Encoder, *mtl.Texture)

// NewLayer makes a CAMetalLayer of the renderer's size on the renderer's
// GPU, for a view to host and Present to draw into.
func (r *Renderer) NewLayer() (*mtl.Layer, error) { return r.d.NewLayer(r.w, r.h) }

// Resize changes the frame to w × h, waiting first for any frame in flight.
// The frame's content is not kept: write the ground again before drawing.
func (r *Renderer) Resize(w, h int) error {
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return gpu2d.ErrSize
	}
	for r.n > 0 {
		_ = r.Wait()
	}
	if w == r.w && h == r.h {
		return nil
	}
	t, err := r.d.NewBuffer(4 * w * h)
	if err != nil {
		return err
	}
	r.target.Close()
	r.target, r.w, r.h = t, w, h
	return nil
}

// raster records the drawing of slot s.
func (r *Renderer) raster(e *mtl.Encoder, s *slot, f *frame) {
	e.Use(r.pipe)
	e.Buffer(0, s.inst)
	e.Buffer(1, s.spans)
	e.Buffer(2, s.list)
	e.Buffer(3, r.target)
	mtl.Constant(e, 4, f)
	e.Buffer(5, s.data)
	e.Dispatch(r.w, r.h)
}

// sent records a submission in flight.
func (r *Renderer) sent(s *slot, p *mtl.Pending, err error) error {
	if err != nil {
		return err
	}
	s.pend = p
	r.n++
	return nil
}

// pack bins the list and copies it into the next free slot.
func (r *Renderer) pack(l *gpu2d.List) (*slot, frame, error) {
	if r.n == slots {
		return nil, frame{}, ErrBusy
	}
	if w, h := l.Size(); w != r.w || h != r.h {
		return nil, frame{}, ErrSize
	}
	s := &r.slot[(r.head+r.n)%slots]
	var err error
	if s.spanIdx, s.listIdx, err = l.Bin(s.spanIdx, s.listIdx); err != nil {
		return nil, frame{}, err
	}
	s.bytes = gpu2d.AppendBytes(s.bytes[:0], l.Instances())
	if s.inst, err = r.fit(s.inst, len(s.bytes)); err != nil {
		return nil, frame{}, err
	}
	if s.spans, err = r.fit(s.spans, 4*len(s.spanIdx)); err != nil {
		return nil, frame{}, err
	}
	if s.list, err = r.fit(s.list, 4*len(s.listIdx)); err != nil {
		return nil, frame{}, err
	}
	if s.data, err = r.fit(s.data, len(l.Data())); err != nil {
		return nil, frame{}, err
	}
	copy(s.data.Bytes(), l.Data())
	copy(s.inst.Bytes(), s.bytes)
	putWords(s.spans.Bytes(), s.spanIdx)
	putWords(s.list.Bytes(), s.listIdx)
	across, _ := l.Tiles()
	return s, frame{W: uint32(r.w), H: uint32(r.h), Across: uint32(across), Tile: gpu2d.Tile}, nil
}

// Wait blocks until the oldest frame in flight has finished and returns the
// GPU's verdict on it; ErrNothingInFlight when there is none.
func (r *Renderer) Wait() error {
	if r.n == 0 {
		return ErrNothingInFlight
	}
	s := &r.slot[r.head]
	err := s.pend.Wait()
	s.pend = nil
	r.head = (r.head + 1) % slots
	r.n--
	return err
}

// InFlight is the number of submitted frames not yet waited for.
func (r *Renderer) InFlight() int { return r.n }

// putWords writes ws into b as little-endian 32-bit words.
func putWords(b []byte, ws []uint32) {
	for i, v := range ws {
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
}

// Close waits for any frame in flight (the GPU must not read memory that is
// being freed), then releases the GPU objects. The renderer must not be used
// after; Close twice is harmless.
func (r *Renderer) Close() {
	for r.n > 0 {
		_ = r.Wait()
	}
	for i := range r.slot {
		s := &r.slot[i]
		for _, b := range []*mtl.Buffer{s.inst, s.spans, s.list, s.data} {
			if b != nil {
				b.Close()
			}
		}
	}
	if r.target != nil {
		r.target.Close()
	}
	if r.pipe != nil {
		r.pipe.Close()
	}
	if r.show != nil {
		r.show.Close()
	}
	if r.lib != nil {
		r.lib.Close()
	}
	r.d.Close()
}
