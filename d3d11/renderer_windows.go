// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build windows

package d3d11

import (
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/go-gfx/gpu2d"
	dx "github.com/go-mswin/d3d11"
)

// kernel is the DXBC New builds the shader from, except in the break check,
// which swaps in a kernel with the blend perturbed.
var kernel = KernelDXBC

// The kernel's registers, as kernel_gen.hlsl declares them: SRVs t0..t2, the
// pixels' UAV u3, the frame's constant buffer b4.
const (
	tInst  = 0 // StructuredBuffer<Inst>, stride 64
	tBins  = 1 // StructuredBuffer<uint>: the tiles' spans, then the list
	tArena = 2 // StructuredBuffer<uint>
	uPix   = 3 // RWStructuredBuffer<uint>
	bFrame = 4 // cbuffer Frame { w, h, across, tile }
)

// instStride is the byte size of the kernel's Inst.
const instStride = gpu2d.InstanceSize

// slot is the per-frame input: the buffers the device reads until the frame is
// waited for, and the host scratch they are filled from.
type slot struct {
	inst, bins, arena *dx.Buffer
	uni               *dx.Buffer // the frame constants, updated each Submit
	srv               [3]*dx.SRV
	spans, list       []uint32 // Bin's storage, kept between frames
	ibytes, bbytes    []byte   // padded host images of the buffers
	abytes            []byte
}

// release drops the frame's buffers and views. Direct3D keeps a resource alive
// until the commands that use it have run, so this is safe with a frame in
// flight; it is done at Wait so that memory does not pile up.
func (s *slot) release() {
	for i, v := range s.srv {
		if v != nil {
			v.Close()
			s.srv[i] = nil
		}
	}
	for _, b := range []**dx.Buffer{&s.inst, &s.bins, &s.arena} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}

// Renderer draws lists into a w × h RGBA frame on a Direct3D 11 device.
//
// Submit and Wait are the asynchronous pair; Draw is both. A Renderer is for
// one goroutine at a time (an immediate context belongs to one).
type Renderer struct {
	dev    *dx.Device
	ctx    *dx.Context
	sh     *dx.ComputeShader
	w, h   int
	pix    []byte     // the host copy of the frame: what Pixels returns
	target *dx.Buffer // the device's frame, made from pix when nothing is in flight
	uav    *dx.UAV
	stage  *dx.Buffer // CPU-readable copy of the target
	slot   [slots]slot
	head   int // the oldest frame in flight
	n      int // frames in flight
	closed bool
	stats  Stats
	last   time.Time // end of the last Submit, for Stats.GPU
}

// New opens the default hardware adapter and allocates a w × h frame.
func New(w, h int) (*Renderer, error) { return NewWith(w, h, Options{}) }

// NewWith is New with options.
func NewWith(w, h int, o Options) (*Renderer, error) {
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return nil, gpu2d.ErrSize
	}
	dev, err := dx.CreateDevice(dx.Options{Software: o.Software, MaxBufferBytes: o.MaxBufferBytes})
	if errors.Is(err, dx.ErrUnsupportedPlatform) {
		return nil, fmt.Errorf("%w: %w", ErrUnsupported, err)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrNoDevice, err)
	}
	r := &Renderer{dev: dev, ctx: dev.Context(), w: w, h: h}
	if err = r.check(w, h); err != nil {
		r.Close()
		return nil, err
	}
	if r.sh, err = dev.CreateComputeShader(kernel); err != nil {
		r.Close()
		return nil, tooLarge(err)
	}
	if err = r.alloc(w, h); err != nil {
		r.Close()
		return nil, err
	}
	for i := range r.slot {
		if r.slot[i].uni, err = dev.CreateBuffer(dx.BufferDesc{Kind: dx.KindConstant, Size: 16}); err != nil {
			r.Close()
			return nil, tooLarge(err)
		}
	}
	return r, nil
}

// tooLarge maps the binding's size refusal to gpu2d's, which is what a caller
// (the window) tests for to draw that frame on the CPU.
func tooLarge(err error) error {
	if errors.Is(err, dx.ErrTooLarge) {
		return fmt.Errorf("%w: %w", gpu2d.ErrTooLarge, err)
	}
	return err
}

// check refuses a frame size the device cannot hold or dispatch: the target
// must fit the device's buffer limit (128 MiB guaranteed, less than a 5793²
// frame) and the thread-group grid 65535 groups per dimension. Both can be
// exceeded by a legal list, so the answer is ErrTooLarge, not a failed call.
func (r *Renderer) check(w, h int) error {
	if 4*uint64(w)*uint64(h) > r.dev.MaxBufferBytes() {
		return gpu2d.ErrTooLarge
	}
	if uint64((w+15)/16) > dx.MaxDispatchGroups || uint64((h+7)/8) > dx.MaxDispatchGroups {
		return gpu2d.ErrTooLarge
	}
	return nil
}

// alloc makes the host copy and the staging buffer for w × h, and forgets the
// device's target: the next Submit with nothing in flight makes it from pix.
func (r *Renderer) alloc(w, h int) error {
	st, err := r.dev.CreateBuffer(dx.BufferDesc{Kind: dx.KindStaging, Size: 4 * uint64(w) * uint64(h)})
	if err != nil {
		return tooLarge(err)
	}
	if r.stage != nil {
		r.stage.Close()
	}
	r.dropTarget()
	r.stage, r.pix, r.w, r.h = st, make([]byte, 4*w*h), w, h
	return nil
}

func (r *Renderer) dropTarget() {
	if r.uav != nil {
		r.uav.Close()
		r.uav = nil
	}
	if r.target != nil {
		r.target.Close()
		r.target = nil
	}
}

// Name is the adapter's own name, such as "Microsoft Basic Render Driver".
func (r *Renderer) Name() string { return r.dev.Adapter().Description }

// Software reports whether the device is WARP or another software adapter.
func (r *Renderer) Software() bool { return r.dev.Adapter().Software() }

// FeatureLevel is the Direct3D feature level of the device.
func (r *Renderer) FeatureLevel() dx.FeatureLevel { return r.dev.FeatureLevel() }

// MaxBufferBytes is the buffer size limit frames are checked against.
func (r *Renderer) MaxBufferBytes() uint64 { return r.dev.MaxBufferBytes() }

// Pixels is the frame, 4·w·h bytes of RGBA rows top to bottom: a host copy.
// Write the ground here before the first Submit (it is uploaded then) and read
// the result here after the Wait that ends the last frame in flight (it is
// copied back then). Between them the frame belongs to the device.
func (r *Renderer) Pixels() []byte { return r.pix }

// Stats returns the time spent in each part of the frames since the last
// ResetStats.
func (r *Renderer) Stats() Stats { return r.stats }

// ResetStats zeroes Stats.
func (r *Renderer) ResetStats() { r.stats = Stats{} }

// InFlight is the number of submitted frames not yet waited for.
func (r *Renderer) InFlight() int { return r.n }

// Draw composites the list over the frame's current content and waits for the
// device to finish: Submit followed by Wait. It is ErrBusy while a frame is in
// flight.
func (r *Renderer) Draw(l *gpu2d.List) error {
	if r.closed {
		return ErrClosed
	}
	if r.n > 0 {
		return ErrBusy
	}
	if err := r.Submit(l); err != nil {
		return err
	}
	return r.Wait()
}

// Submit bins and packs the list into the next free slot, sends it to the
// device and returns. The list is copied into device buffers before Submit
// returns: the caller may Reset and record the next frame at once, which is the
// overlap. Frames run in submission order on the one immediate context, each
// compositing over the previous one's result, which stays on the device. At
// most two frames are in flight; a third Submit is ErrBusy.
//
// With nothing in flight, Submit first uploads Pixels as the frame to draw over.
// The frame's memory belongs to the device until Wait has returned for every
// frame in flight.
func (r *Renderer) Submit(l *gpu2d.List) error {
	if r.closed {
		return ErrClosed
	}
	t0 := time.Now()
	s, err := r.pack(l)
	if err != nil {
		return err
	}
	r.stats.Pack += time.Since(t0)
	if r.n == 0 {
		t0 = time.Now()
		if err := r.upload(); err != nil {
			s.release()
			return err
		}
		r.stats.Upload += time.Since(t0)
	}
	across, _ := l.Tiles()
	if err := r.record(s, r.uav, across); err != nil {
		s.release()
		return err
	}
	r.n++
	r.last = time.Now()
	return nil
}

// upload makes the device's target from the host copy.
func (r *Renderer) upload() error {
	r.dropTarget()
	t, err := r.dev.CreateBuffer(dx.BufferDesc{Kind: dx.KindStructured, Size: uint64(len(r.pix)), Stride: 4, UAV: true, Initial: r.pix})
	if err != nil {
		return tooLarge(err)
	}
	u, err := t.UAV()
	if err != nil {
		t.Close()
		return tooLarge(err)
	}
	r.target, r.uav = t, u
	return nil
}

// record binds the kernel's resources and dispatches it, writing into uav.
func (r *Renderer) record(s *slot, uav *dx.UAV, across int) error {
	var f [16]byte
	binary.LittleEndian.PutUint32(f[0:], uint32(r.w))
	binary.LittleEndian.PutUint32(f[4:], uint32(r.h))
	binary.LittleEndian.PutUint32(f[8:], uint32(across))
	binary.LittleEndian.PutUint32(f[12:], gpu2d.Tile)
	c := r.ctx
	for _, err := range []error{
		c.UpdateBuffer(s.uni, f[:]),
		c.SetShader(r.sh),
		c.SetSRVs(tInst, s.srv[0], s.srv[1], s.srv[2]),
		c.SetUAVs(uPix, uav),
		c.SetConstantBuffers(bFrame, s.uni),
		c.Dispatch(uint32((r.w+15)/16), uint32((r.h+7)/8), 1),
	} {
		if err != nil {
			return tooLarge(err)
		}
	}
	return nil
}

// Wait blocks until the oldest frame in flight has finished and returns the
// device's verdict on it; ErrNothingInFlight when there is none. The frames
// share one in-order context, so the device cannot be asked about one frame
// alone: with another frame still queued, Wait only checks the device is
// healthy and gives the frame's buffers back; the Wait that ends the last
// frame blocks until the device is done and copies the frame into Pixels.
func (r *Renderer) Wait() error {
	if r.n == 0 {
		return ErrNothingInFlight
	}
	s := &r.slot[r.head]
	r.head = (r.head + 1) % slots
	r.n--
	defer s.release()
	if r.n > 0 {
		return r.dev.RemovedReason()
	}
	if err := r.ctx.Wait(waitLimit); err != nil {
		return timeout(err)
	}
	r.stats.GPU += time.Since(r.last)
	r.stats.Frames++
	t0 := time.Now()
	defer func() { r.stats.Readback += time.Since(t0) }()
	return r.readback()
}

// readback copies the target through the staging buffer into Pixels.
func (r *Renderer) readback() error {
	if err := r.ctx.CopyBuffer(r.stage, r.target); err != nil {
		return err
	}
	if err := r.ctx.Wait(waitLimit); err != nil {
		return timeout(err)
	}
	m, err := r.ctx.MapRead(r.stage)
	if err != nil {
		return err
	}
	copy(r.pix, m)
	return r.ctx.Unmap(r.stage)
}

func timeout(err error) error {
	if errors.Is(err, dx.ErrTimeout) {
		return fmt.Errorf("%w: %w", ErrTimeout, err)
	}
	return err
}

// Resize changes the frame to w × h, waiting first for any frame in flight.
// The frame's content is not kept: write the ground again before drawing.
func (r *Renderer) Resize(w, h int) error {
	if r.closed {
		return ErrClosed
	}
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return gpu2d.ErrSize
	}
	for r.n > 0 {
		_ = r.Wait()
	}
	if w == r.w && h == r.h {
		return nil
	}
	if err := r.check(w, h); err != nil {
		return err
	}
	return r.alloc(w, h)
}

// pad returns src in dst's storage, zero-extended to a multiple of mult and at
// least min bytes: a structured buffer holds whole elements, and the kernel
// reads whole words.
func pad(dst *[]byte, src []byte, mult, min int) []byte {
	n := (max(len(src), min) + mult - 1) / mult * mult
	b := append((*dst)[:0], src...)
	for len(b) < n {
		b = append(b, 0)
	}
	*dst = b
	return b
}

// putWords appends ws to b as little-endian 32-bit words.
func putWords(b []byte, ws []uint32) []byte {
	for _, v := range ws {
		b = binary.LittleEndian.AppendUint32(b, v)
	}
	return b
}

// pack bins the list and creates the next free slot's buffers with their data.
func (r *Renderer) pack(l *gpu2d.List) (*slot, error) {
	if r.n == slots {
		return nil, ErrBusy
	}
	if w, h := l.Size(); w != r.w || h != r.h {
		return nil, ErrSize
	}
	s := &r.slot[(r.head+r.n)%slots]
	s.release()
	var err error
	s.spans, s.list, err = l.Bin(s.spans, s.list)
	if err != nil {
		return nil, err
	}
	s.ibytes = gpu2d.AppendBytes(s.ibytes[:0], l.Instances())
	inst := pad(&s.ibytes, s.ibytes, instStride, instStride)
	s.bbytes = putWords(putWords(s.bbytes[:0], s.spans), s.list)
	bins := pad(&s.bbytes, s.bbytes, 4, 4)
	arena := pad(&s.abytes, l.Data(), 4, 4)
	ok := false
	defer func() {
		if !ok {
			s.release()
		}
	}()
	for _, b := range []struct {
		buf    **dx.Buffer
		srv    int
		data   []byte
		stride uint32
	}{{&s.inst, 0, inst, instStride}, {&s.bins, 1, bins, 4}, {&s.arena, 2, arena, 4}} {
		if *b.buf, err = r.dev.CreateBuffer(dx.BufferDesc{Kind: dx.KindStructured, Size: uint64(len(b.data)), Stride: b.stride, SRV: true, Initial: b.data}); err != nil {
			return nil, tooLarge(err)
		}
		if s.srv[b.srv], err = (*b.buf).SRV(); err != nil {
			return nil, tooLarge(err)
		}
	}
	ok = true
	return s, nil
}

// Close waits for any frame in flight (the device must not read memory that is
// being freed), then releases the Direct3D objects. The renderer must not be
// used after; Close twice is harmless.
func (r *Renderer) Close() {
	if r.closed {
		return
	}
	for r.n > 0 {
		_ = r.Wait()
	}
	r.closed = true
	for i := range r.slot {
		r.slot[i].release()
		if r.slot[i].uni != nil {
			r.slot[i].uni.Close()
		}
	}
	r.dropTarget()
	if r.stage != nil {
		r.stage.Close()
	}
	if r.sh != nil {
		r.sh.Close()
	}
	if r.dev != nil {
		r.dev.Close()
	}
}
