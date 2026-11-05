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

// Renderer draws lists into a w × h RGBA frame in GPU-visible memory.
type Renderer struct {
	d                 *mtl.Device
	lib               *mtl.Library
	pipe              *mtl.Pipeline
	w, h              int
	target            *mtl.Buffer
	inst, spans, list *mtl.Buffer
	bytes             []byte
	spanIdx, listIdx  []uint32
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
// the GPU to finish.
func (r *Renderer) Draw(l *gpu2d.List) error {
	if w, h := l.Size(); w != r.w || h != r.h {
		return ErrSize
	}
	var err error
	if r.spanIdx, r.listIdx, err = l.Bin(r.spanIdx, r.listIdx); err != nil {
		return err
	}
	r.bytes = gpu2d.AppendBytes(r.bytes[:0], l.Instances())
	if r.inst, err = r.fit(r.inst, len(r.bytes)); err != nil {
		return err
	}
	if r.spans, err = r.fit(r.spans, 4*len(r.spanIdx)); err != nil {
		return err
	}
	if r.list, err = r.fit(r.list, 4*len(r.listIdx)); err != nil {
		return err
	}
	copy(r.inst.Bytes(), r.bytes)
	putWords(r.spans.Bytes(), r.spanIdx)
	putWords(r.list.Bytes(), r.listIdx)
	across, _ := l.Tiles()
	f := frame{W: uint32(r.w), H: uint32(r.h), Across: uint32(across), Tile: gpu2d.Tile}
	return r.d.Run(func(e *mtl.Encoder) {
		e.Use(r.pipe)
		e.Buffer(0, r.inst)
		e.Buffer(1, r.spans)
		e.Buffer(2, r.list)
		e.Buffer(3, r.target)
		mtl.Constant(e, 4, &f)
		e.Dispatch(r.w, r.h)
	})
}

// putWords writes ws into b as little-endian 32-bit words.
func putWords(b []byte, ws []uint32) {
	for i, v := range ws {
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
}

// Close releases the GPU objects; the renderer must not be used after.
func (r *Renderer) Close() {
	for _, b := range []*mtl.Buffer{r.inst, r.spans, r.list, r.target} {
		if b != nil {
			b.Close()
		}
	}
	if r.pipe != nil {
		r.pipe.Close()
	}
	if r.lib != nil {
		r.lib.Close()
	}
	r.d.Close()
}
