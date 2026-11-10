// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build windows

package d3d11

import (
	"encoding/binary"
	"fmt"

	"github.com/go-gfx/gpu2d"
	dx "github.com/go-mswin/d3d11"
	"github.com/go-mswin/d3d11/dxgi"
)

// SwapChain is a flip-model swap chain (FLIP_DISCARD, R8G8B8A8_UNORM, two
// buffers) on a window, sized as its Renderer's frame. Close it before the
// Renderer.
type SwapChain struct {
	sc   *dx.SwapChain
	w, h int
	// Vsync makes Present wait for the next vertical blank (sync interval 1);
	// the zero value presents at once.
	Vsync bool
}

// NewSwapChain makes the swap chain for the window hwnd (an HWND), as large as
// the renderer's frame. One flip swap chain per window: a window that also has
// its pixels blitted by GDI shows only the swap chain while it exists. A device
// that cannot make a flip swap chain with UAV back buffers refuses here, and the
// window stays on its CPU path.
func (r *Renderer) NewSwapChain(hwnd uintptr) (*SwapChain, error) {
	if r.closed {
		return nil, ErrClosed
	}
	sc, err := r.dev.CreateSwapChain(hwnd, dxgi.SwapChainDesc{Width: uint32(r.w), Height: uint32(r.h)})
	if err != nil {
		return nil, tooLarge(err)
	}
	return &SwapChain{sc: sc, w: r.w, h: r.h}, nil
}

// Size is the back buffers' size in pixels.
func (s *SwapChain) Size() (w, h int) { return s.w, s.h }

// Resize follows a new frame size: call it after Renderer.Resize. Nothing may be
// in flight and no back buffer held; Present holds none between calls.
func (s *SwapChain) Resize(w, h int) error {
	if s.sc == nil {
		return ErrClosed
	}
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return gpu2d.ErrSize
	}
	if err := s.sc.Resize(uint32(w), uint32(h)); err != nil {
		return tooLarge(err)
	}
	s.w, s.h = w, h
	return nil
}

// Close releases the swap chain. It is idempotent.
func (s *SwapChain) Close() {
	if s != nil && s.sc != nil {
		s.sc.Close()
		s.sc = nil
	}
}

// Present composites the list over the frame on the device and puts the result
// on the window: the same kernel as Draw, then a second one copying the frame
// into the swap chain's back buffer as a UAV, then the swap chain's Present. The
// frame is not copied back to Pixels, and Pixels is not uploaded: the device's
// frame, as the last Draw or Present left it, is what the list draws over (a
// window that records its whole frame on an opaque background does not care).
// Pixels is stale afterwards; a Draw uploads it again.
//
// It is ErrBusy while a Submit is in flight, ErrSize when the list or the swap
// chain is not the renderer's size. The raster dispatch and the copy are queued
// behind each other on the one context, so nothing waits for the device here
// except the swap chain itself when its queue is full.
func (r *Renderer) Present(l *gpu2d.List, s *SwapChain) error { return r.present(l, s, nil) }

// PresentAndRead is Present that also copies the back buffer, as the device
// holds it just before the swap chain's Present, into dst (4·w·h bytes of RGBA
// rows): the proof that what goes on the window is the frame, and a screenshot
// that needs no desktop.
func (r *Renderer) PresentAndRead(l *gpu2d.List, s *SwapChain, dst []byte) error {
	if len(dst) != 4*r.w*r.h {
		return fmt.Errorf("%w: dst is %d bytes, the frame %d", gpu2d.ErrSize, len(dst), 4*r.w*r.h)
	}
	return r.present(l, s, dst)
}

// presentShader makes Present's kernel and constants on first use.
func (r *Renderer) presentShader() error {
	if r.psh != nil {
		return nil
	}
	sh, err := r.dev.CreateComputeShader(presentKernel)
	if err != nil {
		return tooLarge(err)
	}
	cb, err := r.dev.CreateBuffer(dx.BufferDesc{Kind: dx.KindConstant, Size: 16})
	if err != nil {
		sh.Close()
		return tooLarge(err)
	}
	r.psh, r.pcb = sh, cb
	return nil
}

// presentKernel is the DXBC of present.hlsl; the break check swaps it.
var presentKernel = PresentDXBC

func (r *Renderer) present(l *gpu2d.List, s *SwapChain, dst []byte) error {
	if r.closed {
		return ErrClosed
	}
	if s == nil || s.sc == nil {
		return ErrClosed
	}
	if r.n > 0 {
		return ErrBusy
	}
	if w, h := l.Size(); w != r.w || h != r.h || s.w != r.w || s.h != r.h {
		return ErrSize
	}
	if err := r.presentShader(); err != nil {
		return err
	}
	sl, err := r.pack(l)
	if err != nil {
		return err
	}
	across, _ := l.Tiles()
	if err := r.record(sl, r.uav, across); err != nil {
		return err
	}
	// Nothing reads this frame's result on the CPU, and the copy below is queued
	// behind it on the same context: the fence has done its job of flushing.
	sl.fence.Close()
	sl.fence = nil
	if err := r.copyToBackBuffer(s, dst); err != nil {
		return err
	}
	return s.sc.Present(vsyncInterval(s.Vsync))
}

func vsyncInterval(v bool) uint32 {
	if v {
		return 1
	}
	return 0
}

// copyToBackBuffer runs the present kernel from the target into the swap
// chain's current back buffer, optionally reading the buffer back, and leaves
// nothing bound that references it (a bound view would make a later
// ResizeBuffers fail).
func (r *Renderer) copyToBackBuffer(s *SwapChain, dst []byte) error {
	bb, err := s.sc.BackBuffer()
	if err != nil {
		return err
	}
	defer bb.Close()
	uav, err := bb.UAV()
	if err != nil {
		return tooLarge(err)
	}
	defer uav.Close()
	var f [16]byte
	binary.LittleEndian.PutUint32(f[0:], uint32(r.w))
	binary.LittleEndian.PutUint32(f[4:], uint32(r.h))
	c := r.ctx
	defer func() {
		// The target goes back to being the raster kernel's UAV at the next frame:
		// drop the views of this one.
		_ = c.SetUAVs(0, nil)
		_ = c.SetSRVs(0, nil)
	}()
	for _, err := range []error{
		c.UpdateBuffer(r.pcb, f[:]),
		c.SetUAVs(uPix, nil), // the target is an input now, never both bound
		c.SetShader(r.psh),
		c.SetSRVs(0, r.tsrv),
		c.SetUAVs(0, uav),
		c.SetConstantBuffers(0, r.pcb),
		c.Dispatch(uint32((r.w+15)/16), uint32((r.h+7)/8), 1),
	} {
		if err != nil {
			return tooLarge(err)
		}
	}
	if dst != nil {
		px, err := c.ReadbackTexture(bb, waitLimit)
		if err != nil {
			return timeout(err)
		}
		copy(dst, px)
	}
	return nil
}
