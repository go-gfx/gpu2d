// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build windows

package d3d11

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"unsafe"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/hlslgen"
	"github.com/go-gfx/gpu2d/internal/scenes"
	"github.com/go-mswin/d3d11/d3dcompiler"
	"golang.org/x/sys/windows"
)

var (
	user32            = windows.NewLazySystemDLL("user32.dll")
	procCreateWindow  = user32.NewProc("CreateWindowExW")
	procDestroyWindow = user32.NewProc("DestroyWindow")
)

// hiddenWindow makes an overlapped window that is never shown: a swap chain
// needs an HWND, not a screen. CI has no desktop to look at.
func hiddenWindow(t testing.TB, w, h int) uintptr {
	t.Helper()
	class, _ := windows.UTF16PtrFromString("STATIC") // a predefined window class
	title, _ := windows.UTF16PtrFromString("gpu2d d3d11 present test")
	const wsOverlappedWindow = 0x00CF0000
	r, _, err := procCreateWindow.Call(0, uintptr(unsafe.Pointer(class)), uintptr(unsafe.Pointer(title)),
		wsOverlappedWindow, 0, 0, uintptr(w), uintptr(h), 0, 0, 0, 0)
	if r == 0 {
		t.Fatalf("CreateWindowExW: %v", err)
	}
	t.Cleanup(func() { procDestroyWindow.Call(r) })
	return r
}

// swapChain makes a swap chain for r on a hidden window, closed before r.
func swapChain(t testing.TB, r *Renderer) *SwapChain {
	t.Helper()
	s, err := r.NewSwapChain(hiddenWindow(t, r.w, r.h))
	if err != nil {
		t.Fatalf("a flip swap chain with UAV back buffers on %s: %v", r.Name(), err)
	}
	t.Cleanup(s.Close)
	return s
}

// putGround makes the device's frame the ground g: Draw of an empty list
// uploads Pixels and reads the result back.
func putGround(t testing.TB, r *Renderer, g []byte) {
	t.Helper()
	copy(r.Pixels(), g)
	empty, _ := gpu2d.NewList(r.w, r.h)
	if err := r.Draw(empty); err != nil {
		t.Fatal(err)
	}
}

// presentParity draws random scenes over both grounds through Present and
// compares the back buffer, read just before the swap chain's Present, with
// gpu2d.Rasterize. It returns the worst |Δ| and the differing channels.
func presentParity(t *testing.T, r *Renderer, s *SwapChain, seeds int) (worst, differing int) {
	t.Helper()
	w, h := r.w, r.h
	got := make([]byte, 4*w*h)
	for _, opaque := range []bool{true, false} {
		g := ground(w, h, opaque)
		for seed := range uint64(seeds) {
			l, _ := gpu2d.NewList(w, h)
			scenes.Random(l, seed, 40)
			want := append([]byte(nil), g...)
			if err := l.Rasterize(want); err != nil {
				t.Fatal(err)
			}
			putGround(t, r, g)
			if err := r.PresentAndRead(l, s, got); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				d := int(want[i]) - int(got[i])
				if d < 0 {
					d = -d
				}
				if d > 0 {
					differing++
				}
				worst = max(worst, d)
			}
		}
	}
	return worst, differing
}

// TestPresentMatchesReference is the D4 proof of gpu2d on WARP: the frame the
// raster kernel leaves on the device is copied into a flip swap chain's back
// buffer by the present kernel, and the back buffer, read through staging just
// before the swap chain presents it, is gpu2d.Rasterize's bytes, on an opaque
// and a translucent ground.
func TestPresentMatchesReference(t *testing.T) {
	const w, h, seeds = 64, 48, 100
	r := renderer(t, w, h)
	s := swapChain(t, r)
	worst, differing := presentParity(t, r, s, seeds)
	t.Logf("%d scenes on each ground through Present on %q: max |Δ| %d, %d channels differ", seeds, r.Name(), worst, differing)
	if worst != 0 {
		t.Fatalf("the back buffer differs from Rasterize: max |Δ| %d, %d channels", worst, differing)
	}
}

// TestBreakCheckPerturbedPresent: the present kernel with red and blue swapped
// (compiled by D3DCompile here) must be caught by the same comparison.
func TestBreakCheckPerturbedPresent(t *testing.T) {
	const old = "float4(p & 255u, (p >> 8) & 255u, (p >> 16) & 255u, p >> 24)"
	if !strings.Contains(PresentHLSL, old) {
		t.Fatal("present.hlsl no longer has the line this check perturbs")
	}
	bad := strings.Replace(PresentHLSL, old, "float4((p >> 16) & 255u, (p >> 8) & 255u, p & 255u, p >> 24)", 1)
	dxbc, err := d3dcompiler.Compile([]byte(bad), "present.hlsl", "present", hlslgen.Target, hlslgen.Flags)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(dxbc, PresentDXBC) {
		t.Fatal("the perturbed kernel compiled to the committed DXBC")
	}
	defer func(k []byte) { presentKernel = k }(presentKernel)
	presentKernel = dxbc
	const w, h = 64, 48
	r := renderer(t, w, h)
	s := swapChain(t, r)
	worst, differing := presentParity(t, r, s, 20)
	t.Logf("red and blue swapped: max |Δ| %d, %d channels differ", worst, differing)
	if worst == 0 {
		t.Fatal("a perturbed present kernel still matched the reference: TestPresentMatchesReference proves nothing")
	}
}

// TestPresentsCompositeOverTheDevicesFrame: two Presents in a row draw over each
// other on the device, with nothing read back between them.
func TestPresentsCompositeOverTheDevicesFrame(t *testing.T) {
	const w, h = 64, 48
	r := renderer(t, w, h)
	s := swapChain(t, r)
	a, b := translucent(w, h, 4), translucent(w, h, 9)
	g := ground(w, h, false)
	want := append([]byte(nil), g...)
	for _, l := range []*gpu2d.List{a, b} {
		if err := l.Rasterize(want); err != nil {
			t.Fatal(err)
		}
	}
	putGround(t, r, g)
	if err := r.Present(a, s); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 4*w*h)
	if err := r.PresentAndRead(b, s, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("the second Present did not draw over the first one's frame")
	}
}

func TestPresentResize(t *testing.T) {
	r := renderer(t, 64, 48)
	s := swapChain(t, r)
	if err := r.Resize(40, 24); err != nil {
		t.Fatal(err)
	}
	if err := s.Resize(40, 24); err != nil {
		t.Fatal(err)
	}
	if w, h := s.Size(); w != 40 || h != 24 {
		t.Fatalf("swap chain size %d x %d", w, h)
	}
	l, _ := gpu2d.NewList(40, 24)
	scenes.Random(l, 7, 40)
	g := ground(40, 24, true)
	want := append([]byte(nil), g...)
	if err := l.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	putGround(t, r, g)
	got := make([]byte, 4*40*24)
	if err := r.PresentAndRead(l, s, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("the resized swap chain's back buffer is not the reference")
	}
	if err := s.Resize(0, 8); !errors.Is(err, gpu2d.ErrSize) {
		t.Fatalf("Resize(0, 8): %v", err)
	}
}

func TestPresentRefusals(t *testing.T) {
	r := renderer(t, 32, 32)
	s := swapChain(t, r)
	l, _ := gpu2d.NewList(32, 32)
	l.FillRect(gpu2d.Rect{W: 32, H: 32}, gpu2d.RGBA{G: 255, A: 255})
	other, _ := gpu2d.NewList(16, 32)
	if err := r.Present(other, s); !errors.Is(err, ErrSize) {
		t.Errorf("a list of another size: %v", err)
	}
	if err := r.PresentAndRead(l, s, make([]byte, 7)); !errors.Is(err, gpu2d.ErrSize) {
		t.Errorf("a short dst: %v", err)
	}
	if err := r.Present(l, nil); !errors.Is(err, ErrClosed) {
		t.Errorf("no swap chain: %v", err)
	}
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	if err := r.Present(l, s); !errors.Is(err, ErrBusy) {
		t.Errorf("Present with a frame in flight: %v", err)
	}
	if err := r.Wait(); err != nil {
		t.Fatal(err)
	}
	if err := r.Present(l, s); err != nil {
		t.Fatal(err)
	}
	r2 := renderer(t, 16, 16) // a swap chain of another size than the renderer's frame
	s2 := swapChain(t, r2)
	if err := r.Present(l, s2); !errors.Is(err, ErrSize) {
		t.Errorf("a swap chain of another size: %v", err)
	}
	s.Close()
	if err := r.Present(l, s); !errors.Is(err, ErrClosed) {
		t.Errorf("a closed swap chain: %v", err)
	}
	if _, err := r.NewSwapChain(0); err == nil {
		t.Error("a swap chain without a window")
	}
	r.Close()
	if _, err := r.NewSwapChain(1); !errors.Is(err, ErrClosed) {
		t.Errorf("NewSwapChain on a closed renderer: %v", err)
	}
	if err := r.Present(l, s2); !errors.Is(err, ErrClosed) {
		t.Errorf("Present on a closed renderer: %v", err)
	}
}
