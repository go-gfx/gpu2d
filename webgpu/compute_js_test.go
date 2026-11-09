// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build js && wasm

package webgpu

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
	wg "github.com/go-gfx/webgpu"
)

// grounds returns an opaque ground (every window surface) and a translucent
// one, w × h, from the same bytes as TestMatchesReference.
func grounds(w, h int) (opaque, translucent []byte) {
	opaque = make([]byte, 4*w*h)
	translucent = make([]byte, 4*w*h)
	for i := range opaque {
		opaque[i] = uint8(i * 7)
		translucent[i] = uint8(i * 7)
		if i%4 == 3 {
			opaque[i] = 255
		}
	}
	return opaque, translucent
}

func (d diff) total() int { return d.differing[0] + d.differing[1] + d.differing[2] + d.differing[3] }

// TestComputeMatchesReference is the K0 proof of docs/NATIVE.md: the compute
// kernel draws byte for byte what Rasterize draws, over an opaque AND a
// translucent ground. Zero differing channels, where the fixed-function path
// differs by up to 4 in alpha over the second.
func TestComputeMatchesReference(t *testing.T) {
	d, ctx := device(t)
	for _, sz := range []struct{ w, h, seeds int }{
		{64, 48, 200}, // the fixed-function test's frame
		{129, 67, 60}, // not a multiple of the 16 × 8 workgroup nor of the tile
		{7, 5, 30},    // smaller than a tile
		{1, 1, 30},    // one pixel
	} {
		r, err := NewCompute(ctx, d, sz.w, sz.h)
		if err != nil {
			t.Fatal(err)
		}
		opaque, translucent := grounds(sz.w, sz.h)
		for _, g := range []struct {
			name   string
			ground []byte
		}{{"opaque ground", opaque}, {"translucent ground", translucent}} {
			t.Run(fmtSize(sz.w, sz.h)+"/"+g.name, func(t *testing.T) {
				m := measure(t, ctx, r, g.ground, sz.seeds, false)
				m.report(t)
				m.drawsEnough(t, 2)
				if n := m.total(); n != 0 {
					t.Errorf("%d differing channels (max |Δ| %v); first: %s", n, m.worst, m.firstBad)
				}
			})
		}
		r.Destroy()
	}
}

func fmtSize(w, h int) string { return fmt.Sprintf("%dx%d", w, h) }

// The parity check can fail: a kernel whose blend is perturbed differs from
// Rasterize. Each perturbation breaks one rule of PixelPainter's blend, and
// the comparison must see it (on a translucent ground, where it shows).
func TestComputeBreakCheck(t *testing.T) {
	d, ctx := device(t)
	const w, h = 64, 48
	opaque, translucent := grounds(w, h)
	for _, p := range []struct {
		name, from, to string
		opaqueToo      bool // the rule is also visible over an opaque ground
	}{
		{"colour rounds half up differently", "+ 127u) / 255u", "+ 128u) / 255u", true},
		{"alpha rounds instead of flooring", "ca + dst.w * ia / 255u", "ca + (dst.w * ia + 127u) / 255u", false},
	} {
		if strings.Count(gpu2d.KernelWGSL, p.from) == 0 {
			t.Fatalf("%s: %q is not in the kernel; the break check breaks nothing", p.name, p.from)
		}
		r, err := newCompute(ctx, d, w, h, strings.ReplaceAll(gpu2d.KernelWGSL, p.from, p.to))
		if err != nil {
			t.Fatal(err)
		}
		on := func(ground []byte) diff { return measure(t, ctx, r, ground, 40, false) }
		if m := on(translucent); m.total() == 0 {
			t.Errorf("%s: the translucent-ground parity did not notice", p.name)
		} else {
			t.Logf("%s: translucent ground, %d differing channels (max %v)", p.name, m.total(), m.worst)
		}
		m := on(opaque)
		if p.opaqueToo && m.total() == 0 {
			t.Errorf("%s: the opaque-ground parity did not notice", p.name)
		}
		t.Logf("%s: opaque ground, %d differing channels (max %v)", p.name, m.total(), m.worst)
		r.Destroy()
	}
}

// What reaches a canvas of either format is the frame, byte for byte (BGRA
// order for the bgra canvas), without the frame leaving the GPU.
func TestComputePresentedFrameIsTheFrame(t *testing.T) {
	const w, h = 64, 48
	d, ctx := device(t)
	r, err := NewCompute(ctx, d, w, h)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()
	for _, f := range []wg.TextureFormat{wg.TextureFormatRGBA8Unorm, wg.TextureFormatBGRA8Unorm} {
		s, err := d.NewSurface(wg.SurfaceDescriptor{Width: w, Height: h, Format: f,
			Usage: wg.TextureUsageRenderAttachment | wg.TextureUsageCopySrc})
		if errors.Is(err, wg.ErrNoCanvasContext) {
			t.Skip(err)
		}
		if err != nil {
			t.Fatal(err)
		}
		ground, _ := grounds(w, h)
		for seed := range uint64(10) {
			l, _ := gpu2d.NewList(w, h)
			scenes.Random(l, seed, 40)
			if err := r.Draw(l, ground); err != nil {
				t.Fatal(err)
			}
			if err := r.Present(s); err != nil {
				t.Fatal(err)
			}
			tex, err := s.CurrentTexture()
			if err != nil {
				t.Fatal(err)
			}
			shown, err := d.ReadTexture(ctx, wg.TexelCopyTextureInfo{Texture: tex}, w, h, 4)
			if err != nil {
				t.Fatal(err)
			}
			frame, err := r.Read(ctx)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < len(frame); i += 4 {
				want := frame[i : i+4]
				if f == wg.TextureFormatBGRA8Unorm {
					want = []byte{frame[i+2], frame[i+1], frame[i], frame[i+3]}
				}
				if got := shown[i : i+4]; string(got) != string(want) {
					p := i / 4
					t.Fatalf("%s seed %d (%d,%d): want %v, canvas %v", f, seed, p%w, p/w, want, got)
				}
			}
			if bm, err := s.Present(); err == nil && bm.Truthy() {
				bm.Call("close")
			}
		}
		s.Destroy()
	}
}

// The compute mode keeps New's contracts: a list of another size is ErrSize, a
// canvas it cannot draw into is ErrSurface, the ground replaces and no ground
// composites over the content, a frame with nothing drawn keeps the ground, a
// buffer that grows (more instances, a bigger arena) keeps working.
func TestComputeContracts(t *testing.T) {
	d, ctx := device(t)
	if _, err := NewCompute(ctx, d, 0, 8); !errors.Is(err, gpu2d.ErrSize) {
		t.Fatalf("a zero width: %v", err)
	}
	r, err := NewCompute(ctx, d, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()
	if r.Texture() != nil {
		t.Error("a compute renderer has a texture")
	}
	big, _ := gpu2d.NewList(9, 8)
	if err := r.Draw(big, nil); err != ErrSize {
		t.Fatalf("a list of another size: %v", err)
	}
	s, err := d.NewSurface(wg.SurfaceDescriptor{Width: 8, Height: 8, Format: "rgba16float", Usage: wg.TextureUsageRenderAttachment})
	if errors.Is(err, wg.ErrNoCanvasContext) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Present(s); !errors.Is(err, ErrSurface) {
		t.Errorf("a float canvas: %v", err)
	}
	s.Destroy()
	ground := make([]byte, 256)
	for i := range ground {
		ground[i] = 9
	}
	empty, _ := gpu2d.NewList(8, 8)
	if err := r.Draw(empty, ground); err != nil {
		t.Fatal(err)
	}
	got, err := r.Read(ctx)
	if err != nil || string(got) != string(ground) {
		t.Fatalf("an empty frame over a ground: %v, %v", got[:8], err)
	}
	// Growth: many instances and masks, against the CPU.
	l, _ := gpu2d.NewList(8, 8)
	for i := range 300 {
		l.FillRect(gpu2d.Rect{X: i % 5, Y: i % 7, W: 3, H: 2}, gpu2d.RGBA{R: uint8(i), G: 40, B: 200, A: uint8(100 + i%100)})
	}
	want := append([]byte(nil), ground...)
	if err := l.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	if err := r.Draw(l, ground); err != nil {
		t.Fatal(err)
	}
	if got, err = r.Read(ctx); err != nil || string(got) != string(want) {
		t.Fatalf("300 instances differ from the CPU (%v)", err)
	}
	// No ground: the next list composites over what is there.
	if err := l.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	if err := r.Draw(l, nil); err != nil {
		t.Fatal(err)
	}
	if got, err = r.Read(ctx); err != nil || string(got) != string(want) {
		t.Fatalf("no ground: differs from the CPU (%v)", err)
	}
}
