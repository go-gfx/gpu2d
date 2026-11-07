// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build js && wasm

package webgpu

import (
	"bytes"
	"errors"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
	wg "github.com/go-gfx/webgpu"
)

// What reaches the canvas is the frame, byte for byte: random scenes are
// drawn, presented to a canvas surface, and the canvas texture is read back
// and compared with the renderer's target. (The target is held to the CPU
// reference by TestMatchesReference.)
func TestAPresentedFrameIsTheFrame(t *testing.T) {
	const w, h = 64, 48
	d, ctx := device(t)
	r, err := New(ctx, d, w, h)
	if err != nil {
		t.Fatal(err)
	}
	// The copy path explicitly: CanvasDescriptor follows the browser's
	// preferred format, which may be bgra8unorm (TestABGRACanvasHoldsTheFrame).
	desc := wg.SurfaceDescriptor{Width: w, Height: h, Format: wg.TextureFormatRGBA8Unorm,
		Usage: wg.TextureUsageRenderAttachment | wg.TextureUsageCopyDst | wg.TextureUsageCopySrc}
	s, err := d.NewSurface(desc)
	if errors.Is(err, wg.ErrNoCanvasContext) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer s.Destroy()
	ground := make([]byte, 4*w*h)
	for i := range ground {
		ground[i] = uint8(i * 7)
		if i%4 == 3 {
			ground[i] = 255
		}
	}
	for seed := range uint64(20) {
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
		if !bytes.Equal(shown, frame) {
			for i := range frame {
				if shown[i] != frame[i] {
					p := i / 4
					t.Fatalf("seed %d (%d,%d) channel %d: frame %d, canvas %d", seed, p%w, p/w, i%4, frame[i], shown[i])
				}
			}
		}
		if bm, err := s.Present(); err == nil && bm.Truthy() {
			bm.Call("close") // a standalone surface hands the frame out; this test drops it
		}
	}
}

// The bgra8unorm canvas, the one browsers prefer on macOS and Windows, gets
// the frame drawn by PresentShader; read back, its bytes are the frame's in
// BGRA order.
func TestABGRACanvasHoldsTheFrame(t *testing.T) {
	const w, h = 64, 48
	d, ctx := device(t)
	r, err := New(ctx, d, w, h)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewSurface(wg.SurfaceDescriptor{Width: w, Height: h, Format: wg.TextureFormatBGRA8Unorm,
		Usage: wg.TextureUsageRenderAttachment | wg.TextureUsageCopySrc})
	if errors.Is(err, wg.ErrNoCanvasContext) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer s.Destroy()
	for seed := range uint64(20) {
		l, _ := gpu2d.NewList(w, h)
		scenes.Random(l, seed, 40)
		ground := make([]byte, 4*w*h)
		for i := range ground {
			ground[i] = uint8(i*11 + int(seed))
			if i%4 == 3 {
				ground[i] = 255
			}
		}
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
			b, g, rr, a := shown[i], shown[i+1], shown[i+2], shown[i+3]
			if rr != frame[i] || g != frame[i+1] || b != frame[i+2] || a != frame[i+3] {
				p := i / 4
				t.Fatalf("seed %d (%d,%d): frame RGBA %v, canvas BGRA %v", seed, p%w, p/w, frame[i:i+4], shown[i:i+4])
			}
		}
		if bm, err := s.Present(); err == nil && bm.Truthy() {
			bm.Call("close")
		}
	}
}

// CanvasDescriptor follows the browser's preferred canvas format, with the
// usage Present needs for it.
func TestCanvasDescriptorIsThePreferredFormat(t *testing.T) {
	nav, err := wg.Navigator()
	if err != nil {
		t.Skip(err)
	}
	desc := CanvasDescriptor()
	switch pref := nav.PreferredCanvasFormat(); pref {
	case wg.TextureFormatBGRA8Unorm:
		if desc.Format != wg.TextureFormatBGRA8Unorm {
			t.Fatalf("preferred %s, descriptor %s", pref, desc.Format)
		}
	default:
		if desc.Format != wg.TextureFormatRGBA8Unorm || desc.Usage&wg.TextureUsageCopyDst == 0 {
			t.Fatalf("preferred %s, descriptor %s usage %#x: want rgba8unorm with COPY_DST", pref, desc.Format, desc.Usage)
		}
	}
	t.Logf("preferred canvas format %s", nav.PreferredCanvasFormat())
}

func TestPresentRefusals(t *testing.T) {
	const w, h = 16, 16
	d, ctx := device(t)
	r, err := New(ctx, d, w, h)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		desc wg.SurfaceDescriptor
		want error
	}{
		{"another size", func() wg.SurfaceDescriptor { c := CanvasDescriptor(); c.Width, c.Height = w+1, h; return c }(), ErrSize},
		{"a float canvas", wg.SurfaceDescriptor{Width: w, Height: h, Format: "rgba16float", Usage: wg.TextureUsageRenderAttachment | wg.TextureUsageCopyDst}, ErrSurface},
		{"no COPY_DST", wg.SurfaceDescriptor{Width: w, Height: h, Format: wg.TextureFormatRGBA8Unorm}, ErrSurface},
	} {
		s, err := d.NewSurface(tc.desc)
		if errors.Is(err, wg.ErrNoCanvasContext) {
			t.Skip(err)
		}
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if err := r.Present(s); !errors.Is(err, tc.want) {
			t.Errorf("%s: Present = %v, want %v", tc.name, err, tc.want)
		}
		s.Destroy()
	}
}
