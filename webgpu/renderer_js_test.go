// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build js && wasm

package webgpu

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
	wg "github.com/go-gfx/webgpu"
)

func device(t *testing.T) (*wg.Device, context.Context) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	nav, err := wg.Navigator()
	if err != nil {
		t.Fatal(err)
	}
	a, err := nav.RequestAdapter(ctx, &wg.RequestAdapterOptions{AllowFallback: os.Getenv("WEBGPU_ALLOW_FALLBACK") == "1"})
	if err != nil {
		t.Fatalf("adapter: %v", err)
	}
	info := a.Info()
	t.Logf("adapter: %q %q, fallback %v", info.Vendor, info.Description, info.Fallback)
	if os.Getenv("WEBGPU_REQUIRE_GPU") == "1" && info.Fallback {
		t.Fatal("a fallback adapter where a GPU is required")
	}
	d, err := a.RequestDevice(ctx, &wg.DeviceDescriptor{Label: t.Name()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Destroy)
	if err := wg.InstallReplay(ctx, ""); err != nil {
		t.Fatal(err)
	}
	return d, ctx
}

// TestMatchesReference renders random scenes on the GPU and on the CPU from
// the same ground. Over an opaque ground, which is every window surface
// (PixelPainter keeps an opaque surface opaque), every channel must be within
// 1. Over a translucent ground the colour channels must be; alpha is
// reported, since PixelPainter floors a + dst·(255−a)/255 where fixed-function
// blending rounds, and the floors add up over stacked translucent paint.
func TestMatchesReference(t *testing.T) {
	const w, h, seeds = 64, 48, 200
	d, ctx := device(t)
	r, err := New(ctx, d, w, h)
	if err != nil {
		t.Fatal(err)
	}
	opaque := make([]byte, 4*w*h)
	translucent := make([]byte, 4*w*h)
	for i := range opaque {
		opaque[i] = uint8(i * 7)
		translucent[i] = uint8(i * 7)
		if i%4 == 3 {
			opaque[i] = 255
		}
	}
	for _, g := range []struct {
		name      string
		ground    []byte
		alphaFree bool
	}{{"opaque ground", opaque, false}, {"translucent ground", translucent, true}} {
		t.Run(g.name, func(t *testing.T) { compare(t, ctx, r, g.ground, seeds, g.alphaFree) })
	}
}

// diff is what a comparison against Rasterize found, per channel R, G, B, A.
type diff struct {
	worst, differing [4]int
	channels, drawn  int
	firstBad         string // the first channel off by more than 1, for a message
	seeds            int
}

func compare(t *testing.T, ctx context.Context, r *Renderer, ground []byte, seeds int, alphaFree bool) {
	t.Helper()
	d := measure(t, ctx, r, ground, seeds, alphaFree)
	d.report(t)
	d.drawsEnough(t, 10)
	if d.firstBad != "" {
		t.Error(d.firstBad)
	}
}

func (d diff) report(t *testing.T) {
	t.Helper()
	for ch, name := range []string{"R", "G", "B", "A"} {
		t.Logf("%s: max |Δ| %d, %d of %d pixels differ (%.4f%%)", name, d.worst[ch], d.differing[ch], d.channels,
			100*float64(d.differing[ch])/float64(d.channels))
	}
	t.Logf("%d scenes, %d instances", d.seeds, d.drawn)
}

// drawsEnough fails a comparison whose scenes barely draw: it proves nothing.
func (d diff) drawsEnough(t *testing.T, perScene int) {
	t.Helper()
	if d.drawn < d.seeds*perScene {
		t.Fatalf("only %d instances in %d scenes: the scenes barely draw", d.drawn, d.seeds)
	}
}

// measure renders the random scenes on r and on the CPU from the same ground.
// alphaFree excuses alpha from firstBad.
func measure(t *testing.T, ctx context.Context, r *Renderer, ground []byte, seeds int, alphaFree bool) diff {
	t.Helper()
	w, h := r.w, r.h
	out := diff{seeds: seeds}
	worst, differing := &out.worst, &out.differing
	channels, drawn := &out.channels, &out.drawn
	for seed := range uint64(seeds) {
		l, _ := gpu2d.NewList(w, h)
		scenes.Random(l, seed, 40)
		*drawn += len(l.Instances())
		want := append([]byte(nil), ground...)
		if err := l.Rasterize(want); err != nil {
			t.Fatal(err)
		}
		if err := r.Draw(l, ground); err != nil {
			t.Fatal(err)
		}
		got, err := r.Read(ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			ch := i % 4
			if ch == 0 {
				*channels++
			}
			dlt := int(want[i]) - int(got[i])
			if dlt < 0 {
				dlt = -dlt
			}
			if dlt > 0 {
				differing[ch]++
			}
			if dlt > worst[ch] {
				worst[ch] = dlt
				if dlt > 1 && !(alphaFree && ch == 3) && out.firstBad == "" {
					p := i / 4
					out.firstBad = fmt.Sprintf("seed %d (%d,%d) channel %d: CPU %d, GPU %d", seed, p%w, p/w, ch, want[i], got[i])
				}
			}
		}
	}
	return out
}

// The shared widget frame on the GPU, and the same frame on the CPU in wasm:
// Draw plus the wait for the GPU to finish, so the time is a whole frame's.
// The list is recorded once outside the loop; BenchmarkRecord in the root
// package measures recording.
func BenchmarkWidgetFrame(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}} {
		l, _ := gpu2d.NewList(s.w, s.h)
		gpu2d.WidgetFrame(l)
		for _, m := range []struct {
			name string
			new  func(context.Context, *wg.Device, int, int) (*Renderer, error)
		}{{"gpu", New}, {"compute", NewCompute}} {
			b.Run(m.name+"/"+s.name, func(b *testing.B) {
				d, ctx := benchDevice(b)
				r, err := m.new(ctx, d, s.w, s.h)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportMetric(float64(len(l.Instances())), "instances")
				for b.Loop() {
					if err := r.Draw(l, nil); err != nil {
						b.Fatal(err)
					}
					if err := d.Queue().OnSubmittedWorkDone(ctx); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
		b.Run("cpu-wasm/"+s.name, func(b *testing.B) {
			buf := make([]byte, 4*s.w*s.h)
			for b.Loop() {
				_ = l.Rasterize(buf)
			}
		})
	}
}

func benchDevice(b *testing.B) (*wg.Device, context.Context) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	b.Cleanup(cancel)
	nav, err := wg.Navigator()
	if err != nil {
		b.Fatal(err)
	}
	// A software adapter is accepted only to be recognised and skipped:
	// without AllowFallback, RequestAdapter refuses it with an error.
	a, err := nav.RequestAdapter(ctx, &wg.RequestAdapterOptions{AllowFallback: true})
	if err != nil {
		b.Fatal(err)
	}
	if a.Info().Fallback {
		b.Skip("fallback adapter: no GPU to measure")
	}
	d, err := a.RequestDevice(ctx, &wg.DeviceDescriptor{Label: b.Name()})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(d.Destroy)
	if err := wg.InstallReplay(ctx, ""); err != nil {
		b.Fatal(err)
	}
	return d, ctx
}

// The comparison can fail: a frame drawn without its last instance differs.
func TestComparisonCanFail(t *testing.T) {
	d, ctx := device(t)
	r, err := New(ctx, d, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	l, _ := gpu2d.NewList(8, 8)
	l.FillRect(gpu2d.Rect{X: 2, Y: 2, W: 3, H: 3}, gpu2d.RGBA{R: 255, A: 255})
	if err := r.Draw(l, make([]byte, 256)); err != nil {
		t.Fatal(err)
	}
	got, err := r.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if o := (3*8 + 3) * 4; got[o] != 255 || got[o+3] != 255 {
		t.Fatalf("the filled pixel reads %v", got[o:o+4])
	}
	if got[0] != 0 {
		t.Fatalf("an untouched pixel reads %v", got[0:4])
	}
	big, _ := gpu2d.NewList(9, 8)
	if err := r.Draw(big, nil); err != ErrSize {
		t.Fatalf("a list of another size: %v", err)
	}
}
