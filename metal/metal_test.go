// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package metal

import (
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/go-gfx/gpu2d"
	mtl "github.com/go-macos/metal"
)

// renderer returns a renderer, or skips where there is no Metal GPU.
func renderer(tb testing.TB, w, h int) *Renderer {
	tb.Helper()
	r, err := New(w, h)
	if errors.Is(err, mtl.ErrUnsupported) || errors.Is(err, mtl.ErrNoDevice) {
		tb.Skip(err)
	}
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(r.Close)
	return r
}

// scene records n random primitives: the same mix as internal/parity.
func scene(l *gpu2d.List, seed uint64, n, w, h int) {
	r := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	coord := func(m int) int { return r.IntN(m+16) - 8 }
	size := func() int { return r.IntN(24) }
	col := func() gpu2d.RGBA {
		a := uint8(255)
		if r.IntN(2) == 0 {
			a = uint8(r.IntN(256))
		}
		return gpu2d.RGBA{R: uint8(r.IntN(256)), G: uint8(r.IntN(256)), B: uint8(r.IntN(256)), A: a}
	}
	rect := func() gpu2d.Rect { return gpu2d.Rect{X: coord(w), Y: coord(h), W: size(), H: size()} }
	for range n {
		switch r.IntN(12) {
		case 0, 1, 2:
			l.FillRect(rect(), col())
		case 3, 4:
			l.FillRoundRect(rect(), r.IntN(16)-2, col())
		case 5:
			l.StrokeRect(rect(), col(), r.IntN(6))
		case 6, 7:
			l.StrokeRoundRect(rect(), r.IntN(16)-2, col(), r.IntN(6))
		case 8:
			c := rect()
			c.W += 16
			c.H += 16
			l.PushClip(c)
		case 9:
			l.PushTranslate(r.IntN(9)-4, r.IntN(9)-4)
		case 10:
			l.PopClip()
		default:
			l.PopTranslate()
		}
	}
}

// TestMatchesReference holds the GPU to gpu2d.Rasterize on random scenes over
// an opaque and a translucent ground. The blend is the reference's integer
// arithmetic, so the only possible difference is coverage in f32 against
// f64; the counts are reported and anything past 1 fails.
func TestMatchesReference(t *testing.T) {
	const w, h, seeds = 64, 48, 300
	r := renderer(t, w, h)
	t.Logf("GPU: %s", r.Name())
	for _, opaque := range []bool{true, false} {
		ground := make([]byte, 4*w*h)
		for i := range ground {
			ground[i] = uint8(i * 7)
			if opaque && i%4 == 3 {
				ground[i] = 255
			}
		}
		var worst, differing, drawn int
		for seed := range uint64(seeds) {
			l, _ := gpu2d.NewList(w, h)
			scene(l, seed, 40, w, h)
			drawn += len(l.Instances())
			want := append([]byte(nil), ground...)
			if err := l.Rasterize(want); err != nil {
				t.Fatal(err)
			}
			copy(r.Pixels(), ground)
			if err := r.Draw(l); err != nil {
				t.Fatal(err)
			}
			got := r.Pixels()
			for i := range want {
				d := int(want[i]) - int(got[i])
				if d < 0 {
					d = -d
				}
				if d > 0 {
					differing++
				}
				if d > worst {
					worst = d
				}
				if d > 1 {
					p := i / 4
					t.Fatalf("opaque=%v seed %d (%d,%d) channel %d: CPU %d, GPU %d", opaque, seed, p%w, p/w, i%4, want[i], got[i])
				}
			}
		}
		t.Logf("opaque ground %v: %d scenes, %d instances, max |Δ| %d, %d of %d channels differ",
			opaque, seeds, drawn, worst, differing, seeds*4*w*h)
		if drawn < seeds*10 {
			t.Fatalf("only %d instances: the scenes barely draw", drawn)
		}
	}
}

func TestRefusals(t *testing.T) {
	if _, err := New(0, 1); !errors.Is(err, gpu2d.ErrSize) {
		t.Fatalf("New(0, 1): %v", err)
	}
	r := renderer(t, 8, 8)
	l, _ := gpu2d.NewList(9, 8)
	if err := r.Draw(l); !errors.Is(err, ErrSize) {
		t.Fatalf("a list of another size: %v", err)
	}
}

// BenchmarkWidgetFrame is the shared widget frame on the GPU: binning,
// upload into shared memory, the dispatch and the wait for it.
func BenchmarkWidgetFrame(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}} {
		b.Run(s.name, func(b *testing.B) {
			r := renderer(b, s.w, s.h)
			l, _ := gpu2d.NewList(s.w, s.h)
			gpu2d.WidgetFrame(l)
			b.ReportMetric(float64(len(l.Instances())), "instances")
			for b.Loop() {
				if err := r.Draw(l); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
