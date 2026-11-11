// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package metal

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
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

// TestMatchesReference holds the GPU to gpu2d.Rasterize on random scenes over
// an opaque and a translucent ground. The blend is the reference's integer
// arithmetic and the corner coverage is decided in integers, so the bar is
// zero differing channels; the counts are reported.
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
		kinds := map[gpu2d.Kind]int{}
		for seed := range uint64(seeds) {
			l, _ := gpu2d.NewList(w, h)
			scenes.Random(l, seed, 40)
			drawn += len(l.Instances())
			for _, in := range l.Instances() {
				kinds[in.Kind]++
			}
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
				if d > 0 {
					p := i / 4
					t.Fatalf("opaque=%v seed %d (%d,%d) channel %d: CPU %d, GPU %d", opaque, seed, p%w, p/w, i%4, want[i], got[i])
				}
			}
		}
		t.Logf("opaque ground %v: %d scenes, %d instances, max |Δ| %d, %d of %d channels differ",
			opaque, seeds, drawn, worst, differing, seeds*4*w*h)
		t.Logf("instances by kind (fill, outline, mask, alpha, image): %d %d %d %d %d",
			kinds[gpu2d.KindFill], kinds[gpu2d.KindOutline], kinds[gpu2d.KindMask], kinds[gpu2d.KindAlpha], kinds[gpu2d.KindImage])
		for k := gpu2d.KindFill; k <= gpu2d.KindImage; k++ {
			if kinds[k] < seeds/4 {
				t.Fatalf("kind %d drawn only %d times", k, kinds[k])
			}
		}
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

// translucent returns a list of overlapping translucent fills: the result
// depends on the order frames are composited in.
func translucent(w, h int, shift int) *gpu2d.List {
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{X: shift, Y: 0, W: w / 2, H: h}, gpu2d.RGBA{R: 200, G: 30, B: 10, A: 140})
	l.FillRoundRect(gpu2d.Rect{X: 3, Y: shift, W: w - 6, H: h / 2}, 5, gpu2d.RGBA{R: 10, G: 40, B: 220, A: 120})
	return l
}

// TestTwoFramesInFlightCompositeInOrder submits two lists back to back, with
// the list reset and re-recorded between them (Submit copies), and holds the
// result to the CPU compositing the same two lists in the same order.
func TestTwoFramesInFlightCompositeInOrder(t *testing.T) {
	const w, h = 64, 48
	r := renderer(t, w, h)
	a, b := translucent(w, h, 4), translucent(w, h, 9)
	want := make([]byte, 4*w*h)
	for i := range want {
		want[i] = uint8(i * 5)
	}
	copy(r.Pixels(), want)
	if err := a.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	if err := b.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	if err := r.Submit(a); err != nil {
		t.Fatal(err)
	}
	if err := r.Submit(b); err != nil {
		t.Fatal(err)
	}
	if r.InFlight() != 2 {
		t.Fatalf("%d in flight", r.InFlight())
	}
	if err := r.Submit(a); !errors.Is(err, ErrBusy) {
		t.Fatalf("a third Submit: %v", err)
	}
	if err := r.Draw(a); !errors.Is(err, ErrBusy) {
		t.Fatalf("Draw while in flight: %v", err)
	}
	for range 2 {
		if err := r.Wait(); err != nil {
			t.Fatal(err)
		}
	}
	if r.InFlight() != 0 {
		t.Fatalf("%d in flight after Wait", r.InFlight())
	}
	got := r.Pixels()
	for i := range want {
		if d := int(want[i]) - int(got[i]); d > 1 || d < -1 {
			t.Fatalf("channel %d: CPU %d, GPU %d", i, want[i], got[i])
		}
	}
}

// TestSubmitCopiesTheList records frame A, submits it, overwrites the list
// with frame B at once, and checks the GPU still drew A.
func TestSubmitCopiesTheList(t *testing.T) {
	const w, h = 32, 32
	r := renderer(t, w, h)
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: w, H: h}, gpu2d.RGBA{R: 255, A: 255})
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	l.Reset()
	l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: w, H: h}, gpu2d.RGBA{B: 255, A: 255})
	if err := r.Wait(); err != nil {
		t.Fatal(err)
	}
	if p := r.Pixels(); p[0] != 255 || p[2] != 0 {
		t.Fatalf("pixel 0 = %v: the GPU drew the list as overwritten", p[:4])
	}
}

func TestMisuse(t *testing.T) {
	r := renderer(t, 8, 8)
	if err := r.Wait(); !errors.Is(err, ErrNothingInFlight) {
		t.Fatalf("Wait without Submit: %v", err)
	}
	l, _ := gpu2d.NewList(8, 8)
	l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: 8, H: 8}, gpu2d.RGBA{G: 255, A: 255})
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	if err := r.Submit(l); err != nil { // the second slot: allowed
		t.Fatal(err)
	}
	other, _ := gpu2d.NewList(9, 8)
	if err := r.Submit(other); !errors.Is(err, ErrBusy) {
		t.Fatalf("Submit with both slots busy: %v", err)
	}
	// Close while two frames are in flight must drain, not crash.
	r.Close()
	if r.InFlight() != 0 {
		t.Fatal("frames in flight after Close")
	}
	r.Close() // twice is harmless
	if err := r.Submit(l); err == nil {
		t.Fatal("Submit after Close succeeded")
	}
	if err := r.Draw(l); err == nil {
		t.Fatal("Draw after Close succeeded")
	}
}

func TestSubmitRefusesAWrongSizeAndATooLargeList(t *testing.T) {
	r := renderer(t, 8, 8)
	other, _ := gpu2d.NewList(9, 8)
	if err := r.Submit(other); !errors.Is(err, ErrSize) {
		t.Fatalf("a list of another size: %v", err)
	}
	if r.InFlight() != 0 {
		t.Fatal("a refused Submit is in flight")
	}
}

// BenchmarkOverlap is the point of Submit: per frame the CPU records the next
// list while the GPU draws the last. "draw" is the synchronous baseline,
// "overlap" records the next frame between Submit and Wait. The CPU work is
// modelled as a spin of cpuMs (0 and 1.5 ms: the gallery's recording through
// gpupainter is about 1.7 ms), so the table does not depend on how fast the
// recorder is. Draw costs cpu + GPU; Submit/Wait costs about max(cpu, GPU)
// plus the binning and packing.
func BenchmarkOverlap(b *testing.B) {
	const w, h = 960, 2834
	for _, cpu := range []time.Duration{0, 1500 * time.Microsecond} {
		for _, overlap := range []bool{false, true} {
			name := "draw"
			if overlap {
				name = "overlap"
			}
			b.Run(fmt.Sprintf("cpu%dus/%s", cpu.Microseconds(), name), func(b *testing.B) {
				r := renderer(b, w, h)
				l, _ := gpu2d.NewList(w, h)
				gpu2d.WidgetFrame(l)
				b.ReportAllocs()
				for b.Loop() {
					if !overlap {
						spin(cpu)
						if err := r.Draw(l); err != nil {
							b.Fatal(err)
						}
						continue
					}
					if err := r.Submit(l); err != nil {
						b.Fatal(err)
					}
					spin(cpu) // the next frame, recorded while the GPU draws
					if err := r.Wait(); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// spin keeps one core busy for d.
func spin(d time.Duration) {
	for t := time.Now(); time.Since(t) < d; {
	}
}
