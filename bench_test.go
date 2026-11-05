// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import "testing"

func TestWidgetFrameDraws(t *testing.T) {
	l := nl(1920, 1080)
	WidgetFrame(l)
	if n := len(l.Instances()); n < 500 {
		t.Fatalf("%d instances", n)
	}
}

// BenchmarkRecord is the CPU cost of describing the widget frame as a list,
// which a GPU back-end pays before it uploads anything.
func BenchmarkRecord(b *testing.B) {
	l := nl(3840, 2160)
	for b.Loop() {
		l.Reset()
		WidgetFrame(l)
	}
	b.ReportMetric(float64(len(l.Instances())), "instances")
}

// BenchmarkRasterize is the CPU reference, which is PixelPainter byte for
// byte, on the shared widget frame: the time a GPU back-end has to beat.
func BenchmarkRasterize(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}} {
		b.Run(s.name, func(b *testing.B) {
			l := nl(s.w, s.h)
			WidgetFrame(l)
			buf := make([]byte, 4*s.w*s.h)
			b.ReportMetric(float64(len(l.Instances())), "instances")
			b.SetBytes(int64(len(buf)))
			for b.Loop() {
				_ = l.Rasterize(buf)
			}
		})
	}
}

// TestSteadyFrameAllocatesNothing holds the per-frame path to zero
// allocations once the storage has grown: Reset, record the widget frame,
// Bin into reused slices, AppendBytes into a reused buffer.
func TestSteadyFrameAllocatesNothing(t *testing.T) {
	if raceEnabled {
		t.Skip("the race detector allocates")
	}
	l := nl(960, 2834)
	var spans, list []uint32
	var buf []byte
	frame := func() {
		l.Reset()
		WidgetFrame(l)
		var err error
		if spans, list, err = l.Bin(spans, list); err != nil {
			t.Fatal(err)
		}
		buf = AppendBytes(buf[:0], l.Instances())
	}
	frame()
	if n := testing.AllocsPerRun(5, frame); n != 0 {
		t.Fatalf("a steady frame allocates %v times", n)
	}
}

// BenchmarkBin is the binning of the widget frame into reused slices.
func BenchmarkBin(b *testing.B) {
	l := nl(960, 2834)
	WidgetFrame(l)
	var spans, list []uint32
	b.ReportAllocs()
	for b.Loop() {
		spans, list, _ = l.Bin(spans, list)
	}
}
