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
