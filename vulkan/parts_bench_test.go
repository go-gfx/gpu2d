// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package vulkan

import (
	"testing"
	"time"

	"github.com/go-gfx/gpu2d"
)

// BenchmarkFrameParts times one WidgetFrame frame in four parts through the
// public path, so it runs unchanged on every layout of the target:
//
//	pack-ns    binning and the copies into the slot (pack, no submission)
//	submit-ns  Submit: pack again, record, queue (the CPU's share)
//	wait-ns    Wait: the device drawing, up to the fence
//	read-ns    Pixels and a copy of the frame into a Go slice, which is what a
//	           consumer of the result pays (the readback of a device-local
//	           target happens inside Pixels)
//
// "ns" because the metrics are per op, and the four add up to a Draw plus the
// read. The pack is measured on its own, so submit-ns contains one more.
func BenchmarkFrameParts(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}} {
		b.Run(s.name, func(b *testing.B) {
			r := renderer(b, s.w, s.h)
			l, _ := gpu2d.NewList(s.w, s.h)
			gpu2d.WidgetFrame(l)
			sink := make([]byte, 4*s.w*s.h)
			var pack, submit, wait, read time.Duration
			for b.Loop() {
				t0 := time.Now()
				if _, _, err := r.pack(l); err != nil {
					b.Fatal(err)
				}
				t1 := time.Now()
				if err := r.Submit(l); err != nil {
					b.Fatal(err)
				}
				t2 := time.Now()
				if err := r.Wait(); err != nil {
					b.Fatal(err)
				}
				t3 := time.Now()
				copy(sink, r.Pixels())
				t4 := time.Now()
				pack += t1.Sub(t0)
				submit += t2.Sub(t1)
				wait += t3.Sub(t2)
				read += t4.Sub(t3)
			}
			n := float64(b.N)
			b.ReportMetric(float64(pack.Nanoseconds())/n, "pack-ns")
			b.ReportMetric(float64(submit.Nanoseconds())/n, "submit-ns")
			b.ReportMetric(float64(wait.Nanoseconds())/n, "wait-ns")
			b.ReportMetric(float64(read.Nanoseconds())/n, "read-ns")
		})
	}
}

// BenchmarkWidgetFrameSync is Draw and Sync: the frame is on the host (the
// download), and not read.
func BenchmarkWidgetFrameSync(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}} {
		b.Run(s.name, func(b *testing.B) {
			r := renderer(b, s.w, s.h)
			l, _ := gpu2d.NewList(s.w, s.h)
			gpu2d.WidgetFrame(l)
			b.ReportAllocs()
			for b.Loop() {
				if err := r.Draw(l); err != nil {
					b.Fatal(err)
				}
				if err := r.Sync(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkWidgetFrameRead is BenchmarkWidgetFrame plus the read of the result
// into a Go slice: Draw, then Pixels copied out. It is the figure to set beside
// gpu2d.Rasterize, which leaves its result in host memory.
func BenchmarkWidgetFrameRead(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}} {
		b.Run(s.name, func(b *testing.B) {
			r := renderer(b, s.w, s.h)
			l, _ := gpu2d.NewList(s.w, s.h)
			gpu2d.WidgetFrame(l)
			sink := make([]byte, 4*s.w*s.h)
			b.ReportAllocs()
			for b.Loop() {
				if err := r.Draw(l); err != nil {
					b.Fatal(err)
				}
				copy(sink, r.Pixels())
			}
		})
	}
}
