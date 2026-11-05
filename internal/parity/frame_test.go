// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package parity

import (
	"bytes"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-widgets/painter"
)

// canvas draws gpu2d.WidgetFrame through PixelPainter.
type canvas struct {
	p    *painter.PixelPainter
	w, h int
}

func pr(r gpu2d.Rect) painter.Rect                   { return painter.Rect{X: r.X, Y: r.Y, W: r.W, H: r.H} }
func pk(c gpu2d.RGBA) painter.RGBA                   { return painter.RGBA{R: c.R, G: c.G, B: c.B, A: c.A} }
func (c canvas) Size() (int, int)                    { return c.w, c.h }
func (c canvas) FillRect(r gpu2d.Rect, k gpu2d.RGBA) { c.p.FillRect(pr(r), pk(k)) }
func (c canvas) FillRoundRect(r gpu2d.Rect, rad int, k gpu2d.RGBA) {
	c.p.FillRoundRect(pr(r), rad, pk(k))
}
func (c canvas) StrokeRoundRect(r gpu2d.Rect, rad int, k gpu2d.RGBA, lw int) {
	c.p.StrokeRoundRect(pr(r), rad, pk(k), lw)
}

// The shared benchmark frame is itself byte-identical between the two, so
// the benchmarks below and the GPU back-ends' compare the same pixels.
func TestWidgetFrameIsPixelPainter(t *testing.T) {
	const w, h = 640, 400
	want := make([]byte, 4*w*h)
	gpu2d.WidgetFrame(canvas{painter.NewPixelPainter(want, w, h), w, h})
	l, err := gpu2d.NewList(w, h)
	if err != nil {
		t.Fatal(err)
	}
	gpu2d.WidgetFrame(l)
	got := make([]byte, 4*w*h)
	if err := l.Rasterize(got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(want, got) {
		t.Fatal("WidgetFrame differs between PixelPainter and gpu2d.Rasterize")
	}
}

// BenchmarkPixelPainter is today's CPU renderer on the shared widget frame:
// the production baseline a GPU back-end must beat.
func BenchmarkPixelPainter(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}} {
		b.Run(s.name, func(b *testing.B) {
			buf := make([]byte, 4*s.w*s.h)
			c := canvas{painter.NewPixelPainter(buf, s.w, s.h), s.w, s.h}
			b.SetBytes(int64(len(buf)))
			for b.Loop() {
				gpu2d.WidgetFrame(c)
			}
		})
	}
}
