// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package parity holds gpu2d.Rasterize to go-widgets/painter's PixelPainter,
// byte for byte. It is its own module so that go-gfx/gpu2d does not depend on
// go-widgets.
package parity

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"github.com/go-gfx/gfx/vector"
	"github.com/go-gfx/gpu2d"
	"github.com/go-widgets/painter"
)

// surface is what both renderers draw on.
type surface interface {
	FillRect(x, y, w, h int, c [4]uint8)
	FillRoundRect(x, y, w, h, rad int, c [4]uint8)
	StrokeRect(x, y, w, h int, c [4]uint8, lw int)
	StrokeRoundRect(x, y, w, h, rad int, c [4]uint8, lw int)
	PushClip(x, y, w, h int)
	PopClip()
	PushTranslate(dx, dy int)
	PopTranslate()
	DrawMask(x, y, w, h int, mask []byte, stride int, c [4]uint8)
	DrawImage(x, y, w, h int, src []byte, sw, sh int)
	FillPath(pth *vector.Path, c [4]uint8, rule vector.FillRule)
	StrokePath(pth *vector.Path, c [4]uint8, width float64)
}

type ref struct{ p *painter.PixelPainter }

func (s ref) DrawMask(x, y, w, h int, mask []byte, stride int, c [4]uint8) {
	s.p.DrawMask(painter.Rect{X: x, Y: y, W: w, H: h}, mask, stride, pc(c))
}
func (s ref) DrawImage(x, y, w, h int, src []byte, sw, sh int) {
	s.p.DrawImage(painter.Rect{X: x, Y: y, W: w, H: h}, src, sw, sh)
}
func (s ref) FillPath(pth *vector.Path, c [4]uint8, rule vector.FillRule) {
	s.p.FillPath(pth, pc(c), rule)
}
func (s ref) StrokePath(pth *vector.Path, c [4]uint8, width float64) {
	s.p.StrokePath(pth, pc(c), width)
}

func pc(c [4]uint8) painter.RGBA { return painter.RGBA{R: c[0], G: c[1], B: c[2], A: c[3]} }
func (s ref) FillRect(x, y, w, h int, c [4]uint8) {
	s.p.FillRect(painter.Rect{X: x, Y: y, W: w, H: h}, pc(c))
}
func (s ref) FillRoundRect(x, y, w, h, rad int, c [4]uint8) {
	s.p.FillRoundRect(painter.Rect{X: x, Y: y, W: w, H: h}, rad, pc(c))
}
func (s ref) StrokeRect(x, y, w, h int, c [4]uint8, lw int) {
	s.p.StrokeRect(painter.Rect{X: x, Y: y, W: w, H: h}, pc(c), lw)
}
func (s ref) StrokeRoundRect(x, y, w, h, rad int, c [4]uint8, lw int) {
	s.p.StrokeRoundRect(painter.Rect{X: x, Y: y, W: w, H: h}, rad, pc(c), lw)
}
func (s ref) PushClip(x, y, w, h int)  { s.p.PushClip(painter.Rect{X: x, Y: y, W: w, H: h}) }
func (s ref) PopClip()                 { s.p.PopClip() }
func (s ref) PushTranslate(dx, dy int) { s.p.PushTranslate(dx, dy) }
func (s ref) PopTranslate()            { s.p.PopTranslate() }

// sub draws through gpu2d. Paths are rasterized by the same go-gfx/gfx/vector
// rasterizer PixelPainter calls, clamped to the surface, and their coverage
// handed to DrawAlpha.
type sub struct {
	l  *gpu2d.List
	rz *vector.Rasterizer
}

func (s sub) DrawMask(x, y, w, h int, mask []byte, stride int, c [4]uint8) {
	s.l.DrawMask(gpu2d.Rect{X: x, Y: y, W: w, H: h}, mask, stride, gc(c))
}
func (s sub) DrawImage(x, y, w, h int, src []byte, sw, sh int) {
	s.l.DrawImage(gpu2d.Rect{X: x, Y: y, W: w, H: h}, src, sw, sh)
}
func (s sub) FillPath(pth *vector.Path, c [4]uint8, rule vector.FillRule) {
	w, h := s.l.Size()
	if cov, ox, oy, bw, bh, ok := s.rz.Fill(pth, rule, w, h); ok && c[3] != 0 {
		s.l.DrawAlpha(cov, ox, oy, bw, bh, gc(c))
	}
}
func (s sub) StrokePath(pth *vector.Path, c [4]uint8, width float64) {
	w, h := s.l.Size()
	if cov, ox, oy, bw, bh, ok := s.rz.Stroke(pth, width, w, h); ok && c[3] != 0 {
		s.l.DrawAlpha(cov, ox, oy, bw, bh, gc(c))
	}
}

func gc(c [4]uint8) gpu2d.RGBA { return gpu2d.RGBA{R: c[0], G: c[1], B: c[2], A: c[3]} }
func (s sub) FillRect(x, y, w, h int, c [4]uint8) {
	s.l.FillRect(gpu2d.Rect{X: x, Y: y, W: w, H: h}, gc(c))
}
func (s sub) FillRoundRect(x, y, w, h, rad int, c [4]uint8) {
	s.l.FillRoundRect(gpu2d.Rect{X: x, Y: y, W: w, H: h}, rad, gc(c))
}
func (s sub) StrokeRect(x, y, w, h int, c [4]uint8, lw int) {
	s.l.StrokeRect(gpu2d.Rect{X: x, Y: y, W: w, H: h}, gc(c), lw)
}
func (s sub) StrokeRoundRect(x, y, w, h, rad int, c [4]uint8, lw int) {
	s.l.StrokeRoundRect(gpu2d.Rect{X: x, Y: y, W: w, H: h}, rad, gc(c), lw)
}
func (s sub) PushClip(x, y, w, h int)  { s.l.PushClip(gpu2d.Rect{X: x, Y: y, W: w, H: h}) }
func (s sub) PopClip()                 { s.l.PopClip() }
func (s sub) PushTranslate(dx, dy int) { s.l.PushTranslate(dx, dy) }
func (s sub) PopTranslate()            { s.l.PopTranslate() }

// scene draws n random operations, the same ones for any surface given the
// same seed: translucent and opaque colours, rectangles past every edge, tiny
// and degenerate sizes, radii past half a side, nested clips and translations.
func scene(s surface, seed uint64, n, w, h int) {
	r := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	coord := func(m int) int { return r.IntN(m+16) - 8 }
	size := func() int { return r.IntN(24) }
	col := func() [4]uint8 {
		a := uint8(255)
		if r.IntN(2) == 0 {
			a = uint8(r.IntN(256))
		}
		return [4]uint8{uint8(r.IntN(256)), uint8(r.IntN(256)), uint8(r.IntN(256)), a}
	}
	bytesOf := func(n int) []byte {
		b := make([]byte, n)
		for i := range b {
			switch r.IntN(4) {
			case 0:
				b[i] = 0
			case 1:
				b[i] = 255
			default:
				b[i] = uint8(r.IntN(256))
			}
		}
		return b
	}
	pathOf := func() *vector.Path {
		p := vector.NewPath().MoveTo(float64(coord(w))+r.Float64(), float64(coord(h))+r.Float64())
		for range 2 + r.IntN(4) {
			x, y := float64(coord(w))+r.Float64(), float64(coord(h))+r.Float64()
			if r.IntN(3) == 0 {
				p.QuadTo(float64(coord(w)), float64(coord(h)), x, y)
			} else {
				p.LineTo(x, y)
			}
		}
		if r.IntN(2) == 0 {
			p.Close()
		}
		return p
	}
	for range n {
		switch r.IntN(16) {
		case 12:
			mw, mh := size(), size()
			stride := mw + r.IntN(3)
			// Sometimes short, so the rows a mask cannot supply are exercised.
			s.DrawMask(coord(w), coord(h), mw, mh, bytesOf(max(stride*mh-r.IntN(8)*stride/2, 0)), stride, col())
			continue
		case 13:
			sw, sh := 1+r.IntN(12), 1+r.IntN(12)
			s.DrawImage(coord(w), coord(h), size(), size(), bytesOf(4*sw*sh), sw, sh)
			continue
		case 14:
			s.FillPath(pathOf(), col(), vector.FillRule(r.IntN(2)))
			continue
		case 15:
			s.StrokePath(pathOf(), col(), 0.5+3*r.Float64())
			continue
		}
		switch r.IntN(12) {
		case 0, 1, 2:
			s.FillRect(coord(w), coord(h), size(), size(), col())
		case 3, 4:
			s.FillRoundRect(coord(w), coord(h), size(), size(), r.IntN(16)-2, col())
		case 5:
			s.StrokeRect(coord(w), coord(h), size(), size(), col(), r.IntN(6))
		case 6, 7:
			s.StrokeRoundRect(coord(w), coord(h), size(), size(), r.IntN(16)-2, col(), r.IntN(6))
		case 8:
			s.PushClip(coord(w), coord(h), size()+16, size()+16)
		case 9:
			s.PushTranslate(r.IntN(9)-4, r.IntN(9)-4)
		default:
			if r.IntN(2) == 0 {
				s.PopClip()
			} else {
				s.PopTranslate()
			}
		}
	}
}

// drawn counts the instances every render recorded, so a run whose scenes all
// clipped to nothing cannot pass as parity.
var (
	drawn int
	kinds = map[gpu2d.Kind]int{}
)

func render(t *testing.T, seed uint64, n, w, h int) (want, got []byte) {
	want = make([]byte, 4*w*h)
	got = make([]byte, 4*w*h)
	// Both start from the same non-trivial ground, so translucent blends show.
	for i := range want {
		want[i] = uint8(i * 7)
	}
	copy(got, want)
	scene(ref{painter.NewPixelPainter(want, w, h)}, seed, n, w, h)
	l, err := gpu2d.NewList(w, h)
	if err != nil {
		t.Fatal(err)
	}
	scene(sub{l, &vector.Rasterizer{}}, seed, n, w, h)
	if err := l.Err(); err != nil {
		t.Fatal(err)
	}
	if err := l.Rasterize(got); err != nil {
		t.Fatal(err)
	}
	drawn += len(l.Instances())
	for _, in := range l.Instances() {
		kinds[in.Kind]++
	}
	return want, got
}

func TestRasterizeIsPixelPainter(t *testing.T) {
	const w, h, seeds = 48, 40, 3000
	drawn = 0
	clear(kinds)
	defer func() {
		t.Logf("%d scenes, %d instances drawn: %d fill, %d outline, %d mask, %d alpha (paths), %d image",
			seeds, drawn, kinds[gpu2d.KindFill], kinds[gpu2d.KindOutline], kinds[gpu2d.KindMask],
			kinds[gpu2d.KindAlpha], kinds[gpu2d.KindImage])
		if drawn < seeds*10 {
			t.Fatalf("only %d instances over %d scenes: the scenes barely draw", drawn, seeds)
		}
		for k := gpu2d.KindFill; k <= gpu2d.KindImage; k++ {
			if kinds[k] < seeds/2 {
				t.Fatalf("kind %d drawn only %d times over %d scenes", k, kinds[k], seeds)
			}
		}
	}()
	for seed := range uint64(seeds) {
		want, got := render(t, seed, 40, w, h)
		if !bytes.Equal(want, got) {
			for i := range want {
				if want[i] != got[i] {
					p := i / 4
					t.Fatalf("seed %d: first difference at (%d,%d) channel %d: PixelPainter %d, gpu2d %d",
						seed, p%w, p/w, i%4, want[i], got[i])
				}
			}
		}
	}
}

// The comparison must be able to fail: the same scene with one extra opaque
// pixel on one side only differs.
func TestParityCanFail(t *testing.T) {
	want, got := render(t, 1, 40, 48, 40)
	got[0] ^= 0xFF
	if bytes.Equal(want, got) {
		t.Fatal("a changed pixel compared equal")
	}
	l, _ := gpu2d.NewList(4, 4)
	l.FillRoundRect(gpu2d.Rect{W: 4, H: 4}, 2, gpu2d.RGBA{A: 255})
	a := make([]byte, 64)
	_ = l.Rasterize(a)
	b := make([]byte, 64)
	painter.NewPixelPainter(b, 4, 4).FillRoundRect(painter.Rect{W: 4, H: 4}, 1, painter.RGBA{A: 255})
	if bytes.Equal(a, b) {
		t.Fatal("radius 2 against radius 1 compared equal")
	}
}
