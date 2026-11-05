// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package scenes records the random frames every back-end's test holds to
// gpu2d.Rasterize, so that they all draw the same mix.
package scenes

import (
	"math/rand/v2"

	"github.com/go-gfx/gpu2d"
)

// Random records n random primitives for a w × h list: translucent and
// opaque colours, rectangles past every edge, degenerate sizes, radii past
// half a side, nested clips and translations, coverage masks (short ones
// too), scaled images with translucent texels, and alpha boxes such as a
// path rasterizer produces.
func Random(l *gpu2d.List, seed uint64, n int) {
	w, h := l.Size()
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
	bytesOf := func(n int) []byte {
		b := make([]byte, max(n, 0))
		for i := range b {
			switch r.IntN(4) {
			case 0:
			case 1:
				b[i] = 255
			default:
				b[i] = uint8(r.IntN(256))
			}
		}
		return b
	}
	for range n {
		switch r.IntN(16) {
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
		case 11:
			l.PopTranslate()
		case 12:
			d := rect()
			stride := d.W + r.IntN(3)
			l.DrawMask(d, bytesOf(stride*d.H-r.IntN(8)*stride/2), stride, col())
		case 13:
			sw, sh := 1+r.IntN(12), 1+r.IntN(12)
			l.DrawImage(rect(), bytesOf(4*sw*sh), sw, sh)
		default:
			d := rect()
			cov := make([]float64, d.W*d.H)
			for i := range cov {
				cov[i] = r.Float64()*1.4 - 0.2
			}
			l.DrawAlpha(cov, d.X, d.Y, d.W, d.H, col())
		}
	}
}
