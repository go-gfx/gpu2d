// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import (
	"errors"
	"math"
)

// ErrBuffer reports a buffer too small for the list's surface.
var ErrBuffer = errors.New("gpu2d: buffer smaller than 4 × width × height")

// Rasterize executes the list into buf, an RGBA surface of the list's size,
// in draw order. It is the reference every GPU back-end is checked against.
func (l *List) Rasterize(buf []byte) error {
	if int64(len(buf)) < 4*int64(l.w)*int64(l.h) {
		return ErrBuffer
	}
	for i := range l.instances {
		in := &l.instances[i]
		switch in.Kind {
		case KindMask, KindAlpha:
			rasterizeMask(buf, l.w, l.data, in)
		case KindImage:
			rasterizeImage(buf, l.w, l.data, in)
		default:
			rasterize(buf, l.w, in)
		}
	}
	return nil
}

// rasterizeMask paints a KindMask or KindAlpha instance, whose rectangle is
// its visible box. A mask's alpha is its coverage, scaled by a translucent
// ink's alpha with PixelPainter's floored product; an alpha instance's is
// the stored byte.
func rasterizeMask(buf []byte, stride int, data []byte, in *Instance) {
	c := unpack(in.Color)
	for y := range int(in.H) {
		row := data[int(in.Data)+y*int(in.Stride):]
		for x := range int(in.W) {
			a := uint32(row[x])
			if a == 0 {
				continue
			}
			if in.Kind == KindMask && c.A != 0xFF {
				a = a * uint32(c.A) / 255
			}
			col := c
			col.A = uint8(a)
			blend(buf[((int(in.Y)+y)*stride+int(in.X)+x)*4:], col)
		}
	}
}

// rasterizeImage draws a KindImage instance: each visible pixel takes the
// source texel PixelPainter's nearest-neighbour mapping lands on.
func rasterizeImage(buf []byte, stride int, data []byte, in *Instance) {
	r := Rect{int(in.X), int(in.Y), int(in.W), int(in.H)}
	x0, y0, x1, y1 := box(in)
	sw, sh := int(in.SrcW), int(in.SrcH)
	for y := y0; y < y1; y++ {
		sy := (y - r.Y) * sh / r.H
		for x := x0; x < x1; x++ {
			o := int(in.Data) + (sy*sw+(x-r.X)*sw/r.W)*4
			blend(buf[(y*stride+x)*4:], RGBA{data[o], data[o+1], data[o+2], data[o+3]})
		}
	}
}

func rasterize(buf []byte, stride int, in *Instance) {
	r := Rect{int(in.X), int(in.Y), int(in.W), int(in.H)}
	c := unpack(in.Color)
	x0, y0, x1, y1 := box(in)
	rad := int(in.Radius)
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			cov := Coverage(in.Kind, x, y, r, rad)
			if cov <= 0 {
				continue
			}
			col := c
			if cov < 1 {
				col.A = uint8(float64(c.A)*cov + 0.5)
			}
			blend(buf[(y*stride+x)*4:], col)
		}
	}
}

// Coverage is the share of pixel (x, y) that a primitive of kind k over r with
// corner radius rad covers, from 0 to 1: PixelPainter's cornerFillCoverage for
// a fill and the ring of its strokeRoundOutline for an outline. A GPU fragment
// shader computes the same expression at the same pixel centre.
func Coverage(k Kind, x, y int, r Rect, rad int) float64 {
	if k == KindFill {
		cx, cy, in := cornerCenter(x, y, r, rad)
		if !in {
			return 1
		}
		return min(max(float64(rad)+0.5-math.Hypot(float64(x)+0.5-cx, float64(y)+0.5-cy), 0), 1)
	}
	if rad <= 0 {
		if y == r.Y || y == r.Y+r.H-1 || x == r.X || x == r.X+r.W-1 {
			return 1
		}
		return 0
	}
	if cx, cy, in := cornerCenter(x, y, r, rad); in {
		d := math.Hypot(float64(x)+0.5-cx, float64(y)+0.5-cy)
		return max(1-math.Abs(d-(float64(rad)-0.5)), 0)
	}
	if y == r.Y || y == r.Y+r.H-1 || x == r.X || x == r.X+r.W-1 {
		return 1
	}
	return 0
}

// cornerCenter returns the circle centre of the corner box (x, y) lies in.
func cornerCenter(x, y int, r Rect, rad int) (cx, cy float64, in bool) {
	left, right := x < r.X+rad, x >= r.X+r.W-rad
	top, bottom := y < r.Y+rad, y >= r.Y+r.H-rad
	switch {
	case left && top:
		return float64(r.X + rad), float64(r.Y + rad), true
	case right && top:
		return float64(r.X + r.W - rad), float64(r.Y + rad), true
	case left && bottom:
		return float64(r.X + rad), float64(r.Y + r.H - rad), true
	case right && bottom:
		return float64(r.X + r.W - rad), float64(r.Y + r.H - rad), true
	}
	return 0, 0, false
}

// blend composites c over the pixel at p[0:4] exactly as PixelPainter's
// blendInto: opaque overwrites, transparent is a no-op, anything else is
// (src·a + dst·(255−a) + 127) / 255 per colour channel.
func blend(p []byte, c RGBA) {
	if c.A == 0xFF {
		p[0], p[1], p[2], p[3] = c.R, c.G, c.B, 0xFF
		return
	}
	if c.A == 0 {
		return
	}
	a := uint32(c.A)
	ia := 255 - a
	mix := func(src, dst uint8) uint8 { return uint8((uint32(src)*a + uint32(dst)*ia + 127) / 255) }
	p[0], p[1], p[2] = mix(c.R, p[0]), mix(c.G, p[1]), mix(c.B, p[2])
	p[3] = uint8(a + uint32(p[3])*ia/255)
}
