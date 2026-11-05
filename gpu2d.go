// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package gpu2d describes a 2D frame as a list of instances that a GPU draws
// as one quad each and that Rasterize executes on the CPU. Rasterize is the
// reference: it ports go-widgets/painter's PixelPainter formulas, and a GPU
// back-end must match it within 1 per channel (see docs/DESIGN.md).
package gpu2d

import "errors"

// Rect is an integer rectangle; the right and bottom edges are exclusive.
type Rect struct{ X, Y, W, H int }

// RGBA is a straight-alpha colour.
type RGBA struct{ R, G, B, A uint8 }

// Kind says how an instance covers its rectangle.
type Kind uint32

const (
	// KindFill covers the rectangle, its corners rounded to Radius.
	KindFill Kind = iota
	// KindOutline covers the one-pixel outline, its corners rounded to Radius.
	KindOutline
)

// Instance is one primitive as a GPU reads it: 48 bytes, every field 32 bits.
// X..H are in surface pixels, translation applied. Clip is the half-open box
// [ClipX0, ClipX1) × [ClipY0, ClipY1) the primitive may touch. Radius is
// already clamped to half the smaller side.
type Instance struct {
	X, Y, W, H                     int32
	ClipX0, ClipY0, ClipX1, ClipY1 int32
	Color                          uint32 // R | G<<8 | B<<16 | A<<24
	Radius                         int32
	Kind                           Kind
	_                              uint32
}

// InstanceSize is the byte size of one Instance.
const InstanceSize = 48

// MaxCoord bounds every coordinate a list records, translation included:
// 2^24 is the last integer up to which float32, which a shader computes in,
// represents every integer exactly. A primitive reaching past it is not
// recorded, and a surface larger than it is refused by NewList.
const MaxCoord = 1 << 24

func pack(c RGBA) uint32 {
	return uint32(c.R) | uint32(c.G)<<8 | uint32(c.B)<<16 | uint32(c.A)<<24
}

func unpack(v uint32) RGBA {
	return RGBA{uint8(v), uint8(v >> 8), uint8(v >> 16), uint8(v >> 24)}
}

// List records a frame. The zero value is not ready: use NewList, which knows
// the surface the clip starts from.
type List struct {
	w, h      int
	instances []Instance
	clips     []Rect
	offsets   [][2]int
}

// NewList starts a frame for a w × h surface. A side below zero or above
// MaxCoord is ErrSize.
func NewList(w, h int) (*List, error) {
	if w < 0 || h < 0 || w > MaxCoord || h > MaxCoord {
		return nil, ErrSize
	}
	return &List{w: w, h: h}, nil
}

// ErrSize reports a surface side outside [0, MaxCoord].
var ErrSize = errors.New("gpu2d: surface side outside [0, MaxCoord]")

// Reset empties the list for the next frame, keeping its storage.
func (l *List) Reset() {
	l.instances = l.instances[:0]
	l.clips = l.clips[:0]
	l.offsets = l.offsets[:0]
}

// Instances returns the recorded instances in draw order.
func (l *List) Instances() []Instance { return l.instances }

// Size returns the surface size.
func (l *List) Size() (w, h int) { return l.w, l.h }

func (l *List) offset() (dx, dy int) {
	if n := len(l.offsets); n > 0 {
		return l.offsets[n-1][0], l.offsets[n-1][1]
	}
	return 0, 0
}

// PushTranslate shifts everything drawn after it by dx, dy, added to the
// enclosing translation. The sum saturates at ±2·MaxCoord, past which nothing
// can land on a surface anyway.
func (l *List) PushTranslate(dx, dy int) {
	ox, oy := l.offset()
	l.offsets = append(l.offsets, [2]int{sat(int64(ox) + int64(dx)), sat(int64(oy) + int64(dy))})
}

// sat clamps v to ±2·MaxCoord, a range every platform's int holds.
func sat(v int64) int { return int(min(max(v, -2*MaxCoord), 2*MaxCoord)) }

// span returns the half-open interval [p, p+n) shifted by d, in int64.
func span(p, n, d int) (lo, hi int64) {
	lo = int64(p) + int64(d)
	return lo, lo + int64(n)
}

// PopTranslate removes the innermost translation; on an empty stack it does
// nothing.
func (l *List) PopTranslate() {
	if n := len(l.offsets); n > 0 {
		l.offsets = l.offsets[:n-1]
	}
}

// PushClip confines drawing to r, translated by the translation in force and
// intersected with the enclosing clip.
func (l *List) PushClip(r Rect) {
	dx, dy := l.offset()
	// Clamping to ±2·MaxCoord changes nothing a surface can show and keeps
	// the arithmetic below inside an int on 32-bit platforms.
	x0, x1 := span(r.X, max(r.W, 0), dx)
	y0, y1 := span(r.Y, max(r.H, 0), dy)
	r = Rect{X: sat(x0), Y: sat(y0)}
	r.W, r.H = sat(x1)-r.X, sat(y1)-r.Y
	if n := len(l.clips); n > 0 {
		r = intersect(r, l.clips[n-1])
	}
	l.clips = append(l.clips, r)
}

// PopClip removes the innermost clip; on an empty stack it does nothing.
func (l *List) PopClip() {
	if n := len(l.clips); n > 0 {
		l.clips = l.clips[:n-1]
	}
}

func intersect(a, b Rect) Rect {
	x0, y0 := max(a.X, b.X), max(a.Y, b.Y)
	x1, y1 := min(a.X+a.W, b.X+b.W), min(a.Y+a.H, b.Y+b.H)
	if x1 <= x0 || y1 <= y0 {
		return Rect{}
	}
	return Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}
}

// add records one primitive. Nothing is recorded when it cannot paint: an
// empty rectangle, a transparent colour, or nothing left after the clip.
func (l *List) add(r Rect, radius int, c RGBA, k Kind) {
	if r.W <= 0 || r.H <= 0 || c.A == 0 {
		return
	}
	dx, dy := l.offset()
	x0, x1 := span(r.X, r.W, dx)
	y0, y1 := span(r.Y, r.H, dy)
	if x0 < -MaxCoord || y0 < -MaxCoord || x1 > MaxCoord || y1 > MaxCoord {
		return
	}
	r = Rect{X: int(x0), Y: int(y0), W: r.W, H: r.H}
	clip := Rect{W: l.w, H: l.h}
	if n := len(l.clips); n > 0 {
		clip = intersect(clip, l.clips[n-1])
	}
	if intersect(r, clip).W == 0 {
		return
	}
	l.instances = append(l.instances, Instance{
		X: int32(r.X), Y: int32(r.Y), W: int32(r.W), H: int32(r.H),
		ClipX0: int32(clip.X), ClipY0: int32(clip.Y),
		ClipX1: int32(clip.X + clip.W), ClipY1: int32(clip.Y + clip.H),
		Color: pack(c), Radius: int32(max(radius, 0)), Kind: k,
	})
}

// clampRadius caps radius at half the smaller side of r, as PixelPainter does.
func clampRadius(radius int, r Rect) int {
	if m := min(r.W, r.H) / 2; radius > m {
		return m
	}
	return radius
}

// FillRect fills r.
func (l *List) FillRect(r Rect, c RGBA) { l.add(r, 0, c, KindFill) }

// FillRoundRect fills r with its corners rounded to radius; a radius that
// clamps to zero or less is a plain fill.
func (l *List) FillRoundRect(r Rect, radius int, c RGBA) {
	l.add(r, clampRadius(radius, r), c, KindFill)
}

// StrokeRect paints lineW concentric one-pixel outlines inward from r's edge.
// lineW is capped at MaxCoord: a wider ring cannot fit any recordable rectangle,
// and the cap bounds the loop whatever the caller passes.
func (l *List) StrokeRect(r Rect, c RGBA, lineW int) {
	for i := range min(max(lineW, 1), MaxCoord) {
		in := Rect{X: r.X + i, Y: r.Y + i, W: r.W - 2*i, H: r.H - 2*i}
		if in.W <= 0 || in.H <= 0 {
			return
		}
		l.outline(in, 0, c)
	}
}

// StrokeRoundRect paints lineW concentric rounded outlines, innermost first,
// each ring's radius one less than the ring outside it; lineW is capped at
// MaxCoord as in StrokeRect.
func (l *List) StrokeRoundRect(r Rect, radius int, c RGBA, lineW int) {
	if r.W <= 0 || r.H <= 0 {
		return
	}
	for i := 1; i < min(max(lineW, 1), MaxCoord); i++ {
		in := Rect{X: r.X + i, Y: r.Y + i, W: r.W - 2*i, H: r.H - 2*i}
		if in.W <= 0 || in.H <= 0 {
			break
		}
		l.outline(in, radius-i, c)
	}
	l.outline(r, radius, c)
}

// outline records one ring. PixelPainter paints a square outline one pixel
// tall or wide by hitting some pixels twice, which a translucent colour shows;
// those cases are recorded as the fills that reproduce exactly that.
func (l *List) outline(r Rect, radius int, c RGBA) {
	rad := clampRadius(radius, r)
	switch {
	case rad > 0:
		l.add(r, rad, c, KindOutline)
	case r.H == 1:
		l.add(r, 0, c, KindFill)
		l.add(r, 0, c, KindFill)
	case r.W == 1:
		l.add(r, 0, c, KindFill)
		l.add(Rect{X: r.X, Y: r.Y + 1, W: 1, H: r.H - 2}, 0, c, KindFill)
	default:
		l.add(r, 0, c, KindOutline)
	}
}
