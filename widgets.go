// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

// Canvas is what WidgetFrame draws on: a List, or an adapter over another
// renderer, so that every renderer's benchmark draws the same frame.
type Canvas interface {
	Size() (w, h int)
	FillRect(r Rect, c RGBA)
	FillRoundRect(r Rect, radius int, c RGBA)
	StrokeRoundRect(r Rect, radius int, c RGBA, lineW int)
}

// WidgetFrame draws a frame shaped like a widget toolkit's: an opaque
// background, then a grid of cards, each a rounded fill, a translucent
// rounded border and a smaller rounded control inside, and every seventh card
// under a translucent hover wash.
func WidgetFrame(c Canvas) {
	w, h := c.Size()
	c.FillRect(Rect{W: w, H: h}, RGBA{236, 236, 236, 255})
	const cell, gap = 96, 12
	i := 0
	for y := gap; y+cell <= h; y += cell + gap {
		for x := gap; x+cell <= w; x += cell + gap {
			r := Rect{X: x, Y: y, W: cell, H: cell}
			c.FillRoundRect(r, 10, RGBA{255, 255, 255, 255})
			c.StrokeRoundRect(r, 10, RGBA{0, 0, 0, 31}, 1)
			c.FillRoundRect(Rect{X: x + 12, Y: y + cell - 36, W: cell - 24, H: 24}, 6, RGBA{0, 122, 255, 255})
			if i%7 == 0 {
				c.FillRoundRect(r, 10, RGBA{0, 0, 0, 20})
			}
			i++
		}
	}
}
