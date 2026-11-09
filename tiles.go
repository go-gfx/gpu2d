// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import "errors"

// Tile is the side, in pixels, of the squares Bin sorts instances into.
const Tile = 16

// MaxBinned bounds the entries Bin may produce, one per instance per tile it
// touches: 2^26 entries are 256 MiB of indices, far past any real frame (a
// 4K frame of 2200 widget instances needs about 0.2 M). A list past it is
// ErrTooLarge, so no list can size an unbounded allocation.
const MaxBinned = 1 << 26

// ErrTooLarge reports a list whose binning would pass MaxBinned.
var ErrTooLarge = errors.New("gpu2d: binned list larger than MaxBinned")

// Tiles returns how many tiles span the surface across and down.
func (l *List) Tiles() (across, down int) {
	return (l.w + Tile - 1) / Tile, (l.h + Tile - 1) / Tile
}

// Bin sorts the list's instances by the Tile × Tile squares their drawable
// box (rectangle cut by clip) touches. spans gets two words per tile, row
// after row: the first entry of the tile in list, and how many. list gets
// instance indices, each tile's in draw order. Both reuse the storage of the
// slices passed in.
//
// Since v0.7.0 a tile omits the instances drawn before the last one that
// hides it entirely: an opaque fill with square corners whose box covers
// the whole tile. Such a fill sets every pixel of the tile to its colour
// whatever lies beneath (Coverage is 1 and the blend overwrites), so a back
// end compositing the tile's list in order draws the same bytes with less
// work.
func (l *List) Bin(spans, list []uint32) ([]uint32, []uint32, error) {
	across, down := l.Tiles()
	total := 0
	for i := range l.instances {
		x0, y0, x1, y1 := box(&l.instances[i])
		total += ((x1-1)/Tile - x0/Tile + 1) * ((y1-1)/Tile - y0/Tile + 1)
		if total > MaxBinned {
			return spans[:0], list[:0], ErrTooLarge
		}
	}
	spans = append(spans[:0], make([]uint32, 2*across*down)...)
	l.cover = append(l.cover[:0], make([]uint32, across*down)...)
	cover := l.cover
	// Two passes over every (instance, tile) pair, counting then placing.
	// They are written out rather than shared through a callback: a frame
	// has one pair per instance per tile it touches (a full-surface fill
	// alone is every tile), and an indirect call per pair was most of the
	// cost of binning (go-gfx/gpu2d#20). The counting pass also finds each
	// tile's last hiding instance and restarts the tile's count there.
	for k := range l.instances {
		in := &l.instances[k]
		x0, y0, x1, y1 := box(in)
		cx0, cx1, cy0, cy1 := 1, 0, 1, 0 // no tile hidden
		if hides(in) {
			cx0, cx1 = wholeTiles(x0, x1, l.w)
			cy0, cy1 = wholeTiles(y0, y1, l.h)
		}
		tx0 := x0 / Tile
		for ty := y0 / Tile; ty <= (y1-1)/Tile; ty++ {
			base := ty * across
			row := spans[2*(base+tx0) : 2*(base+(x1-1)/Tile+1)]
			if ty < cy0 || ty > cy1 {
				for i := 1; i < len(row); i += 2 {
					row[i]++
				}
				continue
			}
			for i, tx := 1, tx0; i < len(row); i, tx = i+2, tx+1 {
				if tx >= cx0 && tx <= cx1 {
					row[i] = 1
					cover[base+tx] = uint32(k) + 1
				} else {
					row[i]++
				}
			}
		}
	}
	first := uint32(0)
	for t := range across * down {
		spans[2*t], first, spans[2*t+1] = first, first+spans[2*t+1], 0
	}
	list = append(list[:0], make([]uint32, first)...)
	for k := range l.instances {
		x0, y0, x1, y1 := box(&l.instances[k])
		tx0, tx1 := x0/Tile, (x1-1)/Tile+1
		for ty := y0 / Tile; ty <= (y1-1)/Tile; ty++ {
			base := ty * across
			row := spans[2*(base+tx0) : 2*(base+tx1)]
			hidden := cover[base+tx0 : base+tx1]
			for i, j := 0, 0; i < len(row); i, j = i+2, j+1 {
				if uint32(k)+1 < hidden[j] {
					continue // a later instance hides this tile
				}
				list[row[i]+row[i+1]] = uint32(k)
				row[i+1]++
			}
		}
	}
	return spans, list, nil
}

// hides reports whether an instance sets every pixel of its box to one
// colour regardless of what lies beneath: an opaque fill with square
// corners (Coverage 1 everywhere, and blend overwrites at alpha 255).
func hides(in *Instance) bool {
	return in.Kind == KindFill && in.Radius <= 0 && in.Color>>24 == 0xFF
}

// wholeTiles returns the first and last tile, along an axis of n pixels,
// that the pixels [p0, p1) cover entirely; last < first when there are
// none. The last tile of the axis ends at the surface edge, not at a
// multiple of Tile.
func wholeTiles(p0, p1, n int) (first, last int) {
	first, last = (p0+Tile-1)/Tile, p1/Tile-1
	if p1 == n {
		last = (n - 1) / Tile
	}
	return first, last
}

// box is an instance's drawable box: its rectangle cut by its clip. A list
// records only instances whose box is not empty, and the clip lies inside
// the surface, so the box does too.
func box(in *Instance) (x0, y0, x1, y1 int) {
	return max(int(in.X), int(in.ClipX0)), max(int(in.Y), int(in.ClipY0)),
		min(int(in.X)+int(in.W), int(in.ClipX1)), min(int(in.Y)+int(in.H), int(in.ClipY1))
}
