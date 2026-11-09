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
	// Two passes over every (instance, tile) pair, counting then placing.
	// They are written out rather than shared through a callback: a frame
	// has one pair per instance per tile it touches (a full-surface fill
	// alone is every tile), and an indirect call per pair was most of the
	// cost of binning (go-gfx/gpu2d#20).
	for k := range l.instances {
		x0, y0, x1, y1 := box(&l.instances[k])
		for ty := y0 / Tile; ty <= (y1-1)/Tile; ty++ {
			row := spans[2*(ty*across+x0/Tile) : 2*(ty*across+(x1-1)/Tile+1)]
			for i := 1; i < len(row); i += 2 {
				row[i]++
			}
		}
	}
	first := uint32(0)
	for t := range across * down {
		spans[2*t], first, spans[2*t+1] = first, first+spans[2*t+1], 0
	}
	list = append(list[:0], make([]uint32, total)...)
	for k := range l.instances {
		x0, y0, x1, y1 := box(&l.instances[k])
		for ty := y0 / Tile; ty <= (y1-1)/Tile; ty++ {
			row := spans[2*(ty*across+x0/Tile) : 2*(ty*across+(x1-1)/Tile+1)]
			for i := 0; i < len(row); i += 2 {
				list[row[i]+row[i+1]] = uint32(k)
				row[i+1]++
			}
		}
	}
	return spans, list, nil
}

// box is an instance's drawable box: its rectangle cut by its clip. A list
// records only instances whose box is not empty, and the clip lies inside
// the surface, so the box does too.
func box(in *Instance) (x0, y0, x1, y1 int) {
	return max(int(in.X), int(in.ClipX0)), max(int(in.Y), int(in.ClipY0)),
		min(int(in.X)+int(in.W), int(in.ClipX1)), min(int(in.Y)+int(in.H), int(in.ClipY1))
}
