// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import (
	"errors"
	"math/rand/v2"
	"testing"
)

// Bin against brute force: every tile lists exactly the instances whose box
// touches it, in draw order.
func TestBinMatchesBruteForce(t *testing.T) {
	var spans, list []uint32
	for seed := range uint64(200) {
		r := rand.New(rand.NewPCG(seed, 1))
		w, h := 1+r.IntN(70), 1+r.IntN(70)
		l := nl(w, h)
		for range 30 {
			rc := Rect{X: r.IntN(w+20) - 10, Y: r.IntN(h+20) - 10, W: r.IntN(40), H: r.IntN(40)}
			if r.IntN(4) == 0 {
				l.PushClip(Rect{X: r.IntN(w), Y: r.IntN(h), W: r.IntN(30) + 1, H: r.IntN(30) + 1})
			}
			l.FillRoundRect(rc, r.IntN(6), RGBA{A: 255})
			if r.IntN(3) == 0 {
				l.PopClip()
			}
		}
		var err error
		spans, list, err = l.Bin(spans, list)
		if err != nil {
			t.Fatal(err)
		}
		across, down := l.Tiles()
		if len(spans) != 2*across*down {
			t.Fatalf("seed %d: %d span words for %d tiles", seed, len(spans), across*down)
		}
		for ty := range down {
			for tx := range across {
				var want []uint32
				for k := range l.instances {
					x0, y0, x1, y1 := box(&l.instances[k])
					if x0 < (tx+1)*Tile && x1 > tx*Tile && y0 < (ty+1)*Tile && y1 > ty*Tile {
						want = append(want, uint32(k))
					}
				}
				tile := ty*across + tx
				got := list[spans[2*tile] : spans[2*tile]+spans[2*tile+1]]
				if len(got) != len(want) {
					t.Fatalf("seed %d tile (%d,%d): %v, want %v", seed, tx, ty, got, want)
				}
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("seed %d tile (%d,%d): %v, want %v", seed, tx, ty, got, want)
					}
				}
			}
		}
	}
}

// A list that would bin past MaxBinned is refused before anything is
// allocated for it.
func TestBinRefusesHugeLists(t *testing.T) {
	l := nl(16384, 16384) // 2^20 tiles
	for range 65 {
		l.FillRect(Rect{W: 16384, H: 16384}, RGBA{A: 255})
	}
	spans, list, err := l.Bin(make([]uint32, 3), make([]uint32, 5))
	if !errors.Is(err, ErrTooLarge) || len(spans) != 0 || len(list) != 0 {
		t.Fatalf("err %v, %d spans, %d entries", err, len(spans), len(list))
	}
}
