// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package metal

import (
	"math/rand/v2"
	"os"
	"strconv"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
)

// TestCornerHeavy draws rounded rectangles and outlines of radii 1 to 64 at
// every position modulo the tile (scenes.Corners) over both grounds: 0
// differing channels, the corner coverage being exact integer arithmetic.
// METAL_WIDE_CORNERS=N runs N scenes (60 by default).
func TestCornerHeavy(t *testing.T) {
	n := 60
	if v := os.Getenv("METAL_WIDE_CORNERS"); v != "" {
		var err error
		if n, err = strconv.Atoi(v); err != nil || n < 1 {
			t.Fatalf("METAL_WIDE_CORNERS=%q", v)
		}
	}
	r := renderer(t, 64, 48)
	differing, channels, instances := 0, 0, 0
	for seed := range uint64(n) {
		rg := rand.New(rand.NewPCG(seed, 0xf00d))
		w, h := 96+rg.IntN(420), 96+rg.IntN(420)
		if err := r.Resize(w, h); err != nil {
			t.Fatal(err)
		}
		l, _ := gpu2d.NewList(w, h)
		scenes.Corners(l, seed, 64)
		instances += len(l.Instances())
		ground := make([]byte, 4*w*h)
		for i := range ground {
			ground[i] = uint8(i * 7)
			if seed%2 == 0 && i%4 == 3 {
				ground[i] = 255
			}
		}
		want := append([]byte(nil), ground...)
		if err := l.Rasterize(want); err != nil {
			t.Fatal(err)
		}
		copy(r.Pixels(), ground)
		if err := r.Draw(l); err != nil {
			t.Fatal(err)
		}
		for i, v := range r.Pixels() {
			if v != want[i] {
				differing++
				if differing <= 5 {
					t.Logf("seed %d %dx%d pixel (%d,%d) channel %d: CPU %d, GPU %d", seed, w, h, i/4%w, i/4/w, i%4, want[i], v)
				}
			}
		}
		channels += len(want)
	}
	t.Logf("%s: %d scenes, %d instances, %d of %d channels differ", r.Name(), n, instances, differing, channels)
	if differing != 0 {
		t.Fatalf("%d channels differ from Rasterize", differing)
	}
	if instances < n*100 {
		t.Fatalf("only %d instances in %d scenes", instances, n)
	}
}
