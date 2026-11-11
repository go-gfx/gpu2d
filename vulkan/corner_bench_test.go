// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package vulkan

import (
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
)

// BenchmarkCornerFrame is the worst case of the corner coverage: 1,280 rounded
// rectangles and outlines of radii 1..64 on a 1080p frame (Draw alone).
func BenchmarkCornerFrame(b *testing.B) {
	const w, h = 1920, 1080
	r := renderer(b, w, h)
	l, _ := gpu2d.NewList(w, h)
	for seed := range uint64(8) {
		scenes.Corners(l, seed, 64)
	}
	b.ReportMetric(float64(len(l.Instances())), "instances")
	b.ReportAllocs()
	for b.Loop() {
		if err := r.Draw(l); err != nil {
			b.Fatal(err)
		}
	}
}
