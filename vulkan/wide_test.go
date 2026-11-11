// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package vulkan

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strconv"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
)

// Two parity runs wider than TestMatchesReference, for a device whose sqrt is
// not correctly rounded (RADV's is 1 ULP off on 18 % of its inputs, measured
// in go-gfx/wgsl's vulkantest): the kernel's corner coverage is integer
// arithmetic and must not care.
//
//	VULKAN_WIDE_SCENES=N   TestMatchesReferenceWide: N random scenes, each of a
//	                       random size from 16 x 16 up to 3840 x 2160
//	VULKAN_WIDE_CORNERS=N  TestCornerHeavy: N scenes of rounded rectangles with
//	                       radii 1..64 at every position modulo the tile
//
// Unset, TestMatchesReferenceWide is skipped and TestCornerHeavy runs 60
// scenes. Both fail on a single differing channel and say where: the kind and
// radius of the topmost instance over each differing pixel.

// cell is what a differing pixel is attributed to: the topmost instance that
// covers it, in draw order.
type cell struct {
	kind   gpu2d.Kind
	radius int32
}

// tally is the outcome of a wide run.
type tally struct {
	scenes, instances, channels, differing, worst int
	pixels                                        map[cell]int // differing pixels by topmost instance
	samples                                       []string
}

func newTally() *tally { return &tally{pixels: map[cell]int{}} }

// compare draws l over g on r and against gpu2d.Rasterize, adding the
// outcome to t.
func (t *tally) compare(tb testing.TB, r *Renderer, l *gpu2d.List, g []byte, tag string) {
	tb.Helper()
	w, h := l.Size()
	want := append([]byte(nil), g...)
	if err := l.Rasterize(want); err != nil {
		tb.Fatal(err)
	}
	copy(r.Pixels(), g)
	if err := r.Draw(l); err != nil {
		tb.Fatal(err)
	}
	got := r.Pixels()
	t.scenes++
	t.instances += len(l.Instances())
	t.channels += len(want)
	for i := range want {
		if want[i] == got[i] {
			continue
		}
		d := int(want[i]) - int(got[i])
		t.differing++
		t.worst = max(t.worst, max(d, -d))
		first := true // attribute once per pixel, at its first differing channel
		for j := i - i%4; j < i; j++ {
			first = first && want[j] == got[j]
		}
		if !first {
			continue
		}
		p := i / 4
		x, y := p%w, p/w
		c := cell{kind: gpu2d.Kind(255)}
		ins := l.Instances()
		for k := len(ins) - 1; k >= 0; k-- {
			in := &ins[k]
			if int32(x) >= max(in.X, in.ClipX0) && int32(x) < min(in.X+in.W, in.ClipX1) &&
				int32(y) >= max(in.Y, in.ClipY0) && int32(y) < min(in.Y+in.H, in.ClipY1) {
				c = cell{in.Kind, in.Radius}
				break
			}
		}
		t.pixels[c]++
		if len(t.samples) < 12 {
			t.samples = append(t.samples, fmt.Sprintf("%s %dx%d pixel (%d,%d) channel %d: want %d got %d, kind %d radius %d",
				tag, w, h, x, y, i%4, want[i], got[i], c.kind, c.radius))
		}
	}
}

// report logs the tally and fails the test when anything differed.
func (t *tally) report(tb testing.TB, what string) {
	tb.Helper()
	tb.Logf("%s: %d scenes, %d instances, max |Δ| %d, %d of %d channels differ",
		what, t.scenes, t.instances, t.worst, t.differing, t.channels)
	if t.differing == 0 {
		return
	}
	keys := make([]cell, 0, len(t.pixels))
	for c := range t.pixels {
		keys = append(keys, c)
	}
	sort.Slice(keys, func(i, j int) bool { return t.pixels[keys[i]] > t.pixels[keys[j]] })
	for _, c := range keys {
		tb.Logf("  differing pixels under kind %d radius %d: %d", c.kind, c.radius, t.pixels[c])
	}
	for _, s := range t.samples {
		tb.Log("  ", s)
	}
	tb.Errorf("%s: %d channels differ from Rasterize (max |Δ| %d)", what, t.differing, t.worst)
}

func envInt(tb testing.TB, name string, def int) int {
	tb.Helper()
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		tb.Fatalf("%s=%q: want a count", name, v)
	}
	return n
}

// logUniform draws from lo..hi so that small frames are the bulk and a 4K
// frame is rare, as a CPU reference at 4K costs 100 ms.
func logUniform(r *rand.Rand, lo, hi int) int {
	return int(math.Round(math.Exp(math.Log(float64(lo)) + r.Float64()*(math.Log(float64(hi))-math.Log(float64(lo))))))
}

// TestMatchesReferenceWide is TestMatchesReference over many more scenes and
// sizes, behind VULKAN_WIDE_SCENES.
func TestMatchesReferenceWide(t *testing.T) {
	n := envInt(t, "VULKAN_WIDE_SCENES", 0)
	if n == 0 {
		t.Skip("VULKAN_WIDE_SCENES=N runs N random scenes up to 3840 x 2160")
	}
	r := rendererRTE(t, 64, 48, false)
	tl := newTally()
	for seed := range uint64(n) {
		rg := rand.New(rand.NewPCG(seed, 0x51ed270b))
		w, h := logUniform(rg, 16, 3840), logUniform(rg, 16, 2160)
		if err := r.Resize(w, h); err != nil {
			t.Fatal(err)
		}
		l, _ := gpu2d.NewList(w, h)
		// Density follows the area: about 40 primitives per 64 x 48, capped.
		scenes.Random(l, 1_000_000+seed, min(40+w*h/4000, 2000))
		tl.compare(t, r, l, ground(w, h, seed%2 == 0), fmt.Sprintf("seed %d", seed))
	}
	tl.report(t, "random scenes, wide")
}

// TestCornerHeavy draws rounded rectangles and outlines of radii 1 to 64 at
// every position modulo the tile (scenes.Corners): the corner coverage is
// decided in integers by the kernel and must be the reference's on every
// device, whatever its sqrt. 60 scenes by default, VULKAN_WIDE_CORNERS=N for
// more.
func TestCornerHeavy(t *testing.T) {
	n, maxRad := envInt(t, "VULKAN_WIDE_CORNERS", 60), 64
	r := rendererRTE(t, 64, 48, false)
	tl := newTally()
	for seed := range uint64(n) {
		rg := rand.New(rand.NewPCG(seed, 0xf00d))
		w, h := 96+rg.IntN(420), 96+rg.IntN(420)
		if err := r.Resize(w, h); err != nil {
			t.Fatal(err)
		}
		l, _ := gpu2d.NewList(w, h)
		scenes.Corners(l, seed, maxRad)
		tl.compare(t, r, l, ground(w, h, seed%2 == 0), fmt.Sprintf("seed %d", seed))
	}
	tl.report(t, fmt.Sprintf("corner-heavy rounded rects, radii 1..%d", maxRad))
	if tl.instances < tl.scenes*100 {
		t.Fatalf("only %d instances in %d scenes: the corners barely draw", tl.instances, tl.scenes)
	}
}
