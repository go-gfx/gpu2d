// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import (
	"fmt"
	"math"
	"testing"
)

// The kernel computes a corner's coverage in f32 and the reference in f64, so
// a byte can differ where the f64 coverage times the alpha lands within f32's
// error of a rounding boundary. These tests enumerate that domain without a
// GPU: the coverage of a corner pixel depends on nothing but (kind, radius,
// the pixel's offset in the corner box, the colour's alpha), so every case up
// to radius 64 is checked, and the kernel is mirrored operation by operation
// in float32 (Go rounds each float32 operation, and the explicit conversions
// below forbid fusing a multiply into an add).
//
// Three evaluations are compared with Coverage's byte:
//
//	kernel      kernel.wgsl in f32 with a correctly rounded sqrt (lavapipe's)
//	kernel+ulp  the same with sqrt off by one ULP on about one input in five,
//	            which is what spirv/vulkantest measures on RADV
//	ds          the corrector proposed in docs/NATIVE.md: the squared distance
//	            exactly in integers, sqrt refined by one Newton step whose
//	            residual is computed exactly (error-free transformations, no
//	            FMA), the coverage and its product with the alpha in
//	            double-single, so the result does not depend on the sqrt's
//	            rounding; with the same perturbed sqrt.

type f32 = float32

// twoSum is Knuth's error-free addition: s + e == a + b exactly.
func twoSum(a, b f32) (s, e f32) {
	s = f32(a + b)
	bb := f32(s - a)
	e = f32(f32(a-f32(s-bb)) + f32(b-bb))
	return
}

// split is Veltkamp's: a == hi + lo, each with at most 12 significant bits.
func split(a f32) (hi, lo f32) {
	c := f32(4097 * a)
	hi = f32(c - f32(c-a))
	lo = f32(a - hi)
	return
}

// twoProd is Dekker's error-free product: p + e == a * b exactly.
func twoProd(a, b f32) (p, e f32) {
	p = f32(a * b)
	ah, al := split(a)
	bh, bl := split(b)
	e = f32(f32(f32(f32(ah*bh)-p)+f32(ah*bl)+f32(al*bh)) + f32(al*bl))
	return
}

// perturb returns s moved by one ULP on about one input in five, by a hash of
// the input, as a sqrt that is not correctly rounded would.
func perturb(in, s f32) f32 {
	h := math.Float32bits(in) * 2654435761
	switch h >> 29 {
	case 0:
		return math.Nextafter32(s, float32(math.Inf(1)))
	case 1:
		return math.Nextafter32(s, 0)
	}
	return s
}

// refByte is the reference: the alpha the CPU blends with, or -1 for a pixel
// it skips (coverage <= 0). It is Coverage's own arithmetic.
func refByte(kind Kind, rad, i, j int, a uint8) int {
	r := Rect{X: 0, Y: 0, W: 2 * rad, H: 2 * rad}
	cov := Coverage(kind, i, j, r, rad)
	if cov <= 0 {
		return -1
	}
	if cov < 1 {
		return int(uint8(float64(a)*cov + 0.5))
	}
	return int(a)
}

// kernelByte is kernel.wgsl's coverage and alpha in f32. sqrt is the sqrt
// under test.
func kernelByte(kind Kind, rad, i, j int, a uint8, sqrt func(f32) f32) int {
	dx, dy := f32(i)+0.5-f32(rad), f32(j)+0.5-f32(rad)
	d := sqrt(f32(f32(dx*dx) + f32(dy*dy)))
	var cov f32
	if kind == KindFill {
		cov = f32(f32(rad)+0.5) - d
		cov = min(max(cov, 0), 1)
	} else {
		cov = max(1-f32(abs32(f32(d-f32(f32(rad)-0.5)))), 0)
	}
	if cov <= 0 {
		return -1
	}
	if cov < 1 {
		return int(uint32(f32(f32(f32(a)*cov) + 0.5)))
	}
	return int(a)
}

func abs32(x f32) f32 { return f32(math.Abs(float64(x))) }

// dsByte is the proposed corrector. D = 4(dx² + dy²) is an integer, exact in
// f32 for radii up to 1448 (checked by the caller); d = sqrt(D) / 2.
func dsByte(kind Kind, rad, i, j int, a uint8, sqrt func(f32) f32) int {
	ui, uj := f32(2*i+1-2*rad), f32(2*j+1-2*rad)
	dd := f32(f32(ui*ui) + f32(uj*uj)) // exact: an integer below 2^24
	s0 := sqrt(dd)                     // within a few ULP of sqrt(dd), however rounded
	// residual dd - s0² exactly: s0² = ph + pl, dd - ph is exact (they are
	// within a factor 2^-12 of each other), then minus pl.
	ph, pl := twoProd(s0, s0)
	res := f32(f32(dd-ph) - pl)
	c := f32(res / f32(2*s0)) // sqrt(dd) = s0 + c to ~2^-48 relative; dd > 0 always
	// d = (s0 + c) / 2, in double-single: halves are exact.
	dh, dl := f32(s0*0.5), f32(c*0.5)
	// e = d - t, t a half-integer-valued f32 (rad + 0.5 or rad - 0.5).
	sub := func(t f32) (hi, lo f32) {
		h, l := twoSum(dh, -t)
		l = f32(l + dl)
		return twoSum(h, l) // renormalised; |l| << |h| except at cancellation
	}
	var chi, clo f32 // coverage = chi + clo
	if kind == KindFill {
		h, l := sub(f32(rad) + 0.5)
		chi, clo = -h, -l // rad + 0.5 - d
	} else {
		h, l := sub(f32(rad) - 0.5)
		if h < 0 || (h == 0 && l < 0) {
			h, l = -h, -l
		}
		// 1 - |e|
		s, e := twoSum(1, -h)
		chi, clo = twoSum(s, f32(e-l))
	}
	// clamp and classify exactly on (chi, clo)
	if chi < 0 || (chi == 0 && clo <= 0) {
		return -1
	}
	if chi > 1 || (chi == 1 && clo >= 0) {
		return int(a)
	}
	// floor(a * cov + 0.5): a * chi exactly as ph + pl, plus a * clo
	p, pe := twoProd(f32(a), chi)
	s, se := twoSum(p, 0.5)
	frac := f32(f32(se+pe) + f32(f32(a)*clo)) // tiny, exact enough
	k := f32(f32(math.Floor(float64(s))))
	rem := f32(f32(s-k) + frac)
	switch {
	case rem >= 1:
		k++
	case rem < 0:
		k--
	}
	return int(uint32(k))
}

func hardSqrt(x f32) f32 { return f32(math.Sqrt(float64(x))) }
func ulpSqrt(x f32) f32  { return perturb(x, hardSqrt(x)) }

// TestF32CornerCoverageDivergence enumerates every corner pixel of every
// radius 1..64 for both kinds and every alpha 1..255, then radii up to 1024
// at six alphas, and counts the bytes that differ from the reference. It logs
// the counts and asserts only what is proved: the double-single corrector has
// none, with an exact or a perturbed sqrt, and the plain f32 kernel is
// affected (so the finding stays visible).
func TestF32CornerCoverageDivergence(t *testing.T) {
	if raceEnabled {
		t.Skip("665 million evaluations: not under the race detector")
	}
	type counts struct{ cases, kernel, ulp, ds, dsUlp int }
	var total counts
	byRad := map[int]*counts{}
	firstDS := ""
	run := func(rad int, alphas []uint8) {
		c := byRad[rad]
		if c == nil {
			c = &counts{}
			byRad[rad] = c
		}
		for _, kind := range []Kind{KindFill, KindOutline} {
			for j := range rad {
				for i := range rad {
					for _, a := range alphas {
						want := refByte(kind, rad, i, j, a)
						c.cases++
						if kernelByte(kind, rad, i, j, a, hardSqrt) != want {
							c.kernel++
						}
						if kernelByte(kind, rad, i, j, a, ulpSqrt) != want {
							c.ulp++
						}
						if g := dsByte(kind, rad, i, j, a, hardSqrt); g != want {
							c.ds++
							if firstDS == "" {
								firstDS = fmt.Sprint("kind/rad/i/j/a/want/got: ", int(kind), rad, i, j, a, want, g)
							}
						}
						if g := dsByte(kind, rad, i, j, a, ulpSqrt); g != want {
							c.dsUlp++
						}
					}
				}
			}
		}
	}
	all := make([]uint8, 255)
	for k := range all {
		all[k] = uint8(k + 1)
	}
	for rad := 1; rad <= 64; rad++ {
		run(rad, all)
	}
	if !testing.Short() {
		for rad := 65; rad <= 1024; rad += 7 {
			run(rad, []uint8{1, 64, 127, 128, 200, 255})
		}
	}
	first, last := 0, 0
	for rad := 1; rad <= 1024; rad++ {
		c := byRad[rad]
		if c == nil {
			continue
		}
		total.cases += c.cases
		total.kernel += c.kernel
		total.ulp += c.ulp
		total.ds += c.ds
		total.dsUlp += c.dsUlp
		if c.kernel > 0 && first == 0 {
			first = rad
		}
		if c.kernel > 0 {
			last = rad
		}
	}
	t.Logf("%d cases: f32 kernel differs on %d, with a 1-ULP sqrt on %d; double-single corrector differs on %d (exact sqrt), %d (1-ULP sqrt)",
		total.cases, total.kernel, total.ulp, total.ds, total.dsUlp)
	t.Logf("the f32 kernel first differs at radius %d (last seen %d)", first, last)
	for _, rad := range []int{1, 8, 13, 16, 24, 32, 40, 48, 56, 64} {
		c := byRad[rad]
		t.Logf("radius %2d: %8d cases, kernel %4d, kernel+ulp %4d, ds %d, ds+ulp %d", rad, c.cases, c.kernel, c.ulp, c.ds, c.dsUlp)
	}
	if total.ds != 0 || total.dsUlp != 0 {
		t.Errorf("the corrector differs from the reference (%s)", firstDS)
	}
	if total.kernel == 0 {
		t.Error("the f32 kernel never differs: the test no longer shows the divergence it documents")
	}
}
