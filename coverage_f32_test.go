// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import (
	"math"
	"testing"
)

// The corner coverage is one mathematical definition: with D the exact squared
// distance from the pixel centre to the corner circle's centre, a fill covers
// clamp(rad + 0.5 - sqrt(D), 0, 1) and an outline ring max(1 - |sqrt(D) -
// (rad - 0.5)|, 0). The offsets are half-integers, so 4·D is an exact integer.
// The CPU reference takes math.Sqrt of it (Coverage). The kernels decide the
// byte floor(alpha·cov + 0.5) from the integer alone, with no rounded value
// (kernel.wgsl, `coverage`); coverageAlpha below is that function, operation
// for operation on 32-bit words, and the tests hold it to Coverage over the
// whole domain a corner pixel can have: (kind, radius, offset in the corner
// box, alpha). The kernels themselves are held to Rasterize on every device by
// the back-ends' tests.

// mul64 is a·b as (low, high) words, from 16-bit halves.
func mul64(a, b uint32) (lo, hi uint32) {
	a0, a1 := a&65535, a>>16
	b0, b1 := b&65535, b>>16
	p00, p01, p10, p11 := a0*b0, a0*b1, a1*b0, a1*b1
	mid := (p00 >> 16) + (p01 & 65535) + (p10 & 65535)
	return (p00 & 65535) | (mid << 16), p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16)
}

func le64(alo, ahi, blo, bhi uint32) bool { return ahi < bhi || (ahi == bhi && alo <= blo) }

// isqrt64 mirrors the kernel's: the estimate is corrected with exact products.
func isqrt64(nlo, nhi, est uint32) uint32 {
	r := est
	for range 16 {
		lo, hi := mul64(r, r)
		if !le64(lo, hi, nlo, nhi) {
			r--
			continue
		}
		lo, hi = mul64(r+1, r+1)
		if le64(lo, hi, nlo, nhi) {
			r++
			continue
		}
		break
	}
	return r
}

// coverageAlpha mirrors kernel.wgsl's coverage for a pixel in a corner box:
// i, j are the pixel's offsets in the box (0..rad-1), the circle's centre at
// (rad, rad); the estimate of the root is f32 with an arbitrary error
// (errEst moves it), which the exact corrections must absorb.
func coverageAlpha(kind Kind, rad, i, j int, ca uint32, errEst int) uint32 {
	dx := int32(2*(i-rad) + 1)
	dy := int32(2*(j-rad) + 1)
	d4 := uint32(dx*dx + dy*dy)
	m := uint32(2*rad - 1)
	nlo, nhi := mul64(ca*ca, d4)
	est := uint32(max(int(float32(ca)*float32(math.Sqrt(float64(float32(d4)))))+errEst, 0)) // u32(f32) is never negative
	fl := isqrt64(nlo, nhi, est)
	sqlo, sqhi := mul64(fl, fl)
	exact := sqlo == nlo && sqhi == nhi
	if kind == KindFill {
		h := uint32(2*rad + 1)
		if d4 >= h*h {
			return 0
		}
		if d4 <= m*m {
			return ca
		}
		t := ca*h + 1
		u := fl
		if !exact {
			u++
		}
		if (t-u)&1 != 0 {
			u++
		}
		return (t - u) / 2
	}
	if d4 == m*m {
		return ca
	}
	if d4 >= (m+2)*(m+2) {
		return 0
	}
	if m >= 2 && d4 <= (m-2)*(m-2) {
		return 0
	}
	if d4 > m*m {
		t := 2*ca + 1 + ca*m
		u := fl
		if !exact {
			u++
		}
		if (t-u)&1 != 0 {
			u++
		}
		return (t - u) / 2
	}
	t := 2*int32(ca) + 1 - int32(ca*m)
	u := int32(fl)
	if (u-t)&1 != 0 {
		u--
	}
	return uint32((u + t) / 2)
}

// referenceAlpha is what Rasterize paints: the alpha floor(a·cov + 0.5) of
// Coverage, or a for cov >= 1; 0 where it paints nothing.
func referenceAlpha(kind Kind, rad, i, j int, a uint8) uint32 {
	cov := Coverage(kind, i, j, Rect{W: 2 * rad, H: 2 * rad}, rad)
	switch {
	case cov <= 0:
		return 0
	case cov < 1:
		return uint32(uint8(float64(a)*cov + 0.5))
	}
	return uint32(a)
}

// oldAlpha is the reference as it was before cornerDist: math.Hypot.
func oldAlpha(kind Kind, rad, i, j int, a uint8) uint32 {
	d := math.Hypot(float64(i)+0.5-float64(rad), float64(j)+0.5-float64(rad))
	var cov float64
	if kind == KindFill {
		cov = min(max(float64(rad)+0.5-d, 0), 1)
	} else {
		cov = max(1-math.Abs(d-(float64(rad)-0.5)), 0)
	}
	switch {
	case cov <= 0:
		return 0
	case cov < 1:
		return uint32(uint8(float64(a)*cov + 0.5))
	}
	return uint32(a)
}

// TestKernelCoverageIsTheReference: the integer algorithm equals Coverage on
// every corner pixel of every radius 1..64 with every alpha, and on a sample of
// radii up to MaxRadius; and it is indifferent to the sqrt estimate it starts
// from (off by up to 3 either way, far more than any device's sqrt). It logs
// how many bytes the old math.Hypot reference painted differently.
func TestKernelCoverageIsTheReference(t *testing.T) {
	if raceEnabled {
		t.Skip("hundreds of millions of evaluations: not under the race detector")
	}
	cases, hypotOff := 0, 0
	check := func(rad, i, j int, a uint8, kind Kind) {
		want := referenceAlpha(kind, rad, i, j, a)
		for _, e := range []int{0, 3, -3} {
			if got := coverageAlpha(kind, rad, i, j, uint32(a), e); got != want {
				t.Fatalf("kind %d radius %d pixel (%d,%d) alpha %d, estimate off by %d: kernel %d, reference %d", kind, rad, i, j, a, e, got, want)
			}
		}
		if oldAlpha(kind, rad, i, j, a) != want {
			hypotOff++
		}
		cases++
	}
	for _, kind := range []Kind{KindFill, KindOutline} {
		for rad := 1; rad <= 64; rad++ {
			for j := range rad {
				for i := range rad {
					for a := 1; a <= 255; a++ {
						check(rad, i, j, uint8(a), kind)
					}
				}
			}
		}
		if testing.Short() {
			continue
		}
		for _, rad := range []int{65, 100, 255, 256, 1000, 1448, 1449, 4095, 4096, 10000, 16382, MaxRadius} {
			for j := 0; j < rad; j += 1 + rad/300 {
				for i := range rad {
					for _, a := range []uint8{1, 2, 64, 127, 128, 200, 254, 255} {
						check(rad, i, j, a, kind)
					}
				}
			}
		}
	}
	t.Logf("%d cases: kernel algorithm == reference on all; the math.Hypot reference differed from it on %d (%.6f%%)",
		cases, hypotOff, 100*float64(hypotOff)/float64(cases))
}

// A radius past MaxRadius is refused, not drawn wrongly: 4·D would pass 32 bits.
func TestRadiusPastMaxRadiusIsRefused(t *testing.T) {
	l, _ := NewList(40000, 40000)
	l.FillRoundRect(Rect{W: 40000, H: 40000}, MaxRadius+1, RGBA{A: 255})
	if len(l.Instances()) != 0 || l.Err() != ErrTooLarge {
		t.Fatalf("%d instances, Err %v", len(l.Instances()), l.Err())
	}
	l.Reset()
	l.FillRoundRect(Rect{W: 40000, H: 40000}, MaxRadius, RGBA{A: 255})
	if len(l.Instances()) != 1 || l.Err() != nil {
		t.Fatalf("a radius of MaxRadius: %d instances, Err %v", len(l.Instances()), l.Err())
	}
}
