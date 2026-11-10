// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package metal

import (
	"errors"
	"math"
	"testing"
	"unsafe"

	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/msl"
	mtl "github.com/go-macos/metal"
)

// docs/NATIVE.md, "sqrt is the one operation no specification pins": the
// kernel's distance is length(), whose sqrt Vulkan and D3D11 leave 1 to 3 ulp
// loose. The proposal is a correctly rounded sqrt (the builtin's result, moved
// one step when Dekker's exact residual says so). These tests measure, on the
// GPU, whether it changes a byte of what the kernel computes: the distances a
// corner pixel can have, exhaustively, and random f32 inputs.

const sqrtKernels = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> u: vec4<u32>; // x: side of the grid, y: 0 builtin, 1 corrected

// x*y = p + the returned error, exactly (Dekker's two-product, Veltkamp's split).
fn twoProdErr(x: f32, y: f32, p: f32) -> f32 {
  let cx = 4097.0 * x;
  let xh = cx - (cx - x);
  let xl = x - xh;
  let cy = 4097.0 * y;
  let yh = cy - (cy - y);
  let yl = y - yh;
  return ((xh * yh - p) + xh * yl + xl * yh) + xl * yl;
}

fn sqrtCR(a: f32) -> f32 { return sqrtFix(a, sqrt(a)); }

// sqrtFix moves s, an approximation of the root of a within one step, to the
// correctly rounded one.
fn sqrtFix(a: f32, s: f32) -> f32 {
  // The residual is exact only while the two-product's error is a normal f32
  // (a above about 2^-60) and nothing overflows (a below 2^126); outside, the
  // builtin's result stands.
  if (!(a > 1e-18) || a > 1e37) { return s; }
  let p = s * s;
  let r = (a - p) - twoProdErr(s, s, p);
  let up = bitcast<f32>(bitcast<u32>(s) + 1u);
  let dn = bitcast<f32>(bitcast<u32>(s) - 1u);
  if (r > s * (up - s)) { return up; }
  if (r < -(s * (s - dn))) { return dn; }
  return s;
}

fn dist(dx: f32, dy: f32) -> f32 {
  return length(vec2f(dx, dy));
}

fn distCR(dx: f32, dy: f32) -> f32 {
  return sqrtCR(dx * dx + dy * dy);
}

@compute @workgroup_size(16, 8)
fn run(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.x || g.y >= u.x) { return; }
  let dx = f32(g.x) + 0.5;
  let dy = f32(g.y) + 0.5;
  var d = dist(dx, dy);
  if (u.y == 1u) { d = distCR(dx, dy); }
  out[g.y * u.x + g.x] = bitcast<u32>(d);
}

@compute @workgroup_size(64)
fn runRandom(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.x) { return; }
  // A spread of magnitudes from 2^-20 to 2^119 (the corrected sqrt's range)
  // with every mantissa pattern the hash produces.
  var h = g.x * 2654435761u + 40503u;
  h = h ^ (h >> 15u);
  h = h * 2246822519u;
  h = h ^ (h >> 13u);
  let e = 107u + (h % 140u);
  let a = bitcast<f32>((e << 23u) | ((h * 3266489917u) >> 9u));
  var d = sqrt(a);
  if (u.y == 1u) { d = sqrtCR(a); }
  // The corrector's own test: handed a root one step too high or too low, as
  // a loose sqrt could, it must still return the correctly rounded one.
  if (u.y == 2u) { d = sqrtFix(a, bitcast<f32>(bitcast<u32>(d) + 1u)); }
  if (u.y == 3u) { d = sqrtFix(a, bitcast<f32>(bitcast<u32>(d) - 1u)); }
  out[2u * g.x] = bitcast<u32>(a);
  out[2u * g.x + 1u] = bitcast<u32>(d);
}
`

func sqrtRun(t *testing.T, d *mtl.Device, entry string, words int, grid [2]int, mode uint32) []uint32 {
	t.Helper()
	m, err := wgsl.Parse(sqrtKernels)
	if err != nil {
		t.Fatal(err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	code, err := msl.Module(m, info, entry)
	if err != nil {
		t.Fatal(err)
	}
	lib, err := d.Compile(code)
	if err != nil {
		t.Fatalf("%v\n%s", err, code)
	}
	defer lib.Close()
	p, err := lib.Pipeline(entry)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	out, err := d.NewBuffer(4 * words)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	u := [4]uint32{uint32(grid[0]), mode, 0, 0}
	if err := d.Run(func(e *mtl.Encoder) {
		e.Use(p)
		e.Buffer(0, out)
		mtl.Constant(e, 1, &u)
		if grid[1] > 0 {
			e.Dispatch(grid[0], grid[1])
		} else {
			e.Dispatch(grid[0])
		}
	}); err != nil {
		t.Fatal(err)
	}
	b := out.Bytes()
	return append([]uint32(nil), unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), words)...)
}

func sqrtDevice(t *testing.T) *mtl.Device {
	t.Helper()
	d, err := mtl.Default()
	if errors.Is(err, mtl.ErrUnsupported) || errors.Is(err, mtl.ErrNoDevice) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	return d
}

// correctlyRounded is the f32 nearest the exact root of an f32: rounding the
// f64 root is correct because f64 carries more than 2·24+2 bits.
func correctlyRounded(a float32) float32 { return float32(math.Sqrt(float64(a))) }

// Every distance a corner pixel can have: the offsets from a corner centre are
// half-integers, and their squares and sum are exact in f32 up to 2048.
func TestSqrtOverEveryCornerDistance(t *testing.T) {
	const n = 2048
	d := sqrtDevice(t)
	var mismatch [2]int
	for mode := range uint32(2) {
		got := sqrtRun(t, d, "run", n*n, [2]int{n, n}, mode)
		for j := range n {
			for i := range n {
				dx, dy := float32(i)+0.5, float32(j)+0.5
				sum := float32(float32(dx*dx) + float32(dy*dy))
				if math.Float32frombits(got[j*n+i]) != correctlyRounded(sum) {
					mismatch[mode]++
				}
			}
		}
	}
	t.Logf("builtin length(): %d of %d distances are not the correctly rounded root; corrected: %d", mismatch[0], n*n, mismatch[1])
	if mismatch[1] != 0 {
		t.Errorf("the corrected sqrt missed %d distances", mismatch[1])
	}
}

// Random f32 inputs over 2^-20 .. 2^119.
func TestSqrtOverRandomInputs(t *testing.T) {
	const n = 1 << 24
	d := sqrtDevice(t)
	var mismatch [4]int
	for mode := range uint32(4) {
		got := sqrtRun(t, d, "runRandom", 2*n, [2]int{n, 0}, mode)
		for i := range n {
			a := math.Float32frombits(got[2*i])
			if math.Float32frombits(got[2*i+1]) != correctlyRounded(a) {
				mismatch[mode]++
			}
		}
	}
	t.Logf("builtin sqrt: %d of %d inputs are not correctly rounded; corrected: %d; corrected from a root one step high: %d, one step low: %d",
		mismatch[0], n, mismatch[1], mismatch[2], mismatch[3])
	// Mode 0 is only measured. Mode 1 must be exact. Modes 2 and 3 are the
	// corrector's break check (Metal's own sqrt is never wrong, so mode 1 alone
	// never exercises it): it must repair a root one step off, except where the
	// true root lies within about 2^-25 ulp of a rounding midpoint, which the
	// f32 residual cannot decide. That is about one input in 16 million, and
	// the reason the corrector is not in kernel.wgsl yet: an exact midpoint
	// test needs more terms (docs/NATIVE.md, risk 1).
	if mismatch[1] != 0 {
		t.Errorf("the corrected sqrt missed %d inputs", mismatch[1])
	}
	for mode := 2; mode < 4; mode++ {
		if mismatch[mode] > 4 {
			t.Errorf("mode %d: %d inputs not repaired", mode, mismatch[mode])
		}
	}
}
