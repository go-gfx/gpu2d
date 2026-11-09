// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package webgpu

import (
	"strings"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/wgsl"
)

// The shader parses and type-checks on every platform, before any browser
// sees it; the browser stays the authority (go-gfx/wgsl's own caveat).
func TestShaderValidates(t *testing.T) {
	if err := wgsl.Validate(Shader); err != nil {
		t.Fatal(err)
	}
	if err := wgsl.Validate(PresentShader); err != nil {
		t.Fatalf("PresentShader: %v", err)
	}
	if err := wgsl.Validate(computePresentShader); err != nil {
		t.Fatalf("computePresentShader: %v", err)
	}
	if err := wgsl.Validate(gpu2d.KernelWGSL); err != nil {
		t.Fatalf("KernelWGSL: %v", err)
	}
}

// The kernel is one compute entry point, 16 × 8, over the bindings its doc
// comment names: what the other emitters (docs/NATIVE.md) are told to expect.
func TestKernelShape(t *testing.T) {
	m, err := wgsl.Parse(gpu2d.KernelWGSL)
	if err != nil {
		t.Fatal(err)
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	if len(info.EntryPoints) != 1 {
		t.Fatalf("%d entry points, want 1", len(info.EntryPoints))
	}
	if ep := info.EntryPoints[0]; ep.Name != "raster" || ep.Stage != wgsl.StageCompute || ep.WorkgroupSize != [3]int64{16, 8, 1} {
		t.Fatalf("entry point %q stage %v workgroup %v", ep.Name, ep.Stage, ep.WorkgroupSize)
	}
	for _, want := range []string{
		"@compute @workgroup_size(16, 8)",
		"@group(0) @binding(0) var<storage, read> insts",
		"@group(0) @binding(1) var<storage, read> bins",
		"@group(0) @binding(2) var<storage, read> arena",
		"@group(0) @binding(3) var<storage, read_write> px",
		"@group(0) @binding(4) var<uniform> f",
	} {
		if !strings.Contains(gpu2d.KernelWGSL, want) {
			t.Errorf("the kernel lacks %q", want)
		}
	}
}

// The kernel's coverage functions are the fixed-function shader's, character
// for character: one arithmetic, two shaders.
func TestKernelCoverageIsTheShaders(t *testing.T) {
	cut := func(src string) string {
		i := strings.Index(src, "// cornerCenter:")
		j := strings.Index(src, "fn coverage(")
		if i < 0 || j < i {
			t.Fatal("no coverage functions in a shader")
		}
		end := strings.Index(src[j:], "\n}\n")
		if end < 0 {
			t.Fatal("coverage does not end")
		}
		return src[i : j+end]
	}
	if a, b := cut(Shader), cut(gpu2d.KernelWGSL); a != b {
		t.Fatalf("kernel.wgsl and shader.wgsl disagree on the coverage functions:\n--- shader.wgsl\n%s\n--- kernel.wgsl\n%s", a, b)
	}
}
