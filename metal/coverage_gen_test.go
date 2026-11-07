// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package metal

import (
	"testing"

	"github.com/go-gfx/gpu2d/internal/mslgen"
	"github.com/go-gfx/gpu2d/webgpu"
)

// coverage_gen.metal is what the WebGPU shader generates today: a change to the
// WGSL that was not regenerated (go generate ./metal) fails here, on every
// platform, before the two back-ends can disagree.
func TestCoverageMetalIsGenerated(t *testing.T) {
	want, err := mslgen.Coverage(webgpu.Shader)
	if err != nil {
		t.Fatal(err)
	}
	if coverageMSL != want {
		t.Fatal("metal/coverage_gen.metal is stale: run go generate ./metal")
	}
}
