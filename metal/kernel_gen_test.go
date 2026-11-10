// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package metal

import (
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/mslgen"
)

// kernel_gen.metal is what kernel.wgsl generates today: a change to the WGSL
// that was not regenerated (go generate ./metal) fails here, on every
// platform, before the back-ends can disagree.
func TestKernelMetalIsGenerated(t *testing.T) {
	want, err := mslgen.Kernel(gpu2d.KernelWGSL)
	if err != nil {
		t.Fatal(err)
	}
	if kernelMSL != want {
		t.Fatal("metal/kernel_gen.metal is stale: run go generate ./metal")
	}
}
