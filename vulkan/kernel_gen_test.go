// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package vulkan

import (
	"bytes"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/spirvgen"
)

// kernel_gen.spv is what kernel.wgsl generates today: a change to the WGSL
// that was not regenerated (go generate ./vulkan) fails here, on every
// platform and with no GPU, before the back-ends can disagree.
func TestKernelSPIRVIsGenerated(t *testing.T) {
	want, err := spirvgen.Kernel(gpu2d.KernelWGSL)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(KernelSPV, want) {
		t.Fatal("vulkan/kernel_gen.spv is stale: run go generate ./vulkan")
	}
}
