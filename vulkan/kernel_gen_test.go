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

// kernel_gen.spv and kernel_rte_gen.spv are what kernel.wgsl generates today:
// a change to the WGSL that was not regenerated (go generate ./vulkan) fails
// here, on every platform and with no GPU, before the back-ends can disagree.
func TestKernelSPIRVIsGenerated(t *testing.T) {
	for _, c := range []struct {
		name string
		rte  bool
		have []byte
	}{{"kernel_gen.spv", false, KernelSPV}, {"kernel_rte_gen.spv", true, KernelRTESPV}} {
		want, err := spirvgen.Kernel(gpu2d.KernelWGSL, c.rte)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(c.have, want) {
			t.Fatalf("vulkan/%s is stale: run go generate ./vulkan", c.name)
		}
	}
	if bytes.Equal(KernelSPV, KernelRTESPV) {
		t.Fatal("the RoundingModeRTE module is the plain one")
	}
}
