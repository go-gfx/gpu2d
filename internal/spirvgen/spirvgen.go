// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package spirvgen makes the Vulkan back-end's raster kernel from kernel.wgsl,
// through go-gfx/wgsl's SPIR-V emitter, so the raster arithmetic is written
// once in WGSL (docs/NATIVE.md, decision 1).
package spirvgen

import (
	"fmt"

	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/spirv"
)

// Entry is the kernel's entry point.
const Entry = "raster"

// Kernel is the SPIR-V module for the compute entry point "raster", as the
// little-endian bytes that vulkan/kernel_gen.spv holds. Float operations carry
// NoContraction. With rte the module also declares RoundingModeRTE for 32-bit
// floats (vulkan/kernel_rte_gen.spv): a device may run it only where it reports
// shaderRoundingModeRTEFloat32, which go-gfx/vulkan v0.2.0 exposes as
// DeviceInfo.RoundingModeRTEFloat32.
func Kernel(kernel string, rte bool) ([]byte, error) {
	m, err := wgsl.Parse(kernel)
	if err != nil {
		return nil, err
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		return nil, fmt.Errorf("spirvgen: the kernel does not check: %v", errs)
	}
	words, err := spirv.Module(m, info, spirv.Options{RoundingModeRTE: rte}, Entry)
	if err != nil {
		return nil, err
	}
	return spirv.Bytes(words), nil
}
