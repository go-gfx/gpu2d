// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package d3d11 will draw a gpu2d list on a Direct3D 11 device with a compute
// kernel, through go-mswin/d3d11 (milestone D3, docs/NATIVE.md). Today it holds
// the raster kernel only, generated from kernel.wgsl so that the arithmetic
// exists once: no back-end yet.
//
// The D3D11 back-end ships opt-in and is proven on WARP in CI only; no Windows
// GPU has run it (docs/NATIVE.md, "Decisions taken").
package d3d11

import _ "embed"

//go:generate go run ../internal/hlslgen/cmd/genhlsl

// KernelHLSL is the HLSL (Shader Model 5.0) of the raster kernel, generated from
// kernel.wgsl by internal/hlslgen through go-gfx/wgsl's hlsl package: entry point
// "raster", a 16 x 8 thread group, the buffers of the WGSL at the registers its
// bindings name (docs/HLSL.md of go-gfx/wgsl). Any platform regenerates it
// (go generate ./d3d11); a test on every platform checks it is current.
//
//go:embed kernel_gen.hlsl
var KernelHLSL string

// KernelDXBC is KernelHLSL compiled for cs_5_0 by D3DCompile. Only Windows can
// remake it (go run ./internal/hlslgen/cmd/gendxbc), so the bytes are committed,
// with KernelDXBCRecord, which says which compiler DLL made them, with which
// flags, from which HLSL. The Windows CI lane recompiles and compares, with no
// tolerance.
//
//go:embed kernel_gen.dxbc
var KernelDXBC []byte

// KernelDXBCRecord is the text of kernel_gen.dxbc.record (internal/hlslgen.Record).
//
//go:embed kernel_gen.dxbc.record
var KernelDXBCRecord string
