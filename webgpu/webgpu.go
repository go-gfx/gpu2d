// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package webgpu draws a gpu2d list with WebGPU in the browser
// (GOOS=js GOARCH=wasm), through go-gfx/webgpu. One quad per instance, the
// coverage of gpu2d.Coverage in the fragment shader, fixed-function blending;
// the result matches gpu2d's CPU reference within 1 per channel.
package webgpu

import _ "embed"

// Shader is the WGSL module: vertex entry point "vs", fragment entry point
// "fs", the instance array at @group(0) @binding(0) and the surface size at
// @binding(1).
//
//go:embed shader.wgsl
var Shader string
