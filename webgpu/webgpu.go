// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package webgpu draws a gpu2d list with WebGPU in the browser
// (GOOS=js GOARCH=wasm), through go-gfx/webgpu. One quad per instance, the
// coverage of gpu2d.Coverage in the fragment shader, fixed-function blending;
// the result matches gpu2d's CPU reference within 1 per channel. NewCompute
// draws with the compute kernel gpu2d.KernelWGSL instead, byte-identical to
// the reference over any ground.
package webgpu

import _ "embed"

// Shader is the WGSL module: vertex entry point "vs", fragment entry point
// "fs", the instance array at @group(0) @binding(0) and the surface size at
// @binding(1).
//
//go:embed shader.wgsl
var Shader string

// PresentShader draws the target into a canvas texture of another format
// (Renderer.Present to a bgra8unorm surface): one triangle covering the
// canvas, each fragment loading the target's texel at its own pixel.
// unorm8 → float → unorm8 gives back the same byte, so the result is the
// copy's, in the canvas' channel order.
const PresentShader = `
@group(0) @binding(0) var frame: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(frame, vec2i(pos.xy), 0);
}
`

// computePresentShader draws a compute-mode renderer's pixel buffer into a
// canvas: one triangle covering it, each fragment unpacking its own pixel.
// unorm8 → float → unorm8 gives back the same byte, in the canvas' channel
// order, whatever the canvas' format.
const computePresentShader = `
@group(0) @binding(0) var<storage, read> px: array<u32>;
@group(0) @binding(1) var<uniform> f: vec4<u32>; // w, h, ...

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return unpack4x8unorm(px[u32(pos.y) * f.x + u32(pos.x)]);
}
`
