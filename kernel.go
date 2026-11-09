// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import _ "embed"

// KernelWGSL is the whole raster kernel as one WGSL compute shader, entry
// point "raster", workgroup 16 × 8, one invocation per pixel. It binds, at
// @group(0): the instances (@binding(0), InstanceSize bytes each), the bins
// (@binding(1): List.Bin's spans followed by its list, one buffer), the arena
// (@binding(2), List.Data padded to a whole word), the pixels (@binding(3),
// one little-endian RGBA word each, read and written) and a uniform
// {w, h, tiles across, Tile} of four words (@binding(4)). It blends in
// integers as Rasterize does, so a frame is byte-identical to Rasterize's
// over any ground. docs/NATIVE.md, decision 1, says why it is one WGSL file.
//
//go:embed kernel.wgsl
var KernelWGSL string
