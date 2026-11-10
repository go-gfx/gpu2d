// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build !windows

// Command gendxbc compiles d3d11/kernel_gen.hlsl with D3DCompile; it can only
// run on Windows.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "gendxbc: D3DCompile only exists on Windows; run it there, or take the dxbc artifact of the CI lane \"d3d11 (windows, WARP)\" (internal/hlslgen: Regenerate)")
	os.Exit(1)
}
