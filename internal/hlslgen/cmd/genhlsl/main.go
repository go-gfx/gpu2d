// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Command genhlsl writes d3d11/kernel_gen.hlsl from kernel.wgsl. It runs on any
// platform and is run by go generate in package d3d11:
//
//	//go:generate go run ../internal/hlslgen/cmd/genhlsl
//
// The DXBC is made by gendxbc, on Windows only.
package main

import (
	"fmt"
	"os"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/hlslgen"
)

func main() {
	out, err := hlslgen.Kernel(gpu2d.KernelWGSL)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.WriteFile("kernel_gen.hlsl", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
