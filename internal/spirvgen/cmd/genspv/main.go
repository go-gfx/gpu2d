// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Command genspv writes vulkan/kernel_gen.spv and vulkan/kernel_rte_gen.spv
// from kernel.wgsl. It is run by go generate in package vulkan:
//
//	//go:generate go run ../internal/spirvgen/cmd/genspv
package main

import (
	"fmt"
	"os"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/spirvgen"
)

func main() {
	for _, f := range []struct {
		name string
		rte  bool
	}{{"kernel_gen.spv", false}, {"kernel_rte_gen.spv", true}} {
		out, err := spirvgen.Kernel(gpu2d.KernelWGSL, f.rte)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := os.WriteFile(f.name, out, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}
