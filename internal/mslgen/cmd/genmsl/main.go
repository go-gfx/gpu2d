// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Command genmsl writes metal/coverage_gen.metal from webgpu/shader.wgsl. It is
// run by go generate in package metal:
//
//	//go:generate go run ../internal/mslgen/cmd/genmsl
package main

import (
	"fmt"
	"os"

	"github.com/go-gfx/gpu2d/internal/mslgen"
	"github.com/go-gfx/gpu2d/webgpu"
)

func main() {
	out, err := mslgen.Coverage(webgpu.Shader)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.WriteFile("coverage_gen.metal", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
