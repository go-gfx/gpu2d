// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build windows

// Command gendxbc compiles d3d11/kernel_gen.hlsl with D3DCompile and writes
// d3d11/kernel_gen.dxbc and d3d11/kernel_gen.dxbc.record. It runs on Windows
// only, from the repository root:
//
//	go run ./internal/hlslgen/cmd/gendxbc
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/go-gfx/gpu2d/internal/hlslgen"
)

func main() {
	if err := run("d3d11"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(dir string) error {
	hlsl, err := os.ReadFile(filepath.Join(dir, "kernel_gen.hlsl"))
	if err != nil {
		return err
	}
	dxbc, rec, err := hlslgen.Build(string(hlsl))
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "kernel_gen.dxbc"), dxbc, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "kernel_gen.dxbc.record"), []byte(rec.String()), 0o644)
}
