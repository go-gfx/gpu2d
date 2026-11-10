// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build windows

package hlslgen

import (
	"fmt"
	"path/filepath"
	"unsafe"

	"github.com/go-mswin/d3d11/d3dcompiler"
	"golang.org/x/sys/windows"
)

// Flags are what the HLSL is compiled with: optimised, with IEEE strictness
// (docs/HLSL.md, rule 1 of go-gfx/wgsl) and every warning an error.
const Flags = d3dcompiler.FlagOptimizationLevel3 | d3dcompiler.FlagIEEEStrictness | d3dcompiler.FlagWarningsAreErrors

// CompilerVersion is the file version of the d3dcompiler DLL the system loads,
// as "d3dcompiler_47.dll 10.0.26100.1".
func CompilerVersion() (string, error) {
	dir, err := windows.GetSystemDirectory()
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, d3dcompiler.Version)
	size, err := windows.GetFileVersionInfoSize(path, nil)
	if err != nil {
		return "", fmt.Errorf("hlslgen: version of %s: %w", path, err)
	}
	block := make([]byte, size)
	if err := windows.GetFileVersionInfo(path, 0, size, unsafe.Pointer(&block[0])); err != nil {
		return "", fmt.Errorf("hlslgen: version of %s: %w", path, err)
	}
	var fixed unsafe.Pointer
	var n uint32
	if err := windows.VerQueryValue(unsafe.Pointer(&block[0]), `\`, unsafe.Pointer(&fixed), &n); err != nil || fixed == nil || n < uint32(unsafe.Sizeof(windows.VS_FIXEDFILEINFO{})) {
		return "", fmt.Errorf("hlslgen: version of %s: no fixed file info (%v)", path, err)
	}
	v := (*windows.VS_FIXEDFILEINFO)(fixed)
	return fmt.Sprintf("%s %d.%d.%d.%d", d3dcompiler.Version, v.FileVersionMS>>16, v.FileVersionMS&0xffff, v.FileVersionLS>>16, v.FileVersionLS&0xffff), nil
}

// Build compiles hlsl for cs_5_0 and returns the DXBC with the record of how it
// was made.
func Build(hlsl string) ([]byte, Record, error) {
	ver, err := CompilerVersion()
	if err != nil {
		return nil, Record{}, err
	}
	dxbc, err := d3dcompiler.Compile([]byte(hlsl), "kernel_gen.hlsl", Entry, Target, Flags)
	if err != nil {
		return nil, Record{}, err
	}
	return dxbc, Record{Compiler: ver, Target: Target, Entry: Entry, Flags: Flags, HLSL: Sum([]byte(hlsl)), DXBC: Sum(dxbc)}, nil
}
