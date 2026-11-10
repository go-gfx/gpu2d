// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build windows

package d3d11

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-gfx/gpu2d/internal/hlslgen"
)

// TestKernelDXBCIsCompiled recompiles kernel_gen.hlsl with the system D3DCompile
// and compares the bytes with kernel_gen.dxbc. There is no tolerance: if the
// runner image's compiler DLL moved, the bytes moved, and the failure says so by
// printing the committed record next to the one just made (hlslgen.Diagnose).
//
// With D3D11_DXBC_OUT=<dir> set it also writes the fresh kernel_gen.dxbc and
// kernel_gen.dxbc.record there (CI uploads the directory as the artifact "dxbc"),
// which is how the committed files are made without a Windows machine.
func TestKernelDXBCIsCompiled(t *testing.T) {
	dxbc, now, err := hlslgen.Build(KernelHLSL)
	if err != nil {
		t.Fatalf("compiling d3d11/kernel_gen.hlsl: %v", err)
	}
	if dir := os.Getenv("D3D11_DXBC_OUT"); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "kernel_gen.dxbc"), dxbc, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "kernel_gen.dxbc.record"), []byte(now.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("compiled with %s: %d bytes of DXBC", now.Compiler, len(dxbc))
	committed, err := hlslgen.ParseRecord(KernelDXBCRecord)
	if err != nil {
		t.Fatalf("d3d11/kernel_gen.dxbc.record: %v\n%s", err, hlslgen.Regenerate)
	}
	if !bytes.Equal(dxbc, KernelDXBC) {
		t.Fatalf("%s\n(committed %d bytes, sha256 %s; fresh %d bytes, sha256 %s)", hlslgen.Diagnose(committed, now), len(KernelDXBC), hlslgen.Sum(KernelDXBC), len(dxbc), now.DXBC)
	}
	// Equal bytes, so the record must be exactly this compile's.
	if committed != now {
		t.Fatalf("the bytes match but the record does not: committed\n%s now\n%s", committed, now)
	}
}
