// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package d3d11

import (
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/hlslgen"
)

// kernel_gen.hlsl is what kernel.wgsl generates today: a change to the WGSL (or
// to go-gfx/wgsl's emitter) that was not regenerated (go generate ./d3d11) fails
// here, on every platform and with no GPU, before the back-ends can disagree.
func TestKernelHLSLIsGenerated(t *testing.T) {
	want, err := hlslgen.Kernel(gpu2d.KernelWGSL)
	if err != nil {
		t.Fatal(err)
	}
	if KernelHLSL != want {
		t.Fatal("d3d11/kernel_gen.hlsl is stale: run go generate ./d3d11 (any platform), then regenerate the DXBC (hlslgen.Regenerate)")
	}
}

// The record says which HLSL the committed DXBC was compiled from, and what the
// DXBC is. A kernel_gen.hlsl that moved on without the DXBC fails here on every
// platform, not only on the Windows lane that can recompile.
func TestKernelDXBCRecord(t *testing.T) {
	rec, err := hlslgen.ParseRecord(KernelDXBCRecord)
	if err != nil {
		t.Fatalf("d3d11/kernel_gen.dxbc.record: %v\n%s", err, hlslgen.Regenerate)
	}
	if rec.Entry != hlslgen.Entry || rec.Target != hlslgen.Target {
		t.Errorf("the record says %s %s, not %s %s", rec.Target, rec.Entry, hlslgen.Target, hlslgen.Entry)
	}
	if len(KernelDXBC) < 32 || string(KernelDXBC[:4]) != "DXBC" {
		t.Fatalf("d3d11/kernel_gen.dxbc is not a DXBC container (%d bytes)\n%s", len(KernelDXBC), hlslgen.Regenerate)
	}
	if got := hlslgen.Sum([]byte(KernelHLSL)); rec.HLSL != got {
		t.Errorf("d3d11/kernel_gen.dxbc was compiled from other HLSL than d3d11/kernel_gen.hlsl (record %s, file %s): the HLSL changed and the DXBC was not remade.\n%s", rec.HLSL, got, hlslgen.Regenerate)
	}
	if got := hlslgen.Sum(KernelDXBC); rec.DXBC != got {
		t.Errorf("d3d11/kernel_gen.dxbc is not the file its record describes (record %s, file %s).\n%s", rec.DXBC, got, hlslgen.Regenerate)
	}
}
