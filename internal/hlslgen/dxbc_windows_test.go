// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build windows

package hlslgen

import (
	"regexp"
	"strings"
	"testing"

	"github.com/go-gfx/gpu2d"
)

func TestCompilerVersion(t *testing.T) {
	v, err := CompilerVersion()
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`^d3dcompiler_47\.dll \d+\.\d+\.\d+\.\d+$`).MatchString(v) {
		t.Fatalf("%q", v)
	}
	t.Log(v)
}

func TestBuild(t *testing.T) {
	hlsl, err := Kernel(gpu2d.KernelWGSL)
	if err != nil {
		t.Fatal(err)
	}
	dxbc, rec, err := Build(hlsl)
	if err != nil {
		t.Fatal(err)
	}
	if string(dxbc[:4]) != "DXBC" || rec.DXBC != Sum(dxbc) || rec.HLSL != Sum([]byte(hlsl)) || rec.Target != Target || rec.Entry != Entry || rec.Flags != Flags {
		t.Fatalf("%+v", rec)
	}
	again, rec2, err := Build(hlsl)
	if err != nil || string(again) != string(dxbc) || rec2 != rec {
		t.Fatal("two compiles of the same HLSL differ")
	}
	if _, _, err := Build(strings.Replace(hlsl, "void raster", "void raster_", 1)); err == nil {
		t.Fatal("a missing entry point compiled")
	}
}
