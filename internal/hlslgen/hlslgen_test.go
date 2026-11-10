// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package hlslgen

import (
	"strings"
	"testing"

	"github.com/go-gfx/gpu2d"
)

func TestKernel(t *testing.T) {
	out, err := Kernel(gpu2d.KernelWGSL)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"[numthreads(16, 8, 1)]", "void raster(", "RWStructuredBuffer<uint> px : register(u3);"} {
		if !strings.Contains(out, want) {
			t.Errorf("the HLSL lacks %q", want)
		}
	}
	if _, err := Kernel("fn ("); err == nil {
		t.Error("a module that does not parse")
	}
	if _, err := Kernel("fn f() -> u32 { return 1.5; }"); err == nil {
		t.Error("a module that does not check")
	}
	if _, err := Kernel("fn f() {}"); err == nil {
		t.Error("a module with no raster entry point")
	}
}

func sample() Record {
	return Record{Compiler: "d3dcompiler_47.dll 10.0.26100.1", Target: Target, Entry: Entry, Flags: 0x28000 | 0x2000, HLSL: Sum([]byte("a")), DXBC: Sum([]byte("b"))}
}

func TestRecordRoundTrip(t *testing.T) {
	r := sample()
	got, err := ParseRecord(r.String())
	if err != nil || got != r {
		t.Fatalf("%+v %v", got, err)
	}
	if !strings.Contains(r.String(), "flags: 0x2a000\n") {
		t.Errorf("flags are written in hex: %q", r.String())
	}
	if len(Sum(nil)) != 64 {
		t.Error("Sum is a hex sha256")
	}
}

func TestParseRecordRefusals(t *testing.T) {
	good := sample().String()
	for name, in := range map[string]string{
		"empty":         "",
		"no colon":      "compiler d3dcompiler_47.dll\n",
		"empty value":   "compiler: \n",
		"unknown key":   good + "extra: 1\n",
		"duplicate":     good + "target: cs_5_0\n",
		"missing key":   strings.Replace(good, "entry: raster\n", "", 1),
		"flags not hex": strings.Replace(good, "flags: 0x2a000", "flags: lots", 1),
	} {
		if _, err := ParseRecord(in); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestDiagnose(t *testing.T) {
	c := sample()
	for name, c2 := range map[string]struct {
		mutate func(*Record)
		want   string
	}{
		"compiler": {func(r *Record) { r.Compiler = "d3dcompiler_47.dll 10.0.26200.2" }, "runner image's d3dcompiler_47.dll moved from \"d3dcompiler_47.dll 10.0.26100.1\" to \"d3dcompiler_47.dll 10.0.26200.2\""},
		"flags":    {func(r *Record) { r.Flags++ }, "compile options differ"},
		"target":   {func(r *Record) { r.Target = "cs_5_1" }, "compile options differ"},
		"hlsl":     {func(r *Record) { r.HLSL = Sum([]byte("c")) }, "kernel_gen.hlsl changed"},
		"same":     {func(r *Record) {}, "not deterministic"},
	} {
		now := c
		c2.mutate(&now)
		msg := Diagnose(c, now)
		if !strings.Contains(msg, c2.want) {
			t.Errorf("%s: %q lacks %q", name, msg, c2.want)
		}
		// Every message shows both sides and ends with the way out.
		for _, must := range []string{"committed with: " + c.Compiler, "compiled now:   " + now.Compiler, "go run ./internal/hlslgen/cmd/gendxbc", "artifact \"dxbc\"", "The tolerance of this test is none"} {
			if !strings.Contains(msg, must) {
				t.Errorf("%s: the message lacks %q", name, must)
			}
		}
	}
}
