// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package hlslgen makes the Direct3D 11 back-end's raster kernel from
// kernel.wgsl, through go-gfx/wgsl's HLSL emitter, so the raster arithmetic is
// written once in WGSL (docs/NATIVE.md, decision 1).
//
// Two files are committed in d3d11/: kernel_gen.hlsl, which any platform can
// regenerate (go generate ./d3d11), and kernel_gen.dxbc, which only the Windows
// lane can (D3DCompile from the system d3dcompiler_47.dll), together with
// kernel_gen.dxbc.record, which says what made the bytes: the compiler DLL's
// version, the flags, and the hashes of the HLSL and of the DXBC. A DXBC drift
// that is only the runner image's compiler having moved is then attributable at
// a glance (Diagnose).
package hlslgen

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/hlsl"
)

// Entry and Target are the kernel's entry point and shader model.
const (
	Entry  = "raster"
	Target = "cs_5_0"
)

// Kernel is the HLSL for the compute entry point "raster", as the text that
// d3d11/kernel_gen.hlsl holds.
func Kernel(kernel string) (string, error) {
	m, err := wgsl.Parse(kernel)
	if err != nil {
		return "", err
	}
	info, errs := wgsl.Check(m)
	if len(errs) > 0 {
		return "", fmt.Errorf("hlslgen: the kernel does not check: %v", errs)
	}
	return hlsl.Module(m, info, Entry)
}

// Record says how a DXBC file was made.
type Record struct {
	Compiler string // the DLL and its file version, e.g. "d3dcompiler_47.dll 10.0.26100.1"
	Target   string
	Entry    string
	Flags    uint32 // D3DCOMPILE_* bits
	HLSL     string // sha256 of the HLSL text compiled, hex
	DXBC     string // sha256 of the DXBC produced, hex
}

// Sum is the hex sha256 of b, the form a Record holds.
func Sum(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// String is the committed form: one `key: value` per line.
func (r Record) String() string {
	return fmt.Sprintf("compiler: %s\ntarget: %s\nentry: %s\nflags: %#x\nhlsl-sha256: %s\ndxbc-sha256: %s\n",
		r.Compiler, r.Target, r.Entry, r.Flags, r.HLSL, r.DXBC)
}

// ParseRecord reads String's form back. Every key must be present, once.
func ParseRecord(s string) (Record, error) {
	var r Record
	seen := map[string]bool{}
	for _, line := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		k, v, ok := strings.Cut(line, ": ")
		if !ok || v == "" {
			return Record{}, fmt.Errorf("hlslgen: record line %q is not `key: value`", line)
		}
		if seen[k] {
			return Record{}, fmt.Errorf("hlslgen: record key %q twice", k)
		}
		seen[k] = true
		switch k {
		case "compiler":
			r.Compiler = v
		case "target":
			r.Target = v
		case "entry":
			r.Entry = v
		case "flags":
			f, err := strconv.ParseUint(v, 0, 32)
			if err != nil {
				return Record{}, fmt.Errorf("hlslgen: record flags %q: %v", v, err)
			}
			r.Flags = uint32(f)
		case "hlsl-sha256":
			r.HLSL = v
		case "dxbc-sha256":
			r.DXBC = v
		default:
			return Record{}, fmt.Errorf("hlslgen: record key %q is unknown", k)
		}
	}
	if len(seen) != 6 {
		return Record{}, fmt.Errorf("hlslgen: the record has %d of its 6 keys", len(seen))
	}
	return r, nil
}

// Regenerate is how the committed DXBC is remade, in the words every drift
// failure ends with.
const Regenerate = `To regenerate (the bytes can only be made on Windows, with D3DCompile):
  - on a Windows machine, from the repository root:
        go run ./internal/hlslgen/cmd/gendxbc
    which rewrites d3d11/kernel_gen.dxbc and d3d11/kernel_gen.dxbc.record;
  - or without one: open the failed run of the CI lane "d3d11 (windows, WARP)",
    download its artifact "dxbc" (it holds both files, made by this runner), copy
    them over d3d11/kernel_gen.dxbc and d3d11/kernel_gen.dxbc.record, and commit.
The tolerance of this test is none: a changed compiler is a changed file.`

// Diagnose explains a DXBC that differs from a fresh compile: committed is the
// record in the repository and now describes the compile that just ran (with
// the HLSL actually compiled). The message names the cause and ends with the
// way to regenerate.
func Diagnose(committed, now Record) string {
	var b strings.Builder
	fmt.Fprintf(&b, "d3d11/kernel_gen.dxbc is not what this runner compiles from d3d11/kernel_gen.hlsl.\n")
	fmt.Fprintf(&b, "  committed with: %s (flags %#x, %s %s)\n", committed.Compiler, committed.Flags, committed.Target, committed.Entry)
	fmt.Fprintf(&b, "  compiled now:   %s (flags %#x, %s %s)\n", now.Compiler, now.Flags, now.Target, now.Entry)
	switch {
	case committed.Compiler != now.Compiler:
		fmt.Fprintf(&b, "Cause: the compiler differs. The runner image's d3dcompiler_47.dll moved from %q to %q; nothing in the repository changed.\n", committed.Compiler, now.Compiler)
	case committed.Flags != now.Flags || committed.Target != now.Target || committed.Entry != now.Entry:
		b.WriteString("Cause: the compile options differ from the committed record (hlslgen.Flags, Target or Entry changed).\n")
	case committed.HLSL != now.HLSL:
		b.WriteString("Cause: d3d11/kernel_gen.hlsl changed (the kernel or the emitter) and the DXBC was not remade.\n")
	default:
		b.WriteString("Cause: unknown; the same compiler, flags and HLSL produced different bytes (the compiler is not deterministic here).\n")
	}
	b.WriteString(Regenerate)
	return b.String()
}
