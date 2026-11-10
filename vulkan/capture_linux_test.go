// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build linux

package vulkan

import (
	"os"
	"strings"
	"syscall"
	"testing"
)

// capture redirects stdout and stderr (fds 1 and 2) to a file and returns the
// function that restores them and gives the text. The validation layer writes
// its messages there, with no callback to hook (go-gfx/vulkan has none).
func capture(t testing.TB) func() string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "out")
	if err != nil {
		t.Fatal(err)
	}
	o1, err1 := syscall.Dup(1)
	o2, err2 := syscall.Dup(2)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}
	_ = syscall.Dup3(int(f.Fd()), 1, 0)
	_ = syscall.Dup3(int(f.Fd()), 2, 0)
	done := false
	return func() string {
		if done {
			return ""
		}
		done = true
		_ = syscall.Dup3(o1, 1, 0)
		_ = syscall.Dup3(o2, 2, 0)
		syscall.Close(o1)
		syscall.Close(o2)
		b, _ := os.ReadFile(f.Name())
		f.Close()
		return string(b)
	}
}

// validationClean fails the test when the validation layer said anything
// while the capture ran.
func validationClean(t testing.TB, stop func() string) {
	t.Helper()
	if out := stop(); strings.Contains(out, "Validation") || strings.Contains(out, "VUID-") {
		t.Fatalf("the validation layer reported:\n%s", out)
	}
}

const canCapture = true
