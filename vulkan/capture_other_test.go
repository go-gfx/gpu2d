// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build !linux

package vulkan

import "testing"

func capture(t testing.TB) func() string { return func() string { return "" } }

func validationClean(t testing.TB, stop func() string) {}

const canCapture = false
