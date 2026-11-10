// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package d3d11

import (
	"errors"
	"runtime"
	"testing"

	"github.com/go-gfx/gpu2d"
)

// TestNewRefusals holds on every platform: an impossible size is
// gpu2d.ErrSize before anything else, and off Windows New says ErrUnsupported
// (the stubs are what lets a program cross-compile).
func TestNewRefusals(t *testing.T) {
	for _, s := range [][2]int{{0, 1}, {1, 0}, {-1, 5}, {gpu2d.MaxCoord + 1, 1}} {
		if _, err := New(s[0], s[1]); !errors.Is(err, gpu2d.ErrSize) {
			t.Fatalf("New(%d, %d): %v", s[0], s[1], err)
		}
	}
	if runtime.GOOS != "windows" {
		if _, err := New(8, 8); !errors.Is(err, ErrUnsupported) {
			t.Fatalf("New(8, 8) off Windows: %v", err)
		}
	}
}
