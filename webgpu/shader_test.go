// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package webgpu

import (
	"testing"

	"github.com/go-gfx/wgsl"
)

// The shader parses and type-checks on every platform, before any browser
// sees it; the browser stays the authority (go-gfx/wgsl's own caveat).
func TestShaderValidates(t *testing.T) {
	if err := wgsl.Validate(Shader); err != nil {
		t.Fatal(err)
	}
	if err := wgsl.Validate(PresentShader); err != nil {
		t.Fatalf("PresentShader: %v", err)
	}
}
