// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build !windows

package d3d11

import "github.com/go-gfx/gpu2d"

// Renderer is the Direct3D 11 renderer. Off Windows none can be made: New
// returns ErrUnsupported, and the methods exist so that code using the type
// builds on every platform.
type Renderer struct{}

// New returns ErrUnsupported (gpu2d.ErrSize first for an impossible size).
func New(w, h int) (*Renderer, error) { return NewWith(w, h, Options{}) }

// NewWith is New with options.
func NewWith(w, h int, _ Options) (*Renderer, error) {
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return nil, gpu2d.ErrSize
	}
	return nil, ErrUnsupported
}

// Name is empty.
func (*Renderer) Name() string { return "" }

// Pixels is nil.
func (*Renderer) Pixels() []byte { return nil }

// Draw returns ErrUnsupported.
func (*Renderer) Draw(*gpu2d.List) error { return ErrUnsupported }

// Submit returns ErrUnsupported.
func (*Renderer) Submit(*gpu2d.List) error { return ErrUnsupported }

// Wait returns ErrUnsupported.
func (*Renderer) Wait() error { return ErrUnsupported }

// InFlight is 0.
func (*Renderer) InFlight() int { return 0 }

// Resize returns ErrUnsupported.
func (*Renderer) Resize(int, int) error { return ErrUnsupported }

// Stats is zero.
func (*Renderer) Stats() Stats { return Stats{} }

// ResetStats does nothing.
func (*Renderer) ResetStats() {}

// Close does nothing.
func (*Renderer) Close() {}
