// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package vulkan

import (
	"bytes"
	"testing"

	"github.com/go-gfx/gpu2d"
)

// The frame is device-local and Pixels its host copy: a Draw moves nothing
// back until Pixels asks, a Pixels call that finds nothing new moves nothing,
// and what the caller wrote there is uploaded once, before the next draw.
func TestPixelsAreTransferredOnDemand(t *testing.T) {
	const w, h = 40, 24
	r := renderer(t, w, h)
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{X: 4, Y: 4, W: 8, H: 8}, gpu2d.RGBA{R: 200, A: 255})
	if !r.up || r.devNewer {
		t.Fatalf("a new renderer: up %v, devNewer %v; want the stage to be uploaded first", r.up, r.devNewer)
	}
	if err := r.Draw(l); err != nil {
		t.Fatal(err)
	}
	if r.up || !r.devNewer {
		t.Fatalf("after Draw: up %v, devNewer %v; want the device ahead and nothing to upload", r.up, r.devNewer)
	}
	if err := r.Draw(l); err != nil {
		t.Fatal(err)
	}
	if !r.devNewer {
		t.Fatal("a second Draw lost the device's lead")
	}
	p := r.Pixels()
	if r.devNewer || !r.up {
		t.Fatalf("after Pixels: devNewer %v, up %v; want it downloaded, and the caller assumed to write", r.devNewer, r.up)
	}
	want := make([]byte, 4*w*h)
	if err := l.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(p, want) {
		t.Fatal("Pixels is not the frame the device drew")
	}
	if err := r.Sync(); err != nil {
		t.Fatal(err)
	}
	// What Pixels returned is uploaded as the ground of the next draw: drawing
	// the same opaque list over its own result changes nothing, and a ground
	// the caller wrote shows through.
	for i := range p {
		p[i] = 0x55
	}
	other, _ := gpu2d.NewList(w, h)
	other.FillRect(gpu2d.Rect{X: 0, Y: 0, W: 1, H: 1}, gpu2d.RGBA{B: 9, A: 255})
	if err := r.Draw(other); err != nil {
		t.Fatal(err)
	}
	got := r.Pixels()
	if got[4] != 0x55 || got[0] != 0 || got[2] != 9 {
		t.Fatalf("the written ground was not uploaded: % x", got[:8])
	}
}

// While a frame is in flight the device owns the frame: Pixels neither waits
// nor downloads, and the next Submit does not upload over the device's work.
func TestPixelsInFlightDoesNotDisturbTheFrame(t *testing.T) {
	const w, h = 40, 24
	r := renderer(t, w, h)
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{W: w, H: h}, gpu2d.RGBA{G: 255, A: 255})
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	_ = r.Pixels() // a peek with a frame in flight
	if r.InFlight() != 1 || r.up {
		t.Fatalf("a peek changed state: in flight %d, up %v", r.InFlight(), r.up)
	}
	if err := r.Sync(); err != nil {
		t.Fatal(err)
	}
	if r.InFlight() != 0 || r.devNewer {
		t.Fatalf("Sync left in flight %d, devNewer %v", r.InFlight(), r.devNewer)
	}
	if p := r.Pixels(); p[1] != 255 || p[3] != 255 {
		t.Fatalf("Sync did not make Pixels current: % x", p[:4])
	}
}
