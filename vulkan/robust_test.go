// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package vulkan

import (
	"errors"
	"testing"
	"time"

	"github.com/go-gfx/gpu2d"
)

// A download that fails leaves the stage stale. The stage must then never be
// uploaded over the device's newer frame, and the failure must not be reported
// once and forgotten: every Submit says so until a Sync succeeds.
func TestFailedDownloadIsNeverUploadedOverTheDeviceFrame(t *testing.T) {
	const w, h = 40, 24
	r := renderer(t, w, h)
	green, _ := gpu2d.NewList(w, h)
	green.FillRect(gpu2d.Rect{W: w, H: h}, gpu2d.RGBA{G: 255, A: 255})
	if err := r.Draw(green); err != nil {
		t.Fatal(err)
	}
	// Break the download: a closed command buffer cannot record.
	good := r.xfer
	good.Close()
	_ = r.Pixels()
	if r.err == nil || r.up {
		t.Fatalf("after a failed download: err %v, up %v; want the error kept and no upload", r.err, r.up)
	}
	red, _ := gpu2d.NewList(w, h)
	red.FillRect(gpu2d.Rect{X: 0, Y: 0, W: 2, H: 2}, gpu2d.RGBA{R: 255, A: 255})
	for i := range 2 { // not once: every Submit
		if err := r.Submit(red); err == nil {
			t.Fatalf("Submit %d after a failed download succeeded, which would upload the stale stage", i)
		}
	}
	if r.InFlight() != 0 {
		t.Fatal("a refused Submit is in flight")
	}
	// Repair the download: the device's frame, not the stale stage, is read.
	var err error
	if r.xfer, err = r.dev.NewCommandBuffer(); err != nil {
		t.Fatal(err)
	}
	if err := r.Sync(); err != nil {
		t.Fatal(err)
	}
	if p := r.Pixels(); p[1] != 255 || p[4*(w*h-1)+1] != 255 {
		t.Fatalf("the device's frame was overwritten by the stale stage: % x", p[:4])
	}
	if err := r.Draw(red); err != nil {
		t.Fatalf("Draw after a successful Sync: %v", err)
	}
}

// A frame that outlives waitLimit may still be running, and the specification
// forbids destroying its fence, its buffers and the device (VUID-vkDestroyFence-
// fence-01120 and others): the renderer is lost, refuses further work, and
// Close destroys nothing. Under the validation layer (VULKAN_VALIDATION=1) any
// destruction of a pending object is a message that fails the test.
func TestFenceTimeoutLosesTheRendererAndDestroysNothing(t *testing.T) {
	const w, h = 3840, 2160
	r := renderer(t, w, h)
	l, _ := gpu2d.NewList(w, h)
	gpu2d.WidgetFrame(l)
	old := waitLimit
	waitLimit = time.Nanosecond
	t.Cleanup(func() { waitLimit = old })
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	err := r.Wait()
	waitLimit = old
	if err == nil {
		t.Skip("the device finished a 4K frame within a nanosecond")
	}
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("Wait: %v, want ErrTimeout", err)
	}
	if !r.lost {
		t.Fatal("a timed-out frame did not mark the renderer lost")
	}
	if err := r.Submit(l); !errors.Is(err, ErrTimeout) {
		t.Fatalf("Submit on a lost renderer: %v", err)
	}
	if err := r.Resize(40, 40); !errors.Is(err, ErrTimeout) {
		t.Fatalf("Resize on a lost renderer: %v", err)
	}
	r.Close() // destroys nothing; the cleanup's second Close is a no-op
	time.Sleep(300 * time.Millisecond)
}
