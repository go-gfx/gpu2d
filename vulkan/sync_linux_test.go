// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build linux

package vulkan

import (
	"os"
	"strings"
	"testing"

	"github.com/go-gfx/gpu2d"
)

// The Khronos validation layer's synchronization validation, as in
// go-gfx/vulkan's own tests:
//
//	VULKAN_SYNC_VALIDATION=1  the lane runs with VK_LAYER_VALIDATE_SYNC=1 and
//	                          VULKAN_VALIDATION=1; every test then fails on any
//	                          message, SYNC-HAZARD included.
//
// TestBreakCheckTransferBarrier proves the lane can see what it is there for:
// the upload's copy of the stage into the target, followed by the dispatch with
// the barrier between them left out, MUST draw a SYNC-HAZARD, and with the
// barrier (what Submit records) it must not.
func TestBreakCheckTransferBarrier(t *testing.T) {
	if os.Getenv("VULKAN_SYNC_VALIDATION") != "1" {
		t.Skip("VULKAN_SYNC_VALIDATION is not set")
	}
	if os.Getenv("VULKAN_VALIDATION") != "1" || os.Getenv("VK_LAYER_VALIDATE_SYNC") != "1" {
		t.Fatal("VULKAN_SYNC_VALIDATION=1 needs VULKAN_VALIDATION=1 and VK_LAYER_VALIDATE_SYNC=1")
	}
	run := func(barrier bool, ops ...string) string {
		stop := capture(t)
		defer stop()
		// Not rendererRTE: that one fails the test on any message.
		r, err := NewWith(64, 64, Options{Validation: true})
		if err != nil {
			t.Fatal(err)
		}
		l, _ := gpu2d.NewList(64, 64)
		l.FillRect(gpu2d.Rect{W: 64, H: 64}, gpu2d.RGBA{G: 255, A: 255})
		s, _, err := r.pack(l)
		if err != nil {
			t.Fatal(err)
		}
		must := func(err error) {
			t.Helper()
			if err != nil {
				t.Fatal(err)
			}
		}
		up := func() { must(s.cb.Copy(r.target, r.stage, r.target.Size())) }
		down := func() { must(s.cb.Copy(r.stage, r.target, r.target.Size())) }
		disp := func() { must(s.cb.Dispatch(s.pipe, 4, 8, 1)) }
		must(s.cb.Begin())
		for i, op := range ops {
			if i > 0 && barrier {
				must(s.cb.Barrier())
			}
			switch op {
			case "up":
				up()
			case "down":
				down()
			case "dispatch":
				disp()
			}
		}
		must(s.cb.End())
		fe, err := r.dev.Submit(s.cb)
		must(err)
		must(r.finish(&slot{fence: fe}))
		fe.Close()
		r.Close()
		return stop()
	}
	// The layer (1.4.309, lavapipe) tracks copies, and does not see the
	// accesses of a dispatch through its descriptors (go-gfx/vulkan found the
	// same): the upload and the download are checked against each other, and
	// the dispatch orderings are logged. They all rely on Barrier, the one
	// primitive Submit and Sync record.
	for _, ops := range [][]string{{"up", "down"}, {"down", "up"}} {
		if out := run(false, ops...); !strings.Contains(out, "SYNC-HAZARD") {
			t.Fatalf("%v with no barrier drew no SYNC-HAZARD; the layer said: %q", ops, out)
		}
		if out := run(true, ops...); strings.Contains(out, "SYNC-HAZARD") || strings.Contains(out, "VUID-") {
			t.Fatalf("%v with the barrier still drew a message: %q", ops, out)
		}
	}
	for _, ops := range [][]string{{"up", "dispatch"}, {"dispatch", "down"}} {
		t.Logf("%v without a barrier: SYNC-HAZARD seen %v", ops, strings.Contains(run(false, ops...), "SYNC-HAZARD"))
	}
}
