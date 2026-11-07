// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build js && wasm

package webgpu

import (
	"testing"

	"github.com/go-gfx/gpu2d"
	wg "github.com/go-gfx/webgpu"
)

// Destroy refunds every byte the renderer charged to the device's quota,
// including the buffers a large list grew, so replacing a renderer on each
// canvas resize does not accumulate memory.
func TestDestroyRefundsTheQuota(t *testing.T) {
	_, ctx := device(t) // the adapter checks and the replay script
	nav, err := wg.Navigator()
	if err != nil {
		t.Fatal(err)
	}
	a, err := nav.RequestAdapter(ctx, &wg.RequestAdapterOptions{AllowFallback: true})
	if err != nil {
		t.Fatal(err)
	}
	// A device keeps accounts only when it has a quota; an empty one limits
	// nothing.
	d, err := a.RequestDevice(ctx, &wg.DeviceDescriptor{Label: t.Name(), Quota: &wg.Quota{}})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Destroy()
	before, _ := d.Quota().Usage()
	r, err := New(ctx, d, 200, 100)
	if err != nil {
		t.Fatal(err)
	}
	l, _ := gpu2d.NewList(200, 100)
	for i := range 500 {
		l.FillRect(gpu2d.Rect{X: i % 200, Y: i % 100, W: 3, H: 3}, gpu2d.RGBA{R: 9, A: 255})
	}
	if err := r.Draw(l, nil); err != nil {
		t.Fatal(err)
	}
	during, _ := d.Quota().Usage()
	if during <= before {
		t.Fatalf("the renderer charged nothing to the quota (%d → %d): the test proves nothing", before, during)
	}
	r.Destroy()
	r.Destroy() // twice is harmless
	if after, _ := d.Quota().Usage(); after != before {
		t.Fatalf("after Destroy the quota holds %d bytes, %d before the renderer", after, before)
	}
}
