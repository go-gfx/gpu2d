// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package vulkan

import (
	"testing"

	"github.com/go-gfx/gpu2d"
)

// Once a slot's buffers have grown to a frame's size, the next pack bins
// that frame straight into them: the indices Bin returns live in the
// buffers' own memory, so nothing is copied. If Bin ever stopped reusing
// the storage it is given, pack would still be correct (place copies) and
// only this test would notice the copy coming back.
func TestPackBinsIntoTheSlotBuffers(t *testing.T) {
	if !littleEndian {
		t.Skip("big-endian host: words is nil by design and every frame copies")
	}
	r := renderer(t, 1920, 1080)
	l, _ := gpu2d.NewList(1920, 1080)
	gpu2d.WidgetFrame(l)
	s, _, err := r.pack(l) // the first pack grows the buffers
	if err != nil {
		t.Fatal(err)
	}
	bins0 := s.bins
	if s, _, err = r.pack(l); err != nil { // the same free slot again
		t.Fatal(err)
	}
	if s.bins != bins0 {
		t.Fatal("a second pack of the same frame replaced the slot's buffer")
	}
	across, down := l.Tiles()
	ns := 2 * across * down
	bw := words(s.bins)
	if len(bw) <= ns {
		t.Fatal("words sees no list in the slot's buffer: pack copies every frame")
	}
	spans, list, err := l.Bin(bw[:0:ns], bw[ns:])
	if err != nil {
		t.Fatal(err)
	}
	if len(spans) == 0 || len(list) == 0 {
		t.Fatal("the widget frame bins to nothing")
	}
	if &spans[0] != &bw[0] || &list[0] != &bw[ns] {
		t.Fatal("Bin did not reuse the buffer it was given: pack copies every frame")
	}
}
