package metal

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
	r := renderer(t, 1920, 1080)
	l, _ := gpu2d.NewList(1920, 1080)
	gpu2d.WidgetFrame(l)
	s, _, err := r.pack(l) // the first pack grows the buffers
	if err != nil {
		t.Fatal(err)
	}
	spans0, list0 := s.spans, s.list
	if s, _, err = r.pack(l); err != nil { // the same free slot again
		t.Fatal(err)
	}
	if s.spans != spans0 || s.list != list0 {
		t.Fatal("a second pack of the same frame replaced the slot's buffers")
	}
	ws, wl := words(s.spans), words(s.list)
	if len(ws) == 0 || len(wl) == 0 {
		t.Fatal("words sees no memory in the slot's buffers: pack copies every frame")
	}
	spans, list, err := l.Bin(ws, wl)
	if err != nil {
		t.Fatal(err)
	}
	if len(spans) == 0 || len(list) == 0 {
		t.Fatal("the widget frame bins to nothing")
	}
	if &spans[0] != &ws[0] || &list[0] != &wl[0] {
		t.Fatal("Bin did not reuse the buffers it was given: pack copies every frame")
	}
}
