// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package metal

import (
	"bytes"
	"errors"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
	mtl "github.com/go-macos/metal"
)

// readBack reads a drawable into bytes, RGBA, in the submission that shows
// it: the only moment the texture exists.
const readBack = `
#include <metal_stdlib>
using namespace metal;
kernel void get(texture2d<float, access::read> in [[texture(0)]],
                constant uint2 &size [[buffer(1)]],
                device uchar4 *px [[buffer(0)]],
                uint2 g [[thread_position_in_grid]]) {
    if (g.x >= size.x || g.y >= size.y) return;
    px[g.y * size.x + g.x] = uchar4(round(in.read(g) * 255.0));
}`

// readBackInto arranges for every Present to copy its drawable into a
// w × h buffer, until the test ends.
func readBackInto(t *testing.T, r *Renderer, w, h int) *mtl.Buffer {
	t.Helper()
	lib, err := r.d.Compile(readBack)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lib.Close)
	get, err := lib.Pipeline("get")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(get.Close)
	out, err := r.d.NewBuffer(4 * w * h)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(out.Close)
	size := [2]uint32{uint32(w), uint32(h)}
	afterPresent = func(e *mtl.Encoder, tex *mtl.Texture) {
		e.Use(get)
		e.Buffer(0, out)
		mtl.Constant(e, 1, &size)
		e.Texture(0, tex)
		e.Dispatch(w, h)
	}
	t.Cleanup(func() { afterPresent = nil })
	return out
}

// What reaches the screen is the frame, byte for byte: random scenes are
// presented to a layer and the drawable is read back on the GPU. The frame
// itself is held to the CPU reference by TestMatchesReference.
func TestAPresentedFrameIsTheFrame(t *testing.T) {
	const w, h = 64, 48
	r := renderer(t, w, h)
	layer, err := r.NewLayer()
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	out := readBackInto(t, r, w, h)
	ground := make([]byte, 4*w*h)
	for i := range ground {
		ground[i] = uint8(i * 7)
		if i%4 == 3 {
			ground[i] = 255 // a window is opaque
		}
	}
	for seed := range uint64(40) {
		l, _ := gpu2d.NewList(w, h)
		scenes.Random(l, seed, 40)
		copy(r.Pixels(), ground)
		clear(out.Bytes())
		if err := r.Present(l, layer); err != nil {
			t.Fatal(err)
		}
		if err := r.Wait(); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(out.Bytes(), r.Pixels()) {
			got, want := out.Bytes(), r.Pixels()
			for i := range want {
				if got[i] != want[i] {
					p := i / 4
					t.Fatalf("seed %d (%d,%d) channel %d: frame %d, on screen %d", seed, p%w, p/w, i%4, want[i], got[i])
				}
			}
		}
	}
}

// Two presents in flight, then a third is refused like a third Submit.
func TestPresentShareSubmitsSlots(t *testing.T) {
	const w, h = 16, 16
	r := renderer(t, w, h)
	layer, err := r.NewLayer()
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{X: 2, Y: 2, W: 8, H: 8}, gpu2d.RGBA{R: 200, A: 255})
	if err := r.Present(l, layer); err != nil {
		t.Fatal(err)
	}
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	if err := r.Present(l, layer); !errors.Is(err, ErrBusy) {
		t.Fatalf("a third frame in flight: %v, want ErrBusy", err)
	}
	for r.InFlight() > 0 {
		if err := r.Wait(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPresentRefusals(t *testing.T) {
	r := renderer(t, 16, 16)
	l, _ := gpu2d.NewList(16, 16)
	other, err := r.d.NewLayer(16, 17)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Present(l, other); !errors.Is(err, ErrSize) {
		t.Fatalf("a layer of another size: %v, want ErrSize", err)
	}
	other.Close()
	if err := r.Present(l, nil); !errors.Is(err, ErrSize) {
		t.Fatalf("no layer: %v, want ErrSize", err)
	}
	layer, err := r.NewLayer()
	if err != nil {
		t.Fatal(err)
	}
	wrong, _ := gpu2d.NewList(16, 15)
	if err := r.Present(wrong, layer); !errors.Is(err, ErrSize) {
		t.Fatalf("a list of another size: %v, want ErrSize", err)
	}
	layer.Close()
	// A closed layer still reports its size, so it reaches go-macos/metal,
	// which refuses it; nothing is left in flight.
	if err := r.Present(l, layer); err == nil {
		t.Fatal("presenting to a closed layer succeeded")
	}
	if r.InFlight() != 0 {
		t.Fatalf("a refused Present left %d frames in flight", r.InFlight())
	}
}

func TestResize(t *testing.T) {
	r := renderer(t, 16, 16)
	l, _ := gpu2d.NewList(16, 16)
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	// Resize waits for the frame in flight before it frees the target.
	if err := r.Resize(16, 16); err != nil {
		t.Fatal(err)
	}
	if r.InFlight() != 0 {
		t.Fatalf("Resize left %d frames in flight", r.InFlight())
	}
	if err := r.Resize(40, 24); err != nil {
		t.Fatal(err)
	}
	if len(r.Pixels()) != 4*40*24 {
		t.Fatalf("after Resize the frame is %d bytes, want %d", len(r.Pixels()), 4*40*24)
	}
	layer, err := r.NewLayer()
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	if w, h := layer.Size(); w != 40 || h != 24 {
		t.Fatalf("NewLayer after Resize is %d × %d", w, h)
	}
	big, _ := gpu2d.NewList(40, 24)
	big.FillRect(gpu2d.Rect{W: 40, H: 24}, gpu2d.RGBA{G: 9, A: 255})
	if err := r.Present(big, layer); err != nil {
		t.Fatal(err)
	}
	if err := r.Wait(); err != nil {
		t.Fatal(err)
	}
	if px := r.Pixels(); px[1] != 9 || px[len(px)-3] != 9 {
		t.Fatal("the resized frame was not drawn")
	}
	for _, s := range [][2]int{{0, 1}, {1, 0}, {gpu2d.MaxCoord + 1, 1}} {
		if err := r.Resize(s[0], s[1]); !errors.Is(err, gpu2d.ErrSize) {
			t.Errorf("Resize(%d, %d): %v, want gpu2d.ErrSize", s[0], s[1], err)
		}
	}
	// A frame the GPU cannot allocate is refused and the old one kept.
	if err := r.Resize(gpu2d.MaxCoord, gpu2d.MaxCoord); err == nil {
		t.Fatal("a 2^24 × 2^24 frame was allocated")
	}
	if len(r.Pixels()) != 4*40*24 {
		t.Fatal("a refused Resize lost the frame")
	}
}
