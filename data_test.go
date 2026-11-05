// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import (
	"errors"
	"testing"
)

func TestDrawMask(t *testing.T) {
	l := nl(4, 3)
	ink := RGBA{200, 100, 50, 255}
	// Nothing to paint: recorded as nothing.
	l.DrawMask(Rect{W: 0, H: 2}, []byte{255}, 1, ink)
	l.DrawMask(Rect{W: 2, H: 2}, []byte{255}, 0, ink)
	l.DrawMask(Rect{W: 2, H: 2}, []byte{255}, 2, RGBA{})
	l.DrawMask(Rect{X: 10, W: 2, H: 2}, []byte{255}, 2, ink)
	if n := len(l.Instances()); n != 0 {
		t.Fatalf("%d instances for nothing", n)
	}
	// A 3×2 mask whose second row is missing: that row is left alone.
	l.DrawMask(Rect{X: 1, Y: 1, W: 3, H: 2}, []byte{255, 0, 128}, 3, ink)
	buf := make([]byte, 4*4*3)
	_ = l.Rasterize(buf)
	if got := px(buf, 4, 1, 1); got != (RGBA{200, 100, 50, 255}) {
		t.Fatalf("full coverage: %v", got)
	}
	if got := px(buf, 4, 2, 1); got != (RGBA{}) {
		t.Fatalf("zero coverage: %v", got)
	}
	if got := px(buf, 4, 3, 1); got.A != 128 {
		t.Fatalf("half coverage: %v", got)
	}
	if got := px(buf, 4, 1, 2); got != (RGBA{}) {
		t.Fatalf("missing row painted: %v", got)
	}
	// A translucent ink scales coverage with a floored product.
	l = nl(1, 1)
	l.DrawMask(Rect{W: 1, H: 1}, []byte{200}, 1, RGBA{255, 255, 255, 100})
	buf = make([]byte, 4)
	_ = l.Rasterize(buf)
	if got := px(buf, 1, 0, 0); got.A != 200*100/255 {
		t.Fatalf("translucent ink: %v, want alpha %d", got, 200*100/255)
	}
}

func TestDrawAlpha(t *testing.T) {
	l := nl(3, 1)
	c := RGBA{0, 0, 255, 200}
	l.DrawAlpha([]float64{1}, 0, 0, 1, 1, RGBA{}) // transparent
	l.DrawAlpha([]float64{1}, 0, 0, 0, 1, c)      // empty box
	l.DrawAlpha([]float64{1}, 0, 0, 2, 1, c)      // short coverage
	l.DrawAlpha([]float64{1, 1}, 10, 0, 2, 1, c)  // off the surface
	l.PushTranslate(5, 0)                         // ignored: surface pixels
	l.DrawAlpha([]float64{0, 0.5, 1.7}, 0, 0, 3, 1, c)
	l.PopTranslate()
	if n := len(l.Instances()); n != 1 {
		t.Fatalf("%d instances, want 1", n)
	}
	buf := make([]byte, 12)
	_ = l.Rasterize(buf)
	if got := px(buf, 3, 0, 0); got != (RGBA{}) {
		t.Fatalf("zero coverage: %v", got)
	}
	if got := px(buf, 3, 1, 0); got.A != 100 { // 200·0.5 + 0.5 rounds down to 100
		t.Fatalf("half coverage: %v", got)
	}
	if got := px(buf, 3, 2, 0); got.A != 200 {
		t.Fatalf("coverage above 1 not clamped: %v", got)
	}
}

func TestDrawImage(t *testing.T) {
	l := nl(4, 2)
	src := []byte{
		255, 0, 0, 255, 0, 255, 0, 128, // red opaque, translucent green
	}
	l.DrawImage(Rect{W: 4, H: 2}, src, 2, 1) // each texel twice across, both rows
	l.DrawImage(Rect{W: 4, H: 2}, src[:7], 2, 1)
	l.DrawImage(Rect{W: 4, H: 2}, src, 0, 1)
	l.DrawImage(Rect{W: 0, H: 2}, src, 2, 1)
	l.DrawImage(Rect{X: 9, W: 4, H: 2}, src, 2, 1)
	if n := len(l.Instances()); n != 1 {
		t.Fatalf("%d instances, want 1", n)
	}
	buf := make([]byte, 32)
	_ = l.Rasterize(buf)
	for _, x := range []int{0, 1} {
		if got := px(buf, 4, x, 1); got != (RGBA{255, 0, 0, 255}) {
			t.Fatalf("(%d,1): %v", x, got)
		}
	}
	if got := px(buf, 4, 3, 0); got.G != (255*128+127)/255 || got.A != 128 {
		t.Fatalf("translucent texel: %v", got)
	}
	if l.Err() != nil {
		t.Fatal(l.Err())
	}
}

func TestRefusalsAreReported(t *testing.T) {
	l := nl(8, 8)
	l.DrawImage(Rect{W: 4, H: 4}, nil, 1<<14, 1<<14) // 1 GiB: judged before the bytes
	if !errors.Is(l.Err(), ErrTooLarge) {
		t.Fatalf("huge image: %v", l.Err())
	}
	l.Reset()
	if l.Err() != nil {
		t.Fatal("Reset kept the error")
	}
	l.DrawImage(Rect{W: 1 << 20, H: 4}, nil, 1<<12, 1) // 2^32 scaling product
	if !errors.Is(l.Err(), ErrImageScale) {
		t.Fatalf("scaling product: %v", l.Err())
	}
	l.DrawImage(Rect{W: 4, H: 4}, nil, 1<<14, 1<<14)
	if !errors.Is(l.Err(), ErrImageScale) {
		t.Fatal("the first refusal is the one kept")
	}

	defer func(m int64) { maxData = m }(maxData)
	maxData = 64
	l = nl(16, 16)
	l.DrawMask(Rect{W: 8, H: 8}, make([]byte, 64), 8, RGBA{A: 255}) // exactly 64
	l.DrawMask(Rect{W: 1, H: 1}, []byte{1}, 1, RGBA{A: 255})        // one more
	l.DrawAlpha([]float64{1}, 0, 0, 1, 1, RGBA{A: 255})
	l.DrawImage(Rect{W: 1, H: 1}, make([]byte, 4), 1, 1)
	if !errors.Is(l.Err(), ErrTooLarge) || len(l.Instances()) != 1 || len(l.Data()) != 64 {
		t.Fatalf("arena past maxData: err %v, %d instances, %d bytes", l.Err(), len(l.Instances()), len(l.Data()))
	}
}
