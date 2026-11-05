// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import (
	"errors"
	"testing"
	"unsafe"
)

// nl is NewList for sizes the test knows are valid.
func nl(w, h int) *List {
	l, err := NewList(w, h)
	if err != nil {
		panic(err)
	}
	return l
}

func TestNewListRefusesBadSizes(t *testing.T) {
	for _, s := range [][2]int{{-1, 1}, {1, -1}, {MaxCoord + 1, 1}, {1, MaxCoord + 1}} {
		if _, err := NewList(s[0], s[1]); !errors.Is(err, ErrSize) {
			t.Fatalf("NewList(%d, %d) err = %v", s[0], s[1], err)
		}
	}
}

// Coordinates past MaxCoord are refused rather than wrapped into int32, and a
// huge translation or clip saturates instead of overflowing.
func TestHugeCoordinatesAreBounded(t *testing.T) {
	l := nl(10, 10)
	l.FillRect(Rect{X: 1 << 30, W: 5, H: 5}, RGBA{A: 255})
	l.FillRect(Rect{X: -MaxCoord - 1, W: 5, H: 5}, RGBA{A: 255})
	l.FillRect(Rect{W: MaxCoord + 1, H: 5}, RGBA{A: 255})
	l.FillRect(Rect{Y: -MaxCoord - 1, W: 5, H: 5}, RGBA{A: 255})
	l.FillRect(Rect{W: 5, H: MaxCoord + 1}, RGBA{A: 255})
	if n := len(l.Instances()); n != 0 {
		t.Fatalf("%d out-of-range instances recorded", n)
	}
	// Two shifts of 2^30 sum past int32; they saturate instead of wrapping.
	l.PushTranslate(1<<30, 0)
	l.PushTranslate(1<<30, 0)
	if dx, _ := l.offset(); dx != 2*MaxCoord {
		t.Fatalf("translation %d, want saturated at %d", dx, 2*MaxCoord)
	}
	l.PopTranslate()
	l.PopTranslate()
	l.PushClip(Rect{X: -1 << 30, Y: -1 << 30, W: 1<<31 - 1, H: 1<<31 - 1})
	l.FillRect(Rect{W: 3, H: 3}, RGBA{A: 255})
	if got := l.Instances(); len(got) != 1 || got[0].ClipX1 != 10 {
		t.Fatalf("huge clip: %+v", got)
	}
	l.Reset()
	l.StrokeRect(Rect{W: 4, H: 4}, RGBA{A: 255}, 1<<30)
	if n := len(l.Instances()); n != 2 {
		t.Fatalf("huge lineW: %d rings", n)
	}
}

func TestInstanceSize(t *testing.T) {
	if got := unsafe.Sizeof(Instance{}); got != InstanceSize {
		t.Fatalf("Instance is %d bytes, InstanceSize says %d", got, InstanceSize)
	}
}

func TestPackRoundTrip(t *testing.T) {
	c := RGBA{1, 2, 3, 4}
	if got := unpack(pack(c)); got != c {
		t.Fatalf("unpack(pack(%v)) = %v", c, got)
	}
	if pack(c) != 0x04030201 {
		t.Fatalf("pack(%v) = %#x, want R in the low byte", c, pack(c))
	}
}

func px(buf []byte, w, x, y int) RGBA {
	o := (y*w + x) * 4
	return RGBA{buf[o], buf[o+1], buf[o+2], buf[o+3]}
}

func TestFillRectOpaqueAndClipped(t *testing.T) {
	l := nl(4, 3)
	l.FillRect(Rect{-1, -1, 3, 3}, RGBA{10, 20, 30, 255})
	buf := make([]byte, 4*4*3)
	if err := l.Rasterize(buf); err != nil {
		t.Fatal(err)
	}
	if got := px(buf, 4, 1, 1); got != (RGBA{10, 20, 30, 255}) {
		t.Fatalf("inside: %v", got)
	}
	if got := px(buf, 4, 2, 1); got != (RGBA{}) {
		t.Fatalf("outside the rect: %v", got)
	}
	if w, h := l.Size(); w != 4 || h != 3 {
		t.Fatalf("Size = %d×%d", w, h)
	}
}

func TestTranslucentBlendIsPixelPainters(t *testing.T) {
	l := nl(1, 1)
	l.FillRect(Rect{0, 0, 1, 1}, RGBA{200, 100, 0, 255})
	l.FillRect(Rect{0, 0, 1, 1}, RGBA{0, 0, 255, 64})
	buf := make([]byte, 4)
	if err := l.Rasterize(buf); err != nil {
		t.Fatal(err)
	}
	// (src·64 + dst·191 + 127) / 255 per channel; alpha 64 + 255·191/255.
	want := RGBA{(200*191 + 127) / 255, (100*191 + 127) / 255, (255*64 + 127) / 255, 255}
	if got := px(buf, 1, 0, 0); got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestNothingRecordedThatCannotPaint(t *testing.T) {
	l := nl(10, 10)
	l.FillRect(Rect{0, 0, 0, 5}, RGBA{A: 255})
	l.FillRect(Rect{0, 0, 5, 5}, RGBA{R: 9})
	l.FillRect(Rect{20, 20, 5, 5}, RGBA{A: 255})
	l.StrokeRect(Rect{0, 0, 0, 0}, RGBA{A: 255}, 1)
	l.StrokeRoundRect(Rect{0, 0, 0, 4}, 2, RGBA{A: 255}, 3)
	l.PushClip(Rect{0, 0, 2, 2})
	l.PushClip(Rect{5, 5, 2, 2}) // disjoint: empty clip
	l.FillRect(Rect{0, 0, 10, 10}, RGBA{A: 255})
	if n := len(l.Instances()); n != 0 {
		t.Fatalf("%d instances recorded, want 0", n)
	}
}

func TestTranslateAndClipNest(t *testing.T) {
	l := nl(10, 10)
	l.PopTranslate() // empty: no-op
	l.PopClip()      // empty: no-op
	l.PushTranslate(2, 3)
	l.PushTranslate(1, 1)
	l.PushClip(Rect{0, 0, 4, 4}) // becomes (3,4)-(7,8)
	l.FillRect(Rect{0, 0, 10, 10}, RGBA{A: 255})
	l.PopClip()
	l.PopTranslate()
	l.FillRect(Rect{0, 0, 1, 1}, RGBA{A: 255})
	got := l.Instances()
	if len(got) != 2 {
		t.Fatalf("%d instances", len(got))
	}
	a := got[0]
	if a.X != 3 || a.Y != 4 || a.ClipX0 != 3 || a.ClipY0 != 4 || a.ClipX1 != 7 || a.ClipY1 != 8 {
		t.Fatalf("nested translate+clip: %+v", a)
	}
	if b := got[1]; b.X != 2 || b.Y != 3 || b.ClipX1 != 10 {
		t.Fatalf("after pops: %+v", b)
	}
	l.Reset()
	if len(l.Instances()) != 0 {
		t.Fatal("Reset kept instances")
	}
}

func TestRasterizeRefusesShortBuffer(t *testing.T) {
	if err := nl(2, 2).Rasterize(make([]byte, 15)); !errors.Is(err, ErrBuffer) {
		t.Fatalf("err = %v", err)
	}
}

// A translucent outline one pixel tall hits its row twice in PixelPainter;
// one pixel wide, its interior twice and its ends once.
func TestDegenerateOutlinesHitTwice(t *testing.T) {
	c := RGBA{255, 0, 0, 128}
	once := RGBA{(255*128 + 127) / 255, 0, 0, 128}
	twice := RGBA{uint8((255*128 + uint32(once.R)*127 + 127) / 255), 0, 0, uint8(128 + 128*127/255)}

	l := nl(3, 1)
	l.StrokeRect(Rect{0, 0, 3, 1}, c, 1)
	buf := make([]byte, 12)
	_ = l.Rasterize(buf)
	if got := px(buf, 3, 1, 0); got != twice {
		t.Fatalf("1-tall: %v, want %v", got, twice)
	}

	l = nl(1, 4)
	l.StrokeRect(Rect{0, 0, 1, 4}, c, 1)
	buf = make([]byte, 16)
	_ = l.Rasterize(buf)
	if got := px(buf, 1, 0, 0); got != once {
		t.Fatalf("1-wide end: %v, want %v", got, once)
	}
	if got := px(buf, 1, 0, 2); got != twice {
		t.Fatalf("1-wide interior: %v, want %v", got, twice)
	}
}

func TestStrokeRectRingsStopWhenTheyMeet(t *testing.T) {
	l := nl(10, 10)
	l.StrokeRect(Rect{0, 0, 4, 4}, RGBA{A: 255}, 5)
	if n := len(l.Instances()); n != 2 {
		t.Fatalf("%d rings, want 2 (the third would be empty)", n)
	}
	l.Reset()
	l.StrokeRect(Rect{0, 0, 4, 4}, RGBA{A: 255}, 0)
	if n := len(l.Instances()); n != 1 {
		t.Fatalf("lineW 0: %d rings, want 1", n)
	}
}

func TestStrokeRoundRectRings(t *testing.T) {
	l := nl(20, 20)
	l.StrokeRoundRect(Rect{0, 0, 10, 10}, 4, RGBA{A: 255}, 3)
	got := l.Instances()
	if len(got) != 3 {
		t.Fatalf("%d rings", len(got))
	}
	// Innermost first: inset 1 radius 3, inset 2 radius 2, then the outer ring.
	if got[0].X != 1 || got[0].Radius != 3 || got[1].X != 2 || got[1].Radius != 2 || got[2].X != 0 || got[2].Radius != 4 {
		t.Fatalf("rings %+v", got)
	}
	l.Reset()
	l.StrokeRoundRect(Rect{0, 0, 4, 4}, 1, RGBA{A: 255}, 9)
	if n := len(l.Instances()); n != 2 {
		t.Fatalf("meeting rings: %d, want 2", n)
	}
}

func TestCoverage(t *testing.T) {
	r := Rect{0, 0, 10, 10}
	if c := Coverage(KindFill, 5, 5, r, 3); c != 1 {
		t.Fatalf("fill interior %v", c)
	}
	if c := Coverage(KindFill, 0, 0, r, 3); c != 0 {
		t.Fatalf("fill outer corner %v", c)
	}
	if c := Coverage(KindFill, 9, 9, r, 3); c != 0 {
		t.Fatalf("fill bottom-right corner %v", c)
	}
	if c := Coverage(KindFill, 9, 0, r, 3); c != 0 || Coverage(KindFill, 0, 9, r, 3) != 0 {
		t.Fatal("fill top-right / bottom-left corner")
	}
	// Centre (0.5, 1.5) to the corner centre (3, 3): 3.5 − √(2.5²+1.5²) ≈ 0.585.
	if c := Coverage(KindFill, 0, 1, r, 3); c <= 0 || c >= 1 {
		t.Fatalf("fill corner edge %v, want partial", c)
	}
	if c := Coverage(KindOutline, 5, 0, r, 0); c != 1 {
		t.Fatalf("square outline edge %v", c)
	}
	if c := Coverage(KindOutline, 5, 5, r, 0); c != 0 {
		t.Fatalf("square outline interior %v", c)
	}
	if c := Coverage(KindOutline, 5, 9, r, 3); c != 1 {
		t.Fatalf("round outline straight edge %v", c)
	}
	if c := Coverage(KindOutline, 5, 5, r, 3); c != 0 {
		t.Fatalf("round outline interior %v", c)
	}
	if c := Coverage(KindOutline, 0, 0, r, 3); c != 0 {
		t.Fatalf("round outline outer corner %v", c)
	}
	if c := Coverage(KindOutline, 1, 1, r, 3); c <= 0 {
		t.Fatalf("round outline ring %v", c)
	}
}

// A faint colour on a partly covered pixel rounds to alpha 0, which leaves the
// pixel alone; a radius past half the smaller side is clamped to it.
func TestFaintCornerAndClampedRadius(t *testing.T) {
	l := nl(10, 10)
	// Pixel (0,1) at radius 4: 4.5 − √(3.5²+2.5²) ≈ 0.199, and 1·0.199 rounds to 0.
	l.FillRoundRect(Rect{0, 0, 10, 10}, 4, RGBA{255, 255, 255, 1})
	buf := make([]byte, 400)
	_ = l.Rasterize(buf)
	if got := px(buf, 10, 0, 1); got != (RGBA{}) {
		t.Fatalf("faint partial pixel changed: %v", got)
	}
	if got := px(buf, 10, 5, 5); got.A != 1 {
		t.Fatalf("interior: %v", got)
	}
	l.Reset()
	l.FillRoundRect(Rect{0, 0, 10, 6}, 50, RGBA{A: 255})
	if r := l.Instances()[0].Radius; r != 3 {
		t.Fatalf("radius %d, want 3 (half of 6)", r)
	}
}

func TestRoundedFillAntialiasesCorners(t *testing.T) {
	l := nl(10, 10)
	l.FillRoundRect(Rect{0, 0, 10, 10}, 4, RGBA{0, 0, 0, 255})
	l.FillRoundRect(Rect{0, 0, 10, 10}, 0, RGBA{0, 0, 0, 255}) // radius 0: plain
	if l.Instances()[1].Radius != 0 {
		t.Fatal("radius 0 is not plain")
	}
	l.Reset()
	l.FillRoundRect(Rect{0, 0, 10, 10}, 4, RGBA{0, 0, 0, 255})
	buf := make([]byte, 400)
	_ = l.Rasterize(buf)
	if a := px(buf, 10, 0, 0).A; a != 0 {
		t.Fatalf("outer corner alpha %d", a)
	}
	if a := px(buf, 10, 1, 1).A; a == 0 || a == 255 {
		t.Fatalf("corner edge alpha %d, want partial", a)
	}
	if a := px(buf, 10, 5, 5).A; a != 255 {
		t.Fatalf("interior alpha %d", a)
	}
}
