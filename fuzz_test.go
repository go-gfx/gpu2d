// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import (
	"encoding/binary"
	"testing"
)

// FuzzList drives a list with arbitrary operations and coordinates: recording
// and Rasterize must neither panic nor write past the surface, whose bytes are
// followed by a guard band that must stay untouched.
func FuzzList(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
	f.Add([]byte{3, 0xff, 0xff, 0xff, 0x7f, 1, 0, 0, 0x80, 2, 0x10, 0, 0, 0, 0, 9, 9, 9})
	f.Fuzz(func(t *testing.T, data []byte) {
		const w, h, guard = 17, 13, 64
		l := nl(w, h)
		next := func() int {
			if len(data) < 4 {
				data = nil
				return 0
			}
			v := int32(binary.LittleEndian.Uint32(data))
			data = data[4:]
			return int(v)
		}
		for len(data) > 0 {
			op := data[0] % 8
			data = data[1:]
			r := Rect{next(), next(), next(), next()}
			c := RGBA{uint8(r.X), uint8(r.Y), uint8(r.W), uint8(next())}
			switch op {
			case 0:
				l.FillRect(r, c)
			case 1:
				l.FillRoundRect(r, next(), c)
			case 2:
				l.StrokeRect(r, c, next()%64)
			case 3:
				l.StrokeRoundRect(r, next(), c, next()%64)
			case 4:
				l.PushClip(r)
			case 5:
				l.PushTranslate(r.X, r.Y)
			case 6:
				l.PopClip()
			default:
				l.PopTranslate()
			}
		}
		for _, in := range l.Instances() {
			if in.X < -MaxCoord || in.Y < -MaxCoord || int64(in.X)+int64(in.W) > MaxCoord || int64(in.Y)+int64(in.H) > MaxCoord {
				t.Fatalf("instance past MaxCoord: %+v", in)
			}
			if in.ClipX0 < 0 || in.ClipY0 < 0 || in.ClipX1 > w || in.ClipY1 > h {
				t.Fatalf("clip outside the surface: %+v", in)
			}
		}
		buf := make([]byte, 4*w*h+guard)
		for i := 4 * w * h; i < len(buf); i++ {
			buf[i] = 0xA5
		}
		if err := l.Rasterize(buf); err != nil {
			t.Fatal(err)
		}
		for i := 4 * w * h; i < len(buf); i++ {
			if buf[i] != 0xA5 {
				t.Fatalf("guard byte %d overwritten", i-4*w*h)
			}
		}
	})
}
