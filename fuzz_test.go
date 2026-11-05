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
		take := func(n int) []byte {
			n = min(max(n, 0), len(data))
			b := data[:n]
			data = data[n:]
			return b
		}
		for len(data) > 0 {
			op := data[0] % 11
			data = data[1:]
			r := Rect{next(), next(), next(), next()}
			c := RGBA{uint8(r.X), uint8(r.Y), uint8(r.W), uint8(next())}
			switch op {
			case 8:
				stride := next() % 64
				l.DrawMask(r, take(next()%256), stride, c)
				continue
			case 9:
				sw, sh := next()%16, next()%16
				l.DrawImage(r, take(next()%1024), sw, sh)
				continue
			case 10:
				raw := take(next() % 256)
				cov := make([]float64, len(raw))
				for i, b := range raw {
					cov[i] = float64(b)/128 - 0.25 // below 0 and above 1 too
				}
				l.DrawAlpha(cov, r.X, r.Y, r.W%32, r.H%32, c)
				continue
			}
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
			var need int64
			switch in.Kind {
			case KindMask, KindAlpha:
				need = int64(in.Data) + int64(in.Stride)*int64(in.H)
			case KindImage:
				need = int64(in.Data) + 4*int64(in.SrcW)*int64(in.SrcH)
			}
			if need > int64(len(l.Data())) {
				t.Fatalf("instance reads %d bytes of a %d-byte arena: %+v", need, len(l.Data()), in)
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
