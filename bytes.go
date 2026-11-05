// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package gpu2d

import "encoding/binary"

// AppendBytes appends the instances to dst as a GPU reads them: InstanceSize
// bytes each, every field a little-endian 32-bit word in declaration order.
// It is the one encoding every back-end uploads.
func AppendBytes(dst []byte, ins []Instance) []byte {
	for i := range ins {
		in := &ins[i]
		for _, v := range [16]uint32{
			uint32(in.X), uint32(in.Y), uint32(in.W), uint32(in.H),
			uint32(in.ClipX0), uint32(in.ClipY0), uint32(in.ClipX1), uint32(in.ClipY1),
			in.Color, uint32(in.Radius), uint32(in.Kind),
			in.Data, in.Stride, in.SrcW, in.SrcH, 0,
		} {
			dst = binary.LittleEndian.AppendUint32(dst, v)
		}
	}
	return dst
}
