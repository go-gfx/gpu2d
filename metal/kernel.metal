// One thread per pixel. A pixel walks the instances of its tile in draw
// order, computes gpu2d.Coverage at its centre in f32, and blends in integers
// exactly as PixelPainter's blendInto does, floored alpha included.
//
// This file follows coverage_gen.metal, generated from webgpu/shader.wgsl: the
// includes, the Inst structure, cornerCenter and coverage come from there.

// blend composites (sr, sg, sb) at alpha ca over (r, g, b, a) exactly as
// PixelPainter's blendInto: opaque overwrites, 0 leaves, else rounded colour
// and floored alpha.
static void blend(thread uint &r, thread uint &g, thread uint &b, thread uint &a,
                  uint sr, uint sg, uint sb, uint ca) {
  if (ca == 255u) {
    r = sr; g = sg; b = sb; a = 255u;
    return;
  }
  if (ca == 0u) { return; }
  uint ia = 255u - ca;
  r = (sr * ca + r * ia + 127u) / 255u;
  g = (sg * ca + g * ia + 127u) / 255u;
  b = (sb * ca + b * ia + 127u) / 255u;
  a = ca + a * ia / 255u;
}

struct Frame {
  uint w, h, across, tile;
};


kernel void raster(device const Inst *insts [[buffer(0)]],
                   device const uint *spans [[buffer(1)]],
                   device const uint *list [[buffer(2)]],
                   device uchar4 *px [[buffer(3)]],
                   constant Frame &f [[buffer(4)]],
                   device const uchar *data [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= f.w || gid.y >= f.h) { return; }
  int x = int(gid.x);
  int y = int(gid.y);
  uint t = (gid.y / f.tile) * f.across + gid.x / f.tile;
  uint first = spans[2 * t];
  uint count = spans[2 * t + 1];
  uint i = gid.y * f.w + gid.x;
  uchar4 p = px[i];
  uint r = p.x, g = p.y, b = p.z, a = p.w;
  for (uint k = 0; k < count; k++) {
    device const Inst &s = insts[list[first + k]];
    if (x < max(s.x, s.cx0) || x >= min(s.x + s.w, s.cx1) || y < max(s.y, s.cy0) || y >= min(s.y + s.h, s.cy1)) {
      continue;
    }
    uint sr = s.color & 255u, sg = (s.color >> 8) & 255u, sb = (s.color >> 16) & 255u;
    uint ca = s.color >> 24;
    if (s.kind == 2u || s.kind == 3u) {
      // A mask's coverage, or a path's stored alpha.
      uint m = data[s.data + uint(y - s.y) * s.stride + uint(x - s.x)];
      if (m == 0u) { continue; }
      if (s.kind == 2u && ca != 255u) { m = m * ca / 255u; }
      blend(r, g, b, a, sr, sg, sb, m);
      continue;
    }
    if (s.kind == 4u) {
      // Nearest neighbour, PixelPainter's integer mapping.
      uint sx = uint(x - s.x) * s.srcW / uint(s.w);
      uint sy = uint(y - s.y) * s.srcH / uint(s.h);
      uint o = s.data + (sy * s.srcW + sx) * 4u;
      blend(r, g, b, a, data[o], data[o + 1], data[o + 2], data[o + 3]);
      continue;
    }
    float cov = coverage(x, y, s);
    if (cov <= 0.0f) { continue; }
    if (cov < 1.0f) { ca = uint(float(ca) * cov + 0.5f); }
    blend(r, g, b, a, sr, sg, sb, ca);
  }
  px[i] = uchar4(uchar(r), uchar(g), uchar(b), uchar(a));
}

// present copies the frame into a layer's drawable. The drawable is BGRA;
// written as RGBA floats, Metal stores the channels in its order, and c/255
// converts back to the byte c exactly.
kernel void present(device const uchar4 *px [[buffer(3)]],
                    constant Frame &f [[buffer(4)]],
                    texture2d<float, access::write> out [[texture(0)]],
                    uint2 g [[thread_position_in_grid]]) {
  if (g.x >= f.w || g.y >= f.h) { return; }
  out.write(float4(px[g.y * f.w + g.x]) / 255.0f, g);
}
