// One thread per pixel. A pixel walks the instances of its tile in draw
// order, computes gpu2d.Coverage at its centre in f32, and blends in integers
// exactly as PixelPainter's blendInto does, floored alpha included.

#include <metal_stdlib>
using namespace metal;

struct Inst {
  int x, y, w, h;
  int cx0, cy0, cx1, cy1;
  uint color;
  int radius;
  uint kind;
  uint pad;
};

struct Frame {
  uint w, h, across, tile;
};

// xy: the circle centre of the corner box (x, y) lies in; z: 1 when in one.
static float3 cornerCenter(int x, int y, device const Inst &s, int rad) {
  bool left = x < s.x + rad;
  bool right = x >= s.x + s.w - rad;
  bool top = y < s.y + rad;
  bool bottom = y >= s.y + s.h - rad;
  if (left && top) { return float3(float(s.x + rad), float(s.y + rad), 1.0f); }
  if (right && top) { return float3(float(s.x + s.w - rad), float(s.y + rad), 1.0f); }
  if (left && bottom) { return float3(float(s.x + rad), float(s.y + s.h - rad), 1.0f); }
  if (right && bottom) { return float3(float(s.x + s.w - rad), float(s.y + s.h - rad), 1.0f); }
  return float3(0.0f);
}

static float coverage(int x, int y, device const Inst &s) {
  int rad = s.radius;
  float3 c = cornerCenter(x, y, s, rad);
  float dx = float(x) + 0.5f - c.x;
  float dy = float(y) + 0.5f - c.y;
  float d = precise::sqrt(dx * dx + dy * dy);
  if (s.kind == 0u) {
    if (c.z == 0.0f) { return 1.0f; }
    return clamp(float(rad) + 0.5f - d, 0.0f, 1.0f);
  }
  if (rad > 0 && c.z != 0.0f) {
    return max(1.0f - precise::fabs(d - (float(rad) - 0.5f)), 0.0f);
  }
  bool edge = y == s.y || y == s.y + s.h - 1 || x == s.x || x == s.x + s.w - 1;
  return edge ? 1.0f : 0.0f;
}

kernel void raster(device const Inst *insts [[buffer(0)]],
                   device const uint *spans [[buffer(1)]],
                   device const uint *list [[buffer(2)]],
                   device uchar4 *px [[buffer(3)]],
                   constant Frame &f [[buffer(4)]],
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
    float cov = coverage(x, y, s);
    if (cov <= 0.0f) { continue; }
    uint ca = s.color >> 24;
    if (cov < 1.0f) { ca = uint(float(ca) * cov + 0.5f); }
    uint sr = s.color & 255u, sg = (s.color >> 8) & 255u, sb = (s.color >> 16) & 255u;
    if (ca == 255u) {
      r = sr; g = sg; b = sb; a = 255u;
      continue;
    }
    if (ca == 0u) { continue; }
    uint ia = 255u - ca;
    r = (sr * ca + r * ia + 127u) / 255u;
    g = (sg * ca + g * ia + 127u) / 255u;
    b = (sb * ca + b * ia + 127u) / 255u;
    a = ca + a * ia / 255u;
  }
  px[i] = uchar4(uchar(r), uchar(g), uchar(b), uchar(a));
}
