// The whole raster kernel as one WGSL compute shader. One invocation per
// pixel: it walks the instances binned into its 16 x 16 tile in draw order,
// computes gpu2d.Coverage at its centre exactly, in integers (the functions of
// webgpu/shader.wgsl, character for character), and blends in integers exactly as
// PixelPainter's blendInto does, floored alpha included. It is the arithmetic
// of the Metal kernel, which is generated from this file
// (metal/kernel_gen.metal), so a frame is byte-identical to List.Rasterize over
// any ground (docs/NATIVE.md, decision 1).
//
// Written inside what the other emitters will express (docs/NATIVE.md): four
// storage buffers and one uniform, a 16 x 8 workgroup, pixels as array<u32>,
// blend returning a vector instead of taking references.

struct Inst {
  x: i32, y: i32, w: i32, h: i32,
  cx0: i32, cy0: i32, cx1: i32, cy1: i32,
  color: u32, radius: i32, kind: u32,
  data: u32, stride: u32, srcW: u32, srcH: u32,
  pad: u32,
}

// Frame: the surface size in pixels, the tiles across, and the tile side.
struct Frame {
  w: u32, h: u32, across: u32, tile: u32,
}

@group(0) @binding(0) var<storage, read> insts: array<Inst>;
// bins is List.Bin's spans (two words per tile: the first entry of the tile in
// the list, and how many) followed by its list (instance indices, each tile's
// in draw order).
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> arena: array<u32>; // masks and images
// px is one little-endian word per pixel: r, g, b, a from the lowest byte.
@group(0) @binding(3) var<storage, read_write> px: array<u32>;
@group(0) @binding(4) var<uniform> f: Frame;

// byteAt reads byte i of the arena (little-endian words).
fn byteAt(i: u32) -> u32 {
  return (arena[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// cornerCenter: xy is the circle centre of the corner box (x, y) lies in,
// z is 1 when it lies in one.
fn cornerCenter(x: i32, y: i32, s: Inst, rad: i32) -> vec3<i32> {
  let left = x < s.x + rad;
  let right = x >= s.x + s.w - rad;
  let top = y < s.y + rad;
  let bottom = y >= s.y + s.h - rad;
  if (left && top) { return vec3<i32>(s.x + rad, s.y + rad, 1); }
  if (right && top) { return vec3<i32>(s.x + s.w - rad, s.y + rad, 1); }
  if (left && bottom) { return vec3<i32>(s.x + rad, s.y + s.h - rad, 1); }
  if (right && bottom) { return vec3<i32>(s.x + s.w - rad, s.y + s.h - rad, 1); }
  return vec3<i32>(0, 0, 0);
}

// mul64 is a·b as (low word, high word), from 16-bit halves.
fn mul64(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 65535u;
  let a1 = a >> 16u;
  let b0 = b & 65535u;
  let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid = (p00 >> 16u) + (p01 & 65535u) + (p10 & 65535u);
  return vec2<u32>((p00 & 65535u) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}

// le64: a <= b for (low, high) pairs.
fn le64(a: vec2<u32>, b: vec2<u32>) -> bool {
  return a.y < b.y || (a.y == b.y && a.x <= b.x);
}

// isqrt64 is floor(sqrt(n)) for n below 2^48, from an estimate that is off by
// a few at most, corrected with exact products.
fn isqrt64(n: vec2<u32>, est: u32) -> u32 {
  var r = est;
  for (var i = 0; i < 16; i++) {
    if (!le64(mul64(r, r), n)) {
      r = r - 1u;
    } else if (le64(mul64(r + 1u, r + 1u), n)) {
      r = r + 1u;
    } else {
      break;
    }
  }
  return r;
}

// coverage is the alpha a primitive of instance s paints at pixel (x, y) with
// colour alpha ca: floor(ca·cov + 0.5) for the corner coverage cov, which is
// ca itself where cov >= 1 and 0 where cov <= 0 (a no-op in blend).
//
// The coverage is the exact one. The offsets from the pixel centre to the
// corner circle's centre are half-integers, so d4 = 4·D = dx² + dy² is an exact
// integer, and with S = sqrt(d4), d = S/2:
//   fill:    cov = (2·rad + 1 - S) / 2
//   outline: cov = 1 - |S - (2·rad - 1)| / 2
// both of which are decided here in integers, with no rounded value: the byte is
// the largest k with 2k <= T -+ ca·S, found from floor(sqrt(ca²·d4)) and its
// exact squares. Equal to the f64 reference (gpu2d.Coverage) on every device,
// whatever its sqrt, rounding mode or fused multiply-add. Valid for rad up to
// 16383 (gpu2d.MaxRadius), where d4 < 2^31.
fn coverage(x: i32, y: i32, s: Inst, ca: u32) -> u32 {
  let rad = s.radius;
  let c = cornerCenter(x, y, s, rad);
  if (c.z == 0) {
    if (s.kind == 0u) { return ca; }
    let edge = y == s.y || y == s.y + s.h - 1 || x == s.x || x == s.x + s.w - 1;
    return select(0u, ca, edge);
  }
  let dx = 2 * (x - c.x) + 1;
  let dy = 2 * (y - c.y) + 1;
  let d4 = u32(dx * dx + dy * dy);
  let m = u32(2 * rad - 1);
  let n = mul64(ca * ca, d4);
  let fl = isqrt64(n, u32(f32(ca) * sqrt(f32(d4))));
  let sq = mul64(fl, fl);
  let exact = sq.x == n.x && sq.y == n.y;
  if (s.kind == 0u) {
    let h = u32(2 * rad + 1);
    if (d4 >= h * h) { return 0u; }
    if (d4 <= m * m) { return ca; }
    let t = ca * h + 1u;
    var u = fl;
    if (!exact) { u = u + 1u; }
    if (((t - u) & 1u) != 0u) { u = u + 1u; }
    return (t - u) / 2u;
  }
  if (d4 == m * m) { return ca; }
  if (d4 >= (m + 2u) * (m + 2u)) { return 0u; }
  if (m >= 2u && d4 <= (m - 2u) * (m - 2u)) { return 0u; }
  if (d4 > m * m) {
    let t = 2u * ca + 1u + ca * m;
    var u = fl;
    if (!exact) { u = u + 1u; }
    if (((t - u) & 1u) != 0u) { u = u + 1u; }
    return (t - u) / 2u;
  }
  let t = 2 * i32(ca) + 1 - i32(ca * m);
  var u = i32(fl);
  if (((u - t) & 1) != 0) { u = u - 1; }
  return u32((u + t) / 2);
}

// blend composites (sr, sg, sb) at alpha ca over dst (r, g, b, a) exactly as
// PixelPainter's blendInto: opaque overwrites, 0 leaves, else rounded colour
// and floored alpha.
fn blend(dst: vec4<u32>, sr: u32, sg: u32, sb: u32, ca: u32) -> vec4<u32> {
  if (ca == 255u) { return vec4<u32>(sr, sg, sb, 255u); }
  if (ca == 0u) { return dst; }
  let ia = 255u - ca;
  return vec4<u32>(
    (sr * ca + dst.x * ia + 127u) / 255u,
    (sg * ca + dst.y * ia + 127u) / 255u,
    (sb * ca + dst.z * ia + 127u) / 255u,
    ca + dst.w * ia / 255u);
}

@compute @workgroup_size(16, 8)
fn raster(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= f.w || gid.y >= f.h) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let down = (f.h + f.tile - 1u) / f.tile;
  let t = (gid.y / f.tile) * f.across + gid.x / f.tile;
  let first = bins[2u * t];
  let count = bins[2u * t + 1u];
  let list = 2u * f.across * down + first;
  let i = gid.y * f.w + gid.x;
  let p = px[i];
  var d = vec4<u32>(p & 255u, (p >> 8u) & 255u, (p >> 16u) & 255u, p >> 24u);
  for (var k = 0u; k < count; k++) {
    let s = insts[bins[list + k]];
    if (x < max(s.x, s.cx0) || x >= min(s.x + s.w, s.cx1) || y < max(s.y, s.cy0) || y >= min(s.y + s.h, s.cy1)) {
      continue;
    }
    let sr = s.color & 255u;
    let sg = (s.color >> 8u) & 255u;
    let sb = (s.color >> 16u) & 255u;
    let ca = s.color >> 24u;
    if (s.kind == 2u || s.kind == 3u) {
      // A mask's coverage, or a path's stored alpha.
      var m = byteAt(s.data + u32(y - s.y) * s.stride + u32(x - s.x));
      if (m == 0u) { continue; }
      if (s.kind == 2u && ca != 255u) { m = m * ca / 255u; }
      d = blend(d, sr, sg, sb, m);
      continue;
    }
    if (s.kind == 4u) {
      // Nearest neighbour, PixelPainter's integer mapping.
      let sx = u32(x - s.x) * s.srcW / u32(s.w);
      let sy = u32(y - s.y) * s.srcH / u32(s.h);
      let o = s.data + (sy * s.srcW + sx) * 4u;
      d = blend(d, byteAt(o), byteAt(o + 1u), byteAt(o + 2u), byteAt(o + 3u));
      continue;
    }
    d = blend(d, sr, sg, sb, coverage(x, y, s, ca));
  }
  px[i] = d.x | (d.y << 8u) | (d.z << 16u) | (d.w << 24u);
}
