// One quad per gpu2d instance. The fragment shader computes gpu2d.Coverage
// at the pixel centre exactly, in integers; fixed-function blending
// composites the result in draw order.

struct Inst {
  x: i32, y: i32, w: i32, h: i32,
  cx0: i32, cy0: i32, cx1: i32, cy1: i32,
  color: u32, radius: i32, kind: u32,
  data: u32, stride: u32, srcW: u32, srcH: u32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> insts: array<Inst>;
@group(0) @binding(1) var<uniform> surface: vec4f; // width, height
@group(0) @binding(2) var<storage, read> arena: array<u32>; // masks and images

// byteAt reads byte i of the arena (little-endian words).
fn byteAt(i: u32) -> u32 {
  return (arena[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) inst: u32,
}

// The quad is the instance's rectangle cut by its clip, two triangles:
// corners (x0,y0) (x1,y0) (x0,y1) (x0,y1) (x1,y0) (x1,y1). Bit v of 0x32
// picks x1, bit v of 0x2C picks y1.
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
  let s = insts[i];
  let x0 = max(s.x, s.cx0);
  let y0 = max(s.y, s.cy0);
  let x1 = min(s.x + s.w, s.cx1);
  let y1 = min(s.y + s.h, s.cy1);
  let px = select(x0, x1, ((0x32u >> v) & 1u) == 1u);
  let py = select(y0, y1, ((0x2Cu >> v) & 1u) == 1u);
  var o: VOut;
  o.pos = vec4f(f32(px) / surface.x * 2.0 - 1.0, 1.0 - f32(py) / surface.y * 2.0, 0.0, 1.0);
  o.inst = i;
  return o;
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

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let s = insts[v.inst];
  let x = i32(floor(v.pos.x));
  let y = i32(floor(v.pos.y));
  let c = s.color;
  let rgb = vec3f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u));
  if (s.kind == 2u || s.kind == 3u) {
    // A mask's coverage (scaled by a translucent ink with a floored
    // product), or a path's stored alpha.
    var m = byteAt(s.data + u32(y - s.y) * s.stride + u32(x - s.x));
    if (m == 0u) { discard; }
    let ink = c >> 24u;
    if (s.kind == 2u && ink != 255u) { m = m * ink / 255u; }
    if (m == 0u) { discard; }
    return vec4f(rgb, f32(m)) / 255.0;
  }
  if (s.kind == 4u) {
    // Nearest neighbour, PixelPainter's integer mapping.
    let sx = u32(x - s.x) * s.srcW / u32(s.w);
    let sy = u32(y - s.y) * s.srcH / u32(s.h);
    let o = s.data + (sy * s.srcW + sx) * 4u;
    let t = vec4f(f32(byteAt(o)), f32(byteAt(o + 1u)), f32(byteAt(o + 2u)), f32(byteAt(o + 3u)));
    if (t.a == 0.0) { discard; }
    return t / 255.0;
  }
  let a = coverage(x, y, s, s.color >> 24u);
  if (a == 0u) { discard; }
  return vec4f(rgb, f32(a)) / 255.0;
}
