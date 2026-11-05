// One quad per gpu2d instance. The fragment shader computes gpu2d.Coverage
// at the pixel centre, step for step, in f32; fixed-function blending
// composites the result in draw order.

struct Inst {
  x: i32, y: i32, w: i32, h: i32,
  cx0: i32, cy0: i32, cx1: i32, cy1: i32,
  color: u32, radius: i32, kind: u32, pad: u32,
}

@group(0) @binding(0) var<storage, read> insts: array<Inst>;
@group(0) @binding(1) var<uniform> surface: vec4f; // width, height

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
fn cornerCenter(x: i32, y: i32, s: Inst, rad: i32) -> vec3f {
  let left = x < s.x + rad;
  let right = x >= s.x + s.w - rad;
  let top = y < s.y + rad;
  let bottom = y >= s.y + s.h - rad;
  if (left && top) { return vec3f(f32(s.x + rad), f32(s.y + rad), 1.0); }
  if (right && top) { return vec3f(f32(s.x + s.w - rad), f32(s.y + rad), 1.0); }
  if (left && bottom) { return vec3f(f32(s.x + rad), f32(s.y + s.h - rad), 1.0); }
  if (right && bottom) { return vec3f(f32(s.x + s.w - rad), f32(s.y + s.h - rad), 1.0); }
  return vec3f(0.0);
}

fn coverage(x: i32, y: i32, s: Inst) -> f32 {
  let rad = s.radius;
  let c = cornerCenter(x, y, s, rad);
  let d = length(vec2f(f32(x) + 0.5 - c.x, f32(y) + 0.5 - c.y));
  if (s.kind == 0u) {
    if (c.z == 0.0) { return 1.0; }
    return clamp(f32(rad) + 0.5 - d, 0.0, 1.0);
  }
  if (rad > 0 && c.z != 0.0) {
    return max(1.0 - abs(d - (f32(rad) - 0.5)), 0.0);
  }
  let edge = y == s.y || y == s.y + s.h - 1 || x == s.x || x == s.x + s.w - 1;
  return select(0.0, 1.0, edge);
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let s = insts[v.inst];
  let x = i32(floor(v.pos.x));
  let y = i32(floor(v.pos.y));
  let cov = coverage(x, y, s);
  if (cov <= 0.0) { discard; }
  var a = s.color >> 24u;
  if (cov < 1.0) { a = u32(f32(a) * cov + 0.5); }
  if (a == 0u) { discard; }
  let c = s.color;
  return vec4f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), f32(a)) / 255.0;
}
