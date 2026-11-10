// Emitted by go-gfx/wgsl/hlsl for Shader Model 5.0 (cs_5_0). Do not edit.
// Compile with D3DCOMPILE_IEEE_STRICTNESS (docs/HLSL.md, rule 1).

// int division by a non-constant warns (X3556); wgsl_div guards its divisor.
#pragma warning(disable: 3556)

// Rule 1: a precise variable restricts every operation that contributes to the
// value assigned to it. Each float +, -, * and / is passed through one, so a
// mul and an add are never fused into a mad and nothing is reassociated.
float wgsl_pf(float x) { precise float r = x; return r; }
float2 wgsl_pf(float2 x) { precise float2 r = x; return r; }
float3 wgsl_pf(float3 x) { precise float3 r = x; return r; }
float4 wgsl_pf(float4 x) { precise float4 r = x; return r; }

static const int wgsl_int_min = -2147483647 - 1;

// WGSL i32 arithmetic wraps (two's complement), through uint.
int wgsl_add(int a, int b) { return asint(asuint(a) + asuint(b)); }
int wgsl_sub(int a, int b) { return asint(asuint(a) - asuint(b)); }
int wgsl_mul(int a, int b) { return asint(asuint(a) * asuint(b)); }
int wgsl_neg(int a) { return asint(0u - asuint(a)); }
// WGSL division and remainder: e2 == 0, and MIN / -1, are defined.
int wgsl_div(int a, int b) {
  bool special = b == 0 || (a == wgsl_int_min && b == -1);
  return a / (special ? 1 : b);
}
int wgsl_rem(int a, int b) {
  bool special = b == 0 || (a == wgsl_int_min && b == -1);
  int d = special ? 1 : b;
  return special ? 0 : wgsl_sub(a, wgsl_mul(a / d, d));
}
uint wgsl_udiv(uint a, uint b) { return a / (b == 0u ? 1u : b); }
uint wgsl_urem(uint a, uint b) { return b == 0u ? 0u : a % (b == 0u ? 1u : b); }
// WGSL shifts: the count modulo 32; << discards high bits; >> on i32 is
// arithmetic.
int wgsl_shl(int a, uint s) { return asint(asuint(a) << (s & 31u)); }
int wgsl_shr(int a, uint s) {
  uint u = asuint(a) >> (s & 31u);
  return asint(a < 0 ? (u | ~(0xFFFFFFFFu >> (s & 31u))) : u);
}
// WGSL f32 to integer: truncate, then the nearest value of the integer type
// that is also exactly representable in f32 (u32 saturates at 4294967040, i32
// at 2147483520); a NaN gives an indeterminate value, here 0.
uint wgsl_f2u(float v) { return v >= 4294967040.0f ? 4294967040u : (v > 0.0f ? uint(v) : 0u); }
int wgsl_f2i(float v) { return v >= 2147483520.0f ? 2147483520 : (v <= -2147483648.0f ? wgsl_int_min : (v == v ? int(v) : 0)); }
// length: the explicit sum of squares in WGSL's order, each step precise.
float wgsl_length(float2 v) {
  precise float xx = v.x * v.x;
  precise float yy = v.y * v.y;
  precise float s = xx + yy;
  return sqrt(s);
}
float wgsl_length(float3 v) {
  precise float xx = v.x * v.x;
  precise float yy = v.y * v.y;
  precise float zz = v.z * v.z;
  precise float s0 = xx + yy;
  precise float s = s0 + zz;
  return sqrt(s);
}
float wgsl_length(float4 v) {
  precise float xx = v.x * v.x;
  precise float yy = v.y * v.y;
  precise float zz = v.z * v.z;
  precise float ww = v.w * v.w;
  precise float s0 = xx + yy;
  precise float s1 = s0 + zz;
  precise float s = s1 + ww;
  return sqrt(s);
}

struct Inst {
  int x;
  int y;
  int w;
  int h;
  int cx0;
  int cy0;
  int cx1;
  int cy1;
  uint color;
  int radius;
  uint kind;
  uint data;
  uint stride;
  uint srcW;
  uint srcH;
  uint pad;
};

struct Frame {
  uint w;
  uint h;
  uint across;
  uint tile;
};

StructuredBuffer<Inst> insts : register(t0);
StructuredBuffer<uint> bins : register(t1);
StructuredBuffer<uint> arena : register(t2);
RWStructuredBuffer<uint> px : register(u3);
cbuffer wgsl_cb4 : register(b4) { Frame f; };

uint byteAt(uint i) {
  return (((arena[(i >> (2u & 31u))] >> (((((i & 3u)) * 8u)) & 31u))) & 255u);
}

float3 cornerCenter(int x, int y, Inst s, int rad) {
  bool left = (x < wgsl_add(s.x, rad));
  bool right = (x >= wgsl_sub(wgsl_add(s.x, s.w), rad));
  bool top = (y < wgsl_add(s.y, rad));
  bool bottom = (y >= wgsl_sub(wgsl_add(s.y, s.h), rad));
  if (((left && top))) {
    return float3(float(wgsl_add(s.x, rad)), float(wgsl_add(s.y, rad)), 1.0f);
  }
  if (((right && top))) {
    return float3(float(wgsl_sub(wgsl_add(s.x, s.w), rad)), float(wgsl_add(s.y, rad)), 1.0f);
  }
  if (((left && bottom))) {
    return float3(float(wgsl_add(s.x, rad)), float(wgsl_sub(wgsl_add(s.y, s.h), rad)), 1.0f);
  }
  if (((right && bottom))) {
    return float3(float(wgsl_sub(wgsl_add(s.x, s.w), rad)), float(wgsl_sub(wgsl_add(s.y, s.h), rad)), 1.0f);
  }
  return ((float3)(0.0f));
}

float coverage(int x, int y, Inst s) {
  int rad = s.radius;
  precise float3 c = cornerCenter(x, y, s, rad);
  precise float d = wgsl_length(float2(wgsl_pf(wgsl_pf(float(x) + 0.5f) - c.x), wgsl_pf(wgsl_pf(float(y) + 0.5f) - c.y)));
  if (((s.kind == 0u))) {
    if (((c.z == 0.0f))) {
      return 1.0f;
    }
    return min(max(wgsl_pf(wgsl_pf(float(rad) + 0.5f) - d), 0.0f), 1.0f);
  }
  if ((((rad > 0) && (c.z != 0.0f)))) {
    return max(wgsl_pf(1.0f - abs(wgsl_pf(d - (wgsl_pf(float(rad) - 0.5f))))), 0.0f);
  }
  bool edge = ((((y == s.y) || (y == wgsl_sub(wgsl_add(s.y, s.h), 1))) || (x == s.x)) || (x == wgsl_sub(wgsl_add(s.x, s.w), 1)));
  return (edge ? 1.0f : 0.0f);
}

uint4 blend(uint4 dst, uint sr, uint sg, uint sb, uint ca) {
  if (((ca == 255u))) {
    return uint4(sr, sg, sb, 255u);
  }
  if (((ca == 0u))) {
    return dst;
  }
  uint ia = (255u - ca);
  return uint4(wgsl_udiv(((((sr * ca) + (dst.x * ia)) + 127u)), 255u), wgsl_udiv(((((sg * ca) + (dst.y * ia)) + 127u)), 255u), wgsl_udiv(((((sb * ca) + (dst.z * ia)) + 127u)), 255u), (ca + wgsl_udiv((dst.w * ia), 255u)));
}

[numthreads(16, 8, 1)]
void raster(uint3 gid : SV_DispatchThreadID) {
  if ((((gid.x >= f.w) || (gid.y >= f.h)))) {
    return;
  }
  int x = asint(gid.x);
  int y = asint(gid.y);
  uint down = wgsl_udiv((((f.h + f.tile) - 1u)), f.tile);
  uint t = (((wgsl_udiv(gid.y, f.tile)) * f.across) + wgsl_udiv(gid.x, f.tile));
  uint first = bins[(2u * t)];
  uint count = bins[((2u * t) + 1u)];
  uint list = (((2u * f.across) * down) + first);
  uint i = ((gid.y * f.w) + gid.x);
  uint p = px[i];
  uint4 d = uint4((p & 255u), (((p >> (8u & 31u))) & 255u), (((p >> (16u & 31u))) & 255u), (p >> (24u & 31u)));
  { [loop] for (uint k = 0u; (k < count); k++) {
    Inst s = insts[bins[(list + k)]];
    if ((((((x < max(s.x, s.cx0)) || (x >= min(wgsl_add(s.x, s.w), s.cx1))) || (y < max(s.y, s.cy0))) || (y >= min(wgsl_add(s.y, s.h), s.cy1))))) {
      continue;
    }
    uint sr = (s.color & 255u);
    uint sg = (((s.color >> (8u & 31u))) & 255u);
    uint sb = (((s.color >> (16u & 31u))) & 255u);
    uint ca = (s.color >> (24u & 31u));
    if ((((s.kind == 2u) || (s.kind == 3u)))) {
      uint m = byteAt(((s.data + (asuint(wgsl_sub(y, s.y)) * s.stride)) + asuint(wgsl_sub(x, s.x))));
      if (((m == 0u))) {
        continue;
      }
      if ((((s.kind == 2u) && (ca != 255u)))) {
        m = wgsl_udiv((m * ca), 255u);
      }
      d = blend(d, sr, sg, sb, m);
      continue;
    }
    if (((s.kind == 4u))) {
      uint sx = wgsl_udiv((asuint(wgsl_sub(x, s.x)) * s.srcW), asuint(s.w));
      uint sy = wgsl_udiv((asuint(wgsl_sub(y, s.y)) * s.srcH), asuint(s.h));
      uint o = (s.data + ((((sy * s.srcW) + sx)) * 4u));
      d = blend(d, byteAt(o), byteAt((o + 1u)), byteAt((o + 2u)), byteAt((o + 3u)));
      continue;
    }
    precise float cov = coverage(x, y, s);
    if (((cov <= 0.0f))) {
      continue;
    }
    if (((cov < 1.0f))) {
      ca = wgsl_f2u(wgsl_pf(wgsl_pf(float(ca) * cov) + 0.5f));
    }
    d = blend(d, sr, sg, sb, ca);
  } }
  px[i] = (((d.x | ((d.y << (8u & 31u)))) | ((d.z << (16u & 31u)))) | ((d.w << (24u & 31u))));
}
