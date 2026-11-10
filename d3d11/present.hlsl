// Hand-written, as metal/present.metal is: the raster kernel is generated from
// kernel.wgsl, but this one writes a texture, which the emitter does not.
//
// It copies the frame (one uint per pixel, the RGBA bytes read little-endian:
// R | G << 8 | B << 16 | A << 24) into a swap chain's back buffer, an
// R8G8B8A8_UNORM texture bound as a typed UAV. A byte c stored as c / 255.0 is
// converted back to unorm8 with round-to-nearest, which gives c again, so the
// copy is exact (Direct3D 11.3 functional spec, "FLOAT -> UNORM" conversion).
//
// Compiled for cs_5_0 on the Windows lane and committed as present.dxbc
// (TestPresentDXBCIsCompiled recompiles and compares).
StructuredBuffer<uint> px : register(t0);
RWTexture2D<unorm float4> back : register(u0);
cbuffer Frame : register(b0) { uint w; uint h; uint pad0; uint pad1; };

[numthreads(16, 8, 1)]
void present(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= w || id.y >= h) return;
    uint p = px[id.y * w + id.x];
    back[id.xy] = float4(p & 255u, (p >> 8) & 255u, (p >> 16) & 255u, p >> 24) / 255.0;
}
