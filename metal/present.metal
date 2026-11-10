// present copies the frame into a layer's drawable. It follows kernel_gen.metal,
// generated from kernel.wgsl, which declares Frame and the pixel buffer's layout:
// the kernel has no WGSL equivalent of this one because the drawable is a
// texture, which the MSL emitter does not write.
//
// The drawable is BGRA; written as RGBA floats, Metal stores the channels in
// its order, and c/255 converts back to the byte c exactly.
kernel void present(device const uchar4 *px [[buffer(3)]],
                    constant Frame &f [[buffer(4)]],
                    texture2d<float, access::write> out [[texture(0)]],
                    uint2 g [[thread_position_in_grid]]) {
  if (g.x >= f.w || g.y >= f.h) { return; }
  out.write(float4(px[g.y * f.w + g.x]) / 255.0f, g);
}
