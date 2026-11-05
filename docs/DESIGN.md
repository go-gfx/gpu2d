# gpu2d design

## Why

A full frame of the go-widgets gallery (960 × 2833 px, every widget family) costs
3.2 ms of CPU on an Apple M4 Max today (`BenchmarkGalleryFrame`, 2026-11-05,
load 7.9, indicative). That fits a 60 Hz budget at this size. It no longer fits
at 8 Mpx (a 4K or Retina full repaint, about 10 ms by proportion), during
animation and scrolling, or when the CPU has other work: on the same machine a 4K
image pipeline took 0.16 ms of CPU on Metal against 82 ms on sixteen cores
(go-macos/metal README). Every go-widgets back-end presents a CPU-drawn RGBA
buffer today (Cocoa `NSBitmapImageRep`, Win32 `StretchDIBits`, Wayland `wl_shm`,
X11 MIT-SHM, browser `putImageData`); none uses the GPU.

## Shape

```
go-widgets/toolkit ─ painter.Painter ─┬─ painter.PixelPainter (CPU, today)
                                      └─ gpu2d adapter (go-widgets side)
                                             │
go-gfx/gpu2d  ── List (instances, backend-neutral)
                   ├─ Rasterize: the CPU reference executor
                   ├─ webgpu back-end (GOOS=js, go-gfx/webgpu Recorder, WGSL)
                   └─ metal back-end (darwin, go-macos/metal, MSL)
```

One **instance list** describes a frame. Each primitive is one 48-byte instance
(rectangle, clip, colour, radius, kind) in draw order. Every back-end consumes
the same list:

- `Rasterize` executes it on the CPU. It is a port of go-widgets/painter's
  `PixelPainter` formulas (`round.go` `cornerFillCoverage`, the ring of
  `strokeRoundOutline`, `pixel.go` `blendInto`) and is the oracle the GPU
  back-ends are checked against.
- A GPU back-end uploads the list as an instance buffer and draws one quad per
  instance (`GPUVertexStepMode` `"instance"`: the instance buffer's address
  advances by `arrayStride` per instance, WebGPU spec, Vertex State). The
  fragment shader computes the same coverage per pixel centre, and fixed-function
  blending with `srcFactor` `"src-alpha"`, `dstFactor` `"one-minus-src-alpha"`
  composites in order (WebGPU spec, §Blend State).

Coordinates are integers in the instance, so no rounding enters before the
shader. A degenerate outline that `PixelPainter` paints twice (a 1-pixel-tall
outline repaints its row) is emitted as two instances, so even multiplicity is
the reference's.

## Correctness contract

- `Rasterize` must equal `painter.PixelPainter` **byte for byte** on every
  primitive it implements. The go-widgets adapter carries that test, since
  go-gfx must not depend on go-widgets.
- A GPU back-end must equal `Rasterize` within **1 per channel** (|Δ| ≤ 1). The
  coverage is the same formula in f32 instead of f64; the blend is unorm
  arithmetic instead of `(src·a + dst·(255−a) + 127) / 255`. The tests report
  the maximum difference and how many channels differ, and fail above 1.

### Measured against the contract (milestone 2)

WebGPU in Chromium on an Apple M4 Max (Metal), 200 random scenes, 2902
instances: over an opaque ground every channel equals `Rasterize`; over a
translucent ground the colour channels do and alpha differs by up to 4, from
PixelPainter's floored `a + dst·(255−a)/255` against the GPU's rounding. The
1-per-channel contract therefore holds for opaque surfaces, which is every
window; alpha over translucent ground is reported, not gated.

## Milestones

1. Instance list and CPU executor: fill, rounded fill, outline, rounded outline,
   translation, clip stack.
2. WebGPU back-end: WGSL shader, offscreen render and read-back, browser test on
   a real GPU against `Rasterize`.
3. Metal back-end: go-macos/metal gains textures, a render pipeline, offscreen
   read-back and `CAMetalLayer` presentation; MSL twin of the WGSL shader.
4. Glyph masks through an atlas (go-gfx/webgpu `atlas.Shelf`), images as
   textures, paths as CPU coverage masks uploaded once.
5. go-widgets adapter implementing `painter.Painter`; webcanvas and the Cocoa
   window present through it. Measured against `PixelPainter` at 1×, 2× and 4K:
   CPU time and wall time per frame.
6. A WGSL→MSL emitter in go-gfx/wgsl (its `Info` already carries types, layouts
   and entry points) retires the hand-written MSL twin.

## Sources opened

- WebGPU specification, gpuweb/gpuweb `spec/index.bs`: Blend State
  (`GPUBlendComponent`, `GPUBlendFactor`), `GPUVertexStepMode`,
  `GPUCanvasAlphaMode`.
- Apple, CAMetalLayer documentation (`nextDrawable`, `pixelFormat`,
  `drawableSize`, `framebufferOnly`, `maximumDrawableCount`).
- go-widgets/painter `round.go`, `pixel.go`, `clip.go`, `translate.go`.
- go-gfx/webgpu README (Recorder, atlas, surfaces), go-macos/metal README
  (compute only), go-gfx/wgsl README (no MSL emitter yet).
