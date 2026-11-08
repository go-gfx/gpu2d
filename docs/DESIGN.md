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

### Metal goes through compute (milestone 3)

go-macos/metal is compute only, and the Metal back-end does not need more:
the CPU bins instances into 16 × 16 tiles (`List.Bin`), and one thread per
pixel walks its tile's instances and blends with PixelPainter's integer
arithmetic. That buys exactness fixed-function blending cannot give (0
differing channels over a translucent ground too, against up to 4 in alpha
for WebGPU) at 1.08 ms for a 1080p widget frame and 2.79 ms at 4K, binning
included. Presenting the frame through a `CAMetalLayer` instead of the CPU
blit moves to milestone 5 with the window integration.

### Asynchronous submit (v0.3.0)

On a small frame the GPU is not the cost: an empty dispatch alone is 0.41 ms
on an M4 Max, and `Draw` blocks the CPU for all of it. go-macos/metal v0.4.0
adds `Device.Submit` (commit without `waitUntilCompleted`) and
`Pending.Wait`/`Done`, from MTLCommandBuffer's `commit`, `waitUntilCompleted`
and `status` (MTLCommandBuffer.h). The Renderer keeps two slots of input
buffers (instances, spans, list, data): `Submit` bins and packs into a free
slot, so the list is free to be reset at once and the CPU records frame N+1
while the GPU draws frame N. The target is one buffer; command buffers on one
queue run in order, so two frames in flight composite in submission order.
`Pixels` is therefore safe to read only after the last `Wait`. Misuse is an
error, not a hang: a third `Submit` is `ErrBusy`, `Wait` with nothing in
flight is `ErrNothingInFlight`, and `Close` drains first.

Measured with `BenchmarkOverlap` (960 × 2834 widget frame, M4 Max, 10 rounds
interleaved, benchstat medians, 1-minute load 8.0 to 8.2, a little above the
usual bound of 8, so times are indicative; allocation counts are exact). The
CPU work between frames is a 1.5 ms spin, the size of the gallery's recording:

| CPU work per frame | `Draw` | `Submit` / `Wait` |
|---|---:|---:|
| 0 | 1.31 ms | 1.33 ms |
| 1.5 ms | 3.17 ms | 1.78 ms |

Each frame cost 132 allocations, nearly all inside go-macos/objc's
reflective message sends. With go-macos/metal v0.5.0, which sends per-frame
messages through objc's `Msg0`–`Msg3`, it costs 30 (`BenchmarkOverlap`,
2026-11-05).

List, Bin and AppendBytes were checked for per-frame allocation: with reused
slices they allocate nothing (`TestSteadyFrameAllocatesNothing`).

### Masks, images and paths (milestone 4)

Everything PixelPainter draws beyond rectangles carries data, so masks and
images live in a per-frame **data arena** next to the instances, and the
instance grew to 64 bytes (`Data`, `Stride`, `SrcW`, `SrcH`):

- `KindMask` is `DrawMask`. Coverage is scaled by a translucent ink with a
  floored product, and a row the caller's mask cannot supply stays untouched.
- `KindAlpha` carries a path's coverage already turned into alpha
  (`uint8(A·cov + 0.5)`), because PixelPainter rounds that differently from a
  mask.
- `KindImage` is `DrawImage`, with the same integer nearest-neighbour mapping.

The parity test drives paths through the same go-gfx/gfx/vector rasterizer
PixelPainter calls. Metal stays byte-identical on both grounds; WebGPU stays
identical over an opaque ground (alpha up to 3 apart over a translucent one).

A defect found on the way: **PixelPainter ignores the translation for paths.**
Under `PushTranslate(20, 0)` a rectangle moves and a path does not, so a
chart inside a scrolled view is drawn where it would be unscrolled.
`DrawAlpha` takes coverage in surface pixels, which keeps today's parity
exact; the fix belongs in go-gfx/gfx/vector (translating a path) and
go-widgets/painter, and the adapter of milestone 5 applies the translation.

## Milestones

1. Instance list and CPU executor: fill, rounded fill, outline, rounded outline,
   translation, clip stack.
2. WebGPU back-end: WGSL shader, offscreen render and read-back, browser test on
   a real GPU against `Rasterize`.
3. Metal back-end: a compute kernel over tiles through go-macos/metal as it
   is (done; see above).
4. Glyph masks, images and paths through a per-frame data arena (done; see
   above). Keeping glyphs and images resident across frames, so that the
   arena carries only what changed, is a later optimisation.
5. go-widgets adapter implementing `painter.Painter` (done: go-widgets/
   gpupainter). Presentation: go-macos/metal v0.6.0 presents a drawable, and
   `Renderer.Present` copies the frame into a `CAMetalLayer`'s drawable with a
   second kernel, so the frame never returns to the CPU (done, v0.4.0). The
   Cocoa window hosts it (go-widgets/window v0.90.0, `Config.GPU`). In a
   browser, `webgpu.Renderer.Present` puts the target into the canvas texture
   in the browser's preferred format (v0.6.0): `copyTextureToTexture` into
   rgba8unorm, `PresentShader` into bgra8unorm (WebGPU spec, supported context
   formats and copy validation). go-widgets/webcanvas v0.9.0 presents through
   it (done).
6. A WGSL→MSL emitter in go-gfx/wgsl retires the hand-written Metal copy of the
   coverage expression (done: wgsl v0.5.0's `msl` package; since gpu2d v0.5.2,
   `metal/coverage_gen.metal` is generated from the WebGPU shader and
   `TestCoverageMetalIsGenerated` catches drift). The coverage now exists in
   Go and WGSL, each held to the CPU reference by test.

## Sources opened

- WebGPU specification, gpuweb/gpuweb `spec/index.bs`: Blend State
  (`GPUBlendComponent`, `GPUBlendFactor`), `GPUVertexStepMode`,
  `GPUCanvasAlphaMode`.
- Apple, CAMetalLayer documentation (`nextDrawable`, `pixelFormat`,
  `drawableSize`, `framebufferOnly`, `maximumDrawableCount`); the macOS SDK's
  `CAMetalLayer.h` (BGRA8Unorm only, `device` nil by default on macOS) and
  `MTLCommandBuffer.h` (`presentDrawable:` before `commit`).
- go-widgets/painter `round.go`, `pixel.go`, `clip.go`, `translate.go`.
- go-gfx/webgpu README (Recorder, atlas, surfaces), go-macos/metal README
  (compute only), go-gfx/wgsl README (no MSL emitter yet).
