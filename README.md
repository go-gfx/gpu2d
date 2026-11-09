# gpu2d

A pure-Go (CGO=0) 2D renderer for the GPU. A frame is one backend-neutral list
of **instances**, one per primitive, which a GPU draws as one quad each and which
`Rasterize` executes on the CPU. `Rasterize` is the reference: it defines every
pixel, and each GPU back-end is checked against it.

**Status: milestone 4 of 6 ([docs/DESIGN.md](docs/DESIGN.md)).** The instance
list, the CPU reference, the WebGPU back-end (package `webgpu`, browser) and
the Metal back-end (package `metal`, macOS) exist, for rectangles, glyph
masks, images and paths. The go-widgets adapter comes next.

## Measured

The same widget-toolkit-shaped frame (`WidgetFrame`: background, a grid of
rounded cards with translucent borders, controls and hover washes) on an Apple
M4 Max, 2026-11-05, 1-minute load about 7 (indicative):

| frame | PixelPainter, native CPU | CPU in wasm | **WebGPU, Chromium on Metal** | **Metal compute, native** |
|---|---:|---:|---:|---:|
| 1920 × 1080, ~700 instances | 13.1 ms | 99.7 ms | **0.99 ms** | **1.08 ms** |
| 3840 × 2160, 2201 instances | 52.5 ms | 390 ms | **1.62 ms** | **2.79 ms** |

The GPU columns are whole frames: `Draw` and the wait for the GPU to finish,
binning on the CPU included for Metal.
Recording the 4K list costs 9.6 µs and no allocation (`BenchmarkRecord`). The
wasm column is `Rasterize`, the reference, which blends every pixel where
PixelPainter copies opaque rows, so it is slower than PixelPainter compiled to
wasm would be.

The GPU's pixels: over 200 random scenes (2902 instances) on an opaque ground,
**every channel of every pixel equals `Rasterize`**. Over a translucent ground
the colour channels still do; alpha differs by up to 4, because PixelPainter
computes `a + dst·(255−a)/255` with a floored division and fixed-function
blending rounds, and the difference grows with stacked translucent paint. A
window surface is opaque, so the case does not arise there. The Metal back-end
blends in integers like PixelPainter and equals `Rasterize` on **every channel
over either ground** (300 scenes, 4372 instances, 0 of 3,686,400 channels
differ on each).

```go
l, err := gpu2d.NewList(960, 600)
l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: 960, H: 600}, gpu2d.RGBA{R: 30, G: 30, B: 30, A: 255})
l.PushClip(gpu2d.Rect{X: 20, Y: 20, W: 400, H: 300})
l.FillRoundRect(gpu2d.Rect{X: 40, Y: 40, W: 120, H: 32}, 8, gpu2d.RGBA{R: 0, G: 122, B: 255, A: 255})
l.StrokeRoundRect(gpu2d.Rect{X: 40, Y: 40, W: 120, H: 32}, 8, gpu2d.RGBA{A: 40}, 1)
l.PopClip()

buf := make([]byte, 4*960*600)
err = l.Rasterize(buf) // CPU reference
gpuInput := l.Instances() // 48 bytes each, what a GPU back-end uploads
```

## What it draws

- Fills and one-pixel outlines of rectangles, square or with anti-aliased
  rounded corners, any number of concentric rings.
- `DrawMask`: one colour through 8-bit coverage, which is what a glyph is.
- `DrawImage`: RGBA pixels scaled by nearest neighbour.
- `DrawAlpha`: per-pixel alpha from a coverage grid, which is what a path
  rasterizer (go-gfx/gfx/vector) produces.

All of them work under nested translations and clips. Masks and images live in
the list's data arena (`Data`), which the GPU reads alongside the instances.

These are the primitives of go-widgets/painter's `PixelPainter`, and
`Rasterize` reproduces it **byte for byte**. `internal/parity` is its own
module, so this one does not depend on go-widgets. It renders 3000 random
scenes with both renderers and compares every byte: 51,947 instances in all,
including 2970 masks, 2999 images and 11,251 filled and stroked paths
rasterized by the same go-gfx/gfx/vector rasterizer PixelPainter uses. The
scenes mix translucent colours, rectangles past every edge, degenerate sizes,
radii past half a side, short masks and nested clips. A control shows the
comparison can fail.

## The contract for a GPU back-end

A back-end draws the instances in order, one quad each, with the fragment
shader computing `Coverage` at the pixel centre and fixed-function blending
(`src-alpha`, `one-minus-src-alpha`). It must match `Rasterize` within **1 per
channel**: the coverage is the same expression in float32, the blend is unorm
arithmetic instead of `(src·a + dst·(255−a) + 127) / 255`.

## WebGPU (package `webgpu`, GOOS=js)

```go
webgpu.InstallReplay(ctx, "")             // go-gfx/webgpu's one-call submit; your URL under a strict CSP
r, err := gpu2dwebgpu.New(ctx, device, w, h)
err = r.Draw(list, nil)                   // one submit; Texture() is the rgba8unorm target
```

One quad per instance from the instance index, no vertex buffer; `Shader` is
the WGSL, validated on every platform by go-gfx/wgsl before a browser sees it.
`TestMatchesReference` runs in a real Chromium worker: on a GPU on macOS in CI,
on SwiftShader on Linux.

To put the frame on a `<canvas>` without reading it back, configure the
canvas's surface with `CanvasDescriptor()` and present after each draw:

```go
desc := gpu2dwebgpu.CanvasDescriptor() // the browser's preferred canvas format
s, err := device.NewSurfaceFromCanvas(offscreen, desc)
err = r.Draw(list, nil)
err = r.Present(s)                      // into the canvas texture, on the GPU
```

`CanvasDescriptor` follows `getPreferredCanvasFormat`, and `Present` handles
both of the formats it can return (both are supported canvas formats in the
WebGPU specification):

- **`bgra8unorm`**, preferred on macOS and Windows. `Present` draws the target
  into it with `PresentShader`: one triangle, a `textureLoad` at each pixel.
  `unorm8 → float → unorm8` gives back the same byte, and
  `TestABGRACanvasHoldsTheFrame` finds the frame's bytes in BGRA order.
- **`rgba8unorm`** elsewhere, with `COPY_DST`. `Present` copies with
  `copyTextureToTexture`; it is the target's own format, so the copy is
  byte-exact (`TestAPresentedFrameIsTheFrame`).
- **Anything else** is `ErrSurface`.

**Why the preferred format.** A canvas in a format the browser does not
prefer costs an extra copy in its compositor; Chrome warns about it.
`internal/presentbench` loads a real page and times `WidgetFrame` at 4K,
`Draw` plus `Present` per animation frame, until the GPU has finished. Each
platform ran 8 interleaved rounds at a 1-minute load of 8 or less, and the
harness logs the adapter with every result:

| 4K frame | rgba8unorm canvas | bgra8unorm canvas |
|---|---:|---:|
| Apple M4 Max, macOS, Chrome (prefers bgra8unorm) | 1.053 ms ± 6 % | **0.792 ms ± 18 %** (−24.8 %, p = 0.000) |
| AMD Radeon PRO W7700 (RDNA 3), Linux, Chrome for Testing 157 on Vulkan (prefers rgba8unorm) | **0.704 ms ± 30 %** | 1.083 ms ± 15 % (+53.7 %, p = 0.000) |

On both, the preferred format wins, even though the formats are opposite.
That is why `CanvasDescriptor` asks the browser rather than fixing a format.

To run it on a host without Go, build the page beforehand
(`PRESENTBENCH_BUILD_SITE=dir`) and serve it there (`PRESENTBENCH_SITE=dir`).
`CHROME` names the browser, and `PRESENTBENCH_FLAGS` adds switches; Linux
needs `--enable-features=Vulkan --use-angle=vulkan --use-vulkan
--ignore-gpu-blocklist --use-webgpu-power-preference=default-high-performance`.
The page asks for the high-performance adapter: without it, a Ryzen 7000
machine gets its integrated RDNA 2 GPU (36 ms per frame).

## Metal (package `metal`, macOS)

```go
r, err := gpu2dmetal.New(w, h) // go-macos/metal's ErrUnsupported off macOS
copy(r.Pixels(), ground)       // on unified memory, the bytes the GPU draws into
err = r.Draw(list)             // bins, dispatches, waits
frame := r.Pixels()
```

`Draw` is `Submit` plus `Wait`. For a small frame, where the time is the
fixed cost of the round trip, overlap the CPU with the GPU:

```go
err = r.Submit(list) // bins and packs a copy of the list, sends it, returns
list.Reset()         // free at once: record the next frame while the GPU draws
record(list)
err = r.Wait()       // the GPU's verdict on the oldest frame in flight
frame := r.Pixels()  // safe to read only when nothing is in flight
```

Two frames may be in flight (each owns its input buffers); they composite in
submission order over the same `Pixels`. A third `Submit` or a `Draw` while
frames are in flight is `ErrBusy`, a `Wait` with none is `ErrNothingInFlight`,
and `Close` waits for what is in flight before releasing memory.
`InFlight()` counts them. `Pixels` belongs to the GPU until the last `Wait`.
`List`, `Bin` and `AppendBytes` allocate nothing in steady state
(`TestSteadyFrameAllocatesNothing`).

`Submit`'s cost is almost all CPU-side packing: binning and copying the list
into the slot. Encoding and Metal's commit add about 14 µs.
- **v0.6.1:** binning runs in plain loops rather than a callback per
  (instance, tile) pair.
- **v0.6.2:** binning writes straight into the slot's Metal buffers, so the
  indices are never copied (once a buffer has grown to the frame).

Measured on an M4 Max, 8 interleaved rounds at a 1-minute load of at most 8
(`BenchmarkBin`, `BenchmarkSubmit`). Each column is a separate A/B run
against the previous tag:

| | v0.6.0 | v0.6.1 | v0.6.2 |
|---|---:|---:|---:|
| `Bin`, 960 × 2834 widget frame | 148.0 µs | 48.3 µs | 48.3 µs |
| `Submit`, `WidgetFrame` at 1920 × 1080 | 164.9 µs | 80.4 µs | 57.3 µs |
| `Submit`, `WidgetFrame` at 1920 × 5668 | 782.9 µs | 364.7 µs | 242.2 µs |

To put the frame on screen without the CPU touching it, present to a
CAMetalLayer that a view hosts:

```go
layer, err := r.NewLayer()    // the renderer's size, on the renderer's GPU
view.Send(objc.Sel("setLayer:"), objc.ID(layer.ID()))
err = r.Present(list, layer)  // Submit, then a second kernel copies the frame into the drawable
err = r.Wait()
err = r.Resize(w, h)          // on a window resize; then layer.Resize(w, h)
```

`Present` takes a frame slot like `Submit`. It waits for the layer's next
drawable (CoreAnimation allows a second), and the drawable is shown once the
GPU is done. The drawable is BGRA; the copy writes RGBA floats and Metal
stores them in the layer's order. `TestAPresentedFrameIsTheFrame` reads the
drawable back on the GPU and finds every byte of `Pixels`, which
`TestMatchesReference` holds to the CPU reference.

A compute kernel through go-macos/metal (pure Go, no cgo): `List.Bin` sorts
the instances into 16 × 16 tiles on the CPU, then one GPU thread per pixel
walks its tile's instances in order, computes `Coverage` (with
`precise::sqrt`, so Metal's fast math cannot change it) and blends with
PixelPainter's integer arithmetic. Binning is bounded (`MaxBinned`) and
refused before anything is allocated.

## Bounds

Every coordinate a list records, translation included, lies within ±2^24
(`MaxCoord`), the range in which float32 holds every integer exactly.

- A primitive reaching past `MaxCoord` is not recorded, rather than wrapped
  into the 32-bit instance.
- Translations and clips saturate, and ring counts are capped.
- A surface side above `MaxCoord` is `ErrSize`.
- The data arena is bounded by `MaxData` (256 MiB). An image is judged by its
  declared size before its bytes are read.
- An image whose scaling products reach `MaxScaled` is refused, because a
  GPU computes them in 32-bit integers.
- Refusals are reported by `List.Err`. `FuzzList` drives arbitrary operations and
checks that nothing is recorded past the bounds and that `Rasterize` writes
nothing past the surface (corpus in `testdata/fuzz`).

## Licence

BSD 3-Clause, see [LICENSE](LICENSE).
