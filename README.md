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

## Metal (package `metal`, macOS)

```go
r, err := gpu2dmetal.New(w, h) // go-macos/metal's ErrUnsupported off macOS
copy(r.Pixels(), ground)       // on unified memory, the bytes the GPU draws into
err = r.Draw(list)             // bins, dispatches, waits
frame := r.Pixels()
```

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
