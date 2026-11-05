# gpu2d

A pure-Go (CGO=0) 2D renderer for the GPU. A frame is one backend-neutral list
of **instances**, one per primitive, which a GPU draws as one quad each and which
`Rasterize` executes on the CPU. `Rasterize` is the reference: it defines every
pixel, and each GPU back-end is checked against it.

**Status: milestone 1 of 6 ([docs/DESIGN.md](docs/DESIGN.md)).** The instance
list and the CPU reference exist; the WebGPU and Metal back-ends do not yet.

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

Fills and one-pixel outlines of rectangles, square or with anti-aliased rounded
corners, any number of concentric rings, under nested translations and clips.
These are the primitives of go-widgets/painter's `PixelPainter`, and
`Rasterize` reproduces it **byte for byte**: `internal/parity` (its own module,
so this one does not depend on go-widgets) renders 3000 random scenes, 42,256
instances in all, with both and compares every byte. The scenes mix
translucent colours, rectangles past every edge, degenerate sizes, radii past
half a side and nested clips; a control shows the comparison can fail.

## The contract for a GPU back-end

A back-end draws the instances in order, one quad each, with the fragment
shader computing `Coverage` at the pixel centre and fixed-function blending
(`src-alpha`, `one-minus-src-alpha`). It must match `Rasterize` within **1 per
channel**: the coverage is the same expression in float32, the blend is unorm
arithmetic instead of `(src·a + dst·(255−a) + 127) / 255`.

## Bounds

Every coordinate a list records, translation included, lies within ±2^24
(`MaxCoord`), the range in which float32 holds every integer exactly. A
primitive reaching past it is not recorded instead of being wrapped into the
32-bit instance; translations and clips saturate; a surface side above it is
`ErrSize`; ring counts are capped. `FuzzList` drives arbitrary operations and
checks that nothing is recorded past the bounds and that `Rasterize` writes
nothing past the surface (corpus in `testdata/fuzz`).

## Licence

BSD 3-Clause, see [LICENSE](LICENSE).
