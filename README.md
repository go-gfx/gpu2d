# gpu2d

A pure-Go (CGO=0) 2D renderer for the GPU. A frame is one backend-neutral list
of **instances**, one per primitive, which a GPU draws as one quad each and which
`Rasterize` executes on the CPU. `Rasterize` is the reference: it defines every
pixel, and each GPU back-end is checked against it.

**Status: milestone 4 of 6 ([docs/DESIGN.md](docs/DESIGN.md)).** The instance
list, the CPU reference, the WebGPU back-end (package `webgpu`, browser) and
the Metal back-end (package `metal`, macOS), the Vulkan back-end (package
`vulkan`, Linux, proven on lavapipe) and the Direct3D 11 back-end (package
`d3d11`, Windows, opt-in, proven on WARP only) exist, for rectangles, glyph
masks, images and paths. The go-widgets adapter comes next. Presentation on
Linux and Windows is designed, not built: [docs/NATIVE.md](docs/NATIVE.md).

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
channel**: the coverage is exact (integer arithmetic, the compute kernel's),
the blend is unorm arithmetic instead of `(src·a + dst·(255−a) + 127) / 255`.

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

### Compute mode (`NewCompute`)

```go
r, err := gpu2dwebgpu.NewCompute(ctx, device, w, h) // same Draw, Present, Read
```

`kernel.wgsl` (`gpu2d.KernelWGSL`) is the whole raster kernel as one WGSL
compute shader: the tile walk over `List.Bin`'s output, the coverage functions
of `shader.wgsl` (a test keeps them character-identical) and PixelPainter's
integer blend, floored alpha included. It is the arithmetic of the Metal
kernel, so a frame is **byte-identical to `Rasterize` over any ground**, where
the fixed-function renderer differs by up to 4 in alpha over a translucent one.
It binds four storage buffers (instances, bins = spans then list, arena,
pixels) and one uniform, and runs in 16 × 8 workgroups, the limits
[docs/NATIVE.md](docs/NATIVE.md) sets so that other back-ends can emit it.

The pixels live in a storage buffer: `Texture()` is nil, and `Present`
(either canvas format, from the buffer) and `Read` keep their contracts.
`TestComputeMatchesReference` reports 0 differing channels on an opaque and a
translucent ground, and `TestComputeBreakCheck` perturbs the blend and shows
the same comparison failing. The default stays the fixed-function renderer;
`internal/presentbench`'s `BenchmarkRaster` measures the two on the same
frames.

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

The kernel is not written in Metal: `metal/kernel_gen.metal` is generated from
`kernel.wgsl` (`go generate ./metal`) by go-gfx/wgsl's `msl` package, the same
WGSL the WebGPU compute mode runs, so the raster arithmetic exists once.
`TestKernelMetalIsGenerated` fails on a stale file, and `TestMatchesReference`
still finds 0 differing channels on both grounds. Only the present kernel
(`metal/present.metal`) is hand-written, since it writes a texture.

## Vulkan (package `vulkan`, Linux)

```go
r, err := vulkan.New(w, h) // errors.Is(err, vulkan.ErrUnsupported) without a Vulkan loader
copy(r.Pixels(), ground)
err = r.Draw(list)         // Submit, then Wait; r.Pixels() holds the frame
```

The same shape as `metal`: `New(w, h)` (and `NewWith` for validation or a
vendor filter), `Draw`, `Submit` and `Wait` over two slots, `Pixels`, `Resize`,
`Close`, `Name`, and `ErrBusy`, `ErrNothingInFlight` and `ErrSize`. Frames
composite in submission order over one target. There is no presentation yet
(milestone L1 of [docs/NATIVE.md](docs/NATIVE.md)).

It runs the same raster kernel through go-gfx/vulkan (pure Go, no cgo, 64-bit
Linux): `vulkan/kernel_gen.spv` is generated from `kernel.wgsl`
(`go generate ./vulkan`) by go-gfx/wgsl's `spirv` package, and
`TestKernelSPIRVIsGenerated` fails on a stale file on every platform. The
frame lives in device-local memory and `Pixels` is a host-visible copy of it:
what you write there is uploaded before the next draw, and what the device has
drawn is downloaded when you next call `Pixels` (or `Sync`), so a loop of
`Draw` calls that never reads the frame moves no pixel across the bus. The
instances, bins and arena stay host-visible; `List.Bin` bins in place into
them only on a software device (lavapipe), because on a discrete GPU that
memory is write-combined and binning reads it. On a Radeon PRO W7700 (RADV) the
widget frame draws in 0.52 ms at 1080p and 1.98 ms at 4K, 52 and 55 times the
CPU `Rasterize` ([docs/NATIVE.md](docs/NATIVE.md), G2, has the table and the
cost of reading the frame back). Elsewhere, and where no loader is installed,
`New` returns an error satisfying `errors.Is(err, vulkan.ErrUnsupported)`.

A frame past a device limit (`maxStorageBufferRange`, guaranteed only at
2^27 bytes, or `maxComputeWorkGroupCount`) is refused with `gpu2d.ErrTooLarge`
before anything is dispatched.

CI runs the package on lavapipe (Mesa's software Vulkan) with the Khronos
validation layer, failing on any message. `TestMatchesReference` finds 0
differing channels on both grounds there; lavapipe proves the bindings and the
kernel, not the arithmetic or the speed of a GPU. The tests honour
`VULKAN_REQUIRE=1`, `VULKAN_VALIDATION=1` and `VULKAN_VENDOR=0x1002`;
`VULKAN_WIDE_SCENES=N` and `VULKAN_WIDE_CORNERS=N` run N random scenes up to 4K
and N corner-heavy scenes with radii 1 to 64 (60 of them are in the default
suite). The corner coverage is decided in 32-bit integers from the exact
squared distance, so the frame is byte-identical to `Rasterize` whatever the
device's `sqrt`; radii past `gpu2d.MaxRadius` (16383) are refused. See G3 in
[docs/NATIVE.md](docs/NATIVE.md).

Two frames are really in flight: `Submit` packs and queues the second frame
while the first runs, and never waits on a fence. Each frame's command buffer
begins with go-gfx/vulkan's `Barrier`, a memory dependency from the earlier
writes in submission order, so the second frame composites over the first's
pixels in order. The frames still run one after the other on the device, since
each needs the other's pixels. On a device that reports
`shaderRoundingModeRTEFloat32` the kernel declares `RoundingModeRTE`
(`kernel_rte_gen.spv`); `robustBufferAccess` is enabled where offered.

## Direct3D 11 (package `d3d11`, Windows, opt-in)

```go
r, err := d3d11.New(w, h) // errors.Is(err, d3d11.ErrUnsupported) off Windows
copy(r.Pixels(), ground)
err = r.Draw(list)        // Submit, then Wait; r.Pixels() holds the frame
```

The same shape as `metal` and `vulkan`: `New(w, h)` (and `NewWith`, whose
`Options.Software` asks for WARP), `Draw`, `Submit` and `Wait` over two slots,
`Pixels`, `Resize`, `Close`, `Name`, and `ErrBusy`, `ErrNothingInFlight`,
`ErrSize`, `ErrUnsupported` and `ErrNoDevice`. It runs the same raster kernel
through go-mswin/d3d11 (pure Go, COM through syscall, no cgo):
`d3d11/kernel_gen.hlsl` is generated from `kernel.wgsl` (`go generate ./d3d11`)
and `d3d11/kernel_gen.dxbc` is that HLSL compiled by `D3DCompile` on the Windows
lane and committed, with a record of the compiler and hashes; nothing is
compiled at run time. Off Windows the constructors return `ErrUnsupported`.

**It ships opt-in, and only WARP has run it.** CI runs the package on WARP, the
software device of a `windows-latest` runner; no Windows GPU has. Parity with
`Rasterize` on WARP is a measurement of WARP's arithmetic, not a proof for any
GPU: a device that disagrees is to be named by `TestMatchesReference`, and the
window falls back to the CPU.

Direct3D 11 has no persistently mapped, GPU-writable buffer, so `Pixels` is a
host copy and `Draw` pays for two copies that Metal and Vulkan do not:

- **Upload.** Each slot keeps its instance, bins and arena buffers across frames
  (DYNAMIC structured buffers written with `Map` `WRITE_DISCARD`, grown on
  demand, go-mswin/d3d11 v0.2.0), and, when nothing is in flight, the target is
  filled in place from `Pixels` with `UpdateSubresource` (the caller may have
  written the ground there).
- **Read-back.** The `Wait` that ends the last frame copies the target to a
  staging buffer, maps it and copies it into `Pixels`.

Frames queued with `Submit` stay on the device between them: the second
composites over the first's result with no read-back, in submission order on the
one immediate context. Each frame is flushed and fenced, so `Wait` of the first
of two returns when that frame is done. `Renderer.Stats` splits the wall time into packing,
target upload, device time and read-back; docs/NATIVE.md has the measured split.

A frame past the device's buffer limit (guaranteed `min(max(128, 0.25 · VRAM),
2048)` MB, 128 MiB on WARP, so a square frame above 5792 pixels) or past 65535
thread groups per dimension is refused with `gpu2d.ErrTooLarge`, as is a frame
whose input buffers would pass it. The tests honour `D3D11_REQUIRE=1` and
`D3D11_HARDWARE=1`.

**Presenting** (v0.11.0): `r.NewSwapChain(hwnd)` makes a flip-model swap chain
(`FLIP_DISCARD`, `R8G8B8A8_UNORM`, two buffers, back buffers usable as UAVs;
go-mswin/d3d11 v0.3.0) on a window, and `r.Present(list, sc)` draws the list over
the device's frame, copies the frame into the back buffer with a second,
hand-written kernel (`d3d11/present.hlsl`, `present.dxbc` made on the Windows
lane and committed, as the raster kernel's is) and presents it. Nothing is
uploaded or read back: `Pixels` is neither read nor updated by `Present`, and the
list draws over the frame the last `Draw` or `Present` left on the device.
`PresentAndRead` also copies the back buffer, just before the swap chain's
`Present`, into a slice: the proof that the window gets the frame, and a
screenshot that needs no desktop. The swap chain is sized as the frame
(`SwapChain.Resize` after `Renderer.Resize`); close it before the renderer. A
window with GDI-blitted pixels shows only the swap chain while it exists (flip
model). CI proves it on WARP with a hidden window: the back buffer equals
`Rasterize` on an opaque and a translucent ground, and a present kernel with
red and blue swapped is caught.

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
