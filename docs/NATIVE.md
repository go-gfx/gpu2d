# GPU-presented windows on Linux and Windows: design

Status: **proposal, nothing implemented.** It extends [DESIGN.md](DESIGN.md)
(milestones 1–6, WebGPU and Metal) to Linux and Windows.

Already decided by the user (2026-11-09): **Vulkan** on Linux (Wayland and
X11) and **Direct3D 11** on Windows. Both in pure Go without cgo, reaching the
system libraries the way go-macos/metal reaches Metal (`purego.Dlopen` and
`RegisterLibFunc`, `metal_darwin.go:25-30`), or through `syscall` on Windows.
No third-party native library: no wgpu-native, no Dawn, no glslang.

Each decision below gives the specification text that forces it. Only
documents that were actually opened are cited; they are listed at the end.
Where a fact is still to be measured, the text says so.

## Where we start

- **gpu2d v0.6.2.** One instance list, binned on the CPU into 16 × 16 tiles
  (`List.Bin`, `tiles.go:31`).
  - **The Metal back-end** runs one thread per pixel (`metal/kernel.metal`).
    It is byte-identical to `Rasterize` because blending is done in integers in
    a compute kernel.
  - **The WebGPU back-end** uses fixed-function blending, and alpha can differ
    by up to 4 over a translucent ground.
  - **Generated code:** the coverage functions are generated from
    `webgpu/shader.wgsl` by go-gfx/wgsl's `msl` package
    (`internal/mslgen/mslgen.go:31`). The rest of the Metal kernel (the tile
    walk, `blend`, `present`) is hand-written: 86 lines.
  - **Frames in flight:** Submit and Wait work over two slots.
- **go-gfx/wgsl v0.5.0.**
  - **The front end is complete.** The parser rejects none of the 130,057
    valid CTS cases (`docs/CONFORMANCE.md`), and the checker hands a back end
    what it needs (`Info`, `check.go:158`):
    - resolved types;
    - struct layouts (`Structs`);
    - entry points with their workgroup size, IO and statically used resources
      (`EntryPoint`, `check.go:134`).
  - **The only emitter is `msl`, and it is narrow** (README, docs/MSL.md). It
    handles plain functions over scalars, vectors and structs, with `let`,
    `var`, assignment, `if` and `return`. It refuses entry points, module-scope
    variables, loops, arrays and conversions from `f32` to an integer.
- **go-widgets/window v0.90.1.**
  - **Only Cocoa has a GPU path:** `Config.GPU`, `internal/cocoa/gpu_darwin.go`.
    It falls back to the CPU on any failure and records why (`GPUError`).
  - **Linux and Windows present CPU buffers:** `wl_shm`, MIT-SHM, `StretchDIBits`.
  - **Both Linux clients already send file descriptors over their own
    sockets:**
    - Wayland: `syscall.UnixRights` in `internal/wayland/transport_linux.go:46`;
    - X11: `sendRequestFD` for MIT-SHM `AttachFd` in `internal/x11/shm.go:97`.
- **go-mswin/screencapture** calls DXGI and D3D11 through hand-numbered COM
  vtable slots with `syscall.SyscallN` (`dxgi_windows.go:8-105`). Every slot
  is written out with its inheritance chain, because a wrong slot does not
  crash: it calls a different method. This is the pattern to reuse.
- **No Vulkan, D3D, EGL or DRM binding exists** in the 2230 repositories
  (fleetfind, 2026-11-09).

## Decision 1: the shader source of truth is one WGSL compute kernel

### What the kernel needs

Byte-identity with `Rasterize` needs two things:

- **PixelPainter's integer blend**, which includes the floored alpha
  `a = ca + a·(255−ca)/255`. No fixed-function blend state expresses it, so the
  blend must be done in a compute kernel, as on Metal.
- **The same f32 coverage arithmetic on every device.**

Every native back-end therefore runs the Metal kernel's algorithm. The
question is where that kernel is written.

### Chosen

Write the **whole raster kernel in WGSL** as a `@compute` entry point
(`kernel.wgsl` at the module root, beside `webgpu/shader.wgsl`). Generate
every native language from it with go-gfx/wgsl:

| Target | Emitter | When it runs | What is committed |
|---|---|---|---|
| Metal | `msl`, extended | `go generate` | `metal/kernel_gen.metal`, replacing `kernel.metal` and `coverage_gen.metal` |
| Vulkan | new `spirv` package | `go generate` | `vulkan/kernel.spv` (bytes) |
| D3D11 | new `hlsl` package, then `D3DCompile` (cs_5_0) | `go generate` on Windows | `d3d11/kernel.hlsl` and `d3d11/kernel.dxbc` |
| WebGPU | none (the WGSL itself) | — | — |

Each generated file gets a drift test, the way `TestCoverageMetalIsGenerated`
works today. The SPIR-V and HLSL tests run on every OS (pure Go). The DXBC test
runs on the Windows CI lane, which recompiles with the system compiler and
compares.

The kernel is written so that every emitter can express it:

- **Pixels are `array<u32>`,** one little-endian word per pixel. These are the
  same bytes as Metal's `uchar4` (`pack`, `gpu2d.go:84`).
- **The arena is read through `byteAt`,** which `webgpu/shader.wgsl:18`
  already has.
- **`blend` returns a `vec4<u32>`** instead of taking references, so no emitter
  needs pointers.
- **`spans` and `list` share one buffer.** The kernel then binds four storage
  buffers (instances, bins, arena, pixels) and one uniform. Four is the core
  minimum of `maxPerStageDescriptorStorageBuffers`.
- **The workgroup is 16 × 8 = 128 invocations,** the core minimum of
  `maxComputeWorkGroupInvocations`. 16 × 16 = 256 is guaranteed only from
  Roadmap 2022 / Vulkan 1.4 (Vulkan spec, Limits, Required Limits table). A
  tile stays 16 × 16; two workgroups share its span.

### Why one WGSL kernel

1. **It is the only source that every target can consume.** WGSL is already
   the source of the coverage functions, and the checker is CTS-backed. The
   browser can run the same kernel, so a **WebGPU compute mode** is possible.
   It would be byte-identical over a translucent ground too, closing the
   "alpha ≤ 4" gap of DESIGN.md. This is milestone K0 below. It is also how
   the WGSL kernel is proved before any new emitter exists.
2. **The arithmetic must be pinned per target, and the emitters are where
   that is done.**
   - **Vulkan** (spec, appendix "SPIR-V Environment",
     [`#spirvenv-precision-operation`](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation)):
     - `OpFAdd` and `OpFMul` are "Correctly rounded" only under the
       `RoundingModeRTE` execution mode. Otherwise the rounding is
       implementation-defined.
     - Operations may be contracted unless they carry `NoContraction`
       (`#spirvenv-evaluation-expressions`).
     - `Sqrt` is "Inherited from 1.0 / inversesqrt()", and `InverseSqrt` is
       "2 ULP".
     - `OpFDiv` is 2.5 ULP.
   - **D3D11** (Functional Spec §3.1.3):
     - add, subtract and multiply are within 0.5 ULP;
     - "sqrt and rcp have 1 ULP tolerance";
     - denormals "MUST be flushed";
     - fusing into `mad` is permitted.
   - **HLSL:** `precise` keeps "separate operations … separate" and preserves
     their order (HLSL variable syntax), and `D3DCOMPILE_IEEE_STRICTNESS`
     "avoids optimizations that may break IEEE rules".

   So the SPIR-V emitter puts `NoContraction` on every float operation and
   declares `RoundingModeRTE` when the device reports
   `shaderRoundingModeRTEFloat32`
   (`VUID-RuntimeSpirv-shaderRoundingModeRTEFloat32-06303`). The HLSL emitter
   declares every float temporary `precise` and compiles with
   `IEEE_STRICTNESS`. The MSL emitter already does the equivalent
   (`math_mode(safe)`, `contract(off)`, `precise::`).
3. **The kernel has no float division,** so the 2.5 ULP of `OpFDiv` does not
   matter here. All of its divisions are integer divisions, which the
   emitters guard the way docs/MSL.md rule 5 does.

### sqrt is the one operation no specification pins

Under the rules above, Vulkan allows `sqrt` an error of about 2 to 3 ULP and
D3D11 allows 1 ULP. Byte-identity with `Rasterize` (which uses f64 `Hypot`) is
already **empirical** on Metal: the f32 result lands on the same byte because
no tested pixel sits on a rounding boundary. Nothing promises that it holds on
another GPU.

Proposal: add a **correctly rounded `sqrt`** to `kernel.wgsl`. It takes the
builtin's result and corrects it by one ULP, using an exact residual computed
with Dekker's two-product. That is valid only where multiplication and
addition are correctly rounded and not contracted, which point 2 secures on
Vulkan with RTE and in HLSL with `precise`. Every conforming device then
computes **the same f32 bits**, so a byte-identity measured on one GPU
carries over to all of them. Cost: about ten flops per corner pixel only.

**K1 measured it on Metal** (`metal/sqrt_test.go`, M4 Max): the builtin
`length()` is already correctly rounded on every distance a corner pixel can
have (all 4,194,304 offset pairs up to 2047.5) and on 16,777,216 random f32
inputs, so a correction changes no byte there. The corrector written as
proposed (f32 residual from Dekker's two-product) repairs a root one step off
on all but about one input in 16 million, the ones within about 2^-25 ulp of a
rounding midpoint, which an f32 residual cannot decide; an exact midpoint test
needs more terms. It also moves a root by one step only, while Vulkan allows
`sqrt` more than that. It is therefore not in `kernel.wgsl` yet: where it
matters (lavapipe, RADV, WARP) it is to be finished and measured with W2.

**Deferred to W2.** Nothing is labelled correctly rounded until the corrector
is exact near midpoints and has been measured on lavapipe, RADV and WARP.

The plan as written before K1: it enters in K1 behind a measurement: the
parity suite runs on lavapipe,
RADV, WARP and Metal with and without it. It is kept only if it changes no
byte on Metal. Where RTE is not offered, the back-end does not claim
identity. It reports the tolerance (|Δ| ≤ 1 per channel, the DESIGN.md
contract).

### What the emitters must add

Today `msl` emits the coverage functions only. The kernel also needs:

| Construct | msl today | spirv | hlsl |
|---|---|---|---|
| `@compute @workgroup_size(16, 8)`, `@builtin(global_invocation_id)` | refused | new | new |
| `var<storage, read>` / `read_write` / `var<uniform>`, `@group` / `@binding` | refused | new (Block, Offset, ArrayStride, DescriptorSet, Binding from `Info.Structs`) | new (`StructuredBuffer`, `ByteAddressBuffer`, `RWByteAddressBuffer`, `cbuffer`) |
| runtime-sized arrays, indexing, member access through a reference | refused | new | new |
| `for` / `loop`, `break`, `continue` | refused | new (`OpLoopMerge`, SPIR-V §2.11) | new |
| `u32(f32)` | refused | new (`OpConvertFToU` is undefined out of range; WGSL clamps, so the emitter clamps or proves the range — here [0, 255.5]) | new |
| everything in docs/MSL.md | done | port the same rules | port the same rules |

Rough size, from the 802 lines of `msl/msl.go`:

- **msl:** +500 to 800 lines.
- **hlsl:** about 1,200 lines. It is textual and close to `msl`. The shared
  C-like core (division guards, signed wrap) is factored out of `msl` first.
- **spirv:** about 3,000 lines. It is the large one: a binary encoder, the
  module layout of SPIR-V §2.4 (capabilities, …, types, constants and globals
  before functions), interning of types and constants, structured control
  flow, decorations from the layouts, and locals as Function-storage
  `OpVariable`s so that no φ construction is needed.

Every emitter keeps the msl discipline: it refuses an out-of-scope construct
with its position and never emits an approximation.

### Alternatives rejected

- **Hand-written GLSL compiled with glslang, or HLSL compiled with DXC.**
  Both are third-party native tools, which the user excludes, and each adds
  another hand-kept copy of the kernel.
- **Hand-written SPIR-V and HLSL beside a hand-written MSL.** That makes three
  copies of the kernel. DESIGN.md milestone 6 already removed one copy for
  exactly this reason.
- **Building DXBC from HLSL at run time** with the system `d3dcompiler_47.dll`
  (in-box since Windows 8.1, per Walbourn, "HLSL, FXC, and D3DCompile"). It is
  workable, but the same post recommends compiling at build time. A run-time
  compile also makes the numerics depend on whichever compiler the user's
  Windows ships. Committing the DXBC produced on CI pins it. The cost is that
  regenerating it needs the Windows lane. A contributor on macOS or Linux
  takes the bytes from the CI artifact.
- **Emitting DXBC directly from Go.** It needs a reimplementation of FXC's
  register allocation, plus the container checksum. That is out of proportion
  to one kernel.
- **Emitting SPIR-V at run time.** The module would link go-gfx/wgsl into
  every program for one constant blob. Generating it once and committing it
  matches `coverage_gen.metal`.

## Decision 2: two new binding modules, minimal surface

### `github.com/go-gfx/vulkan` (new repository)

**Why go-gfx:** Vulkan is cross-vendor and cross-OS, a peer of go-gfx/webgpu.
It is not part of an OS the way Metal belongs to go-macos.

**Loader.**
- It opens `libvulkan.so.1`, the versioned name. The Khronos loader doc
  ("Linux Library Name") tells programs that `dlopen()` the loader to use the
  versioned name.
- It resolves only `vkGetInstanceProcAddr` by `dlsym`.
- Device functions come from `vkGetDeviceProcAddr`, which "point directly to
  the driver" when no layer is enabled (same doc, "Obtaining Function
  Pointers").
- On Windows the same package would open `vulkan-1.dll`. It is not planned
  here.

**Surface.** It is exactly what gpu2d needs, nothing speculative:
- `vkCreateInstance` with API version 1.2: float controls, used for RTE.
- Enumeration of physical devices, with properties2 and limits.
- **Device choice:**
  - On Wayland, compare `VkPhysicalDeviceDrmPropertiesEXT`
    (`VK_EXT_physical_device_drm`: `primaryMajor` / `primaryMinor` and
    `renderMajor` / `renderMinor`) with the compositor's `main_device`
    (decision 3).
  - On X11, compare it with the `st_rdev` of the fd that `DRI3Open` returns.
  - With neither, pick the first discrete GPU, then the first integrated one.
- One device with one compute queue.
- **Buffers:** device-local for the target, and host-visible and coherent for
  the per-slot inputs and for read-back.
- A descriptor set layout, a pool and sets; a pipeline layout; a compute
  pipeline from SPIR-V bytes.
- Command pool and command buffers, `vkCmdDispatch`, `vkCmdCopyBuffer`,
  `vkCmdCopyBufferToImage`, and pipeline barriers.
- Fences, through `vkWaitForFences` and `vkGetFenceStatus`. These map onto
  the existing `Submit` / `Wait` / `Done`.
- **Export (decision 3):**
  - `VK_KHR_external_memory_fd`, with `vkGetMemoryFdKHR`;
  - `VK_EXT_external_memory_dma_buf`, which "enables applications … to export
    VkDeviceMemory as a dma_buf" and requires `VK_KHR_external_memory_fd`;
  - `VK_EXT_image_drm_format_modifier`. The application "queries the
    implementation-chosen modifier with
    vkGetImageDrmFormatModifierPropertiesEXT", then each plane's layout with
    `vkGetImageSubresourceLayout`.
- **No WSI extensions at all:** no `VK_KHR_surface`, no
  `VK_KHR_swapchain` (decision 3 says why).

**Struct layout.**
- Vulkan passes every create-info **by pointer**. The planned surface passes
  no struct by value, so purego's "no structs by value" footnotes do not
  apply. Not one call in the surface takes a struct argument by value.
- Go mirrors of the C structs are generated by an `internal/vkgen` from the
  Khronos registry (`vk.xml`), with offsets computed by the C rules.
- A test pins `unsafe.Sizeof` and `unsafe.Offsetof` for every mirrored struct
  against the generator's table. A hand-typed struct has no witness; the
  generator's arithmetic does.

**Callbacks.** None. `VK_EXT_debug_utils` would need `purego.NewCallback`. The
validation layer is enabled in tests only, through `VK_INSTANCE_LAYERS`, and
reports on stderr.

**Platforms: 64-bit only** (linux/amd64 and linux/arm64, purego Tier 1). On
32-bit targets, non-dispatchable handles are `uint64_t` passed in two words,
and purego's 32-bit Linux targets are Tier 2. go-mswin/win32 already targets
amd64/arm64 only. Other targets build a stub whose `New` returns
`ErrUnsupported`.

**Errors.** A negative `VkResult` becomes a typed error (`ErrDeviceLost`,
`ErrOutOfDeviceMemory`, `ErrInitializationFailed`, …) that carries the code.
`VK_ERROR_DEVICE_LOST` makes the window drop to the CPU path, as Cocoa does
(`gpu_darwin.go:107`).

**Threading.** A queue is externally synchronized, so one `Renderer` belongs
to one goroutine, as on Metal.

### `github.com/go-mswin/d3d11` (new repository)

**Why go-mswin:** D3D11 and DXGI are OS components, so they sit in the OS
organisation, as go-macos/metal does.

**Calls.** It builds on go-mswin/win32 (x/sys/windows). Calls go through
`syscall.SyscallN` on vtable slots, with screencapture's discipline: every
slot constant carries its full inheritance chain in a comment. A small
`internal/com` package (`Release`, `QueryInterface`, slot call, HRESULT
mapping) is lifted from screencapture and winrt, not reinvented. Package
`dxgi` holds the factory, the adapter and the swap chain. screencapture may
migrate to it later; it does not have to.

**Surface:**
- `D3D11CreateDevice`, with the hardware driver type or
  `D3D_DRIVER_TYPE_WARP`. The WARP guide: "the runtime creates a WARP device
  and does not initialize a hardware device".
- Buffers: structured, raw and constant buffers, and staging buffers for
  read-back.
- Views: SRV and UAV.
- Compute: `CreateComputeShader` from DXBC, then `CSSet*` and `Dispatch`.
- `CopyResource`, `Map` and `Unmap`.
- Completion through `D3D11_QUERY_EVENT`, polled with `GetData`. It is
  available on every Windows 10 and needs no `ID3D11Fence`.
- **dxgi:** `CreateDXGIFactory2`, then `IDXGIFactory2::CreateSwapChainForHwnd`,
  `GetBuffer`, `Present1`, `ResizeBuffers`.

**Errors.** A failing HRESULT becomes a typed error. On
`DXGI_ERROR_DEVICE_REMOVED` / `RESET` the error carries
`GetDeviceRemovedReason`, and the window drops to the CPU path.

## Decision 3: presentation goes over our own protocol connections, by dma-buf

### Why the Vulkan WSI is ruled out on Wayland and X11

`VK_KHR_wayland_surface` takes the client's connection, and so does
`VK_KHR_xcb_surface`:

- **What `VkWaylandSurfaceCreateInfoKHR` requires:** "`display` must point to a
  valid Wayland `wl_display`" (VUID-…-display-01304), and "`surface` must point
  to a valid Wayland `wl_surface`" (-01305).
- **What a `wl_display` is:** libwayland-client's object, which "represents a
  client connection to a Wayland compositor". `wl_display_connect_to_fd`
  "takes ownership of the fd" (Wayland client API, appendix B).

Our Wayland client is pure Go and owns its socket and its object-id space.
There is no `wl_display*` to hand over. A second connection made by libwayland
cannot use our `wl_surface`, because protocol objects belong to the
connection that created them. The only way to use the WSI would be to give
**the whole Wayland connection** to libwayland-client through purego. That
means the event loop, the input and the clipboard, and it means abandoning the
pure-Go client.

The same holds for X11: `VK_KHR_xcb_surface` takes libxcb's
`xcb_connection_t*`, which owns the socket.

### Chosen: export from Vulkan, import over our own sockets

**Wayland: `zwp_linux_dmabuf_v1`, version 4** (linux-dmabuf-v1 protocol).
1. **Choose the device.** `get_default_feedback` and `get_surface_feedback`
   (since 4) give `main_device`, "the main device that the server prefers to
   use", and tranches with `tranche_target_device` and format / modifier
   tables. The physical device whose DRM node matches `main_device` is the one
   chosen.
2. **Allocate.** gpu2d's Vulkan renderer allocates, per present slot, an
   exportable image:
   - tiling `DRM_FORMAT_MODIFIER`;
   - the modifier list taken from the intersection of the compositor's tranche
     and `VkDrmFormatModifierPropertiesListEXT`;
   - format `R8G8B8A8_UNORM`.

   It then exports the memory as a dma-buf fd.
3. **The format needs no swizzle.** `DRM_FORMAT_XBGR8888` is "[31:0] x:B:G:R
   8:8:8:8 little endian" (`drm_fourcc.h`). The bytes in memory are R, G, B, x:
   exactly gpu2d's pixel words. The present step is therefore a plain
   `vkCmdCopyBufferToImage` from the target. It is not a kernel. If the
   compositor offers only `XRGB8888`, a two-line present kernel swaps R and B,
   as Metal's `present` does.
4. **Create the buffer.** `zwp_linux_buffer_params_v1.add(fd, plane_idx,
   offset, stride, modifier_hi, modifier_lo)`, with the plane layout from
   `vkGetImageSubresourceLayout`, then `create_immed` (since 2), which
   creates a `wl_buffer`. The fds "will stay valid for the whole lifetime of
   the wl_buffer". The fds go through the existing SCM_RIGHTS path.
5. **Synchronization.** "Unless specified otherwise in another protocol
   extension, implicit synchronization is used." The
   `VK_EXT_external_memory_dma_buf` page opened here says nothing about
   attaching an implicit fence to exported memory written by
   `vkQueueSubmit`. Without a promise, the window **waits for the slot's
   VkFence before `wl_surface.commit`**. The window already holds frames
   until `Wait`, so this costs latency, not correctness. Explicit sync is a
   later milestone: `linux-drm-syncobj-v1` on Wayland, `DRI3ImportSyncobj` on
   X11 (dri3proto; its version entry was not in the part read).
6. **Reuse.** A slot is reused only after `wl_buffer.release`. The window
   keeps a ring of 2 or 3 present images.

**X11: DRI3 1.2 and Present** (dri3proto, presentproto).
1. `DRI3GetSupportedModifiers` gives the window and screen modifier lists.
2. `DRI3PixmapFromBuffers` takes up to 4 fds with stride, offset and an
   explicit `modifier`. It came in DRI3 1.2. `PixmapFromBuffer` (1.0) carries
   no modifier, and with `DRM_FORMAT_MOD_INVALID` "the driver may make its own
   inference through unspecified means", so the 1.2 request is required.
3. `PresentPixmap` presents. The pixmap is reused after `PresentIdleNotify`:
   "If 'idle-fence' is None, then the X server must arrange for the pixmap to
   be re-usable by the client as soon as the PresentIdleNotify event has been
   received."
4. Synchronization "is considered to follow the implicit model" (dri3proto
   §12), so the same CPU wait on the fence applies.
5. The device is matched through `DRI3Open`'s fd.

**Why this way:**
- It keeps both Linux clients pure Go and unchanged in ownership.
- The fd passing it needs is already written and tested on both sockets.
- It needs no WSI library, and the frame never comes back to the CPU.

**Fallback chain.** Each step is taken only when the previous one fails, with
the reason recorded in `GPUError`:
1. dma-buf (above).
2. **Vulkan raster with CPU present.** The target is copied into a
   host-visible buffer, and the window presents it through its existing
   `wl_shm` / MIT-SHM path. That costs one PCIe read-back per frame (to be
   measured on fractal). The variant that imports the shm mapping with
   `VK_EXT_external_memory_host` (alignment `minImportedHostPointerAlignment`,
   VUID-…-pHostPointer-01749) would remove the memcpy. It is tried on RADV
   first, because whether a memfd mapping counts as a host allocation for that
   extension is not established here.
3. The CPU path as today.

### Windows: DXGI flip model on the HWND

- **The swap chain.** `IDXGIFactory2::CreateSwapChainForHwnd` with
  `DXGI_SWAP_EFFECT_FLIP_DISCARD` and `R8G8B8A8_UNORM`. The flip model allows
  only R16G16B16A16_FLOAT, B8G8R8A8_UNORM, R8G8B8A8_UNORM or
  R10G10B10A2_UNORM, `SampleDesc` 1/0, and `BufferCount` "from two to
  sixteen" (DXGI_SWAP_CHAIN_DESC1, Remarks). R8G8B8A8 is gpu2d's byte order.
  `AlphaMode` is IGNORE, because windows are opaque (DESIGN.md contract).
- **Writing the back buffer.** It is requested with
  `DXGI_USAGE_UNORDERED_ACCESS`, a documented swap-chain usage ("Use the
  surface or resource for unordered access", DXGI_USAGE). The present kernel
  then writes it as a `RWTexture2D<unorm float4>` with `c/255`, as Metal's
  `present` does.
  - If a driver refuses a UAV back buffer, the kernel writes our own UAV
    texture, followed by `CopyResource`.
  - D3D11 has no buffer-to-texture copy, so a kernel is needed either way.
- **Exactness of the store.** That `c/255` stored into a UNORM becomes `c`
  exactly is **tested, not cited**: the test reads the back buffer through a
  staging copy.
- **Constraints of the flip model:** one flip swap chain per HWND, and "only
  Direct3D content in flip model swap chains … are visible" (DXGI flip model).
  So the GPU path is refused for a window that hosts GDI-drawn native
  controls, as Cocoa refuses materials (`cocoa_darwin.go:1036`).
- **Resize:** `ResizeBuffers` after releasing every back-buffer reference.

## Decision 4: milestones, each one or two PRs, each with its proof

The repository is named first. "Parity" means gpu2d's existing random-scene
suite against `Rasterize`, over an opaque and a translucent ground, reporting
the maximum difference and the count of differing channels.

| # | Repository | Content | Proof |
|---|---|---|---|
| K0 | gpu2d | `kernel.wgsl` (compute) and a WebGPU compute mode | Parity in Chromium on the M4 Max: 0 differing channels on **both** grounds (today alpha differs by up to 4 on a translucent one) |
| K1 | wgsl, gpu2d | `msl` gains entry points, buffers, loops and `u32(f32)`; `metal/kernel_gen.metal` replaces both hand files; the correctly rounded sqrt is measured | `TestMatchesReference` unchanged and byte-identical; the msl GPU semantics tests extended to the new constructs |
| W1 | wgsl | `spirv` package, a subset large enough for a fill kernel | Structural tests of the module layout (SPIR-V §2.4); then V1 runs it |
| V1 | vulkan (new) | Loader, instance, device, buffers, compute pipeline, fences; `vkgen` | On CI, ubuntu with lavapipe (Mesa's software Vulkan) and the Khronos validation layer, with zero validation messages; on fractal, RADV on the W7700 (cross-compiled `go test -c`, scp, run) |
| W2 | wgsl | `spirv` complete for the kernel | Every docs/MSL.md edge case (NaN, ±0, i32 overflow, division by zero) on lavapipe and RADV, compared with the checker's constant evaluator |
| G1 | gpu2d | `vulkan` back-end: Draw, Submit/Wait, Pixels | Parity on lavapipe (CI) and RADV (fractal); `BenchmarkOverlap` and frame benchmarks against the CPU path on fractal (below) |
| L1 | window | `Config.GPU` on Wayland and X11 through fallback 2 (Vulkan raster, shm present) | Live window on fractal; frame equal to the CPU path's (the test compares the shm bytes) |
| L2 | window, gpu2d | `zwp_linux_dmabuf_v1` v4 with feedback; export images | Protocol tests against the in-repo fake compositor (`internal/wayland/virtual.go`); live on fractal under a compositor whose screenshot can be read back (weston headless is the candidate, to be confirmed there) |
| L3 | window | DRI3 1.2 and Present | Same, under Xorg or Xwayland on fractal |
| D1 | d3d11 (new) | Device (hardware or WARP), buffers, views, compute, query, staging | windows-latest CI on WARP |
| D2 | wgsl, gpu2d | `hlsl` package; `kernel.dxbc` generated on CI | Drift test on the Windows lane; edge-case semantics on WARP |
| D3 | gpu2d | `d3d11` back-end | Parity on WARP |
| D4 | window | Swap chain on the HWND, `Config.GPU` on Win32 | Live window on the CI runner when it has a desktop session; back buffer read through staging and compared |

### Outcomes of K0 and K1 (2026-11-10)

K0 (gpu2d v0.7.0, #25) and K1 (go-gfx/wgsl v0.6.0, v0.6.1 and v0.6.2; gpu2d
v0.8.0, #26) are done. What they changed in this design:

- **Compute mode stays opt-in.** `webgpu.NewCompute` is byte-identical to
  `Rasterize` on both grounds (0 differing channels in Chromium on the M4 Max,
  and on the macOS and SwiftShader CI lanes), but it is slower than the
  fixed-function path on every frame measured: +57 % at 1080p, +36 % at 4K,
  +53 % on the gallery frame (CPU binning and the bins upload are in the
  frame). By the rule that compute replaces fixed-function only if not slower,
  it does not.
- **The generated Metal kernel is identical and not slower.**
  `metal/kernel_gen.metal` comes from `kernel.wgsl` through `msl.Module`;
  `TestMatchesReference` still finds 0 differing channels, and `WidgetFrame`
  and `Submit` against v0.7.0 are equal or 2 % faster. `kernel.metal` and
  `coverage_gen.metal` are gone.
- **`present.metal` stays hand-written.** The present kernel writes a
  texture, which `msl` does not emit and `kernel.wgsl` has no equivalent for.
- **Spans and list are one `bins` buffer.** As the kernel binds them (four
  storage buffers and one uniform); a slot still bins in place, without a copy.
  Buffer indices are the WGSL bindings: instances 0, bins 1, arena 2, pixels 3,
  frame 4.
- **`u32(f32)` saturates at 4294967040, and `i32(f32)` at 2147483520.** The
  specification takes the closest value that is also representable in f32
  (`wgsl/index.bs`, "Floating Point Conversion"). The checker's constant
  evaluator said 4294967295 and 2147483647; fixed in go-gfx/wgsl v0.6.2. The
  emitters (SPIR-V, HLSL) must follow the specification, not the old value.
- **Out-of-bounds indexing is unchecked in `msl`.** WGSL bounds the outcomes of
  an out-of-bounds access (a trap, a zero, any location of the buffer); MSL
  does not, and a runtime-sized array's length is not known to the kernel. The
  kernel's indices are in range by construction, and the host sizes the
  buffers. The SPIR-V and HLSL emitters will have the same choice to make.
- **`main` is a reserved name** in `msl` (MSL refuses a kernel called `main`),
  and a function may call one declared below it (prototypes first, v0.6.1).
- **The correctly rounded sqrt:** Metal's `length()` is already correctly
  rounded on the sets tested, and the proposed corrector is not exact near
  midpoints; see "sqrt is the one operation no specification pins". It is
  deferred to W2.

### Outcomes of V1 (2026-11-10)

V1 is done: go-gfx/vulkan v0.1.0 (#1). On lavapipe, in CI, with the validation
layer on, the fill kernel emitted by go-gfx/wgsl `spirv` writes the expected
bytes with zero validation messages; a wrong descriptor binding fails. **The
fractal run (RADV, W7700) is pending:** the host did not answer SSH. What it
changed in this design:

- **Destroy entry points are dlsym'd too.** Besides `vkGetInstanceProcAddr`,
  the loader resolves `vkDestroyInstance` and `vkDestroyDevice` by `dlsym`, so
  an object can still be destroyed when resolving the rest of its table fails.
- **`maxMemoryAllocationSize` is not in `VkPhysicalDeviceProperties`.** It is in
  `VkPhysicalDeviceMaintenance3Properties`, read through properties2. The
  instance therefore asks for API 1.2 and the binding refuses a device that
  reports less than 1.1.
- **Validation is enabled through `InstanceCreateInfo`, not
  `VK_INSTANCE_LAYERS`.** `Options.Validation` names the layer and fails with
  `ErrLayerNotPresent` when it is absent, so a run cannot validate nothing
  silently. The tests capture stdout and stderr and fail on any message.
- **The registry holds Vulkan SC variants.** `vk.xml` repeats members and
  enumerators with `api="vulkansc"`; `vkgen` keeps only those for `vulkan`.
  The registry is committed gzipped (0.3 MB of 3.3 MB), as the push gate
  refuses blobs of 1 MiB or more.
- **No `dlclose`.** `Close` keeps the loader mapped for the life of the
  process: drivers may keep threads alive after their instance is destroyed.

**Benchmarks (G1, L2, D3).**
- The reference is the CPU path, PixelPainter on one core, measured against
  the GPU back-end.
- Frames: the gallery frame (960 × 2834), a 1080p widget frame and 4K.
- Each frame is reported for `Draw`, for `Submit`/`Wait` overlapped with
  1.5 ms of CPU work, and with present included.
- Runs use interleaved rounds and benchstat medians on fractal, never at a
  1-minute load above about 8. Allocation counts are reported exactly.
- **Each back-end is compared with the Metal figures in DESIGN.md:** 1.08 ms
  at 1080p and 2.79 ms at 4K, binning included. A Vulkan back-end slower than
  the CPU path on the 7950X is an open defect.

**What CI can and cannot prove.**
- **lavapipe and WARP are software implementations.**
  - They prove the bindings (slots, layouts, call order), the emitters'
    semantics, and byte-identity under *their* arithmetic.
  - WARP "passes all the standard WHQL conformance tests" (WARP guide).
  - They prove neither speed nor the arithmetic of real hardware: a GPU's
    `sqrt` error is precisely what a software rasterizer does not exhibit.
- **The Linux hardware claims rest on fractal:** RDNA3 with RADV, and one
  GPU. NVIDIA and Intel stay unmeasured until someone runs the suite there.
- **Windows hardware is unproven.** No machine with a GPU is available: CI
  is WARP only. Until one is, a hardware D3D11 path ships as opt-in and says
  so.

**Security and resource bounds.** They extend README "Bounds".
- **Device limits are checked at `New` and per frame.** Both limits below can
  be exceeded by a legal list, so the frame is refused with `ErrTooLarge`
  instead of being dispatched, and the window draws that frame on the CPU.
  - `maxStorageBufferRange` is guaranteed only at 2^27 bytes, and
    `maxMemoryAllocationSize` at 2^30 (Vulkan Limits table). `MaxData` is
    2^28. `MaxBinned` entries take 2^28 bytes as `u32`.
  - The D3D11 buffer-size limit is
    `min(max(128, 0.25 · dedicated VRAM), 2048)` MB, guaranteed: 128 MiB
    (2^27 bytes, the same as Vulkan's `maxStorageBufferRange` floor) on WARP
    and any adapter under 512 MiB. See "Outcomes of D1".
- **32-bit arithmetic in the kernel.**
  - A pixel index `y·w + x` fits in `u32` because the target is bounded by
    `maxStorageBufferRange`.
  - Image products are already bounded by `MaxScaled` (2^31).
  - Arena offsets are bounded by `MaxData` and are checked when the list is
    recorded.
  - The `u32(f32)` conversion is range-proved: the value is in [0, 255.5].
- **Out-of-bounds GPU access.** Every index the kernel reads comes from the
  CPU (`Bin`, recorded offsets), not from a guest shader. The Vulkan
  back-end enables `robustBufferAccess` when offered, as a second line. On
  D3D11 an out-of-range structured-buffer **index** reads 0, but an
  out-of-range byte offset inside a struct is undefined. See "Outcomes of D1".
- **No untrusted shader enters the path.** The kernel is fixed and generated
  at build time, so go-wgsl's "not a security boundary" caveat does not
  apply.
- **File descriptors.** An exported dma-buf fd is closed on every error path.
  The Wayland transport already closes stray received fds
  (`transport_linux.go:156`).
- **Fuzzing.** The new protocol encoders get fuzz targets, as the existing
  wire code has: dmabuf feedback parsing (a format table arrives as an fd
  plus size from the compositor, so it is untrusted and size-bounded) and the
  DRI3 / Present replies. `vkgen`'s registry reader is fuzzed too.

### Outcomes of G1 (2026-11-10)

G1 is done except what needs fractal: package `vulkan` (go-gfx/gpu2d, this
pull request) over go-gfx/vulkan (v0.1.0 at first, v0.2.0 since the overlap change below) and
go-gfx/wgsl v0.8.0. **Pending, because
fractal has been down since 2026-11-10 10:00:** the RADV run on the W7700 and the
fractal benchmarks. Nothing below is a statement about a GPU.

- **Parity on lavapipe, in CI.** `TestMatchesReference` (300 random scenes,
  4,769 instances, over an opaque and a translucent ground) finds **max |Δ| 0
  and 0 of 3,686,400 channels** differing on each ground, on llvmpipe
  (LLVM 19.1.7, Mesa 25.0.7), with the validation layer on and no message.
  The two-frames-in-flight composition test (50 rounds, byte for byte), the
  copy-semantics test and the in-place binning test pass.
- **Break check.** The kernel regenerated with the red channel's rounding
  perturbed (`+ 127u` becomes `+ 100u`) fails parity: max |Δ| 2, 6,936
  channels differ in 60 scenes (`TestBreakCheckPerturbedBlend`). The pixels'
  descriptor pointed at a decoy leaves `Pixels` untouched and draws on the
  decoy (`TestBreakCheckWrongBuffer`).
- **Generated, with a drift test.** `vulkan/kernel_gen.spv` (12,608 bytes) and
  `vulkan/kernel_rte_gen.spv` (12,632 bytes) come from `internal/spirvgen`;
  `.gitattributes` marks them binary.
- **Wrong assumption, then fixed: two frames in flight.** go-gfx/vulkan v0.1.0
  records its only barrier *after* the dispatch (`compute -> host`), so a
  second submission had no memory dependency on the first, and the first
  version of this back-end (v0.9.0) made `Submit` wait for the previous
  frame's fence on the host: the frames were queued one at a time. go-gfx/vulkan
  v0.2.0 adds `CommandBuffer.Barrier`, a compute/transfer memory barrier. The
  specification (`vkCmdPipelineBarrier`) defines both its scopes in submission
  order ("all commands that occur earlier ... later in submission order") and
  a memory dependency between commands "submitted to the same queue" before and
  after it, so a barrier at the start of each frame's command buffer orders its
  dispatch after the previous frame's writes with no semaphore. `Submit` no
  longer waits on a fence: on lavapipe, with 1920 x 2834 frames, the second
  `Submit` returns in about 40 µs while the first frame is still running (its
  fence is not signalled, `TestFramesReallyOverlap`), and the two-frame
  composite is the CPU's, byte for byte. The frames still run one after the
  other on the device, since each composites over the other's pixels; what
  overlaps is the queueing and the CPU packing.
- **The barrier's break check is the binding's.** The validation layer's
  synchronization validation (1.4.309, lavapipe) reported no hazard between
  compute dispatches for this kernel, only between copies, so gpu2d cannot
  fail on a missing barrier here; go-gfx/vulkan's `TestBreakMissingBarrier`
  shows two unordered submissions drawing a `SYNC-HAZARD`, clean once the
  second begins with `Barrier`. lavapipe also runs its queue serially, so a race
  would not show; a real GPU is where it matters (fractal, pending).
- **RTE is now used where offered.** go-gfx/vulkan v0.2.0 reports
  `shaderRoundingModeRTEFloat32` (`DeviceInfo.RoundingModeRTEFloat32`). The
  renderer runs `vulkan/kernel_rte_gen.spv`, the same kernel with the
  `RoundingModeRTE` execution mode declared (every float operation keeps
  `NoContraction`), on a device that reports it, and `kernel_gen.spv`
  elsewhere. lavapipe reports it: parity is **max |Δ| 0 with both modules**
  (`Options.NoRTE` selects the plain one in the tests). Whether the mode
  changes anything on RADV's hardware is for fractal.
- **`robustBufferAccess` is enabled where offered**, by go-gfx/vulkan v0.2.0's
  `Open` (`Renderer.RobustBufferAccess`); lavapipe offers it. It is the second
  line behind the kernel's in-range indices, as the bounds section promised.
- **Device-local target: not decided.** The binding now has device-local
  buffers and a staging `Copy`, but a lavapipe has one memory type that is both
  host-visible and device-local, so it cannot show whether a device-local target
  with a copy beats a mapped one, and no discrete GPU is reachable (fractal is
  down). The target stays host-visible; to be measured on the W7700.
- **One descriptor set per `Pipeline`** in the binding, so each slot has its
  own pipeline and command buffer, and a slot rebinds a buffer only when it
  was replaced (growth, `Resize`).
- **Limits are real on lavapipe.** It reports `maxStorageBufferRange`
  134,217,728 (2^27, the guaranteed minimum) and `maxComputeWorkGroupCount`
  65,535, so `ErrTooLarge` is tested against live limits at `New`, `Resize` and
  a slot buffer: a target past 2^27 bytes is refused, 1080p and 4K are not.
  Slot buffers double up to the device's range, never past it.
- **Without a loader the package skips, not fails**, unless `VULKAN_REQUIRE=1`;
  it builds and vets on windows, js/wasm and 386 (the binding returns
  `ErrUnsupported` there).

**Software benchmarks** (taken on v0.9.0, before the overlap change, which
adds one recorded command per frame and removes a wait; not re-measured, so
`Submit` in particular may differ). These are lavapipe (llvmpipe, 16 threads) in a
linux/arm64 container on the M4 Max host: **a CPU rasterizer's speed, not a
GPU's**, and not comparable with the Metal figures in DESIGN.md. Three
interleaved rounds (`WidgetFrame`, `Submit`, `WidgetFrame`, ...), 30
iterations each, median reported. The 1-minute load of the host was 6.9 at the
start and 10.8 at the end (other sessions plus the 16 lavapipe threads), so
**the times are indicative only**; the allocation counts are exact.

| Benchmark | Median | Rounds | Allocs/op |
|---|---|---|---|
| `WidgetFrame/1080p` (`Draw`) | 12.76 ms | 12.73, 12.76, 14.73 | 60 |
| `WidgetFrame/4K` (`Draw`) | 43.5 ms | 40.7, 43.5, 46.9 | 61 |
| `Submit/empty/64x64` | 21.6 µs | 21.6, 28.0, 21.3 | 50 |
| `Submit/empty/1920x5668` | 64.4 µs | 63.3, 64.4, 75.6 | 50 |
| `Submit/widget/1920x1080` | 93.8 µs | 93.8, 89.1, 130.6 | 51 |
| `Submit/widget/1920x5668` | 334 µs | 328, 334, 351 | 52 |

`Submit` alone costs about 20 µs of fixed work (the binding's calls) plus
binning and copies that grow with the frame, and it returns before the device
has drawn. The fractal numbers, the comparison with the CPU path on the 7950X
and with cairo stay **pending**.

### Outcomes of L1 (2026-11-10)

L1 is done except what needs fractal: go-widgets/window #157 (open for review)
puts `Config.GPU` on the X11 and Wayland back-ends through fallback 2, over
gpu2d v0.9.1. **Pending, because fractal is down:** the run on the W7700 and
every hardware number. The live proof ran under lavapipe in a linux/arm64
container (`test/gpu-linux/Dockerfile` in the window repository) instead.
Nothing below is a statement about a GPU.

- **Live byte equality, per protocol, under lavapipe (llvmpipe, LLVM 19.1.7).**
  - X11 (Xvfb, MIT-SHM): the window read back with `XGetImage` equals a window
    that did not ask for the GPU, and PixelPainter's frame: 192,000 bytes at
    320 × 200, 312,000 after a resize to 400 × 260.
  - Wayland (headless sway): the `wl_shm` buffer the window committed equals
    the CPU window's and PixelPainter's packed as ARGB8888 (1,920,000 bytes,
    800 × 600; sway resized the toplevel from 320 × 200, so the renderer's
    `Resize` ran live), and grim's capture of the output matches at all 480,000
    pixels.
  - Break check: flipping one bit of the copied frame fails both, naming the
    byte.
- **Assumption corrected: the target is already host-visible, so fallback 2 is
  one memcpy, not a `vkCmdCopyBuffer` read-back.** G1's `Pixels()` is the mapped
  target (it stayed host-visible); the window copies it into its own
  framebuffer and the existing present path packs from there. The
  `VK_EXT_external_memory_host` variant, which would remove that copy by
  importing the shm mapping, was not tried (it needs a device that is not
  lavapipe to mean anything).
- **Assumption corrected: "the window already holds frames until `Wait`".** The
  window calls `Draw` (`Submit` then `Wait`) once per frame: it does not use the
  two frames in flight, because the present path needs the pixels before it can
  commit. Overlapping the next frame's recording with the device is available
  and not used yet.
- **Assumption corrected: `Pixels()` is not zeroed.** The kernel composites over
  the frame's current contents, and Vulkan promises nothing about fresh device
  memory, so the window clears `Pixels()` when the path starts and after every
  `Resize` (the contract already said the content is not kept). A translucent
  theme background would otherwise differ from the CPU path.
- **Assumption corrected: "other targets build a stub" for go-gfx/vulkan.** Its
  constraint is `linux && (amd64 || arm64)`, and `GOOS=android` satisfies
  `linux`, so the real binding is compiled on android/arm64, where purego has no
  CGO-free `Dlopen` (`undefined: cgo.Dlopen`). window's `android` lane builds
  CGO-free and caught it. Fixed upstream: go-gfx/vulkan v0.2.1 (#6) constrains
  the loader to `linux && !android && (amd64 || arm64)`, android gets the
  `ErrUnsupported` stub, and its CI vets and builds android/arm64 CGO-free.
  gpu2d v0.9.2 requires it, and the window's temporary workaround is gone.
- **The fallback is for good, as on Cocoa, not "for that frame".** A frame the
  renderer refuses (`ErrTooLarge`, a lost device, a timeout, a frame of the wrong
  size) releases the renderer, records the error (`GPUError`) and is drawn on
  the CPU, as is every later frame. A refused `Resize` does the same. The first
  CPU frame after a drop is a whole one even for an incremental root, because
  the framebuffer was not kept current while the GPU drew. Vulkan absent
  (`ErrUnsupported`) leaves the window on the CPU path from the start with the
  reason in `GPUError`.
- **sway, not weston, was the headless compositor.** It is the one the window's
  Wayland CI lane already uses, it ran as an unprivileged user in the container
  (it will not run as root), and it served `wl_shm` and the capture. weston was
  not tried. Whether this sway's renderer (pixman in the lane) accepts dma-buf
  buffers is **not examined**; L2's proof depends on it.
- **Software numbers** (lavapipe, 16 threads, linux/arm64 container on the
  M4 Max; 6 interleaved rounds under abbench `--max-load 8`, 30 frames, benchstat
  medians; the host's 1-minute load stayed under 8; allocation counts are
  exact). One whole frame painted and presented in a live window; `cpu` is the
  process's CPU time per frame, every thread. A Wayland frame includes one round
  trip with the compositor, the same for both paths.

  | window | protocol | CPU path wall / cpu | GPU path wall / cpu | allocs/op CPU / GPU |
  |---|---|---:|---:|---:|
  | 960 × 640 | X11 | 2.47 ms / 2.48 ms | 3.62 ms / 25.8 ms | 30 / 89 |
  | 960 × 640 | Wayland | 2.14 ms / 2.13 ms | 3.26 ms / 23.3 ms | 64 / 127 |
  | 1920 × 1200 | X11 | 8.41 ms / 8.46 ms | 11.7 ms / 89.7 ms | 30 / 89 |
  | 1920 × 1200 | Wayland | 7.75 ms / 7.71 ms | 10.5 ms / 86.9 ms | 64 / 128 |

  Under lavapipe the GPU path costs about ten times the CPU time and 35 to
  52 % more wall time than the CPU path, as it must: the "device" is a CPU
  rasterizer and the copy is on top. Whether it pays on a real GPU, and whether
  the copy is what remains, is for fractal. By the rule above (a Vulkan back-end
  slower than the CPU path on the 7950X is an open defect) that comparison
  decides.
- **CI.** window gains a `live GPU (lavapipe, X11 + Wayland)` lane running the
  same tests with `WINDOW_GPU_REQUIRE=1`, so a runner without a driver fails
  instead of skipping.

### Outcomes of D1 (2026-11-10)

`github.com/go-mswin/d3d11` v0.1.0 (PR #1) is released, with packages `dxgi`
and `d3dcompiler`. Pure Go, no cgo. On windows-latest, a **WARP** device
("Microsoft Basic Render Driver", feature level 11_1) dispatches a fill shader
and the staging read-back holds the expected bytes. No GPU has run it.

**Proof.** The break check passes: with the UAV bound at the wrong slot the
same dispatch leaves the buffer zero and the comparison fails. A raw
(`ByteAddressBuffer`) SRV/UAV dispatch matches too. Every COM object is counted,
and the count returns to zero after each test, error paths included. The D1
row says "windows-latest CI on WARP"; that is what is proven, nothing more.

**The two open items.** Source: Direct3D 11.3 Functional Specification
(microsoft.github.io/DirectX-Specs, opened 2026-11-10).
- **Buffer size.** `min(max(A, B · VRAM), C)` MB with
  `D3D11_REQ_RESOURCE_SIZE_IN_MEGABYTES_EXPRESSION_A_TERM` = 128,
  `_B_TERM` = 0.25f, `_C_TERM` = 2048. It is a required minimum, not a ceiling.
  WARP reports no dedicated memory, so its limit is 128 MiB (a 128 MiB buffer
  was created in CI). 2048 MiB needs 8 GiB of VRAM. Constant buffers are
  limited to 4096 four-component elements (64 KiB). `MaxData` (2^28) and
  `MaxBinned` (2^28 bytes) exceed the 128 MiB floor, so a frame is refused with
  `ErrTooLarge` as on Vulkan, unless the device's limit is higher.
- **Out-of-bounds reads.** `ld_structured` (22.4.12): an out-of-bounds read of
  `t#` or `u#` returns 0 per 32-bit component, "except: if srcByteOffset (plus
  swizzle) is what causes out of bounds access … the returned value … is
  undefined". So an out-of-range struct index reads 0 and an out-of-range byte
  offset inside a struct is undefined. `ld_raw` returns 0 without the
  exception. Out-of-bounds struct-index writes are dropped. WARP returned 0 for
  an out-of-range index. The kernel never relies on either: every index comes
  from the CPU.

**What this document got wrong or left out.**
- **Windows layer:** Decision 2 says d3d11 builds on go-mswin/win32. It needs
  only `golang.org/x/sys/windows`, and that is all it imports.
- **DXBC route:** the table says `go generate` on Windows. The module has no
  `go generate`: `TestDXBCNotDrifted` recompiles with `D3DCompile`
  (`d3dcompiler_47.dll`, through syscall; not `fxc.exe`) and compares. With
  `D3D11_DXBC_OUT` set it writes the fresh bytes, and CI uploads them as an
  artifact. That is how the committed files were made.
- **Drift risk:** DXBC depends on the runner's `d3dcompiler_47.dll`. If the
  image updates it, the drift test goes red until the files are regenerated.
- **DXGI:** WARP appears twice in the adapter enumeration, once with the
  software flag clear and once with it set, both named "Microsoft Basic Render
  Driver". A device made with `D3D_DRIVER_TYPE_WARP` reports the software one.
- **A slot list is not enough.** The first WARP run failed with `E_INVALIDARG`
  because `D3D11_QUERY_EVENT` is enumerator 0, not 1: enumerator values need
  the same anchors as vtable slots.

### Outcomes of D2 (2026-11-10)

go-gfx/wgsl v0.9.0 has package `hlsl` (HLSL for Shader Model 5.0), and this
repository has `d3d11/kernel_gen.hlsl` and `d3d11/kernel_gen.dxbc` generated
from `kernel.wgsl`. No back-end yet: that is D3. Nothing here has run on a GPU.

**The emitter** takes the whole of `kernel.wgsl`. Its rules and their sources
are in go-gfx/wgsl `docs/HLSL.md`.
- **Resources:** `StructuredBuffer`, not `ByteAddressBuffer`. The kernel reads an
  array of a 64-byte `Inst` and three arrays of `u32`; a structured buffer
  indexes them as WGSL does, and go-mswin/d3d11 creates them with a stride. A
  binding is the register index of its class: `read` is `t<binding>`,
  `read_write` is `u<binding>` (at most 8), a uniform is `b<binding>` (at most
  14). The kernel's bindings give `t0 t1 t2 u3 b4`, so the host binds SRVs at 0,
  1, 2, a UAV at 3 and the `Frame` constant buffer at 4.
- **Integers and conversions:** wrap through `uint`, WGSL's division and shift
  results, `u32(f32)` and `i32(f32)` saturating at 4294967040, 2147483520 and
  -2147483648, NaN to 0. DXBC's own `ftou` and `ftoi` clamp at 2^32 and 2^31 (not
  at the values `f32` can hold) and `udiv` by zero is 0xffffffff, so none of them
  could be left to the compiler.
- **`precise`:** every float variable is `precise`, and each float `+ - * /` is
  passed through a `precise` variable (`wgsl_pf`), because "qualifying a
  variable precise doesn't make operations that use the variable precise"; it
  propagates only to what contributes to the value assigned (Microsoft Learn,
  "Variable syntax"). Break check on windows-latest: `a * b + c` compiles to a
  `mul` and an `add` as emitted, even without `D3DCOMPILE_IEEE_STRICTNESS`, and
  to a `mad` with the pins removed. The whole kernel compiles with no `mad`.

**WARP** ("Microsoft Basic Render Driver", feature level 11_1), through
go-mswin/d3d11 v0.1.0, in the nested module `hlsl/hlsltest`:
- every emitted file is compiled by `D3DCompile` (cs_5_0, IEEE strictness,
  warnings as errors) and the integer, float, conversion, NaN, loop and
  structure edge cases are held to WGSL's definition, including the checker's
  constant evaluator for `u32(f32)` and `i32(f32)`; all pass;
- `sqrt` and `length` are correctly rounded on WARP (0 of 261480 and 0 of 217072
  inputs differ); the D3D11 specification allows 1 ULP and the test fails beyond
  it. That is WARP's arithmetic, not a GPU's;
- a second break check: with the divisor guard removed, `7 / 0` is 0xFFFFFFFF
  where WGSL says 7, and the edge cases see it.

**The DXBC lane** (this repository, `d3d11 (windows, WARP)`):
- `kernel_gen.hlsl` is generated by `go generate ./d3d11` on any platform, with a
  drift test on every platform.
- `kernel_gen.dxbc` can only be made on Windows. `kernel_gen.dxbc.record` sits
  next to it and holds the compiler DLL's file version (today
  `d3dcompiler_47.dll 10.0.26100.33648`), the flags, the target and entry, and
  the sha256 of the HLSL and of the DXBC. A test on every platform checks the
  record against both files, so an HLSL change without a new DXBC fails
  everywhere. The Windows lane recompiles and compares the bytes, with no
  tolerance.
- **The failure names its cause.** If the runner image's compiler DLL moves, the
  message prints both versions ("the runner image's d3dcompiler_47.dll moved
  from ... to ...; nothing in the repository changed") and how to regenerate:
  `go run ./internal/hlslgen/cmd/gendxbc` on Windows, or the lane's `dxbc`
  artifact, copied over the two files. The committed files were made that way.

**What this document got wrong or left out.**
- **Where `go generate` runs:** the table said `go generate` on Windows for the
  whole D3D11 row. The HLSL is generated anywhere; only the DXBC needs Windows.
- **A drift test needs a record.** Bytes alone say that something changed, not
  what. The record is what makes a red lane attributable at a glance.
- **fxc is stricter than WGSL about scopes.** Two sibling loops that declare the
  same variable fail with X3078 (an error under warnings-as-errors), so each
  loop is emitted in a block of its own. HLSL has no `float3(x)` either, and
  `&&`, `||` and `?:` do not short-circuit, so a user function call on the right
  of `&&` is refused.
- **Names:** `out`, `mul`, `line`, `sample` and every intrinsic are reserved in
  HLSL, so a WGSL name that is one is refused rather than renamed.
- **Line endings:** a Windows checkout turned `kernel_gen.hlsl` into CRLF and the
  first record held the hash of that. `.gitattributes` now pins the generated
  files, and the committed record was remade from the corrected run.

### Outcomes of D3 (2026-11-10)

Package `d3d11` of gpu2d is a back-end with the shape of `metal` and `vulkan`
(`New`, `NewWith`, `Draw`, `Submit`/`Wait` over two slots, `Pixels`, `Resize`,
`Close`, `Name`, the same errors), running the committed `kernel_gen.dxbc`
through go-mswin/d3d11 v0.1.0. It ships opt-in. Non-Windows builds have stubs
returning `ErrUnsupported`; linux, darwin, windows (amd64, 386), js/wasm and
linux/386 vet. Nothing here has run on a GPU: WARP only, on `windows-latest`
(run 42846149477, "Microsoft Basic Render Driver", feature level 11_1, buffer
limit 134217728 bytes).

**Parity on WARP.** `TestMatchesReference` (300 random scenes, 64 × 48, 4769
instances per ground, all five kinds): max |Δ| 0, 0 of 3,686,400 channels
differ, on the opaque and on the translucent ground. That is WARP's arithmetic,
not a GPU's.

**Break checks.**
- A kernel with the red rounding changed in the WGSL, emitted as HLSL and
  compiled by `D3DCompile` in the test, run in the same parity test: max |Δ| 2,
  6936 of 1,474,560 channels differ.
- `TestBreakCheckWrongUAV` binds u3 to a decoy: `Pixels` keeps its 0xAA fill,
  and the decoy is drawn on.
- `TestSecondSubmitDrawsOverTheDevicesFrame` scribbles on `Pixels` between two
  Submits; the second frame must not see it (it draws over the device's frame).

**Upload and read-back (what was chosen).**
- **First version (go-mswin/d3d11 v0.1.0, merged as the first commit of #35):**
  a structured buffer could be filled only at creation (`BufferDesc.Initial`)
  and `UpdateBuffer` took constant buffers only, so each Submit created the
  instance, bins and arena buffers and their SRVs, and the target from `Pixels`.
- **Now (go-mswin/d3d11 v0.2.0, released from go-mswin/d3d11#2):** a slot keeps
  its three input buffers across frames. They are `Dynamic` structured buffers,
  written by `UpdateBuffer` (Map `WRITE_DISCARD`) and grown on demand like
  metal's `fit` (doubling, up to the device limit, never shrinking). The choice
  is from Microsoft Learn's D3D11_USAGE page (opened 2026-11-10): DYNAMIC is "a
  good choice for a resource that will be updated by the CPU at least once per
  frame", GPU read only and "yes" as an input to a stage; DEFAULT is the only
  usage a stage can write, and its page and UpdateSubresource's say a dynamic
  resource cannot be an `UpdateSubresource` destination. So the three inputs
  (CPU-written every frame, read through SRVs) are DYNAMIC; the target (a UAV)
  is DEFAULT, created once per size and filled in place by `UpdateSubresource`
  when nothing is in flight. The 16-byte frame constant buffer is per slot.
- `Pixels` is a host copy. The caller may have written the ground there, so with
  nothing in flight Submit uploads it; a second queued frame draws over the
  device's frame. The Wait that ends the last frame copies the target to a
  staging buffer, maps it and copies into `Pixels`.
- Each frame is flushed and fenced (`Context.Flush`, `Context.Signal`, one event
  query per frame): Wait of the first of two frames waits for that frame.
- Binning is not in place: `List.Bin` fills host slices kept per slot, written
  as bytes into the slot's buffer. (Metal and Vulkan bin straight into
  device-visible memory; D3D11 has none.)
- A frame past the buffer limit (`4·w·h` > limit, so 5793² on WARP), past 65535
  groups per dimension, or a slot buffer past the limit, is `gpu2d.ErrTooLarge`
  with nothing in flight; `Options.MaxBufferBytes` lowers the limit for the test.

**Measured: WARP on a CI VM, never GPU speed** (4 vCPU `windows-latest`
runner, 20 iterations, one run each, load unknown; `BenchmarkWidgetFrame`,
`Stats` per frame). Before is the per-frame creation of v0.1.0 (run
42846149477), after is buffer reuse and per-frame fences (run 42893588088):

| frame | | WARP `Draw` | pack | target upload | device | read-back | CPU `Rasterize`, same run | allocs/op |
|---|---|---|---|---|---|---|---|---|
| 1080p | before | 96.9 ms | 0.256 ms | 0.529 ms | 84.7 ms | 11.3 ms | 39.9 ms | 227 |
| 1080p | after | 112.2 ms | 0.089 ms | 0.368 ms | 103.3 ms | 8.4 ms | 54.8 ms | 203 |
| 4K | before | 370.4 ms | 1.205 ms | 2.568 ms | 348.7 ms | 17.9 ms | 161.5 ms | 255 |
| 4K | after | 440.7 ms | 0.898 ms | 1.955 ms | 416.3 ms | 21.4 ms | 221.6 ms | 238 |

What the change touches moved the right way on both sizes: packing, which was
buffer creation, fell 65 % at 1080p and 25 % at 4K, and the target upload (an
in-place `UpdateSubresource` instead of a creation) fell 30 % and 24 %. Those
are 0.5 % of a frame, so the frame time does not show it: the device column
(WARP's software rasterizing) is over 90 % of it and moved with the runner, which the
CPU reference in each run shows (+37 % between the two runs of the same code).
Relative to that reference, WARP went from 2.4x to 2.0x slower at 1080p and from
2.3x to 2.0x at 4K, within what the runner's variance allows; none of it is a
claim about a GPU. The copies (upload plus read-back) are about 8 % of a 1080p
frame and 5 % of a 4K one on WARP; on a discrete GPU they cross PCIe and will
weigh more, which only a hardware run can say. The read-back is a staging copy
and a memcpy of the frame; it is unchanged in kind. Parity is still 0 of
3,686,400 channels on both grounds, and the break checks still fail the way they
did.

**What this document got wrong or left out.**
- **"In-place binning into GPU memory may not apply"** is confirmed: it does not.
  The CPU-to-device copy is mandatory (a dynamic buffer written by `Map`).
- **The binding was the gap, not Direct3D.** v0.1.0 had no way to update a
  structured buffer, no `Flush`, and a `Wait` that was an event query on the
  whole context. The first version of #35 worked round that (a buffer creation
  per input per frame; the first of two Waits only checked `RemovedReason`).
  v0.2.0 adds `Dynamic` buffers, `UpdateBuffer` on structured and raw buffers,
  `Flush` and `Context.Signal`/`Fence`, and the back-end now reuses its buffers
  and fences each frame. A CPU-written, shader-read input and a shader-written
  target need two different usages, which the usage table dictates.
- **Windows CI cannot say how much overlap there is.** A frame's fence passing is
  proven (`TestFenceWaitsForItsOwnFrame` in go-mswin/d3d11); whether WARP or a
  GPU starts frame 1 before frame 2 is queued is not measured here.
- **`Software` is not the default.** `New` asks for a hardware adapter; the tests
  and CI pass `Options.Software` (WARP), and `D3D11_HARDWARE=1` selects the
  default adapter.

### Outcomes of D4 (2026-11-10)

D4 has three parts: the swap chain in go-mswin/d3d11 (v0.3.0), `Present` in
gpu2d's `d3d11` (this repository), and `Config.GPU` on Win32 in go-widgets/window
(its outcome is recorded below once that pull request has run).

**What the Microsoft documentation says, and what it does not** (Microsoft Learn,
opened 2026-11-10: "DXGI flip model", "For best performance, use DXGI flip model",
`IDXGIFactory2::CreateSwapChainForHwnd`, `IDXGISwapChain::ResizeBuffers`,
`DXGI_USAGE`, `DXGI_SWAP_CHAIN_DESC`):
- **Formats.** A flip-model swap chain takes `R16G16B16A16_FLOAT`,
  `B8G8R8A8_UNORM` or `R8G8B8A8_UNORM`, one sample, and at least two buffers
  (two to sixteen). `R8G8B8A8_UNORM` is what the frame already is, so no
  conversion is needed.
- **`DXGI_USAGE_UNORDERED_ACCESS`** exists in the usage table. Nothing opened says
  that a flip-model back buffer may use it: `DXGI_SWAP_CHAIN_DESC` says "the back
  buffer can be used for shader input or render-target output". The binding asks
  for it and reports a refusal as the error it is; **on WARP it is accepted**, the
  UAV is created and written (`TestSwapChainBackBufferUAV`), so Present's kernel
  writes the back buffer directly. Whether a given GPU's driver accepts a typed
  UAV of `R8G8B8A8_UNORM` (Direct3D 11.0 requires typed stores only for R32
  formats) is **not known**; a device that refuses gives an error at
  `NewSwapChain`, which the window turns into its CPU path.
- **Resize.** `ResizeBuffers` "can't resize a swap chain unless you release all
  outstanding references to its back buffers", views and bindings included.
  Measured on WARP: with a back buffer referenced it fails with `0x887A0001`
  (`DXGI_ERROR_INVALID_CALL`); the binding therefore holds no back buffer between
  calls, unbinds its views after each Present and calls `ClearState` in `Resize`.
- **After Present** the back buffer "needs to explicitly be re-bound": `BackBuffer`
  is asked for again (buffer 0) every frame. `FLIP_DISCARD` leaves it undefined,
  so the read-back for the proof is made before the swap chain's Present.
- **One flip swap chain per HWND, and nothing else may draw on it:** "only Direct3D
  content in flip model swap chains … are visible". A window that falls back to the
  CPU after its swap chain was made must release the swap chain (and Flush) before
  GDI draws again.

**`Present`** copies nothing between CPU and device: the raster kernel draws over
the device's frame, then `present.hlsl` (hand-written, as `metal/present.metal`
is, since the emitter does not write textures) copies the frame into the back
buffer as an `RWTexture2D<unorm float4>`; `c / 255.0` stored as unorm8 rounds back
to `c`, so the copy is byte-exact. The target is read through an SRV there and
never bound with its UAV. `present.dxbc` is made on the Windows lane
(`TestPresentDXBCIsCompiled` recompiles and compares, with no tolerance; the
artifact `dxbc` holds the file) and committed; it has no record file, since the
compiler version is the raster kernel's.

**Proof on WARP** (run 42897557147, hidden window, "Microsoft Basic Render
Driver"): the back buffer read through staging just before the swap chain's
Present is `Rasterize`'s bytes, max |Δ| 0 and 0 channels differing over 100 random
scenes on each of an opaque and a translucent ground. Break check: a present
kernel with red and blue swapped, compiled in the test, gives max |Δ| 255 and
245,376 channels differing. Two Presents in a row composite on the device;
resize and refusals pass. go-mswin/d3d11's own tests, with a break check, show
the same on the binding: `ResizeBuffers` fails while a back buffer is referenced.

**What this document got wrong or left out.**
- **"UAV back buffer"** was an open question in the D4 row and in risk 6; on WARP
  the answer is yes, and the documentation neither promises nor forbids it.
- **A second kernel and a second DXBC** are needed: the swap chain's buffer is a
  texture, and the target stays a structured buffer, because the raster kernel's
  bindings (and its HLSL) were generated for a buffer. `CopyResource` cannot copy a
  buffer into a texture.
- **`Present` does not synchronize `Pixels`.** The window never reads them; a
  program that mixes `Draw` and `Present` must know that `Draw` uploads `Pixels`.
- **Hidden windows present.** `Present` on a never-shown `HWND` succeeds (the
  status `DXGI_STATUS_OCCLUDED` is a success), which is what CI runs; what the
  compositor then shows has not been observed by anything.

## Risks

1. **Biggest: identity across vendors.** The specifications permit
   2–3 ULP (Vulkan) or 1 ULP (D3D11) on `sqrt`. Today's byte-identity is
   empirical on one GPU. The correctly rounded sqrt (decision 1) is the
   remedy, but it depends on RTE and on no contraction actually holding in
   the driver's compiler, and the specification states that only for SPIR-V
   decorations. If a driver ignores them, we fall back to the ≤ 1 contract
   for that device, and the parity test names the device.
2. **Compositor acceptance of our modifiers and formats.** XBGR8888 with a
   modifier from the feedback is the common case, not a guarantee. This is
   why fallback 2 exists and is milestone L1.
3. **purego on Linux with CGO_ENABLED=0 inside a test binary that is
   cross-compiled and copied to fractal.** It is the first thing V1 checks,
   before any binding work.
4. **The size of the `spirv` emitter** (about 3,000 lines) and its validation
   without spirv-val. The validation layer catches invalid modules at
   pipeline creation; whether that is enough is question 2.
5. **The CPU wait before commit or present** adds up to one frame of latency
   until explicit sync lands.
6. **Windows hardware behaviour** (a UAV back buffer, driver numerics) cannot
   be observed at all before a GPU machine is available.

## Decisions taken (2026-11-09)

The user answered questions 1–4 below:
1. **Windows:** the D3D11 back-end ships **opt-in** (`Config.GPU`, as on
   macOS), its identity proven on WARP in CI, falling back to the CPU on any
   failure. Real Windows GPUs stay unproven until someone runs the suite on
   one.
2. **Native validators as CI oracles only:** spirv-val (SPIRV-Tools) and
   lavapipe may run in CI. Nothing shipped or run at run time depends on
   them.
3. **Measured identity is accepted:** `Rasterize` stays the oracle. Each
   GPU's identity is a parity measurement; a GPU that disagrees falls back
   to the CPU and is named by the test.
4. **WebGPU compute (K0) replaces the fixed-function path if it is not
   slower,** measured on the M4 Max and on fractal's W7700. Otherwise it
   stays an option.

Taken as proposed, open to objection:
5. **Repository homes:** `go-gfx/vulkan` and `go-mswin/d3d11` (with
   `dxgi`). Moving screencapture onto `dxgi` is left for later.
6. **External reference:** cairo's image back-end joins the CPU path as the
   C reference in the G1 and D3 benchmarks.

## Questions for the user (as asked)

1. **Windows hardware.** Is there a Windows machine with a real GPU, or can
   one be offered as a self-hosted runner? Without one, should the D3D11 path
   ship as opt-in, proven on WARP only?
2. **spirv-val as a CI test oracle.** SPIRV-Tools is a third-party native
   tool. Can it run in CI only, never shipped? Otherwise the Khronos
   validation layer (also native, also CI only) is the only external judge.
   Is that acceptable?
3. **The correctly rounded sqrt.** Is it acceptable that `Rasterize`
   (f64 `Hypot`) stays the oracle, while GPUs share one f32 result that
   equals it by measurement, not by proof?
4. **WebGPU compute mode (K0).** Should it replace the fixed-function WebGPU
   path, which would make the browser byte-identical as well? Or should it
   stay an option because of its cost on weak integrated GPUs, which has not
   been measured?
5. **Repository homes.** Are `go-gfx/vulkan` and `go-mswin/d3d11` (with
   package `dxgi`) the right names, and should screencapture move onto
   `dxgi`?
6. **The performance reference.** Beyond the CPU path, should a C 2D
   rasterizer (for example cairo's image back-end) be the external reference,
   per the fleet's performance rule?

## Sources opened

- **gpu2d** (this repository, v0.6.2): `docs/DESIGN.md`, `metal/kernel.metal`,
  `webgpu/shader.wgsl`, `internal/mslgen/mslgen.go`, `gpu2d.go` (bounds,
  `pack`), `tiles.go`, `cpu.go` (`Coverage` in f64 with `math.Hypot`).
- **go-gfx/wgsl v0.5.0:** `README.md`, `docs/MSL.md`, `docs/CONFORMANCE.md`,
  `check.go` (`Info`, `EntryPoint`), and the function list of `msl/msl.go`.
- **go-widgets/window v0.90.1:**
  - `internal/wayland/transport_linux.go` and `shm.go`;
  - `internal/x11/shm.go`;
  - `internal/cocoa/gpu_darwin.go`;
  - `window.go` (`Config.GPU`);
  - the file lists of `internal/win32` and `internal/wayland`.
- **go-mswin/screencapture:** `dxgi_windows.go`. go-mswin/win32 and
  go-mswin/winrt: repository descriptions and file trees.
- **go-macos/metal:** `metal_darwin.go`, `layer_darwin.go` (purego loading).
- **Vulkan specification and reference pages** (docs.vulkan.org, fetched
  2026-11-09):
  - `VkWaylandSurfaceCreateInfoKHR`;
  - appendix SPIR-V Environment, `#spirvenv-precision-operation`,
    `#spirvenv-evaluation-expressions`, `#spirvenv-op-prec`, and the RTE
    VUIDs;
  - Limits, Required Limits table;
  - `VK_EXT_external_memory_dma_buf`;
  - `VkPhysicalDeviceDrmPropertiesEXT`;
  - `VkImportMemoryHostPointerInfoEXT`;
  - `VK_EXT_image_drm_format_modifier`.
- **Khronos Vulkan-Loader** `docs/LoaderApplicationInterface.md`: library
  names, function pointers.
- **SPIR-V specification** (unified1): §2.4 "Logical Layout of a Module",
  §2.11 "Structured Control Flow" and §2.16.2 (merge instructions).
- **Wayland:** linux-dmabuf-v1 protocol (wayland.app rendering of
  `linux-dmabuf-v1.xml`); libwayland client API, appendix B (`wl_display`,
  `wl_display_connect_to_fd`, event queues, `wl_proxy_create_wrapper`).
- **X.Org** (xorgproto mirror GlassOnTin/xorgproto, since freedesktop's
  GitLab refused the fetch):
  - `dri3proto.txt`: `PixmapFromBuffer(s)`, `GetSupportedModifiers`,
    `FenceFromFD`, `ImportSyncobj`, §10 versions, §12 Synchronization;
  - `presentproto.txt`: `PresentPixmap`, `PresentIdleNotify`.
- **Linux** `include/uapi/drm/drm_fourcc.h`: `XBGR8888`, `ABGR8888`,
  `MOD_LINEAR`, `MOD_INVALID`.
- **Microsoft Learn:**
  - `DXGI_SWAP_CHAIN_DESC1`, `DXGI_USAGE`, "DXGI flip model";
  - "WARP Guide";
  - HLSL "Variable syntax" (`precise`), "D3DCOMPILE Constants";
  - Direct3D 11.3 Functional Specification §3.1.3 (32-bit floating point),
    the resource-size constants and limits table, and 22.4.10-22.4.13
    (`ld_raw`, `store_raw`, `ld_structured`, `store_structured`).
- **Chuck Walbourn,** "HLSL, FXC, and D3DCompile" (d3dcompiler_47 in-box,
  build-time compilation recommended).
- **ebitengine/purego** README: platform tiers, structs by value, CGO_ENABLED=0.
