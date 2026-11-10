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

The first paragraph's plan, as written before K1:
It enters in K1 behind a measurement: the parity suite runs on lavapipe,
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
  - The D3D11 buffer-size limit is still to be read: it is an **open item**,
    not yet opened.
- **32-bit arithmetic in the kernel.**
  - A pixel index `y·w + x` fits in `u32` because the target is bounded by
    `maxStorageBufferRange`.
  - Image products are already bounded by `MaxScaled` (2^31).
  - Arena offsets are bounded by `MaxData` and are checked when the list is
    recorded.
  - The `u32(f32)` conversion is range-proved: the value is in [0, 255.5].
- **Out-of-bounds GPU access.** Every index the kernel reads comes from the
  CPU (`Bin`, recorded offsets), not from a guest shader. The Vulkan
  back-end enables `robustBufferAccess` when offered, as a second line. What
  D3D11 guarantees for out-of-range buffer reads is an **open item**, not yet
  opened.
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
  - Direct3D 11.3 Functional Specification §3.1.3 (32-bit floating point).
- **Chuck Walbourn,** "HLSL, FXC, and D3DCompile" (d3dcompiler_47 in-box,
  build-time compilation recommended).
- **ebitengine/purego** README: platform tiers, structs by value, CGO_ENABLED=0.
