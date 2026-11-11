// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package d3d11 draws a gpu2d list on a Direct3D 11 device with a compute
// kernel, through go-mswin/d3d11 (pure Go, COM through syscall, no cgo). It has
// the shape of packages metal and vulkan: instances are binned into tiles on
// the CPU; one GPU invocation per pixel walks its tile's instances in draw
// order, computes gpu2d.Coverage exactly in integers and blends in integers as the CPU
// reference does. The kernel is kernel.wgsl, emitted as HLSL by go-gfx/wgsl and
// compiled to DXBC at generation time (kernel_gen.hlsl, kernel_gen.dxbc): the
// arithmetic exists once, and nothing is compiled at run time.
//
// Upload and read-back. Direct3D 11 has no persistently mapped, GPU-writable
// buffer as Metal and Vulkan have in host-visible memory, so Pixels is a host
// copy of the frame, not the device's memory. A frame is uploaded when Submit
// finds nothing in flight (the caller may have written the ground into Pixels),
// stays on the device while frames are queued, and is copied back through a
// staging buffer by the Wait that ends the last frame in flight. Each slot keeps
// its input buffers (instances, bins, arena) from frame to frame, DYNAMIC
// structured buffers rewritten with Map WRITE_DISCARD and grown on demand; the
// target is a DEFAULT buffer filled with UpdateSubresource (go-mswin/d3d11
// v0.2.0). Each frame has its own fence, so Wait of the first of two frames
// returns when that frame is done. Stats says what each part costs.
//
// The back-end ships opt-in and is proven on WARP in CI only (docs/NATIVE.md,
// "Decisions taken"); no Windows GPU has run it. There is no presentation
// here: that is milestone D4. Off Windows, and where no device can be created,
// New returns an error that satisfies errors.Is(err, ErrUnsupported) or
// ErrNoDevice, so a program cross-compiles without build tags.
package d3d11

import (
	_ "embed"
	"errors"
	"time"
)

//go:generate go run ../internal/hlslgen/cmd/genhlsl

// KernelHLSL is the HLSL (Shader Model 5.0) of the raster kernel, generated from
// kernel.wgsl by internal/hlslgen through go-gfx/wgsl's hlsl package: entry point
// "raster", a 16 x 8 thread group, the buffers of the WGSL at the registers its
// bindings name (docs/HLSL.md of go-gfx/wgsl): instances t0, bins (the tiles'
// spans, then the list) t1, arena t2, pixels u3, frame b4. Any platform
// regenerates it (go generate ./d3d11); a test on every platform checks it is
// current.
//
//go:embed kernel_gen.hlsl
var KernelHLSL string

// KernelDXBC is KernelHLSL compiled for cs_5_0 by D3DCompile. Only Windows can
// remake it (go run ./internal/hlslgen/cmd/gendxbc), so the bytes are committed,
// with KernelDXBCRecord, which says which compiler DLL made them, with which
// flags, from which HLSL. The Windows CI lane recompiles and compares, with no
// tolerance.
//
//go:embed kernel_gen.dxbc
var KernelDXBC []byte

// KernelDXBCRecord is the text of kernel_gen.dxbc.record (internal/hlslgen.Record).
//
//go:embed kernel_gen.dxbc.record
var KernelDXBCRecord string

// PresentHLSL is the hand-written kernel that copies the frame into a swap
// chain's back buffer (a typed UAV): entry point "present", a 16 x 8 thread
// group, the frame t0, the back buffer u0, the size b0. It is hand-written, as
// metal/present.metal is, because the emitter does not write textures.
//
//go:embed present.hlsl
var PresentHLSL string

// PresentDXBC is PresentHLSL compiled for cs_5_0 by D3DCompile with the flags of
// the raster kernel (internal/hlslgen.Flags). Like KernelDXBC it is made on
// Windows and committed; TestPresentDXBCIsCompiled recompiles and compares, and
// writes the fresh bytes to D3D11_DXBC_OUT for the lane's "dxbc" artifact.
//
//go:embed present.dxbc
var PresentDXBC []byte

// ErrUnsupported reports that Direct3D 11 is not available: a platform other
// than Windows.
var ErrUnsupported = errors.New("gpu2d/d3d11: Direct3D 11 is not available")

// ErrNoDevice reports that Windows has Direct3D 11 but no device could be
// created (no adapter at feature level 11_0, or the driver refused).
var ErrNoDevice = errors.New("gpu2d/d3d11: no Direct3D 11 device")

// ErrSize reports a list whose surface is not the renderer's.
var ErrSize = errors.New("gpu2d/d3d11: list size differs from the renderer's")

// ErrBusy reports a Submit with every slot in flight, or a Draw while work
// is in flight: Wait first.
var ErrBusy = errors.New("gpu2d/d3d11: frames in flight; Wait first")

// ErrNothingInFlight reports a Wait with no Submit to wait for.
var ErrNothingInFlight = errors.New("gpu2d/d3d11: Wait without a Submit")

// ErrTimeout reports a frame the device did not finish within the wait limit.
var ErrTimeout = errors.New("gpu2d/d3d11: the device did not finish the frame")

// ErrClosed reports a use of a closed renderer.
var ErrClosed = errors.New("gpu2d/d3d11: renderer closed")

// slots is how many frames may be in flight: the device draws one while the CPU
// bins and packs the next.
const slots = 2

// waitLimit bounds a wait on the device, so that a hung GPU is an error and
// not a hung program.
const waitLimit = time.Minute

// Options picks the device.
type Options struct {
	// Software asks for a WARP device (the software rasterizer), which is what
	// CI runs. The zero value asks for the default hardware adapter.
	Software bool
	// MaxBufferBytes overrides the buffer size limit the renderer checks
	// against; 0 is the documented guarantee for the adapter,
	// min(max(128, 0.25 * VRAM), 2048) MB (128 MiB on WARP).
	MaxBufferBytes uint64
}

// Stats is the time a Renderer has spent in each part of a frame since it was
// made (or since ResetStats). Pack bins the list and writes it into the slot's
// input buffers (growing them when needed); Upload copies the host frame into
// the device's target; GPU is, per frame, from its queueing to its fence
// passing; Readback copies the target to the staging buffer, maps it and copies
// it into Pixels. Every one is wall time on the calling goroutine.
type Stats struct {
	Frames                      int
	Pack, Upload, GPU, Readback time.Duration
}
