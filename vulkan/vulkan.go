// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package vulkan draws a gpu2d list on a Vulkan device with a compute kernel,
// through go-gfx/vulkan (pure Go, no cgo). It has the shape of package metal:
// instances are binned into tiles on the CPU, straight into a host-visible
// buffer; one GPU invocation per pixel walks its tile's instances in draw
// order, computes gpu2d.Coverage exactly in integers, and blends in integers as the CPU
// reference does. The kernel is kernel.wgsl, compiled to SPIR-V by go-gfx/wgsl
// at go generate time (kernel_gen.spv): the arithmetic exists once.
//
// The frame lives in device-local memory and Pixels is a host-visible copy of
// it: what the caller writes there is uploaded before the next draw, and what
// the device has drawn is downloaded when Pixels is next called, so a loop of
// Draw calls that never reads the frame moves no pixel across the bus. On a
// device with one memory type (lavapipe, integrated GPUs) the copies are
// memcpys. docs/NATIVE.md measures the alternatives on a discrete GPU.
//
// There is no presentation here: that is milestone L1. Where Vulkan is absent,
// and on every platform go-gfx/vulkan does not support, New returns an error
// that satisfies errors.Is(err, ErrUnsupported), so a program cross-compiles
// without build tags.
package vulkan

import (
	"encoding/binary"
	"errors"
	"fmt"
	"time"
	"unsafe"

	_ "embed"

	"github.com/go-gfx/gpu2d"
	vk "github.com/go-gfx/vulkan"
)

//go:generate go run ../internal/spirvgen/cmd/genspv

// KernelSPV is the SPIR-V module of the raster kernel, generated from
// kernel.wgsl by internal/spirvgen: entry point "raster", its bindings the
// WGSL's (instances 0, bins 1, arena 2, pixels 3, frame 4).
//
//go:embed kernel_gen.spv
var KernelSPV []byte

// KernelRTESPV is KernelSPV with the RoundingModeRTE execution mode declared
// for 32-bit floats, so that + - * are correctly rounded (the kernel's only
// float is the estimate of a root that integer arithmetic then corrects, so
// the result does not depend on it). The renderer runs it on a device
// that reports shaderRoundingModeRTEFloat32 (a module may declare the mode only
// there: VUID-RuntimeSpirv-shaderRoundingModeRTEFloat32-06303), and KernelSPV
// elsewhere; both keep NoContraction.
//
//go:embed kernel_rte_gen.spv
var KernelRTESPV []byte

// kernels are the modules New builds the pipeline from, except in the break
// check, which swaps in kernels with the blend perturbed.
var kernels = struct{ plain, rte []byte }{KernelSPV, KernelRTESPV}

// ErrUnsupported is go-gfx/vulkan's: no Vulkan loader, or a platform without
// the binding.
var ErrUnsupported = vk.ErrUnsupported

// ErrNoDevice reports that no Vulkan device matched.
var ErrNoDevice = vk.ErrNoDevice

// ErrSize reports a list whose surface is not the renderer's.
var ErrSize = errors.New("gpu2d/vulkan: list size differs from the renderer's")

// ErrBusy reports a Submit with every slot in flight, or a Draw while work
// is in flight: Wait first.
var ErrBusy = errors.New("gpu2d/vulkan: frames in flight; Wait first")

// ErrNothingInFlight reports a Wait with no Submit to wait for.
var ErrNothingInFlight = errors.New("gpu2d/vulkan: Wait without a Submit")

// ErrTimeout reports a frame the device did not finish within waitLimit.
var ErrTimeout = errors.New("gpu2d/vulkan: the device did not finish the frame")

// waitLimit bounds a wait on the device, so that a hung GPU is an error and
// not a hung program.
var waitLimit = time.Minute // a variable for the test of the timeout

// slots is how many frames may be in flight: the GPU draws one while the CPU
// bins and packs the next.
const slots = 2

// frame is the kernel's Frame, field for field.
type frame struct{ W, H, Across, Tile uint32 }

// The kernel's bindings, as kernel.wgsl declares them.
const (
	bInst = iota
	bBins
	bArena
	bPixels
	bFrame
	nBind
)

// slot is the per-frame input memory. The device reads it until the slot's
// fence is waited on, so each in-flight frame owns its own, with its own
// pipeline (and so descriptor set) and command buffer.
type slot struct {
	inst, bins, data *vk.Buffer // bins: the tiles' spans, then the list
	uni              *vk.Buffer // the frame uniform
	pipe             *vk.Pipeline
	cb               *vk.CommandBuffer
	bound            [nBind]*vk.Buffer // what the descriptor set points at
	bytes            []byte
	fence            *vk.Fence
}

// Options picks the instance and the device.
type Options struct {
	// Validation enables the Khronos validation layer (the tests use it).
	Validation bool
	// Filter narrows the choice of physical device; the zero Filter takes the
	// best one (discrete, integrated, ..., software).
	Filter vk.Filter
	// NoRTE runs the kernel without RoundingModeRTE even on a device that
	// reports it (the tests run both).
	NoRTE bool
}

// Renderer draws lists into a w × h RGBA frame in host-visible memory.
//
// Submit and Wait are the asynchronous pair; Draw is both. A Renderer is
// for one goroutine at a time.
type Renderer struct {
	in     *vk.Instance
	dev    *vk.Device
	info   vk.DeviceInfo
	shader *vk.Shader
	w, h   int
	target *vk.Buffer // the frame the kernel draws into: device-local memory
	stage  *vk.Buffer // host-visible copy of it, which is Pixels
	xfer   *vk.CommandBuffer
	slot   [slots]slot
	rte    bool // the kernel declares RoundingModeRTE
	head   int  // the oldest frame in flight
	n      int  // frames in flight
	closed bool

	lost     bool  // a frame timed out and may still run: nothing is destroyed (release)
	up       bool  // stage may differ from target: upload it before the next dispatch
	devNewer bool  // target is ahead of stage: Pixels downloads it first
	err      error // a download Pixels could not report, for the next call that can
	inplace  bool  // bin straight into the slots' memory (see pack)

	spans, list []uint32 // the binning's own storage, where it is not in place
}

// New opens the best Vulkan device and allocates a w × h frame.
func New(w, h int) (*Renderer, error) { return NewWith(w, h, Options{}) }

// NewWith is New with options.
func NewWith(w, h int, o Options) (*Renderer, error) {
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return nil, gpu2d.ErrSize
	}
	in, err := vk.Open(vk.Options{ApplicationName: "go-gfx/gpu2d", Validation: o.Validation})
	if errors.Is(err, vk.ErrIncompatibleDriver) {
		// A loader with no driver behind it (a CI runner): as good as no Vulkan.
		return nil, fmt.Errorf("%w: %w", ErrUnsupported, err)
	}
	if err != nil {
		return nil, err
	}
	r := &Renderer{in: in, w: w, h: h}
	pd, err := in.Pick(o.Filter)
	if err != nil {
		r.Close()
		return nil, err
	}
	if r.dev, err = pd.Open(); err != nil {
		r.Close()
		return nil, err
	}
	r.info = r.dev.Info()
	if err = r.check(w, h); err != nil {
		r.Close()
		return nil, err
	}
	kernel := kernels.plain
	if r.info.RoundingModeRTEFloat32 && !o.NoRTE {
		kernel, r.rte = kernels.rte, true
	}
	if len(kernel)%4 != 0 {
		r.Close()
		return nil, errors.New("gpu2d/vulkan: kernel_gen.spv is not a whole number of words")
	}
	spv := make([]uint32, len(kernel)/4)
	for i := range spv {
		spv[i] = binary.LittleEndian.Uint32(kernel[4*i:])
	}
	if r.shader, err = r.dev.NewShader(spv); err != nil {
		r.Close()
		return nil, tooLarge(err)
	}
	if err = r.newTarget(w, h); err != nil {
		r.Close()
		return nil, err
	}
	// Bin in place only into memory the CPU may read back cheaply: a software
	// rasterizer's. Binning counts into the spans, which reads them, and on a
	// discrete GPU host-visible memory is write-combined: those reads cost 21.6
	// ms a frame on a W7700 against 47 µs in ordinary memory (docs/NATIVE.md).
	r.inplace = r.info.Type == vk.DeviceTypeCPU
	if r.xfer, err = r.dev.NewCommandBuffer(); err != nil {
		r.Close()
		return nil, err
	}
	binds := []vk.Binding{
		{Binding: bInst, Kind: vk.StorageBuffer}, {Binding: bBins, Kind: vk.StorageBuffer},
		{Binding: bArena, Kind: vk.StorageBuffer}, {Binding: bPixels, Kind: vk.StorageBuffer},
		{Binding: bFrame, Kind: vk.UniformBuffer},
	}
	for i := range r.slot {
		s := &r.slot[i]
		if s.pipe, err = r.dev.NewComputePipeline(r.shader, "raster", binds); err != nil {
			r.Close()
			return nil, err
		}
		if s.cb, err = r.dev.NewCommandBuffer(); err != nil {
			r.Close()
			return nil, err
		}
		if s.uni, err = r.dev.NewBuffer(16, vk.UsageUniform); err != nil {
			r.Close()
			return nil, err
		}
	}
	return r, nil
}

// newTarget allocates the device-local frame and its host-visible stage for a
// w × h frame, replacing (and closing) the ones in use. The caller has waited
// for every frame in flight. The new frame is to be uploaded from the stage.
func (r *Renderer) newTarget(w, h int) error {
	n := 4 * uint64(w) * uint64(h)
	t, err := r.dev.NewDeviceBuffer(n, vk.UsageStorage|vk.UsageTransferSrc|vk.UsageTransferDst)
	if err != nil {
		return tooLarge(err)
	}
	// The stage is what the CPU reads back: host-cached memory where there is
	// any. NewBuffer's first host-visible type is write-combined on a discrete
	// GPU, which the CPU reads at about 220 MB/s (go-gfx/vulkan#7).
	st, err := r.dev.NewReadbackBuffer(n, vk.UsageTransferSrc|vk.UsageTransferDst)
	if err != nil {
		t.Close()
		return tooLarge(err)
	}
	if r.target != nil {
		r.target.Close()
		r.stage.Close()
	}
	r.target, r.stage = t, st
	r.up, r.devNewer = true, false
	return nil
}

// tooLarge maps the binding's size refusal to gpu2d's, which is what a caller
// (the window) tests for to draw that frame on the CPU.
func tooLarge(err error) error {
	if errors.Is(err, vk.ErrTooLarge) {
		return gpu2d.ErrTooLarge
	}
	return err
}

// check refuses a frame size the device cannot dispatch or hold: the target
// must fit maxStorageBufferRange (guaranteed only at 2^27 bytes) and the
// workgroup grid maxComputeWorkGroupCount. Both can be exceeded by a legal
// list, so the answer is ErrTooLarge, not a failed dispatch.
func (r *Renderer) check(w, h int) error {
	lim := r.info
	if 4*uint64(w)*uint64(h) > uint64(lim.MaxStorageBufferRange) || 4*uint64(w)*uint64(h) > lim.MaxMemoryAllocationSize {
		return gpu2d.ErrTooLarge
	}
	if uint64((w+15)/16) > uint64(lim.MaxComputeWorkGroupCount[0]) || uint64((h+7)/8) > uint64(lim.MaxComputeWorkGroupCount[1]) {
		return gpu2d.ErrTooLarge
	}
	return nil
}

// Name is the device's own name, such as "llvmpipe (LLVM 19.1.7, 256 bits)".
func (r *Renderer) Name() string { return r.info.Name }

// Info describes the physical device the renderer runs on.
func (r *Renderer) Info() vk.DeviceInfo { return r.info }

// RoundingModeRTE reports whether the kernel runs with RoundingModeRTE
// declared: true on a device that reports shaderRoundingModeRTEFloat32.
func (r *Renderer) RoundingModeRTE() bool { return r.rte }

// RobustBufferAccess reports whether the device was created with
// robustBufferAccess enabled, which go-gfx/vulkan does whenever it is offered:
// the second line of defence behind the kernel's in-range indices.
func (r *Renderer) RobustBufferAccess() bool { return r.dev.RobustBufferAccess() }

// Pixels is the frame, 4·w·h bytes of RGBA rows top to bottom: write the ground
// here before Draw and read the result here after it. It is a host-visible copy
// of the frame, which lives in device-local memory: with no frame in flight,
// Pixels first downloads whatever the device has drawn since the last call, and
// assumes the caller may write, so the next Submit uploads it. Calling it costs
// nothing when neither happened (a loop of Draw calls never calls it), and the
// slice stays valid until Resize or Close. While frames are in flight it
// returns the copy as it is and the device owns the frame: it is not current
// until the last Wait. A download that fails leaves the stage stale: it is then
// never uploaded (the device's frame is the newer one), every Submit and Draw
// returns the error until a Sync succeeds, and Pixels returns the stale copy.
func (r *Renderer) Pixels() []byte {
	if r.n == 0 && !r.closed {
		if err := r.Sync(); err != nil {
			r.err = err
		} else {
			r.up = true
		}
	}
	return r.stage.Bytes()
}

// Sync waits for every frame in flight and makes Pixels current: it downloads
// the device's frame into it if the device is ahead. Pixels does as much,
// without a way to return the error.
func (r *Renderer) Sync() error {
	if r.closed {
		return vk.ErrClosed
	}
	var first error
	for r.n > 0 {
		if err := r.Wait(); err != nil && first == nil {
			first = err
		}
	}
	if first != nil {
		return first
	}
	if r.lost {
		return ErrTimeout
	}
	if !r.devNewer {
		r.err = nil
		return nil
	}
	if err := r.xfer.Begin(); err != nil {
		return err
	}
	// Order the copy after the last dispatch (finished, but cheap) and make the
	// bytes visible to the host.
	if err := r.xfer.Barrier(); err != nil {
		return err
	}
	if err := r.xfer.Copy(r.stage, r.target, r.target.Size()); err != nil {
		return err
	}
	if err := r.xfer.End(); err != nil {
		return err
	}
	fe, err := r.dev.Submit(r.xfer)
	if err != nil {
		return err
	}
	if err := r.finish(&slot{fence: fe}); err != nil {
		r.release(fe, err)
		return err
	}
	fe.Close()
	r.devNewer = false
	r.err = nil
	return nil
}

// fit returns b if it holds n bytes, else a new buffer of at least n
// (doubling, up to the device's limit), closing b.
func (r *Renderer) fit(b *vk.Buffer, n int) (*vk.Buffer, error) {
	n = (max(n, 4) + 3) &^ 3 // the kernel reads whole words
	if b != nil && len(b.Bytes()) >= n {
		return b, nil
	}
	c := uint64(n)
	if b != nil {
		c = max(c, 2*uint64(len(b.Bytes())))
	}
	if c > uint64(r.info.MaxStorageBufferRange) && uint64(n) <= uint64(r.info.MaxStorageBufferRange) {
		c = uint64(r.info.MaxStorageBufferRange) // grow no further than the device allows
	}
	nb, err := r.dev.NewBuffer(c, vk.UsageStorage)
	if err != nil {
		return b, tooLarge(err)
	}
	if b != nil {
		b.Close()
	}
	return nb, nil
}

// Draw composites the list over the frame's current content and waits for
// the device to finish: Submit followed by Wait. It is ErrBusy while a frame
// is in flight.
func (r *Renderer) Draw(l *gpu2d.List) error {
	if r.closed {
		return vk.ErrClosed
	}
	if r.n > 0 {
		return ErrBusy
	}
	if err := r.Submit(l); err != nil {
		return err
	}
	return r.Wait()
}

// Submit bins and packs the list into the next free slot, sends it to the
// device and returns. The list is copied: the caller may Reset and record the
// next frame at once, which is the overlap. Frames run in submission order,
// each compositing over the previous one's result. At most two frames are in
// flight; a third Submit is ErrBusy.
//
// Every frame composites over the previous frame's pixels, so the device
// runs them one after the other, but both are queued: the second frame is
// packed and submitted while the first runs, and Submit never waits on a
// fence. The order is the device's: each command buffer begins with a
// go-gfx/vulkan Barrier (a memory dependency from the earlier writes, in
// submission order, to this frame's reads and writes of the pixels), so the
// second dispatch starts only when the first's writes are visible.
//
// The frame's memory (Pixels) belongs to the device until Wait has returned for
// every frame in flight: write the ground before the first Submit, read the
// result after the last Wait.
func (r *Renderer) Submit(l *gpu2d.List) error {
	if r.closed {
		return vk.ErrClosed
	}
	if r.lost {
		return ErrTimeout
	}
	if err := r.err; err != nil {
		return err // sticky: the stage is stale until a Sync succeeds
	}
	s, _, err := r.pack(l)
	if err != nil {
		return err
	}
	if err := s.cb.Begin(); err != nil {
		return err
	}
	// The pixels this frame reads and writes are the previous frame's: the
	// barrier orders this dispatch after every earlier write on the queue, a
	// frame still running included.
	if err := s.cb.Barrier(); err != nil {
		return err
	}
	if r.up {
		// The host may have written the frame (the ground): send it, and order
		// the dispatch after the copy.
		if err := s.cb.Copy(r.target, r.stage, r.target.Size()); err != nil {
			return err
		}
		if err := s.cb.Barrier(); err != nil {
			return err
		}
	}
	if err := s.cb.Dispatch(s.pipe, uint32((r.w+15)/16), uint32((r.h+7)/8), 1); err != nil {
		return tooLarge(err)
	}
	if err := s.cb.End(); err != nil {
		return err
	}
	fe, err := r.dev.Submit(s.cb)
	if err != nil {
		return err
	}
	s.fence = fe
	r.n++
	r.up, r.devNewer = false, true
	return nil
}

// release disposes of a fence after finish returned err. A fence that timed out
// may still be pending, and the specification forbids destroying it
// (vkDestroyFence, VUID-vkDestroyFence-fence-01120: "All queue submission
// commands that refer to fence must have completed execution"), as it forbids
// destroying the buffers and the device that pending work uses
// (VUID-vkDestroyBuffer-buffer-00922, VUID-vkDestroyDevice-device-00378). The
// binding has no vkDeviceWaitIdle to make them safe, so the renderer is lost:
// the fence is leaked on purpose, every later call returns ErrTimeout, and Close
// destroys nothing. Any other error (VK_ERROR_DEVICE_LOST included) means the
// work is over, and the fence is destroyed.
func (r *Renderer) release(fe *vk.Fence, err error) {
	if errors.Is(err, ErrTimeout) {
		r.lost = true
		return
	}
	fe.Close()
}

// finish blocks until s's frame has run. It leaves the fence in place.
func (r *Renderer) finish(s *slot) error {
	ok, err := s.fence.Wait(waitLimit)
	if err != nil {
		return err
	}
	if !ok {
		return ErrTimeout
	}
	return nil
}

// Resize changes the frame to w × h, waiting first for any frame in flight.
// The frame's content is not kept: write the ground again before drawing.
func (r *Renderer) Resize(w, h int) error {
	if r.closed {
		return vk.ErrClosed
	}
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return gpu2d.ErrSize
	}
	for r.n > 0 {
		_ = r.Wait()
	}
	if r.lost {
		return ErrTimeout
	}
	if w == r.w && h == r.h {
		return nil
	}
	if err := r.check(w, h); err != nil {
		return err
	}
	if err := r.newTarget(w, h); err != nil {
		return err
	}
	r.w, r.h = w, h
	for i := range r.slot {
		r.slot[i].bound[bPixels] = nil
	}
	return nil
}

// pack bins the list and copies it into the next free slot, then points the
// slot's descriptor set at its buffers.
func (r *Renderer) pack(l *gpu2d.List) (*slot, frame, error) {
	if r.n == slots {
		return nil, frame{}, ErrBusy
	}
	if w, h := l.Size(); w != r.w || h != r.h {
		return nil, frame{}, ErrSize
	}
	s := &r.slot[(r.head+r.n)%slots]
	across, down := l.Tiles()
	// Bin straight into the slot's buffer, which no frame in flight reads: the
	// spans fill its first 2·across·down words and the list follows. Bin
	// reuses the storage it is given when it is large enough, so the indices
	// land in device-visible memory with no copy; place copies only after Bin
	// had to grow past the buffer, and the grown buffer is reused afterwards.
	ns := 2 * across * down
	bw := words(s.bins)
	spans0, list0 := r.spans[:0], r.list[:0]
	if r.inplace && len(bw) >= ns {
		spans0, list0 = bw[:0:ns], bw[ns:]
	}
	spans, list, err := l.Bin(spans0, list0)
	if err != nil {
		return nil, frame{}, err
	}
	if !r.inplace {
		r.spans, r.list = spans, list // kept for the next frame; place copies them
	}
	if s.bins, err = r.place(s.bins, spans, list); err != nil {
		return nil, frame{}, err
	}
	s.bytes = gpu2d.AppendBytes(s.bytes[:0], l.Instances())
	if s.inst, err = r.fit(s.inst, len(s.bytes)); err != nil {
		return nil, frame{}, err
	}
	if s.data, err = r.fit(s.data, len(l.Data())); err != nil {
		return nil, frame{}, err
	}
	copy(s.data.Bytes(), l.Data())
	copy(s.inst.Bytes(), s.bytes)
	f := frame{W: uint32(r.w), H: uint32(r.h), Across: uint32(across), Tile: gpu2d.Tile}
	u := s.uni.Bytes()
	binary.LittleEndian.PutUint32(u[0:], f.W)
	binary.LittleEndian.PutUint32(u[4:], f.H)
	binary.LittleEndian.PutUint32(u[8:], f.Across)
	binary.LittleEndian.PutUint32(u[12:], f.Tile)
	for b, buf := range [nBind]*vk.Buffer{bInst: s.inst, bBins: s.bins, bArena: s.data, bPixels: r.target, bFrame: s.uni} {
		if s.bound[b] == buf {
			continue
		}
		if err := s.pipe.Bind(uint32(b), buf); err != nil {
			return nil, frame{}, err
		}
		s.bound[b] = buf
	}
	return s, f, nil
}

// Wait blocks until the oldest frame in flight has finished and returns the
// device's verdict on it; ErrNothingInFlight when there is none.
func (r *Renderer) Wait() error {
	if r.n == 0 {
		return ErrNothingInFlight
	}
	s := &r.slot[r.head]
	err := ErrTimeout
	if !r.lost {
		err = r.finish(s)
	}
	r.release(s.fence, err)
	s.fence = nil
	r.head = (r.head + 1) % slots
	r.n--
	return err
}

// InFlight is the number of submitted frames not yet waited for.
func (r *Renderer) InFlight() int { return r.n }

// littleEndian is whether the host's byte order is the device's. On a
// big-endian host the in-place view of the buffer as words would be wrong, so
// words returns nil there and every frame takes place's copy.
var littleEndian = func() bool { x := uint32(1); return *(*byte)(unsafe.Pointer(&x)) == 1 }()

// words views a buffer's memory as 32-bit words, nil for no buffer. A mapped
// Vulkan allocation is aligned far past four bytes (minMemoryMapAlignment is
// at least 64), so the view is aligned.
func words(b *vk.Buffer) []uint32 {
	if b == nil || !littleEndian {
		return nil
	}
	m := b.Bytes()
	return unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(m))), len(m)/4)
}

// place makes buf hold spans followed by list. When both already live in buf
// (Bin reused it) there is nothing to do; when buf is large enough both are
// copied in; otherwise a larger buffer is allocated and both are copied in,
// before the old one, which one of them may live in, is closed.
func (r *Renderer) place(buf *vk.Buffer, spans, list []uint32) (*vk.Buffer, error) {
	v := words(buf)
	if len(v) >= len(spans) && len(spans) > 0 && &v[0] == &spans[0] &&
		(len(list) == 0 || (len(v) >= len(spans)+len(list) && &v[len(spans)] == &list[0])) {
		return buf, nil
	}
	n := 4 * uint64(len(spans)+len(list))
	if buf != nil && uint64(len(buf.Bytes())) >= n {
		putWords(buf.Bytes(), spans)
		putWords(buf.Bytes()[4*len(spans):], list)
		return buf, nil
	}
	c := max(n, 4)
	if buf != nil {
		c = max(c, 2*uint64(len(buf.Bytes())))
	}
	if c > uint64(r.info.MaxStorageBufferRange) && n <= uint64(r.info.MaxStorageBufferRange) {
		c = uint64(r.info.MaxStorageBufferRange)
	}
	nb, err := r.dev.NewBuffer(c, vk.UsageStorage)
	if err != nil {
		return buf, tooLarge(err)
	}
	putWords(nb.Bytes(), spans)
	putWords(nb.Bytes()[4*len(spans):], list)
	if buf != nil {
		buf.Close()
	}
	return nb, nil
}

// putWords writes ws into b as little-endian 32-bit words.
func putWords(b []byte, ws []uint32) {
	b = b[:4*len(ws)] // one bounds check here instead of four per word
	for i, v := range ws {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
}

// Close waits for any frame in flight (the device must not read memory that is
// being freed), then releases the Vulkan objects. The renderer must not be used
// after; Close twice is harmless.
func (r *Renderer) Close() {
	if r.closed {
		return
	}
	for r.n > 0 {
		_ = r.Wait()
	}
	r.closed = true
	if r.lost {
		// A frame may still be running: destroy nothing (see release).
		return
	}
	for i := range r.slot {
		s := &r.slot[i]
		for _, b := range []*vk.Buffer{s.inst, s.bins, s.data, s.uni} {
			if b != nil {
				b.Close()
			}
		}
		if s.pipe != nil {
			s.pipe.Close()
		}
		if s.cb != nil {
			s.cb.Close()
		}
	}
	if r.target != nil {
		r.target.Close()
	}
	if r.stage != nil {
		r.stage.Close()
	}
	if r.xfer != nil {
		r.xfer.Close()
	}
	if r.shader != nil {
		r.shader.Close()
	}
	if r.dev != nil {
		_ = r.dev.Close()
	}
	_ = r.in.Close()
}
