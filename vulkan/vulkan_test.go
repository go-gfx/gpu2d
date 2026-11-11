// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package vulkan

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
	"github.com/go-gfx/gpu2d/internal/spirvgen"
	vk "github.com/go-gfx/vulkan"
)

// The tests drive a real Vulkan implementation: lavapipe in CI, RADV on
// fractal. Where there is none they skip, unless VULKAN_REQUIRE=1 says the
// machine must have one (the CI lane that installs lavapipe sets it).
//
//	VULKAN_REQUIRE=1     a missing loader or device is a failure, not a skip
//	VULKAN_VALIDATION=1  enable the Khronos validation layer and fail on ANY message
//	VULKAN_VENDOR=0x1002 only a device of this PCI vendor

// renderer returns a renderer, or skips where there is no Vulkan device.
func renderer(tb testing.TB, w, h int) *Renderer { return rendererRTE(tb, w, h, false) }

// rendererRTE is renderer, with the kernel without RoundingModeRTE when noRTE.
func rendererRTE(tb testing.TB, w, h int, noRTE bool) *Renderer {
	tb.Helper()
	o := Options{Validation: os.Getenv("VULKAN_VALIDATION") == "1", NoRTE: noRTE}
	if v := os.Getenv("VULKAN_VENDOR"); v != "" {
		id, err := strconv.ParseUint(v, 0, 32)
		if err != nil {
			tb.Fatalf("VULKAN_VENDOR=%q: %v", v, err)
		}
		o.Filter.VendorID = uint32(id)
	}
	if o.Validation {
		if !canCapture {
			tb.Skip("no way to capture the validation layer's output here")
		}
		stop := capture(tb)
		tb.Cleanup(func() { validationClean(tb, stop) }) // runs after Close below
	}
	r, err := NewWith(w, h, o)
	if (errors.Is(err, ErrUnsupported) || errors.Is(err, ErrNoDevice)) && os.Getenv("VULKAN_REQUIRE") != "1" {
		tb.Skip(err)
	}
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(r.Close)
	return r
}

// ground is a deterministic frame to draw over, opaque or not.
func ground(w, h int, opaque bool) []byte {
	g := make([]byte, 4*w*h)
	for i := range g {
		g[i] = uint8(i * 7)
		if opaque && i%4 == 3 {
			g[i] = 255
		}
	}
	return g
}

// parity draws random scenes on r over both grounds and compares each with
// gpu2d.Rasterize. It returns the worst |Δ| and the count of differing
// channels (summed over both grounds), and fails when a scene barely draws.
func parity(t *testing.T, r *Renderer, w, h, seeds int) (worst, differing int) {
	t.Helper()
	for _, opaque := range []bool{true, false} {
		g := ground(w, h, opaque)
		var w0, d0, drawn int
		kinds := map[gpu2d.Kind]int{}
		for seed := range uint64(seeds) {
			l, _ := gpu2d.NewList(w, h)
			scenes.Random(l, seed, 40)
			drawn += len(l.Instances())
			for _, in := range l.Instances() {
				kinds[in.Kind]++
			}
			want := append([]byte(nil), g...)
			if err := l.Rasterize(want); err != nil {
				t.Fatal(err)
			}
			copy(r.Pixels(), g)
			if err := r.Draw(l); err != nil {
				t.Fatal(err)
			}
			got := r.Pixels()
			for i := range want {
				d := int(want[i]) - int(got[i])
				if d < 0 {
					d = -d
				}
				if d > 0 {
					d0++
				}
				w0 = max(w0, d)
			}
		}
		t.Logf("opaque ground %v: %d scenes, %d instances, max |Δ| %d, %d of %d channels differ",
			opaque, seeds, drawn, w0, d0, seeds*4*w*h)
		t.Logf("instances by kind (fill, outline, mask, alpha, image): %d %d %d %d %d",
			kinds[gpu2d.KindFill], kinds[gpu2d.KindOutline], kinds[gpu2d.KindMask], kinds[gpu2d.KindAlpha], kinds[gpu2d.KindImage])
		for k := gpu2d.KindFill; k <= gpu2d.KindImage; k++ {
			if kinds[k] < seeds/4 {
				t.Fatalf("kind %d drawn only %d times", k, kinds[k])
			}
		}
		if drawn < seeds*10 {
			t.Fatalf("only %d instances: the scenes barely draw", drawn)
		}
		worst, differing = max(worst, w0), differing+d0
	}
	return worst, differing
}

// TestMatchesReference holds the device to gpu2d.Rasterize on random scenes
// over an opaque and a translucent ground. The blend is the reference's
// integer arithmetic, and the coverage is f32 against f64: the bar is zero
// differing channels, as on Metal. A device that misses it is named here.
//
// It runs twice: the kernel with RoundingModeRTE declared, which the renderer
// picks on a device that reports it, and the plain one, which it picks
// elsewhere. Both must be exact.
func TestMatchesReference(t *testing.T) {
	const w, h, seeds = 64, 48, 300
	for _, noRTE := range []bool{false, true} {
		r := rendererRTE(t, w, h, noRTE)
		i := r.Info()
		t.Logf("device %q type=%v vendor=%#x api=%v driver=%q %q; RoundingModeRTE offered %v, kernel declares it %v; robustBufferAccess %v",
			i.Name, i.Type, i.VendorID, i.APIVersion, i.DriverName, i.DriverInfo,
			i.RoundingModeRTEFloat32, r.RoundingModeRTE(), r.RobustBufferAccess())
		if want := i.RoundingModeRTEFloat32 && !noRTE; r.RoundingModeRTE() != want {
			t.Fatalf("noRTE=%v, offered %v, but the kernel declares RoundingModeRTE=%v", noRTE, i.RoundingModeRTEFloat32, r.RoundingModeRTE())
		}
		worst, differing := parity(t, r, w, h, seeds)
		if worst != 0 {
			t.Fatalf("%s (RTE %v): max |Δ| %d, %d channels differ from Rasterize", i.Name, r.RoundingModeRTE(), worst, differing)
		}
		r.Close()
	}
}

// TestBreakCheckPerturbedBlend proves the parity test can fail: the generated
// kernel with the rounding of its red channel changed must be caught.
func TestBreakCheckPerturbedBlend(t *testing.T) {
	const old = "(sr * ca + dst.x * ia + 127u) / 255u"
	if !strings.Contains(gpu2d.KernelWGSL, old) {
		t.Fatal("kernel.wgsl no longer has the blend line this check perturbs")
	}
	bad := strings.Replace(gpu2d.KernelWGSL, old, "(sr * ca + dst.x * ia + 100u) / 255u", 1)
	plain, err := spirvgen.Kernel(bad, false)
	if err != nil {
		t.Fatal(err)
	}
	rte, err := spirvgen.Kernel(bad, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func(k struct{ plain, rte []byte }) { kernels = k }(kernels)
	kernels = struct{ plain, rte []byte }{plain, rte}
	const w, h = 64, 48
	r := renderer(t, w, h)
	worst, differing := parity(t, r, w, h, 60)
	t.Logf("perturbed blend: max |Δ| %d, %d channels differ", worst, differing)
	if worst == 0 {
		t.Fatal("a perturbed blend still matched the reference: TestMatchesReference proves nothing")
	}
}

// TestBreakCheckPerturbedCorner proves the corner-heavy parity can fail: the
// kernel's fill coverage rounded the other way (+ 2 for + 1, which moves the
// byte by one where a pixel is antialiased) must be caught by TestCornerHeavy's
// scenes, and by the random ones, which now reach radius 64.
func TestBreakCheckPerturbedCorner(t *testing.T) {
	const old = "let t = ca * h + 1u;"
	if strings.Count(gpu2d.KernelWGSL, old) != 1 {
		t.Fatal("kernel.wgsl no longer has the coverage line this check perturbs")
	}
	bad := strings.Replace(gpu2d.KernelWGSL, old, "let t = ca * h + 2u;", 1)
	plain, err := spirvgen.Kernel(bad, false)
	if err != nil {
		t.Fatal(err)
	}
	rte, err := spirvgen.Kernel(bad, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func(k struct{ plain, rte []byte }) { kernels = k }(kernels)
	kernels = struct{ plain, rte []byte }{plain, rte}
	r := rendererRTE(t, 64, 48, false)
	tl := newTally()
	for seed := range uint64(10) {
		w, h := 200, 200
		if err := r.Resize(w, h); err != nil {
			t.Fatal(err)
		}
		l, _ := gpu2d.NewList(w, h)
		scenes.Corners(l, seed, 64)
		tl.compare(t, r, l, ground(w, h, seed%2 == 0), "seed")
	}
	t.Logf("perturbed corner: max |Δ| %d, %d channels differ", tl.worst, tl.differing)
	if tl.differing == 0 {
		t.Fatal("a perturbed corner coverage still matched the reference: TestCornerHeavy proves nothing")
	}
}

// TestBreakCheckWrongBuffer binds the pixels' descriptor to a decoy: the
// reference result must not appear in Pixels, and the decoy must be drawn on.
func TestBreakCheckWrongBuffer(t *testing.T) {
	const w, h = 32, 32
	r := renderer(t, w, h)
	decoy, err := r.dev.NewBuffer(4*w*h, vk.UsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer decoy.Close()
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: w, H: h}, gpu2d.RGBA{R: 255, A: 255})
	s, _, err := r.pack(l)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.pipe.Bind(bPixels, decoy); err != nil {
		t.Fatal(err)
	}
	s.bound[bPixels] = decoy
	for i := range r.Pixels() {
		r.Pixels()[i] = 0xAA
	}
	// Submit packs again into the same free slot and rebinds the real target,
	// so record by hand what Submit does after pack.
	if err := s.cb.Begin(); err != nil {
		t.Fatal(err)
	}
	if err := s.cb.Dispatch(s.pipe, uint32((w+15)/16), uint32((h+7)/8), 1); err != nil {
		t.Fatal(err)
	}
	if err := s.cb.End(); err != nil {
		t.Fatal(err)
	}
	if s.fence, err = r.dev.Submit(s.cb); err != nil {
		t.Fatal(err)
	}
	r.n++
	if err := r.Wait(); err != nil {
		t.Fatal(err)
	}
	for _, b := range r.Pixels() {
		if b != 0xAA {
			t.Fatalf("Pixels was written (%#x) although its binding pointed at the decoy", b)
		}
	}
	if decoy.Bytes()[0] != 255 || decoy.Bytes()[3] != 255 {
		t.Fatalf("the decoy was not drawn on: % x", decoy.Bytes()[:4])
	}
}

func TestRefusals(t *testing.T) {
	if _, err := New(0, 1); !errors.Is(err, gpu2d.ErrSize) {
		t.Fatalf("New(0, 1): %v", err)
	}
	r := renderer(t, 8, 8)
	l, _ := gpu2d.NewList(9, 8)
	if err := r.Draw(l); !errors.Is(err, ErrSize) {
		t.Fatalf("a list of another size: %v", err)
	}
	if err := r.Resize(0, 8); !errors.Is(err, gpu2d.ErrSize) {
		t.Fatalf("Resize(0, 8): %v", err)
	}
}

// TestDeviceLimits: a frame past a limit of the device is gpu2d.ErrTooLarge,
// at New, at Resize, and when a slot's buffers would pass the storage range.
func TestDeviceLimits(t *testing.T) {
	r := renderer(t, 8, 8)
	lim := r.Info()
	t.Logf("maxStorageBufferRange %d, maxMemoryAllocationSize %d, maxComputeWorkGroupCount %v",
		lim.MaxStorageBufferRange, lim.MaxMemoryAllocationSize, lim.MaxComputeWorkGroupCount)
	side := 1
	for 4*uint64(side)*uint64(side) <= uint64(lim.MaxStorageBufferRange) {
		side *= 2
	}
	if side <= gpu2d.MaxCoord {
		if err := r.Resize(side, side); !errors.Is(err, gpu2d.ErrTooLarge) {
			t.Fatalf("Resize(%d, %d) past maxStorageBufferRange: %v", side, side, err)
		}
		// The same device as r (VULKAN_VENDOR), not New's best one.
		o := Options{Filter: vk.Filter{VendorID: r.Info().VendorID}}
		if nr, err := NewWith(side, side, o); !errors.Is(err, gpu2d.ErrTooLarge) {
			if nr != nil {
				nr.Close()
			}
			t.Fatalf("New(%d, %d) past maxStorageBufferRange: %v", side, side, err)
		}
	}
	if w := 16*uint64(lim.MaxComputeWorkGroupCount[0]) + 1; w <= gpu2d.MaxCoord && 4*w <= uint64(lim.MaxStorageBufferRange) {
		if err := r.Resize(int(w), 1); !errors.Is(err, gpu2d.ErrTooLarge) {
			t.Fatalf("Resize(%d, 1) past maxComputeWorkGroupCount: %v", w, err)
		}
	}
	if r.w != 8 || r.h != 8 {
		t.Fatalf("a refused Resize changed the frame to %d x %d", r.w, r.h)
	}
	if _, err := r.fit(nil, int(lim.MaxStorageBufferRange)+4); !errors.Is(err, gpu2d.ErrTooLarge) {
		t.Fatalf("a slot buffer past maxStorageBufferRange: %v", err)
	}
	// Still usable after the refusals.
	l, _ := gpu2d.NewList(8, 8)
	l.FillRect(gpu2d.Rect{W: 8, H: 8}, gpu2d.RGBA{G: 255, A: 255})
	if err := r.Draw(l); err != nil {
		t.Fatal(err)
	}
}

func TestResize(t *testing.T) {
	r := renderer(t, 16, 16)
	if err := r.Resize(16, 16); err != nil {
		t.Fatal(err)
	}
	if err := r.Resize(40, 24); err != nil {
		t.Fatal(err)
	}
	if len(r.Pixels()) != 4*40*24 {
		t.Fatalf("%d bytes after Resize(40, 24)", len(r.Pixels()))
	}
	l, _ := gpu2d.NewList(40, 24)
	l.FillRect(gpu2d.Rect{W: 40, H: 24}, gpu2d.RGBA{B: 255, A: 255})
	if err := r.Draw(l); err != nil {
		t.Fatal(err)
	}
	if p := r.Pixels(); p[4*(40*24-1)+2] != 255 {
		t.Fatalf("last pixel % x", p[4*(40*24-1):])
	}
	// Resize with a frame in flight waits for it.
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	if err := r.Resize(8, 8); err != nil {
		t.Fatal(err)
	}
	if r.InFlight() != 0 {
		t.Fatal("Resize left a frame in flight")
	}
}

// translucent returns a list of overlapping translucent fills: the result
// depends on the order frames are composited in.
func translucent(w, h int, shift int) *gpu2d.List {
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{X: shift, Y: 0, W: w / 2, H: h}, gpu2d.RGBA{R: 200, G: 30, B: 10, A: 140})
	l.FillRoundRect(gpu2d.Rect{X: 3, Y: shift, W: w - 6, H: h / 2}, 5, gpu2d.RGBA{R: 10, G: 40, B: 220, A: 120})
	return l
}

// TestTwoFramesInFlightCompositeInOrder submits two lists back to back and
// holds the result to the CPU compositing the same two lists in the same
// order, byte for byte, over many rounds (an ordering race would show up as a
// difference on some round).
func TestTwoFramesInFlightCompositeInOrder(t *testing.T) {
	const w, h = 64, 48
	r := renderer(t, w, h)
	a, b := translucent(w, h, 4), translucent(w, h, 9)
	for round := range 50 {
		want := make([]byte, 4*w*h)
		for i := range want {
			want[i] = uint8(i*5 + round)
		}
		copy(r.Pixels(), want)
		if err := a.Rasterize(want); err != nil {
			t.Fatal(err)
		}
		if err := b.Rasterize(want); err != nil {
			t.Fatal(err)
		}
		if err := r.Submit(a); err != nil {
			t.Fatal(err)
		}
		if err := r.Submit(b); err != nil {
			t.Fatal(err)
		}
		if r.InFlight() != 2 {
			t.Fatalf("%d in flight", r.InFlight())
		}
		if err := r.Submit(a); !errors.Is(err, ErrBusy) {
			t.Fatalf("a third Submit: %v", err)
		}
		if err := r.Draw(a); !errors.Is(err, ErrBusy) {
			t.Fatalf("Draw while in flight: %v", err)
		}
		for range 2 {
			if err := r.Wait(); err != nil {
				t.Fatal(err)
			}
		}
		if r.InFlight() != 0 {
			t.Fatalf("%d in flight after Wait", r.InFlight())
		}
		got := r.Pixels()
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("round %d channel %d: CPU %d, GPU %d", round, i, want[i], got[i])
			}
		}
	}
}

// TestFramesReallyOverlap: Submit of the second frame returns while the first
// is still running on the device (its fence is not signalled), Submit does not
// wait on a fence, and the composite is still the CPU's, byte for byte. The
// frames are large enough that the device takes far longer than a Submit.
func TestFramesReallyOverlap(t *testing.T) {
	const w, h = 1920, 2834
	r := renderer(t, w, h)
	a, _ := gpu2d.NewList(w, h)
	gpu2d.WidgetFrame(a)
	b := translucent(w, h, 40)
	ground := make([]byte, 4*w*h)
	for i := range ground {
		ground[i] = uint8(i*5 + i>>9)
	}
	want := append([]byte(nil), ground...)
	if err := a.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	if err := b.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	overlapped := 0
	const rounds = 5
	for round := range rounds {
		copy(r.Pixels(), ground)
		if err := r.Submit(a); err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		if err := r.Submit(b); err != nil {
			t.Fatal(err)
		}
		took := time.Since(t0)
		done, err := r.slot[r.head].fence.Done()
		if err != nil {
			t.Fatal(err)
		}
		if !done {
			overlapped++
		}
		t.Logf("round %d: Submit of the second frame took %v; the first still running: %v", round, took, !done)
		for range 2 {
			if err := r.Wait(); err != nil {
				t.Fatal(err)
			}
		}
		if !bytes.Equal(r.Pixels(), want) {
			t.Fatalf("round %d: the overlapped frames are not the CPU's composite", round)
		}
	}
	if overlapped == 0 {
		t.Fatal("the first frame had always finished when the second was submitted: nothing overlapped")
	}
}

// TestSubmitCopiesTheList records frame A, submits it, overwrites the list
// with frame B at once, and checks the device still drew A.
func TestSubmitCopiesTheList(t *testing.T) {
	const w, h = 32, 32
	r := renderer(t, w, h)
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: w, H: h}, gpu2d.RGBA{R: 255, A: 255})
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	l.Reset()
	l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: w, H: h}, gpu2d.RGBA{B: 255, A: 255})
	if err := r.Wait(); err != nil {
		t.Fatal(err)
	}
	if p := r.Pixels(); p[0] != 255 || p[2] != 0 {
		t.Fatalf("pixel 0 = %v: the device drew the list as overwritten", p[:4])
	}
}

func TestMisuse(t *testing.T) {
	r := renderer(t, 8, 8)
	if err := r.Wait(); !errors.Is(err, ErrNothingInFlight) {
		t.Fatalf("Wait without Submit: %v", err)
	}
	l, _ := gpu2d.NewList(8, 8)
	l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: 8, H: 8}, gpu2d.RGBA{G: 255, A: 255})
	if err := r.Submit(l); err != nil {
		t.Fatal(err)
	}
	if err := r.Submit(l); err != nil { // the second slot: allowed
		t.Fatal(err)
	}
	other, _ := gpu2d.NewList(9, 8)
	if err := r.Submit(other); !errors.Is(err, ErrBusy) {
		t.Fatalf("Submit with both slots busy: %v", err)
	}
	// Close while two frames are in flight must drain, not crash.
	r.Close()
	if r.InFlight() != 0 {
		t.Fatal("frames in flight after Close")
	}
	r.Close() // twice is harmless
	if err := r.Submit(l); err == nil {
		t.Fatal("Submit after Close succeeded")
	}
	if err := r.Draw(l); err == nil {
		t.Fatal("Draw after Close succeeded")
	}
	if err := r.Resize(4, 4); err == nil {
		t.Fatal("Resize after Close succeeded")
	}
}

func TestSubmitRefusesAWrongSizeList(t *testing.T) {
	r := renderer(t, 8, 8)
	other, _ := gpu2d.NewList(9, 8)
	if err := r.Submit(other); !errors.Is(err, ErrSize) {
		t.Fatalf("a list of another size: %v", err)
	}
	if r.InFlight() != 0 {
		t.Fatal("a refused Submit is in flight")
	}
}

// BenchmarkWidgetFrame is the shared widget frame on the device: binning, the
// copies into host-visible memory, the dispatch and the wait for it.
func BenchmarkWidgetFrame(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}} {
		b.Run(s.name, func(b *testing.B) {
			r := renderer(b, s.w, s.h)
			l, _ := gpu2d.NewList(s.w, s.h)
			gpu2d.WidgetFrame(l)
			b.ReportMetric(float64(len(l.Instances())), "instances")
			b.ReportAllocs()
			for b.Loop() {
				if err := r.Draw(l); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkOverlap is the point of Submit: per frame the CPU records the next
// list while the device draws the last. "draw" is the synchronous baseline,
// "overlap" records the next frame between Submit and Wait. The CPU work is
// modelled as a spin of cpuMs (0 and 1.5 ms), so the table does not depend on
// how fast the recorder is.
func BenchmarkOverlap(b *testing.B) {
	const w, h = 960, 2834
	for _, cpu := range []time.Duration{0, 1500 * time.Microsecond} {
		for _, overlap := range []bool{false, true} {
			name := "draw"
			if overlap {
				name = "overlap"
			}
			b.Run(fmt.Sprintf("cpu%dus/%s", cpu.Microseconds(), name), func(b *testing.B) {
				r := renderer(b, w, h)
				l, _ := gpu2d.NewList(w, h)
				gpu2d.WidgetFrame(l)
				b.ReportAllocs()
				for b.Loop() {
					if !overlap {
						spin(cpu)
						if err := r.Draw(l); err != nil {
							b.Fatal(err)
						}
						continue
					}
					if err := r.Submit(l); err != nil {
						b.Fatal(err)
					}
					spin(cpu) // the next frame, recorded while the device draws
					if err := r.Wait(); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// spin keeps one core busy for d.
func spin(d time.Duration) {
	for t := time.Now(); time.Since(t) < d; {
	}
}
