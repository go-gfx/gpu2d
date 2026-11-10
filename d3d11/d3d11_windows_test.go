// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build windows

package d3d11

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/hlslgen"
	"github.com/go-gfx/gpu2d/internal/scenes"
	dx "github.com/go-mswin/d3d11"
)

// The tests run on WARP, the software device, so that CI is deterministic;
// D3D11_HARDWARE=1 runs them on the default adapter instead. Where there is no
// device they skip, unless D3D11_REQUIRE=1 says the machine must have one (the
// CI lane sets it).

// renderer returns a renderer, or skips where there is no device.
func renderer(tb testing.TB, w, h int) *Renderer { return rendererWith(tb, w, h, Options{}) }

func rendererWith(tb testing.TB, w, h int, o Options) *Renderer {
	tb.Helper()
	o.Software = os.Getenv("D3D11_HARDWARE") != "1"
	r, err := NewWith(w, h, o)
	if (errors.Is(err, ErrUnsupported) || errors.Is(err, ErrNoDevice)) && os.Getenv("D3D11_REQUIRE") != "1" {
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
// differing channels, as on Metal and Vulkan. A device that misses it is named
// here.
func TestMatchesReference(t *testing.T) {
	const w, h, seeds = 64, 48, 300
	r := renderer(t, w, h)
	t.Logf("device %q software=%v feature level %v, buffer limit %d bytes", r.Name(), r.Software(), r.FeatureLevel(), r.MaxBufferBytes())
	worst, differing := parity(t, r, w, h, seeds)
	if worst != 0 {
		t.Fatalf("%s: max |Δ| %d, %d channels differ from Rasterize", r.Name(), worst, differing)
	}
}

// TestBreakCheckPerturbedKernel proves the parity test can fail: the kernel
// with the rounding of its red channel changed in the WGSL, emitted as HLSL and
// compiled to DXBC here by D3DCompile, must be caught.
func TestBreakCheckPerturbedKernel(t *testing.T) {
	const old = "(sr * ca + dst.x * ia + 127u) / 255u"
	if !strings.Contains(gpu2d.KernelWGSL, old) {
		t.Fatal("kernel.wgsl no longer has the blend line this check perturbs")
	}
	hlsl, err := hlslgen.Kernel(strings.Replace(gpu2d.KernelWGSL, old, "(sr * ca + dst.x * ia + 100u) / 255u", 1))
	if err != nil {
		t.Fatal(err)
	}
	dxbc, _, err := hlslgen.Build(hlsl)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(dxbc, KernelDXBC) {
		t.Fatal("the perturbed kernel compiled to the committed DXBC")
	}
	defer func(k []byte) { kernel = k }(kernel)
	kernel = dxbc
	const w, h = 64, 48
	r := renderer(t, w, h)
	worst, differing := parity(t, r, w, h, 60)
	t.Logf("perturbed blend: max |Δ| %d, %d channels differ", worst, differing)
	if worst == 0 {
		t.Fatal("a perturbed blend still matched the reference: TestMatchesReference proves nothing")
	}
}

// TestBreakCheckWrongUAV binds the pixels' register u3 to a decoy: the
// reference result must not appear in Pixels, and the decoy must be drawn on.
func TestBreakCheckWrongUAV(t *testing.T) {
	const w, h = 32, 32
	r := renderer(t, w, h)
	decoy, err := r.dev.CreateBuffer(dx.BufferDesc{Kind: dx.KindStructured, Size: 4 * w * h, Stride: 4, UAV: true, Initial: make([]byte, 4*w*h)})
	if err != nil {
		t.Fatal(err)
	}
	defer decoy.Close()
	duav, err := decoy.UAV()
	if err != nil {
		t.Fatal(err)
	}
	defer duav.Close()
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{X: 0, Y: 0, W: w, H: h}, gpu2d.RGBA{R: 255, A: 255})
	s, err := r.pack(l)
	if err != nil {
		t.Fatal(err)
	}
	for i := range r.Pixels() {
		r.Pixels()[i] = 0xAA
	}
	if err := r.upload(); err != nil {
		t.Fatal(err)
	}
	across, _ := l.Tiles()
	// What Submit does after upload, with the wrong view.
	if err := r.record(s, duav, across); err != nil {
		t.Fatal(err)
	}
	r.n++
	if err := r.Wait(); err != nil {
		t.Fatal(err)
	}
	for _, b := range r.Pixels() {
		if b != 0xAA {
			t.Fatalf("Pixels was written (%#x) although u3 pointed at the decoy", b)
		}
	}
	d, err := r.ctx.Readback(decoy, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if d[0] != 255 || d[3] != 255 {
		t.Fatalf("the decoy was not drawn on: % x", d[:4])
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
// at New, at Resize, and when a slot's buffers would pass the buffer limit.
func TestDeviceLimits(t *testing.T) {
	r := renderer(t, 8, 8)
	lim := r.MaxBufferBytes()
	t.Logf("device buffer limit %d bytes (%d MiB)", lim, lim>>20)
	if r.Software() && lim != 128<<20 {
		t.Fatalf("WARP's guaranteed limit is 128 MiB, got %d", lim)
	}
	side := 1
	for 4*uint64(side)*uint64(side) <= lim {
		side++ // the first square frame that does not fit
	}
	if side <= gpu2d.MaxCoord {
		if err := r.Resize(side, side); !errors.Is(err, gpu2d.ErrTooLarge) {
			t.Fatalf("Resize(%d, %d) past the buffer limit: %v", side, side, err)
		}
		if _, err := NewWith(side, side, Options{Software: r.Software()}); !errors.Is(err, gpu2d.ErrTooLarge) {
			t.Fatalf("New(%d, %d) past the buffer limit: %v", side, side, err)
		}
	}
	if w := 16*dx.MaxDispatchGroups + 1; uint64(w) <= gpu2d.MaxCoord && 4*uint64(w) <= lim {
		if err := r.Resize(w, 1); !errors.Is(err, gpu2d.ErrTooLarge) {
			t.Fatalf("Resize(%d, 1) past the dispatch grid: %v", w, err)
		}
	}
	if r.w != 8 || r.h != 8 {
		t.Fatalf("a refused Resize changed the frame to %d x %d", r.w, r.h)
	}
	l, _ := gpu2d.NewList(8, 8)
	l.FillRect(gpu2d.Rect{W: 8, H: 8}, gpu2d.RGBA{G: 255, A: 255})
	if err := r.Draw(l); err != nil {
		t.Fatalf("not usable after the refusals: %v", err)
	}
}

// TestSlotBufferPastTheLimit lowers the limit with Options.MaxBufferBytes: the
// frame itself fits, an instance buffer does not, and Submit says ErrTooLarge,
// leaves nothing in flight and the renderer usable.
func TestSlotBufferPastTheLimit(t *testing.T) {
	const w, h = 128, 128 // 65536 bytes: exactly the limit
	r := rendererWith(t, w, h, Options{MaxBufferBytes: 4 * w * h})
	if _, err := NewWith(w+1, h, Options{Software: true, MaxBufferBytes: 4 * w * h}); !errors.Is(err, gpu2d.ErrTooLarge) {
		t.Fatalf("a frame one column past the limit: %v", err)
	}
	big, _ := gpu2d.NewList(w, h)
	for range 4*w*h/gpu2d.InstanceSize + 1 { // more instances than the limit holds
		big.FillRect(gpu2d.Rect{W: 1, H: 1}, gpu2d.RGBA{R: 1, A: 255})
	}
	if err := r.Submit(big); !errors.Is(err, gpu2d.ErrTooLarge) {
		t.Fatalf("an instance buffer past the limit: %v", err)
	}
	if r.InFlight() != 0 {
		t.Fatal("a refused Submit is in flight")
	}
	l, _ := gpu2d.NewList(w, h)
	l.FillRect(gpu2d.Rect{W: w, H: h}, gpu2d.RGBA{B: 255, A: 255})
	if err := r.Draw(l); err != nil {
		t.Fatal(err)
	}
	if p := r.Pixels(); p[2] != 255 {
		t.Fatalf("pixel 0 % x", p[:4])
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
// order, byte for byte, over many rounds. The second frame draws over the
// first's result on the device, with no read-back between them.
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

// TestSecondSubmitDrawsOverTheDevicesFrame: with the first frame still in
// flight, Pixels (the host copy) is not read: a later write to it must not
// reach the second frame. Then, with nothing in flight, a write to Pixels is
// the next frame's ground.
func TestSecondSubmitDrawsOverTheDevicesFrame(t *testing.T) {
	const w, h = 32, 32
	r := renderer(t, w, h)
	a, b := translucent(w, h, 4), translucent(w, h, 9)
	g := ground(w, h, true)
	want := append([]byte(nil), g...)
	if err := a.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	if err := b.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	copy(r.Pixels(), g)
	if err := r.Submit(a); err != nil {
		t.Fatal(err)
	}
	for i := range r.Pixels() { // scribble on the host copy mid-flight
		r.Pixels()[i] = 0x55
	}
	if err := r.Submit(b); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := r.Wait(); err != nil {
			t.Fatal(err)
		}
	}
	if !bytes.Equal(r.Pixels(), want) {
		t.Fatal("the second frame did not draw over the device's first frame")
	}
	// A fresh ground written with nothing in flight is uploaded.
	copy(r.Pixels(), g)
	want = append(want[:0], g...)
	if err := b.Rasterize(want); err != nil {
		t.Fatal(err)
	}
	if err := r.Draw(b); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(r.Pixels(), want) {
		t.Fatal("a ground written to Pixels was not the next frame's")
	}
}

// TestBuffersAreReusedAndGrow: a slot's input buffers are kept from frame to
// frame, grow when a list needs more (never shrink), and every frame, before
// and after a growth, is still the reference's.
func TestBuffersAreReusedAndGrow(t *testing.T) {
	const w, h = 96, 64
	r := renderer(t, w, h)
	g := ground(w, h, false)
	frame := func(seed uint64, n int) *gpu2d.List {
		l, _ := gpu2d.NewList(w, h)
		scenes.Random(l, seed, n)
		return l
	}
	check := func(l *gpu2d.List) {
		t.Helper()
		want := append([]byte(nil), g...)
		if err := l.Rasterize(want); err != nil {
			t.Fatal(err)
		}
		copy(r.Pixels(), g)
		if err := r.Draw(l); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(r.Pixels(), want) {
			t.Fatal("a frame differs from the reference")
		}
	}
	check(frame(1, 40)) // slot 0
	first := r.slot[0].in[0].buf
	size := first.Size()
	for seed := uint64(2); seed < 12; seed++ {
		check(frame(seed, 10)) // slot 0 again, no more than before
	}
	if r.slot[0].in[0].buf != first {
		t.Fatal("the instance buffer was recreated although it was large enough")
	}
	check(frame(99, 4000)) // far more instances: it must grow
	if r.slot[0].in[0].buf == first || r.slot[0].in[0].buf.Size() <= size {
		t.Fatalf("the instance buffer did not grow: %d then %d", size, r.slot[0].in[0].buf.Size())
	}
	grown := r.slot[0].in[0].buf
	check(frame(3, 10))
	if r.slot[0].in[0].buf != grown {
		t.Fatal("the instance buffer shrank or was recreated")
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
	if err := r.Submit(l); !errors.Is(err, ErrClosed) {
		t.Fatalf("Submit after Close: %v", err)
	}
	if err := r.Draw(l); !errors.Is(err, ErrClosed) {
		t.Fatalf("Draw after Close: %v", err)
	}
	if err := r.Resize(4, 4); !errors.Is(err, ErrClosed) {
		t.Fatalf("Resize after Close: %v", err)
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

// TestEmptyList: no instance, no data: the buffers are padded to one element
// and the frame comes back as it went.
func TestEmptyList(t *testing.T) {
	const w, h = 20, 12
	r := renderer(t, w, h)
	g := ground(w, h, false)
	copy(r.Pixels(), g)
	l, _ := gpu2d.NewList(w, h)
	if err := r.Draw(l); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(r.Pixels(), g) {
		t.Fatal("an empty list changed the frame")
	}
}

// BenchmarkWidgetFrame is the shared widget frame on the device (WARP on CI
// unless D3D11_HARDWARE=1): binning, the input buffers' creation with their
// data, the target's upload, the dispatch, the wait, and the read-back into
// Pixels. The custom metrics split the time per frame into those parts, so the
// cost of Direct3D 11's copies is read off, not guessed.
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
			r.ResetStats()
			for b.Loop() {
				if err := r.Draw(l); err != nil {
					b.Fatal(err)
				}
			}
			st := r.Stats()
			per := func(d time.Duration) float64 { return float64(d.Microseconds()) / float64(max(st.Frames, 1)) }
			b.ReportMetric(per(st.Pack), "pack-µs")
			b.ReportMetric(per(st.Upload), "upload-µs")
			b.ReportMetric(per(st.GPU), "gpu-µs")
			b.ReportMetric(per(st.Readback), "readback-µs")
		})
	}
}
