// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build js && wasm

// The page presentbench loads: it draws gpu2d.WidgetFrame into the page's
// canvas through webgpu.Renderer, Draw then Present, frame after frame on
// requestAnimationFrame, and reports the time per frame in
// window.__nsPerFrame. ?format=rgba configures the canvas as rgba8unorm
// (Present copies), ?format=bgra as bgra8unorm (Present draws); ?w=&h= set
// the size, ?frames= the frames timed after a warm-up. ?mode=compute draws
// with the compute kernel (webgpu.NewCompute) instead of fixed-function
// blending.
package main

import (
	"context"
	"strconv"
	"syscall/js"
	"time"

	"github.com/go-gfx/gpu2d"
	gpu2dwebgpu "github.com/go-gfx/gpu2d/webgpu"
	wg "github.com/go-gfx/webgpu"
)

func param(q js.Value, name string, def int) int {
	if v := q.Call("get", name); !v.IsNull() {
		if n, err := strconv.Atoi(v.String()); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func fail(err error) {
	js.Global().Set("__error", err.Error())
}

func main() {
	q := js.Global().Get("URLSearchParams").New(js.Global().Get("location").Get("search"))
	w, h, frames := param(q, "w", 3840), param(q, "h", 2160), param(q, "frames", 120)
	format := wg.TextureFormatRGBA8Unorm
	usage := wg.TextureUsageRenderAttachment | wg.TextureUsageCopyDst
	if f := q.Call("get", "format"); !f.IsNull() && f.String() == "bgra" {
		format, usage = wg.TextureFormatBGRA8Unorm, wg.TextureUsageRenderAttachment
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	nav, err := wg.Navigator()
	if err != nil {
		fail(err)
		return
	}
	// The discrete GPU where there is one: a machine with an integrated GPU
	// too (a Ryzen 7000 has one) otherwise gets the low-power adapter.
	a, err := nav.RequestAdapter(ctx, &wg.RequestAdapterOptions{PowerPreference: wg.PowerPreferenceHighPerformance})
	if err != nil {
		fail(err)
		return
	}
	// Which GPU drew: a software Vulkan (Mesa's lavapipe) is not flagged as
	// a fallback adapter, so the harness reports this with every result.
	info := a.Info()
	js.Global().Set("__adapter", info.Vendor+" / "+info.Architecture+" / "+info.Device+" / "+info.Description+
		" / preferred canvas format "+string(nav.PreferredCanvasFormat()))
	d, err := a.RequestDevice(ctx, &wg.DeviceDescriptor{Label: "presentbench"})
	if err != nil {
		fail(err)
		return
	}
	if err := wg.InstallReplay(ctx, ""); err != nil {
		fail(err)
		return
	}
	newRenderer := gpu2dwebgpu.New
	if m := q.Call("get", "mode"); !m.IsNull() && m.String() == "compute" {
		newRenderer = gpu2dwebgpu.NewCompute
	}
	r, err := newRenderer(ctx, d, w, h)
	if err != nil {
		fail(err)
		return
	}
	canvas := js.Global().Get("document").Call("getElementById", "screen")
	canvas.Set("width", w)
	canvas.Set("height", h)
	s, err := d.NewSurfaceFromCanvas(canvas, wg.SurfaceDescriptor{Format: format, Usage: usage})
	if err != nil {
		fail(err)
		return
	}
	l, _ := gpu2d.NewList(w, h)
	gpu2d.WidgetFrame(l)

	const warmup = 30
	n := 0
	var start float64
	done := make(chan struct{})
	var frame js.Func
	frame = js.FuncOf(func(_ js.Value, args []js.Value) any {
		if n == warmup {
			start = args[0].Float()
		}
		if n == warmup+frames {
			close(done) // main waits for the GPU, then stops the clock
			return nil
		}
		if err := r.Draw(l, nil); err != nil {
			fail(err)
			close(done)
			return nil
		}
		if err := r.Present(s); err != nil {
			fail(err)
			close(done)
			return nil
		}
		n++
		js.Global().Call("requestAnimationFrame", frame)
		return nil
	})
	js.Global().Call("requestAnimationFrame", frame)
	<-done
	if js.Global().Get("__error").Truthy() {
		return
	}
	// requestAnimationFrame times only how fast frames were SUBMITTED; the
	// clock stops once the GPU has finished all of them.
	if err := d.Queue().OnSubmittedWorkDone(ctx); err != nil {
		fail(err)
		return
	}
	end := js.Global().Get("performance").Call("now").Float()
	js.Global().Set("__nsPerFrame", (end-start)*1e6/float64(frames))
}
