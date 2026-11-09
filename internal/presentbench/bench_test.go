// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package presentbench measures webgpu.Renderer.Present into a real page's
// canvas, end to end: the frame time a page sees includes the browser
// compositor's work, which no in-worker benchmark does. Chrome warns that a
// canvas in a format other than the device's preferred one "requires an
// extra copy"; this is how that copy is measured.
//
//	go test -c -o presentbench.test .
//	abbench --base presentbench.test --head presentbench.test --max-load 8 \
//	    --base-arg=-test.bench=Present/rgba --head-arg=-test.bench=Present/bgra \
//	    --rename-base rgba=fmt --rename-head bgra=fmt -- -test.run '^$' -test.benchtime 3x
//
// Headless Chrome runs with its frame-rate limit and vsync off, so a frame
// costs what it costs rather than a display interval. Skipped without Chrome
// or without a WebGPU adapter.
package presentbench

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/chromedp/chromedp"
)

const index = `<!doctype html><meta charset="utf-8">
<canvas id="screen"></canvas>
<script src="wasm_exec.js"></script>
<script>
const go = new Go();
WebAssembly.instantiateStreaming(fetch("page.wasm"), go.importObject).then(r => go.run(r.instance));
</script>`

var site string // the served page, built once

// On a host without Go (a cross-compiled test binary copied to another
// machine), the page is built beforehand: PRESENTBENCH_BUILD_SITE=dir builds
// it into dir and exits, and PRESENTBENCH_SITE=dir serves that dir instead of
// building. CHROME names the browser executable and PRESENTBENCH_FLAGS adds
// space-separated Chrome switches (a Linux GPU may need Vulkan ones).
func TestMain(m *testing.M) {
	if dir := os.Getenv("PRESENTBENCH_SITE"); dir != "" {
		srv := httptest.NewServer(http.FileServer(http.Dir(dir)))
		site = srv.URL
		code := m.Run()
		srv.Close()
		os.Exit(code)
	}
	dir := os.Getenv("PRESENTBENCH_BUILD_SITE")
	keep := dir != ""
	var err error
	if keep {
		err = os.MkdirAll(dir, 0o755)
	} else {
		dir, err = os.MkdirTemp("", "presentbench")
	}
	if err != nil {
		panic(err)
	}
	build := exec.Command("go", "build", "-o", filepath.Join(dir, "page.wasm"), "./page")
	// From this file's own directory: a compiled test binary (go test -c,
	// run by abbench) runs wherever it is started, not in the package.
	_, self, _, _ := runtime.Caller(0)
	build.Dir = filepath.Dir(self)
	build.Env = append(os.Environ(), "GOOS=js", "GOARCH=wasm")
	if out, err := build.CombinedOutput(); err != nil {
		// The error too, not only the output: a source directory that no
		// longer exists (the checkout moved to another branch) fails before
		// the compiler prints anything.
		panic(fmt.Sprintf("building the page in %s: %v\n%s", build.Dir, err, out))
	}
	execJS, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"))
	if err != nil {
		panic(err)
	}
	_ = os.WriteFile(filepath.Join(dir, "wasm_exec.js"), execJS, 0o644)
	_ = os.WriteFile(filepath.Join(dir, "index.html"), []byte(index), 0o644)
	if keep {
		fmt.Println("presentbench: site built in", dir)
		os.Exit(0)
	}
	srv := httptest.NewServer(http.FileServer(http.Dir(dir)))
	site = srv.URL
	code := m.Run()
	srv.Close()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

// frameTime loads the page with a query string (format=, mode=, w=, h=) and
// returns its time per frame.
func frameTime(b *testing.B, query string) float64 {
	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", true),
		chromedp.Flag("disable-gpu", false),
		chromedp.Flag("enable-unsafe-webgpu", true),
		chromedp.Flag("disable-gpu-vsync", true),
		chromedp.Flag("disable-frame-rate-limit", true))
	if p := os.Getenv("CHROME"); p != "" {
		opts = append(opts, chromedp.ExecPath(p))
	}
	for _, f := range strings.Fields(os.Getenv("PRESENTBENCH_FLAGS")) {
		name, value, hasValue := strings.Cut(strings.TrimPrefix(f, "--"), "=")
		if hasValue {
			opts = append(opts, chromedp.Flag(name, value))
		} else {
			opts = append(opts, chromedp.Flag(name, true))
		}
	}
	actx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()
	ctx, cancel2 := chromedp.NewContext(actx)
	defer cancel2()
	ctx, cancel3 := context.WithTimeout(ctx, 120*time.Second)
	defer cancel3()
	if _, err := chromedp.Run(ctx, chromedp.Navigate(site+"/?"+query)); err != nil {
		b.Skipf("no Chrome: %v", err)
	}
	for deadline := time.Now().Add(100 * time.Second); ; time.Sleep(100 * time.Millisecond) {
		if e, err := chromedp.Run(ctx, chromedp.Evaluate[string](`window.__error || ""`)); err == nil && e != "" {
			b.Skipf("the page could not draw: %s", e)
		}
		if ns, err := chromedp.Run(ctx, chromedp.Evaluate[float64](`window.__nsPerFrame || 0`)); err == nil && ns > 0 {
			if a, err := chromedp.Run(ctx, chromedp.Evaluate[string](`window.__adapter || ""`)); err == nil {
				b.Logf("adapter: %s", a) // which GPU drew it
			}
			return ns
		}
		if time.Now().After(deadline) {
			b.Fatal("the page never finished its frames")
		}
	}
}

// BenchmarkPresent is one page load per iteration, 120 timed 4K frames each;
// ns/frame is their mean.
func BenchmarkPresent(b *testing.B) {
	for _, format := range []string{"rgba", "bgra"} {
		b.Run(format, func(b *testing.B) {
			var total float64
			n := 0
			for b.Loop() {
				total += frameTime(b, "format="+format)
				n++
			}
			b.ReportMetric(total/float64(n), "ns/frame")
		})
	}
}

// BenchmarkRaster compares the fixed-function renderer with the compute
// kernel on the shared widget frame, Draw plus Present into a bgra8unorm
// canvas, the format macOS prefers. Interleave the two with abbench:
//
//	go test -c -o presentbench.test .
//	abbench --base presentbench.test --head presentbench.test --max-load 8 \
//	    --base-arg=-test.bench=Raster/4K/fixed --head-arg=-test.bench=Raster/4K/compute \
//	    --rename-base fixed=mode --rename-head compute=mode -- -test.run '^$' -test.benchtime 3x
func BenchmarkRaster(b *testing.B) {
	for _, s := range []struct {
		name string
		w, h int
	}{{"1080p", 1920, 1080}, {"4K", 3840, 2160}, {"gallery", 960, 2834}} {
		for _, mode := range []string{"fixed", "compute"} {
			b.Run(s.name+"/"+mode, func(b *testing.B) {
				var total float64
				n := 0
				for b.Loop() {
					total += frameTime(b, fmt.Sprintf("format=bgra&mode=%s&w=%d&h=%d", mode, s.w, s.h))
					n++
				}
				b.ReportMetric(total/float64(n), "ns/frame")
			})
		}
	}
}
