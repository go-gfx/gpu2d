//go:build darwin

// Metal exists only on macOS, and Getrusage is not in syscall on js/wasm or
// Windows.

package metal

import (
	"syscall"
	"testing"
	"time"

	"github.com/go-gfx/gpu2d"
	mtl "github.com/go-macos/metal"
)

// BenchmarkSubmit times Submit alone (binning, packing, encoding and the
// commit), the GPU's work and the wait kept outside the timer
// (go-gfx/gpu2d#20: in a 2x gallery frame Submit takes 1.15 ms, while a bare
// go-macos/metal commit takes 7 us). The cases split what grows with the
// frame's size from what grows with its instances.
func BenchmarkSubmit(b *testing.B) {
	for _, c := range []struct {
		name   string
		w, h   int
		widget bool
	}{
		{"empty/64x64", 64, 64, false},
		{"empty/1920x5668", 1920, 5668, false},
		{"widget/1920x1080", 1920, 1080, true},
		{"widget/1920x5668", 1920, 5668, true},
	} {
		b.Run(c.name, func(b *testing.B) {
			r := renderer(b, c.w, c.h)
			l, _ := gpu2d.NewList(c.w, c.h)
			if c.widget {
				gpu2d.WidgetFrame(l)
			}
			b.ReportMetric(float64(len(l.Instances())), "instances")
			b.ReportAllocs()
			// The whole process's CPU time inside Submit, Metal's own threads
			// included: far below the wall time means Submit waits, not works.
			var cpu time.Duration
			for b.Loop() {
				c0 := processCPU(b)
				if err := r.Submit(l); err != nil {
					b.Fatal(err)
				}
				cpu += processCPU(b) - c0
				b.StopTimer()
				if err := r.Wait(); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
			}
			b.ReportMetric(float64(cpu.Nanoseconds())/float64(b.N), "cpu-ns/op")
		})
	}
}

// BenchmarkSubmitParts splits Submit on the 2x gallery-sized WidgetFrame:
// pack (binning and copies into the slot, pure Go) and device (encoding the
// raster dispatch and go-macos/metal's commit, on a slot packed once).
func BenchmarkSubmitParts(b *testing.B) {
	const w, h = 1920, 5668
	r := renderer(b, w, h)
	l, _ := gpu2d.NewList(w, h)
	gpu2d.WidgetFrame(l)
	b.Run("pack", func(b *testing.B) {
		for b.Loop() {
			if _, _, err := r.pack(l); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("device", func(b *testing.B) {
		s, f, err := r.pack(l)
		if err != nil {
			b.Fatal(err)
		}
		for b.Loop() {
			p, err := r.d.Submit(func(e *mtl.Encoder) { r.raster(e, s, &f) })
			if err != nil {
				b.Fatal(err)
			}
			b.StopTimer()
			if err := p.Wait(); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
		}
	})
}

// processCPU is the process's CPU time so far (user + system, all threads).
func processCPU(tb testing.TB) time.Duration {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		tb.Fatal(err)
	}
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}
