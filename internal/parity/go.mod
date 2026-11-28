module github.com/go-gfx/gpu2d/internal/parity

go 1.27.4

require (
	github.com/go-gfx/gfx v0.42.0
	github.com/go-gfx/gpu2d v0.12.1
	github.com/go-widgets/painter v0.22.0
)

require (
	github.com/go-simd/boxfilter v0.2.0 // indirect
	golang.org/x/sys v0.50.0 // indirect
)

replace github.com/go-gfx/gpu2d => ../..
