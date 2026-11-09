package metal

import (
	"bytes"
	"testing"

	"github.com/go-gfx/gpu2d"
	"github.com/go-gfx/gpu2d/internal/scenes"
)

// Binning leaves out what a later opaque square-cornered fill hides in a
// whole tile. The frame must not change: random scenes (translucent, rounded,
// outlined, masked, imaged) are drawn, then opaque fills land over them,
// some covering whole tiles and some not, then more random content on top.
// The GPU result equals the CPU reference byte for byte, and binning did
// leave entries out.
func TestCulledTilesDrawTheSameBytes(t *testing.T) {
	const w, h = 128, 96
	r := renderer(t, w, h)
	culledScenes := 0
	for seed := range uint64(60) {
		l, _ := gpu2d.NewList(w, h)
		// First, before anything can leave a clip or a translation pushed:
		// a rounded translucent fill, then an opaque one hiding its tiles.
		// Binning must drop the first wherever the second covers a tile.
		l.FillRoundRect(gpu2d.Rect{X: 8, Y: 8, W: 72, H: 56}, 5, gpu2d.RGBA{R: 90, G: 90, A: 140})
		l.FillRect(gpu2d.Rect{X: 16, Y: 16, W: 48, H: 32}, gpu2d.RGBA{G: 200, B: 90, A: 255})
		scenes.Random(l, seed, 30)
		// Then over random content (which may have left a clip pushed):
		// opaque square fills, one on the tile grid and one off it, one
		// rounded (never hides), one translucent (never hides).
		l.FillRect(gpu2d.Rect{X: 16, Y: 16, W: 48, H: 32}, gpu2d.RGBA{R: 200, G: 10, B: 40, A: 255})
		l.FillRect(gpu2d.Rect{X: 70 + int(seed%5), Y: 3, W: 41, H: 50}, gpu2d.RGBA{R: 5, G: 120, B: 9, A: 255})
		l.FillRoundRect(gpu2d.Rect{X: 0, Y: 48, W: 64, H: 48}, 6, gpu2d.RGBA{B: 255, A: 255})
		l.FillRect(gpu2d.Rect{X: 64, Y: 48, W: 64, H: 48}, gpu2d.RGBA{R: 255, A: 128})
		scenes.Random(l, seed+1000, 10)

		want := make([]byte, 4*w*h)
		for i := range want {
			want[i] = uint8(i * 13)
		}
		ground := append([]byte(nil), want...)
		if err := l.Rasterize(want); err != nil {
			t.Fatal(err)
		}
		copy(r.Pixels(), ground)
		if err := r.Draw(l); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(want, r.Pixels()) {
			got := r.Pixels()
			for i := range want {
				if want[i] != got[i] {
					p := i / 4
					t.Fatalf("seed %d (%d,%d) channel %d: CPU %d, GPU %d", seed, p%w, p/w, i%4, want[i], got[i])
				}
			}
		}

		// Entries binned against every (instance, tile) pair the boxes touch.
		_, list, err := l.Bin(nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(list) < touched(l) {
			culledScenes++
		}
	}
	if culledScenes != 60 {
		t.Fatalf("binning left entries out in %d of 60 scenes, want all", culledScenes)
	}
}

// touched counts the (instance, tile) pairs the instances' boxes touch.
func touched(l *gpu2d.List) int {
	n := 0
	for _, in := range l.Instances() {
		x0, y0 := max(int(in.X), int(in.ClipX0)), max(int(in.Y), int(in.ClipY0))
		x1, y1 := min(int(in.X)+int(in.W), int(in.ClipX1)), min(int(in.Y)+int(in.H), int(in.ClipY1))
		n += ((x1-1)/gpu2d.Tile - x0/gpu2d.Tile + 1) * ((y1-1)/gpu2d.Tile - y0/gpu2d.Tile + 1)
	}
	return n
}
