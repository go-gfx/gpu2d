// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build js && wasm

package webgpu

import (
	"context"
	"encoding/binary"
	"unsafe"

	"github.com/go-gfx/gpu2d"
	wg "github.com/go-gfx/webgpu"
)

// computeState is a Renderer's compute mode: gpu2d.KernelWGSL over a pixel
// storage buffer, instead of one quad per instance and fixed-function
// blending.
type computeState struct {
	pipe    *wg.ComputePipeline
	bgl     *wg.BindGroupLayout
	frame   *wg.Buffer // the kernel's uniform: w, h, tiles across, Tile
	bins    *wg.Buffer // spans, then list
	binsCap uint64
	px      *wg.Buffer // 4·w·h bytes, RGBA words
	group   *wg.BindGroup
	spans   []uint32
	list    []uint32
	show    map[wg.TextureFormat]*computePresent
}

// computePresent draws the pixel buffer into a canvas of one format.
type computePresent struct {
	pipe  *wg.RenderPipeline
	group *wg.BindGroup
}

// NewCompute is New with the raster kernel as one compute shader
// (gpu2d.KernelWGSL). Integer blending is PixelPainter's, so a frame is
// byte-identical to Rasterize's over any ground, where New's fixed-function
// blending differs by up to 4 in alpha over a translucent one. The pixels
// live in a storage buffer: Texture returns nil, and Present, Read and Draw
// keep their contracts.
func NewCompute(ctx context.Context, d *wg.Device, w, h int) (*Renderer, error) {
	return newCompute(ctx, d, w, h, gpu2d.KernelWGSL)
}

// newCompute takes the kernel source, so a test can perturb it.
func newCompute(ctx context.Context, d *wg.Device, w, h int, kernel string) (*Renderer, error) {
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return nil, gpu2d.ErrSize
	}
	if !wg.ReplayInstalled() {
		return nil, ErrNoReplay
	}
	r := &Renderer{d: d, w: w, h: h, c: &computeState{show: map[wg.TextureFormat]*computePresent{}}}
	c := r.c
	var err error
	rd := func(b uint32) wg.BindGroupLayoutEntry {
		return wg.BindGroupLayoutEntry{Binding: b, Visibility: wg.ShaderStageCompute, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeReadOnlyStorage}}
	}
	if c.bgl, err = d.CreateBindGroupLayout(wg.BindGroupLayoutDescriptor{Entries: []wg.BindGroupLayoutEntry{
		rd(0), rd(1), rd(2),
		{Binding: 3, Visibility: wg.ShaderStageCompute, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeStorage}},
		{Binding: 4, Visibility: wg.ShaderStageCompute, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeUniform}},
	}}); err != nil {
		return nil, err
	}
	pl, err := d.CreatePipelineLayout(wg.PipelineLayoutDescriptor{BindGroupLayouts: []*wg.BindGroupLayout{c.bgl}})
	if err != nil {
		return nil, err
	}
	mod, err := d.CreateShaderModule(wg.ShaderModuleDescriptor{Label: "gpu2d kernel", Code: kernel})
	if err != nil {
		return nil, err
	}
	if c.pipe, err = d.CreateComputePipelineAsync(ctx, wg.ComputePipelineDescriptor{
		Label: "gpu2d kernel", Layout: pl, Compute: wg.ProgrammableStage{Module: mod, EntryPoint: "raster"},
	}); err != nil {
		return nil, err
	}
	if c.frame, err = d.CreateBuffer(wg.BufferDescriptor{Label: "gpu2d frame", Size: 16, Usage: wg.BufferUsageUniform | wg.BufferUsageCopyDst}); err != nil {
		return nil, err
	}
	tl, _ := gpu2d.NewList(w, h)
	ac, _ := tl.Tiles()
	var u [16]byte
	binary.LittleEndian.PutUint32(u[0:], uint32(w))
	binary.LittleEndian.PutUint32(u[4:], uint32(h))
	binary.LittleEndian.PutUint32(u[8:], uint32(ac))
	binary.LittleEndian.PutUint32(u[12:], gpu2d.Tile)
	if err = d.Queue().WriteBuffer(c.frame, 0, u[:]); err != nil {
		return nil, err
	}
	if c.px, err = d.CreateBuffer(wg.BufferDescriptor{Label: "gpu2d pixels", Size: 4 * uint64(w) * uint64(h),
		Usage: wg.BufferUsageStorage | wg.BufferUsageCopyDst | wg.BufferUsageCopySrc}); err != nil {
		return nil, err
	}
	if err = r.growCompute(64, 4096, 2*ac*((h+gpu2d.Tile-1)/gpu2d.Tile)+64); err != nil {
		return nil, err
	}
	r.rec = d.NewRecorder()
	return r, nil
}

// growCompute is grow for the compute mode: the bins buffer holds binWords
// words (spans and list).
func (r *Renderer) growCompute(n, data, binWords int) error {
	c := r.c
	inst, instCap, err := r.fit(r.inst, r.instCap, uint64(n)*gpu2d.InstanceSize, "gpu2d instances")
	if err != nil {
		return err
	}
	arena, dataCap, err := r.fit(r.arena, r.dataCap, uint64(data+3)&^3, "gpu2d arena")
	if err != nil {
		return err
	}
	bins, binsCap, err := r.fit(c.bins, c.binsCap, 4*uint64(binWords), "gpu2d bins")
	if err != nil {
		return err
	}
	if inst == r.inst && arena == r.arena && bins == c.bins {
		return nil
	}
	g, err := r.d.CreateBindGroup(wg.BindGroupDescriptor{Layout: c.bgl, Entries: []wg.BindGroupEntry{
		{Binding: 0, Buffer: inst},
		{Binding: 1, Buffer: bins},
		{Binding: 2, Buffer: arena},
		{Binding: 3, Buffer: c.px},
		{Binding: 4, Buffer: c.frame},
	}})
	if err != nil {
		return err
	}
	for _, old := range []struct{ was, now *wg.Buffer }{{r.inst, inst}, {r.arena, arena}, {c.bins, bins}} {
		if old.was != nil && old.was != old.now {
			old.was.Destroy()
		}
	}
	r.inst, r.instCap, r.arena, r.dataCap, c.bins, c.binsCap, c.group = inst, instCap, arena, dataCap, bins, binsCap, g
	return nil
}

// words views a []uint32 as its bytes, for a little-endian host (wasm is).
func words(w []uint32) []byte {
	if len(w) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), 4*len(w))
}

func (r *Renderer) drawCompute(l *gpu2d.List, ground []byte) error {
	c := r.c
	ins := l.Instances()
	data := l.Data()
	var err error
	if c.spans, c.list, err = l.Bin(c.spans, c.list); err != nil {
		return err
	}
	if err = r.growCompute(len(ins), len(data), len(c.spans)+len(c.list)); err != nil {
		return err
	}
	r.rec.Reset()
	if ground != nil {
		r.rec.WriteBuffer(c.px, 0, ground)
	}
	if len(ins) > 0 {
		r.bytes = gpu2d.AppendBytes(r.bytes[:0], ins)
		r.rec.WriteBuffer(r.inst, 0, r.bytes)
		r.rec.WriteBuffer(c.bins, 0, words(c.spans))
		if len(c.list) > 0 {
			r.rec.WriteBuffer(c.bins, 4*uint64(len(c.spans)), words(c.list))
		}
		if len(data) > 0 {
			r.padded = append(append(r.padded[:0], data...), make([]byte, (4-len(data)%4)%4)...)
			r.rec.WriteBuffer(r.arena, 0, r.padded)
		}
		p := r.rec.BeginComputePass()
		p.SetPipeline(c.pipe)
		p.SetBindGroup(0, c.group)
		p.DispatchWorkgroups(uint32((r.w+15)/16), uint32((r.h+7)/8), 1)
		p.End()
	}
	_, err = r.rec.Submit()
	return err
}

// presentCompute draws the pixel buffer into a canvas of either format.
func (r *Renderer) presentCompute(s *wg.Surface) error {
	c := r.c
	f := s.Format()
	if f != wg.TextureFormatRGBA8Unorm && f != wg.TextureFormatBGRA8Unorm {
		return ErrSurface
	}
	pp := c.show[f]
	if pp == nil {
		bgl, err := r.d.CreateBindGroupLayout(wg.BindGroupLayoutDescriptor{Entries: []wg.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wg.ShaderStageFragment, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wg.ShaderStageFragment, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeUniform}},
		}})
		if err != nil {
			return err
		}
		pl, err := r.d.CreatePipelineLayout(wg.PipelineLayoutDescriptor{BindGroupLayouts: []*wg.BindGroupLayout{bgl}})
		if err != nil {
			return err
		}
		mod, err := r.d.CreateShaderModule(wg.ShaderModuleDescriptor{Label: "gpu2d compute present", Code: computePresentShader})
		if err != nil {
			return err
		}
		pipe, err := r.d.CreateRenderPipeline(wg.RenderPipelineDescriptor{
			Label:     "gpu2d compute present",
			Layout:    pl,
			Vertex:    wg.VertexState{Module: mod, EntryPoint: "vs"},
			Primitive: wg.PrimitiveState{Topology: wg.PrimitiveTopologyTriangleList, CullMode: wg.CullModeNone},
			Fragment:  &wg.FragmentState{Module: mod, EntryPoint: "fs", Targets: []wg.ColorTargetState{{Format: f}}},
		})
		if err != nil {
			return err
		}
		group, err := r.d.CreateBindGroup(wg.BindGroupDescriptor{Layout: bgl, Entries: []wg.BindGroupEntry{
			{Binding: 0, Buffer: c.px}, {Binding: 1, Buffer: c.frame},
		}})
		if err != nil {
			return err
		}
		pp = &computePresent{pipe: pipe, group: group}
		c.show[f] = pp
	}
	v, err := s.CurrentView()
	if err != nil {
		return err
	}
	r.rec.Reset()
	p := r.rec.BeginRenderPass(wg.RenderPassDescriptor{ColorAttachments: []*wg.RenderPassColorAttachment{
		{View: v, LoadOp: wg.LoadOpClear, StoreOp: wg.StoreOpStore},
	}})
	p.SetPipeline(pp.pipe)
	p.SetBindGroup(0, pp.group)
	p.Draw(3, 1, 0, 0)
	p.End()
	_, err = r.rec.Submit()
	return err
}

// readCompute copies the pixel buffer to Go memory through a mappable one.
func (r *Renderer) readCompute(ctx context.Context) ([]byte, error) {
	n := 4 * uint64(r.w) * uint64(r.h)
	buf, err := r.d.CreateBuffer(wg.BufferDescriptor{Label: "gpu2d read", Size: n, Usage: wg.BufferUsageMapRead | wg.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	defer buf.Destroy()
	enc, err := r.d.CreateCommandEncoder("gpu2d read")
	if err != nil {
		return nil, err
	}
	if err := enc.CopyBufferToBuffer(r.c.px, 0, buf, 0, n); err != nil {
		return nil, err
	}
	cb, err := enc.Finish()
	if err != nil {
		return nil, err
	}
	if err := r.d.Queue().Submit(cb); err != nil {
		return nil, err
	}
	if err := buf.MapAsync(ctx, wg.MapModeRead, 0, 0); err != nil {
		return nil, err
	}
	out := make([]byte, n)
	if err := buf.ReadMapped(0, out); err != nil {
		return nil, err
	}
	return out, buf.Unmap()
}
