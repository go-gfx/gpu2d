// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build js && wasm

package webgpu

import (
	"context"
	"encoding/binary"
	"errors"
	"math"

	"github.com/go-gfx/gpu2d"
	wg "github.com/go-gfx/webgpu"
)

// ErrSize reports a list whose surface is not the renderer's.
var ErrSize = errors.New("gpu2d/webgpu: list size differs from the renderer's")

// ErrNoReplay reports that go-gfx/webgpu's replay script, which submits a
// frame in one JavaScript call, is not loaded. The caller installs it with
// webgpu.InstallReplay: from its own URL under a strict script-src, or ""
// for the embedded copy.
var ErrNoReplay = errors.New("gpu2d/webgpu: webgpu.InstallReplay has not run")

// Renderer draws lists into an rgba8unorm texture of a fixed size.
type Renderer struct {
	d       *wg.Device
	w, h    int
	pipe    *wg.RenderPipeline
	bgl     *wg.BindGroupLayout
	uniform *wg.Buffer
	inst    *wg.Buffer
	instCap uint64
	arena   *wg.Buffer
	dataCap uint64
	group   *wg.BindGroup
	target  *wg.Texture
	view    *wg.TextureView
	rec     *wg.Recorder
	show    *presentPass // drawn into a bgra8unorm canvas; nil until needed
	bytes   []byte
	padded  []byte // the arena, padded to whole words
}

// New builds the pipeline and a w × h target on d.
func New(ctx context.Context, d *wg.Device, w, h int) (*Renderer, error) {
	if w <= 0 || h <= 0 || w > gpu2d.MaxCoord || h > gpu2d.MaxCoord {
		return nil, gpu2d.ErrSize
	}
	if !wg.ReplayInstalled() {
		return nil, ErrNoReplay
	}
	r := &Renderer{d: d, w: w, h: h}
	var err error
	if r.bgl, err = d.CreateBindGroupLayout(wg.BindGroupLayoutDescriptor{Entries: []wg.BindGroupLayoutEntry{
		{Binding: 0, Visibility: wg.ShaderStageVertex | wg.ShaderStageFragment, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeReadOnlyStorage}},
		{Binding: 1, Visibility: wg.ShaderStageVertex, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeUniform}},
		{Binding: 2, Visibility: wg.ShaderStageFragment, Buffer: &wg.BufferBindingLayout{Type: wg.BufferBindingTypeReadOnlyStorage}},
	}}); err != nil {
		return nil, err
	}
	pl, err := d.CreatePipelineLayout(wg.PipelineLayoutDescriptor{BindGroupLayouts: []*wg.BindGroupLayout{r.bgl}})
	if err != nil {
		return nil, err
	}
	mod, err := d.CreateShaderModule(wg.ShaderModuleDescriptor{Label: "gpu2d", Code: Shader})
	if err != nil {
		return nil, err
	}
	blend := &wg.BlendState{
		Color: wg.BlendComponent{Operation: wg.BlendOperationAdd, SrcFactor: wg.BlendFactorSrcAlpha, DstFactor: wg.BlendFactorOneMinusSrcAlpha},
		Alpha: wg.BlendComponent{Operation: wg.BlendOperationAdd, SrcFactor: wg.BlendFactorOne, DstFactor: wg.BlendFactorOneMinusSrcAlpha},
	}
	if r.pipe, err = d.CreateRenderPipelineAsync(ctx, wg.RenderPipelineDescriptor{
		Label:     "gpu2d",
		Layout:    pl,
		Vertex:    wg.VertexState{Module: mod, EntryPoint: "vs"},
		Primitive: wg.PrimitiveState{Topology: wg.PrimitiveTopologyTriangleList, CullMode: wg.CullModeNone},
		Fragment:  &wg.FragmentState{Module: mod, EntryPoint: "fs", Targets: []wg.ColorTargetState{{Format: wg.TextureFormatRGBA8Unorm, Blend: blend}}},
	}); err != nil {
		return nil, err
	}
	if r.uniform, err = d.CreateBuffer(wg.BufferDescriptor{Label: "gpu2d surface", Size: 16, Usage: wg.BufferUsageUniform | wg.BufferUsageCopyDst}); err != nil {
		return nil, err
	}
	var u [16]byte
	binary.LittleEndian.PutUint32(u[0:], math.Float32bits(float32(w)))
	binary.LittleEndian.PutUint32(u[4:], math.Float32bits(float32(h)))
	if err = d.Queue().WriteBuffer(r.uniform, 0, u[:]); err != nil {
		return nil, err
	}
	if r.target, err = d.CreateTexture(wg.TextureDescriptor{Label: "gpu2d target", Size: wg.Extent3D{Width: uint32(w), Height: uint32(h)},
		Format: wg.TextureFormatRGBA8Unorm, Usage: wg.TextureUsageRenderAttachment | wg.TextureUsageCopySrc | wg.TextureUsageCopyDst | wg.TextureUsageTextureBinding}); err != nil {
		return nil, err
	}
	if r.view, err = r.target.CreateView(nil); err != nil {
		return nil, err
	}
	if err = r.grow(64, 4096); err != nil {
		return nil, err
	}
	r.rec = d.NewRecorder()
	return r, nil
}

// grow makes the instance buffer hold at least n instances and the arena
// buffer at least data bytes, each doubling, and rebuilds the bind group
// when either changed.
func (r *Renderer) grow(n, data int) error {
	inst, instCap, err := r.fit(r.inst, r.instCap, uint64(n)*gpu2d.InstanceSize, "gpu2d instances")
	if err != nil {
		return err
	}
	arena, dataCap, err := r.fit(r.arena, r.dataCap, uint64(data+3)&^3, "gpu2d arena")
	if err != nil {
		return err
	}
	if inst == r.inst && arena == r.arena {
		return nil
	}
	g, err := r.d.CreateBindGroup(wg.BindGroupDescriptor{Layout: r.bgl, Entries: []wg.BindGroupEntry{
		{Binding: 0, Buffer: inst},
		{Binding: 1, Buffer: r.uniform},
		{Binding: 2, Buffer: arena},
	}})
	if err != nil {
		return err
	}
	for _, old := range []struct{ was, now *wg.Buffer }{{r.inst, inst}, {r.arena, arena}} {
		if old.was != nil && old.was != old.now {
			old.was.Destroy()
		}
	}
	r.inst, r.instCap, r.arena, r.dataCap, r.group = inst, instCap, arena, dataCap, g
	return nil
}

// fit returns b when its capacity holds need bytes, else a new storage
// buffer of at least need (doubling); the caller destroys the old one once
// the bind group no longer names it.
func (r *Renderer) fit(b *wg.Buffer, capacity, need uint64, label string) (*wg.Buffer, uint64, error) {
	if b != nil && need <= capacity {
		return b, capacity, nil
	}
	c := max(capacity*2, need, 4)
	nb, err := r.d.CreateBuffer(wg.BufferDescriptor{Label: label, Size: c, Usage: wg.BufferUsageStorage | wg.BufferUsageCopyDst})
	if err != nil {
		return nil, 0, err
	}
	return nb, c, nil
}

// Texture is the render target, for a caller that presents or samples it.
func (r *Renderer) Texture() *wg.Texture { return r.target }

// Draw composites the list over the target's current content, in one
// submit. ground, when not nil, first replaces the content (4·w·h bytes,
// RGBA rows top to bottom).
func (r *Renderer) Draw(l *gpu2d.List, ground []byte) error {
	if w, h := l.Size(); w != r.w || h != r.h {
		return ErrSize
	}
	ins := l.Instances()
	data := l.Data()
	if err := r.grow(len(ins), len(data)); err != nil {
		return err
	}
	r.rec.Reset()
	if ground != nil {
		r.rec.WriteTexture(wg.TexelCopyTextureInfo{Texture: r.target}, ground,
			wg.TexelCopyBufferLayout{BytesPerRow: uint32(4 * r.w)}, wg.Extent3D{Width: uint32(r.w), Height: uint32(r.h)})
	}
	if len(ins) > 0 {
		r.bytes = gpu2d.AppendBytes(r.bytes[:0], ins)
		r.rec.WriteBuffer(r.inst, 0, r.bytes)
		if len(data) > 0 {
			// WriteBuffer wants a multiple of 4 bytes; the arena's last entry
			// may end short of one.
			r.padded = append(append(r.padded[:0], data...), make([]byte, (4-len(data)%4)%4)...)
			r.rec.WriteBuffer(r.arena, 0, r.padded)
		}
		p := r.rec.BeginRenderPass(wg.RenderPassDescriptor{ColorAttachments: []*wg.RenderPassColorAttachment{
			{View: r.view, LoadOp: wg.LoadOpLoad, StoreOp: wg.StoreOpStore},
		}})
		p.SetPipeline(r.pipe)
		p.SetBindGroup(0, r.group)
		p.Draw(6, uint32(len(ins)), 0, 0)
		p.End()
	}
	_, err := r.rec.Submit()
	return err
}

// ErrSurface reports a canvas surface Present cannot draw into: neither
// rgba8unorm with COPY_DST nor bgra8unorm. CanvasDescriptor makes one that
// can.
var ErrSurface = errors.New("gpu2d/webgpu: the surface is neither rgba8unorm with COPY_DST nor bgra8unorm")

// presentPass is the pipeline and bind group PresentShader draws with,
// built the first time a bgra8unorm surface is presented.
type presentPass struct {
	pipe  *wg.RenderPipeline
	group *wg.BindGroup
}

func (r *Renderer) presentPass() (*presentPass, error) {
	if r.show != nil {
		return r.show, nil
	}
	bgl, err := r.d.CreateBindGroupLayout(wg.BindGroupLayoutDescriptor{Entries: []wg.BindGroupLayoutEntry{
		{Binding: 0, Visibility: wg.ShaderStageFragment, Texture: &wg.TextureBindingLayout{SampleType: wg.TextureSampleTypeFloat, ViewDimension: wg.TextureViewDimension2D}},
	}})
	if err != nil {
		return nil, err
	}
	pl, err := r.d.CreatePipelineLayout(wg.PipelineLayoutDescriptor{BindGroupLayouts: []*wg.BindGroupLayout{bgl}})
	if err != nil {
		return nil, err
	}
	mod, err := r.d.CreateShaderModule(wg.ShaderModuleDescriptor{Label: "gpu2d present", Code: PresentShader})
	if err != nil {
		return nil, err
	}
	pipe, err := r.d.CreateRenderPipeline(wg.RenderPipelineDescriptor{
		Label:     "gpu2d present",
		Layout:    pl,
		Vertex:    wg.VertexState{Module: mod, EntryPoint: "vs"},
		Primitive: wg.PrimitiveState{Topology: wg.PrimitiveTopologyTriangleList, CullMode: wg.CullModeNone},
		Fragment:  &wg.FragmentState{Module: mod, EntryPoint: "fs", Targets: []wg.ColorTargetState{{Format: wg.TextureFormatBGRA8Unorm}}},
	})
	if err != nil {
		return nil, err
	}
	group, err := r.d.CreateBindGroup(wg.BindGroupDescriptor{Layout: bgl, Entries: []wg.BindGroupEntry{{Binding: 0, TextureView: r.view}}})
	if err != nil {
		return nil, err
	}
	r.show = &presentPass{pipe: pipe, group: group}
	return r.show, nil
}

// CanvasDescriptor is the configuration a canvas surface needs for Present:
// rgba8unorm, which the WebGPU specification lists among the supported
// context formats and which is the target's own format, so the copy is
// byte-exact; and COPY_DST on top of the default RENDER_ATTACHMENT, which
// copyTextureToTexture requires of its destination. Alpha stays "opaque".
func CanvasDescriptor() wg.SurfaceDescriptor {
	return wg.SurfaceDescriptor{
		Format: wg.TextureFormatRGBA8Unorm,
		Usage:  wg.TextureUsageRenderAttachment | wg.TextureUsageCopyDst,
	}
}

// Present puts the target into the surface's current texture, on the GPU:
// the frame goes to the canvas without coming back to Go memory. Call it
// after Draw; the browser shows the canvas at the end of the task. The
// surface must be the renderer's size, and either rgba8unorm with COPY_DST
// (a copyTextureToTexture) or bgra8unorm (a pass drawing PresentShader).
func (r *Renderer) Present(s *wg.Surface) error {
	if w, h := s.Size(); w != r.w || h != r.h {
		return ErrSize
	}
	switch s.Format() {
	case wg.TextureFormatRGBA8Unorm:
	case wg.TextureFormatBGRA8Unorm:
		return r.presentDrawn(s)
	default:
		return ErrSurface
	}
	t, err := s.CurrentTexture()
	if err != nil {
		return err
	}
	if t.Descriptor().Usage&wg.TextureUsageCopyDst == 0 {
		return ErrSurface
	}
	r.rec.Reset()
	r.rec.CopyTextureToTexture(wg.TexelCopyTextureInfo{Texture: r.target}, wg.TexelCopyTextureInfo{Texture: t},
		wg.Extent3D{Width: uint32(r.w), Height: uint32(r.h), DepthOrArrayLayers: 1})
	_, err = r.rec.Submit()
	return err
}

// presentDrawn draws the target into a bgra8unorm surface.
func (r *Renderer) presentDrawn(s *wg.Surface) error {
	pp, err := r.presentPass()
	if err != nil {
		return err
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

// Destroy releases the renderer's buffers and target texture, refunding them
// to the device's quota. A caller that replaces a renderer (a canvas resized)
// destroys the old one, or each replacement keeps its memory charged. The
// renderer must not be used after; Destroy twice is harmless. (Its pipeline
// is not refunded: WebGPU has no way to destroy one.)
func (r *Renderer) Destroy() {
	for _, b := range []*wg.Buffer{r.uniform, r.inst, r.arena} {
		if b != nil {
			b.Destroy()
		}
	}
	if r.target != nil {
		r.target.Destroy()
	}
}

// Read copies the target back to Go memory: 4·w·h bytes, RGBA rows top to
// bottom. It waits for the GPU; it is for tests and screenshots.
func (r *Renderer) Read(ctx context.Context) ([]byte, error) {
	return r.d.ReadTexture(ctx, wg.TexelCopyTextureInfo{Texture: r.target}, r.w, r.h, 4)
}
