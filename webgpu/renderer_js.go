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
	group   *wg.BindGroup
	target  *wg.Texture
	view    *wg.TextureView
	rec     *wg.Recorder
	bytes   []byte
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
		Format: wg.TextureFormatRGBA8Unorm, Usage: wg.TextureUsageRenderAttachment | wg.TextureUsageCopySrc | wg.TextureUsageCopyDst}); err != nil {
		return nil, err
	}
	if r.view, err = r.target.CreateView(nil); err != nil {
		return nil, err
	}
	if err = r.grow(64); err != nil {
		return nil, err
	}
	r.rec = d.NewRecorder()
	return r, nil
}

// grow makes the instance buffer hold at least n instances, doubling.
func (r *Renderer) grow(n int) error {
	need := uint64(n) * gpu2d.InstanceSize
	if need <= r.instCap {
		return nil
	}
	c := max(r.instCap*2, need)
	b, err := r.d.CreateBuffer(wg.BufferDescriptor{Label: "gpu2d instances", Size: c, Usage: wg.BufferUsageStorage | wg.BufferUsageCopyDst})
	if err != nil {
		return err
	}
	g, err := r.d.CreateBindGroup(wg.BindGroupDescriptor{Layout: r.bgl, Entries: []wg.BindGroupEntry{
		{Binding: 0, Buffer: b},
		{Binding: 1, Buffer: r.uniform},
	}})
	if err != nil {
		b.Destroy()
		return err
	}
	if r.inst != nil {
		r.inst.Destroy()
	}
	r.inst, r.instCap, r.group = b, c, g
	return nil
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
	if err := r.grow(len(ins)); err != nil {
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

// Read copies the target back to Go memory: 4·w·h bytes, RGBA rows top to
// bottom. It waits for the GPU; it is for tests and screenshots.
func (r *Renderer) Read(ctx context.Context) ([]byte, error) {
	return r.d.ReadTexture(ctx, wg.TexelCopyTextureInfo{Texture: r.target}, r.w, r.h, 4)
}
