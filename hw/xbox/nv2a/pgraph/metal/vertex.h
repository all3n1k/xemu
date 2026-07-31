/*
 * Geforce NV2A PGRAPH Metal Renderer - vertex attributes
 *
 * Copyright (c) 2026 xemu Metal backend contributors
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef XEMU_NV2A_PGRAPH_METAL_VERTEX_H
#define XEMU_NV2A_PGRAPH_METAL_VERTEX_H

#include "renderer.h"

/* Buffer slot 0 carries the uniform struct (MSL_UNIFORM_BUFFER_INDEX), so
 * vertex attribute buffers start at 1. */
#define METAL_VERTEX_BUFFER_BASE 1

/* One block of 16 float4 constant attributes, claimed per draw. */
#define METAL_CONST_ATTR_BLOCK (NV2A_VERTEXSHADER_ATTRIBUTES * 16)

size_t pgraph_metal_claim_const_block(PGRAPHMetalState *r);

void pgraph_metal_init_vertex(NV2AState *d);
void pgraph_metal_finalize_vertex(PGRAPHState *pg);

MTLVertexDescriptor *pgraph_metal_build_vertex_descriptor(NV2AState *d);
void pgraph_metal_bind_vertex_buffers(NV2AState *d,
                                      id<MTLRenderCommandEncoder> enc);
unsigned int pgraph_metal_bind_inline_buffer(NV2AState *d,
                                             id<MTLRenderCommandEncoder> enc,
                                             MTLVertexDescriptor *vd);
unsigned int pgraph_metal_bind_inline_array(NV2AState *d,
                                            id<MTLRenderCommandEncoder> enc,
                                            MTLVertexDescriptor *vd);

#endif /* XEMU_NV2A_PGRAPH_METAL_VERTEX_H */
