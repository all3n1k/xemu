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

/*
 * Vertex attribute setup.
 *
 * The GL backend maintains a GL buffer object mirroring guest VRAM and
 * re-uploads whichever region a draw touches (update_memory_buffer). On
 * Apple Silicon that copy is unnecessary: memory is unified, so a single
 * MTLBuffer created with newBufferWithBytesNoCopy: can alias the entire
 * guest VRAM mapping, and attributes bind as offsets into it. No per-draw
 * upload, and the guest's own writes are visible to the GPU immediately.
 *
 * Each NV2A attribute has its own base address and stride, so each gets its
 * own Metal buffer binding slot (1..16; slot 0 is the uniform struct).
 * Attributes the guest supplies as a constant rather than an array use
 * MTLVertexStepFunctionConstant against a small scratch buffer.
 */

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "renderer.h"
#include "vertex.h"

void pgraph_metal_init_vertex(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    size_t vram_size = memory_region_size(d->vram);

    /*
     * Zero-copy alias of guest VRAM. newBufferWithBytesNoCopy: requires a
     * page-aligned pointer and length; the VRAM mapping satisfies this. The
     * deallocator is nil because QEMU owns the memory.
     */
    size_t page = getpagesize();
    if (((uintptr_t)d->vram_ptr % page) == 0 && (vram_size % page) == 0) {
        r->vram_buffer = [r->device newBufferWithBytesNoCopy:d->vram_ptr
                                                      length:vram_size
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    }

    if (r->vram_buffer == nil) {
        /* Fall back to a private copy rather than failing outright; the draw
         * path then has to upload, which is what the GL backend does anyway. */
        fprintf(stderr,
                "nv2a: metal: VRAM not aliasable (ptr %p size %zu, page %zu), "
                "falling back to a copy\n",
                d->vram_ptr, vram_size, page);
        r->vram_buffer = [r->device newBufferWithLength:vram_size
                                                options:MTLResourceStorageModeShared];
        r->vram_buffer_is_copy = true;
    }

    /* Scratch buffer for constant (non-array) attributes: 16 slots x float4. */
    r->const_attr_buffer =
        [r->device newBufferWithLength:NV2A_VERTEXSHADER_ATTRIBUTES * 16
                               options:MTLResourceStorageModeShared];
}

void pgraph_metal_finalize_vertex(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    r->vram_buffer = nil;
    r->const_attr_buffer = nil;
    r->index_buffer = nil;
}

/* NV2A attribute format -> Metal vertex format. */
static MTLVertexFormat metal_vertex_format(unsigned int format,
                                           unsigned int count, bool *swizzle,
                                           bool *compressed)
{
    *swizzle = false;
    *compressed = false;

    switch (format) {
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_D3D:
        /* D3D colour order: BGRA in memory. Metal has no BGRA vertex format,
         * so it is read as RGBA and swizzled in the shader, which is exactly
         * what the generator's `swizzle_attrs` path already emits. */
        *swizzle = true;
        return MTLVertexFormatUChar4Normalized;

    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_OGL:
        switch (count) {
        case 1: return MTLVertexFormatUCharNormalized;
        case 2: return MTLVertexFormatUChar2Normalized;
        case 3: return MTLVertexFormatUChar3Normalized;
        case 4: return MTLVertexFormatUChar4Normalized;
        }
        break;

    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S1:
        switch (count) {
        case 1: return MTLVertexFormatShortNormalized;
        case 2: return MTLVertexFormatShort2Normalized;
        case 3: return MTLVertexFormatShort3Normalized;
        case 4: return MTLVertexFormatShort4Normalized;
        }
        break;

    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_F:
        switch (count) {
        case 1: return MTLVertexFormatFloat;
        case 2: return MTLVertexFormatFloat2;
        case 3: return MTLVertexFormatFloat3;
        case 4: return MTLVertexFormatFloat4;
        }
        break;

    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S32K:
        switch (count) {
        case 1: return MTLVertexFormatShort;
        case 2: return MTLVertexFormatShort2;
        case 3: return MTLVertexFormatShort3;
        case 4: return MTLVertexFormatShort4;
        }
        break;

    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_CMP:
        /* 3 signed normalized components packed in 32 bits (11,11,10). The
         * shader unpacks it, so hand it over as a plain int. */
        *compressed = true;
        return MTLVertexFormatInt;

    default:
        break;
    }

    fprintf(stderr, "nv2a: metal: unhandled vertex format 0x%x count %u\n",
            format, count);
    return MTLVertexFormatInvalid;
}

MTLVertexDescriptor *pgraph_metal_build_vertex_descriptor(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    float *const_data = (float *)r->const_attr_buffer.contents;

    pg->compressed_attrs = 0;
    pg->swizzle_attrs = 0;

    for (int i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *attr = &pg->vertex_attributes[i];
        NSUInteger slot = METAL_VERTEX_BUFFER_BASE + i;

        bool use_constant = (attr->count == 0) || (attr->stride == 0);
        bool swizzle = false, compressed = false;

        if (!use_constant) {
            MTLVertexFormat fmt = metal_vertex_format(attr->format,
                                                      attr->count, &swizzle,
                                                      &compressed);
            if (fmt == MTLVertexFormatInvalid) {
                use_constant = true;
            } else {
                hwaddr dma_len;
                uint8_t *attr_data = (uint8_t *)nv_dma_map(
                    d, attr->dma_select ? pg->dma_vertex_b : pg->dma_vertex_a,
                    &dma_len);
                if (attr->offset >= dma_len) {
                    use_constant = true;
                } else {
                    hwaddr base = attr_data + attr->offset - d->vram_ptr;

                    vd.attributes[i].format = fmt;
                    vd.attributes[i].offset = 0;
                    vd.attributes[i].bufferIndex = slot;
                    vd.layouts[slot].stride = attr->stride;
                    vd.layouts[slot].stepFunction =
                        MTLVertexStepFunctionPerVertex;
                    vd.layouts[slot].stepRate = 1;

                    r->attr_buffer_offset[i] = base;
                    r->attr_is_constant[i] = false;

                    if (swizzle) {
                        pg->swizzle_attrs |= (1 << i);
                    }
                    if (compressed) {
                        pg->compressed_attrs |= (1 << i);
                    }
                    continue;
                }
            }
        }

        /*
         * Constant attribute. Metal has no per-attribute constant, so point
         * the slot at a scratch buffer entry with a constant step function:
         * every vertex then reads the same value.
         */
        memcpy(&const_data[i * 4], attr->inline_value, sizeof(float) * 4);

        vd.attributes[i].format = MTLVertexFormatFloat4;
        vd.attributes[i].offset = 0;
        vd.attributes[i].bufferIndex = slot;
        vd.layouts[slot].stride = 16;
        vd.layouts[slot].stepFunction = MTLVertexStepFunctionConstant;
        vd.layouts[slot].stepRate = 0;

        r->attr_buffer_offset[i] = i * 16;
        r->attr_is_constant[i] = true;
    }

    return vd;
}

/*
 * Inline buffer: the guest supplies vertices immediately, one float4 per
 * attribute per vertex, rather than pointing at an array in memory. This is
 * what a fullscreen composite quad uses, which is why the dashboard's final
 * blit-to-screen draws went missing while everything else rendered.
 */
unsigned int pgraph_metal_bind_inline_buffer(NV2AState *d,
                                             id<MTLRenderCommandEncoder> enc,
                                             MTLVertexDescriptor *vd)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    unsigned int count = pg->inline_buffer_length;
    if (!count) {
        return 0;
    }

    for (int i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *attr = &pg->vertex_attributes[i];
        NSUInteger slot = METAL_VERTEX_BUFFER_BASE + i;

        vd.attributes[i].format = MTLVertexFormatFloat4;
        vd.attributes[i].offset = 0;
        vd.attributes[i].bufferIndex = slot;

        if (attr->inline_buffer_populated) {
            size_t bytes = (size_t)count * sizeof(float) * 4;
            id<MTLBuffer> b =
                [r->device newBufferWithBytes:attr->inline_buffer
                                       length:bytes
                                      options:MTLResourceStorageModeShared];
            vd.layouts[slot].stride = 16;
            vd.layouts[slot].stepFunction = MTLVertexStepFunctionPerVertex;
            vd.layouts[slot].stepRate = 1;
            [enc setVertexBuffer:b offset:0 atIndex:slot];

            attr->inline_buffer_populated = false;
            memcpy(attr->inline_value,
                   attr->inline_buffer + (count - 1) * 4,
                   sizeof(attr->inline_value));
        } else {
            /* Constant for every vertex. */
            float *cd = (float *)r->const_attr_buffer.contents;
            memcpy(&cd[i * 4], attr->inline_value, sizeof(float) * 4);
            vd.layouts[slot].stride = 16;
            vd.layouts[slot].stepFunction = MTLVertexStepFunctionConstant;
            vd.layouts[slot].stepRate = 0;
            [enc setVertexBuffer:r->const_attr_buffer
                          offset:i * 16
                         atIndex:slot];
        }
    }

    return count;
}

/*
 * Inline array: attributes interleaved in one guest-supplied block, packed in
 * attribute order with each element aligned to its own size.
 */
unsigned int pgraph_metal_bind_inline_array(NV2AState *d,
                                            id<MTLRenderCommandEncoder> enc,
                                            MTLVertexDescriptor *vd)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    unsigned int offset = 0;
    for (int i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *attr = &pg->vertex_attributes[i];
        if (attr->count == 0) {
            continue;
        }
        offset = ROUND_UP(offset, attr->size);
        attr->inline_array_offset = offset;
        offset += attr->size * attr->count;
        offset = ROUND_UP(offset, attr->size);
    }

    unsigned int vertex_size = offset;
    if (!vertex_size) {
        return 0;
    }

    unsigned int index_count =
        pg->inline_array_length * sizeof(uint32_t) / vertex_size;

    if (getenv("XEMU_METAL_TRACE_INLINE")) {
        static int n;
        if (n++ < 4) {
            fprintf(stderr,
                    "inline_array: len=%u words (%u bytes) vertex_size=%u "
                    "-> %u vertices\n",
                    pg->inline_array_length,
                    (unsigned)(pg->inline_array_length * sizeof(uint32_t)),
                    vertex_size, index_count);
            for (int k = 0; k < NV2A_VERTEXSHADER_ATTRIBUTES; k++) {
                VertexAttribute *a = &pg->vertex_attributes[k];
                if (a->count) {
                    fprintf(stderr,
                            "   attr%d fmt=0x%x size=%u count=%u off=%u\n",
                            k, a->format, a->size, a->count,
                            a->inline_array_offset);
                }
            }
        }
    }

    if (!index_count) {
        return 0;
    }

    id<MTLBuffer> b = [r->device
        newBufferWithBytes:pg->inline_array
                    length:pg->inline_array_length * sizeof(uint32_t)
                   options:MTLResourceStorageModeShared];

    for (int i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *attr = &pg->vertex_attributes[i];
        NSUInteger slot = METAL_VERTEX_BUFFER_BASE + i;

        if (attr->count == 0) {
            float *cd = (float *)r->const_attr_buffer.contents;
            memcpy(&cd[i * 4], attr->inline_value, sizeof(float) * 4);
            vd.attributes[i].format = MTLVertexFormatFloat4;
            vd.attributes[i].offset = 0;
            vd.attributes[i].bufferIndex = slot;
            vd.layouts[slot].stride = 16;
            vd.layouts[slot].stepFunction = MTLVertexStepFunctionConstant;
            vd.layouts[slot].stepRate = 0;
            [enc setVertexBuffer:r->const_attr_buffer
                          offset:i * 16
                         atIndex:slot];
            continue;
        }

        bool sw = false, cmp = false;
        MTLVertexFormat fmt = metal_vertex_format(attr->format, attr->count,
                                                  &sw, &cmp);
        if (fmt == MTLVertexFormatInvalid) {
            fmt = MTLVertexFormatFloat4;
        }
        if (sw) {
            pg->swizzle_attrs |= (1 << i);
        }
        if (cmp) {
            pg->compressed_attrs |= (1 << i);
        }

        vd.attributes[i].format = fmt;
        vd.attributes[i].offset = 0;
        vd.attributes[i].bufferIndex = slot;
        vd.layouts[slot].stride = vertex_size;
        vd.layouts[slot].stepFunction = MTLVertexStepFunctionPerVertex;
        vd.layouts[slot].stepRate = 1;
        [enc setVertexBuffer:b
                      offset:attr->inline_array_offset
                     atIndex:slot];
    }

    return index_count;
}

void pgraph_metal_bind_vertex_buffers(NV2AState *d,
                                      id<MTLRenderCommandEncoder> enc)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    for (int i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        id<MTLBuffer> buf = r->attr_is_constant[i] ? r->const_attr_buffer
                                                   : r->vram_buffer;
        [enc setVertexBuffer:buf
                      offset:r->attr_buffer_offset[i]
                     atIndex:METAL_VERTEX_BUFFER_BASE + i];
    }
}
