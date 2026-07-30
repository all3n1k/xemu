/*
 * Geforce NV2A PGRAPH Metal Renderer
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

#ifndef XEMU_NV2A_PGRAPH_METAL_RENDERER_H
#define XEMU_NV2A_PGRAPH_METAL_RENDERER_H

/*
 * This header is intended to be included only from the Metal backend's
 * own .m / .mm source files (which already include qemu/osdep.h and
 * nv2a_int.h via renderer.m). We avoid pulling osdep.h in again here to
 * stay consistent with the GL backend's renderer.h.
 */

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "qemu/queue.h"

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/nv2a/nv2a_regs.h"
#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "hw/xbox/nv2a/pgraph/surface.h"
#include "hw/xbox/nv2a/pgraph/glsl/shaders.h"

typedef struct MetalSurfaceBinding MetalSurfaceBinding;

/* A generated + compiled shader pair for one ShaderState. */
typedef struct MetalShaderBinding {
    ShaderState        state;
    bool               valid;
    id<MTLLibrary>     vsh_library;
    id<MTLLibrary>     psh_library;
    id<MTLFunction>    vsh_function;
    id<MTLFunction>    psh_function;
} MetalShaderBinding;

/*
 * The Metal renderer state. This is the structural counterpart of
 * PGRAPHGLState / PGRAPHVkState. For now it only holds the core Metal
 * device/queue pair plus the display-texture handle used by
 * get_framebuffer_surface().
 *
 * As subsystems (shaders, surfaces, textures, draws) are implemented, each
 * will add its own state here, mirroring the layout of PGRAPHGLState.
 */
typedef struct PGRAPHMetalState {
    /* Core device + queue. */
    id<MTLDevice>            device;
    id<MTLCommandQueue>      queue;

    /* Display compositor output, returned to the UI as the framebuffer
     * "texture" id. On Metal this is a MTLTexture, but the rest of xemu
     * expects an integer handle (legacy GLuint). We hand back the opaque
     * pointer cast to int for now; a Metal-aware display path will be
     * added in a later stage. */
    id<MTLTexture>           display_texture;

    /* Stamp so callers can detect when device initialization finished. */
    bool                     initialized;

    /* Render targets. Mirrors the GL backend's surface cache: bindings are
     * keyed by guest VRAM address and kept until evicted or invalidated. */
    QTAILQ_HEAD(, MetalSurfaceBinding) surfaces;
    MetalSurfaceBinding      *color_binding;
    MetalSurfaceBinding      *zeta_binding;
    bool                     downloads_pending;
    QemuEvent                downloads_complete;
    bool                     download_dirty_surfaces_pending;
    QemuEvent                dirty_surfaces_download_complete;

    /* Shaders generated from live pgraph state, keyed on ShaderState. */
    GHashTable               *shader_cache;
    MetalShaderBinding       *shader_binding;
    unsigned int             shader_gen_successes;
    unsigned int             shader_gen_failures;

    /* Vertex data. vram_buffer aliases guest VRAM zero-copy where possible. */
    id<MTLBuffer>            vram_buffer;
    bool                     vram_buffer_is_copy;
    id<MTLBuffer>            const_attr_buffer;
    id<MTLBuffer>            index_buffer;
    size_t                   attr_buffer_offset[NV2A_VERTEXSHADER_ATTRIBUTES];
    bool                     attr_is_constant[NV2A_VERTEXSHADER_ATTRIBUTES];

    /* Pipeline states, keyed on shader + attachment + blend state. */
    GHashTable               *pipeline_cache;
    unsigned int             pipeline_count;
    unsigned int             pipeline_failures;

    /*
     * Depth target used when the guest binds none. The generated fragment
     * shader always writes gl_FragDepth -- the NV2A w-buffering emulation
     * needs it -- and Metal rejects a pipeline that writes depth with no
     * depth attachment, where GL simply discards the write. So an
     * unbound-zeta draw gets a throwaway target of the right size.
     */
    id<MTLTexture>           scratch_depth;

    /* Textures. */
    GHashTable               *texture_cache;
    GHashTable               *sampler_cache;
    id<MTLTexture>           white_texture;
    id<MTLTexture>           white_texture_cube;
    id<MTLTexture>           white_texture_3d;
    /*
     * Ratio of host texels to guest texels for whatever is bound to each
     * stage: 1.0 normally, the surface scale factor when the stage samples a
     * live render target. The generated shader divides unnormalized (linear)
     * texture coordinates by textureSize/texScale, so leaving this zero makes
     * every linear texture sample at the wrong rate.
     */
    float                    texture_scale[NV2A_MAX_TEXTURES];
    unsigned long            texture_type_mismatch;
    unsigned long            texture_uploads;
    unsigned long            texture_unsupported;
    unsigned long            surface_as_texture;
    bool                     reported_format[64];

    /* Uniform staging. setVertexBytes: caps at 4 KB and the vertex block
     * is larger than that, so it needs a real buffer. */
    id<MTLBuffer>            vsh_uniform_buffer;
    id<MTLBuffer>            psh_uniform_buffer;

    /* Current draw. */
    id<MTLCommandBuffer>          command_buffer;
    id<MTLRenderCommandEncoder>   encoder;
    /*
     * Attachments the open encoder was created against. A render pass is
     * expensive on a tile GPU -- it loads and stores the whole target -- so
     * consecutive draws share one encoder and it is only restarted when the
     * attachments actually change.
     */
    id<MTLTexture>                encoder_color;
    id<MTLTexture>                encoder_depth;
    unsigned long                 encoder_draws;
    unsigned long                 passes;
    unsigned long                 submits;
    unsigned long            draws_issued;
    unsigned long            expanded_prims;
    unsigned long            unsupported_prims;
    unsigned long            unsupported_draws;
    unsigned long            inline_draws;
    unsigned long            cmdbuf_errors;
    MTLVertexDescriptor      *pending_vd;

    /* Diagnostics: XEMU_METAL_DEBUG_POS makes the vertex shader also write
     * its computed clip position here, so the transform can be read back. */
    /*
     * Occlusion queries. Metal counts samples that pass depth/stencil into a
     * visibility buffer attached to the render pass; one slot per counted
     * draw, summed when the guest asks for the report.
     */
    id<MTLBuffer>            visibility_buffer;
    unsigned int             visibility_slot;
    uint64_t                 zpass_pixel_count_result;

    id<MTLBuffer>            debug_pos_buffer;
    unsigned int             debug_pos_count;
} PGRAPHMetalState;

#endif /* XEMU_NV2A_PGRAPH_METAL_RENDERER_H */
