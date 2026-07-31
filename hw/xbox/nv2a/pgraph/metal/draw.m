/*
 * Geforce NV2A PGRAPH Metal Renderer - draw path
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
 * Shader binding for the Metal backend.
 *
 * A survey of a real BIOS boot (120k draws over 60s) showed the vertex
 * submission mix is overwhelmingly one path:
 *
 *   inline_elements 99.7%   draw_arrays 0.3%   inline_array 0.1%
 *   inline_buffer   0%
 *
 * so the draw dispatch that follows this stage targets indexed drawing
 * first. What lives here now is the piece that has to work before any of
 * that matters: turning the *guest's actual* shader state into a compiled
 * Metal library.
 *
 * That is worth doing on its own because up to this point the MSL generator
 * had only ever been run against hand-built states in the self-test. This
 * exercises it against whatever a real title asks for, and reports the
 * failures with the generated source so they can be diagnosed.
 */

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "hw/xbox/nv2a/pgraph/glsl/shaders.h"
#include "renderer.h"
#include "shaders.h"
#include "surface.h"
#include "draw.h"
#include "vertex.h"
#include "texture.h"
#include "hw/xbox/nv2a/pgraph/glsl/msl.h"

static guint shader_state_hash(gconstpointer key)
{
    /* ShaderState is memset to zero before population precisely so it can be
     * hashed as raw bytes; see pgraph_glsl_get_shader_state(). */
    return g_bytes_hash(key);
}

static gboolean shader_state_equal(gconstpointer a, gconstpointer b)
{
    return g_bytes_equal(a, b);
}

static void shader_binding_free(gpointer data)
{
    MetalShaderBinding *b = data;
    b->vsh_library = nil;
    b->psh_library = nil;
    b->vsh_function = nil;
    b->psh_function = nil;
    g_free(b);
}

void pgraph_metal_init_shader_cache(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    r->shader_cache = g_hash_table_new_full(shader_state_hash,
                                            shader_state_equal,
                                            (GDestroyNotify)g_bytes_unref,
                                            shader_binding_free);
    r->shader_binding = NULL;
    r->shader_gen_failures = 0;
    r->shader_gen_successes = 0;

    r->pipeline_cache = g_hash_table_new_full(
        g_bytes_hash, g_bytes_equal, (GDestroyNotify)g_bytes_unref,
        (GDestroyNotify)CFRelease);
}

void pgraph_metal_finalize_shader_cache(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    if (r->pipeline_cache) {
        g_hash_table_destroy(r->pipeline_cache);
        r->pipeline_cache = NULL;
    }
    if (r->shader_cache) {
        g_hash_table_destroy(r->shader_cache);
        r->shader_cache = NULL;
    }
    r->shader_binding = NULL;
}

/*
 * Compile one stage. On failure the generated source is dumped once per
 * distinct failure so the offending construct can be found; repeat failures
 * of the same state are silent because a stuck title would otherwise flood
 * the log at draw rate.
 */
/* The out-params are explicitly __strong: they are written into a heap
 * struct, and ARC otherwise infers __autoreleasing for indirect object
 * parameters, which it refuses for non-local storage. */
static bool compile_stage(PGRAPHMetalState *r, MString *src, const char *what,
                          __strong id<MTLLibrary> *out_library,
                          __strong id<MTLFunction> *out_function)
{
    Error *err = NULL;
    id<MTLLibrary> lib = pgraph_metal_compile_shader(r->device, src, &err);

    if (lib == nil) {
        if (r->shader_gen_failures < 8) {
            fprintf(stderr, "nv2a: metal: %s compile failed: %s\n", what,
                    error_get_pretty(err));
            fprintf(stderr, "--- generated %s ---\n%s\n--------------------\n",
                    what, mstring_get_str(src));
        }
        error_free(err);
        return false;
    }

    id<MTLFunction> fn =
        [lib newFunctionWithName:@(PGRAPH_METAL_SHADER_ENTRY)];
    if (fn == nil) {
        fprintf(stderr, "nv2a: metal: %s has no entry point '%s'\n", what,
                PGRAPH_METAL_SHADER_ENTRY);
        return false;
    }

    *out_library = lib;
    *out_function = fn;
    return true;
}

MetalShaderBinding *pgraph_metal_bind_shaders(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    ShaderState state = pgraph_glsl_get_shader_state(pg);

    GBytes *key = g_bytes_new(&state, sizeof(state));
    MetalShaderBinding *binding = g_hash_table_lookup(r->shader_cache, key);

    if (binding) {
        g_bytes_unref(key);
        r->shader_binding = binding;
        return binding;
    }

    binding = g_new0(MetalShaderBinding, 1);
    binding->state = state;

    MString *vsh_src = pgraph_metal_gen_vsh(&state.vsh);
    MString *psh_src = pgraph_metal_gen_psh(&state.psh);

    bool ok = compile_stage(r, vsh_src, "vertex shader", &binding->vsh_library,
                            &binding->vsh_function) &&
              compile_stage(r, psh_src, "fragment shader",
                            &binding->psh_library, &binding->psh_function);

    mstring_unref(vsh_src);
    mstring_unref(psh_src);

    binding->valid = ok;

    if (ok) {
        r->shader_gen_successes++;
    } else {
        r->shader_gen_failures++;
    }

    /* One line per *new* state, not per draw: the interesting number is how
     * many distinct shaders a title asks for, which is bounded. */
    if ((r->shader_gen_successes + r->shader_gen_failures) % 25 == 0 ||
        !ok) {
        fprintf(stderr,
                "nv2a: metal: shader states compiled=%u failed=%u "
                "(latest: %s vsh, %d psh stages)\n",
                r->shader_gen_successes, r->shader_gen_failures,
                state.vsh.is_fixed_function ? "fixed-function" :
                                              "programmable",
                state.psh.combiner_control & 0xFF);
    }

    /* Cache failures too: a state that does not compile will not start
     * compiling later, and re-running the generator every draw is expensive. */
    g_hash_table_insert(r->shader_cache, key, binding);

    r->shader_binding = binding;
    return binding;
}


/* ------------------------------------------------------------------ */
/* Uniform upload.                                                     */
/* ------------------------------------------------------------------ */

/*
 * The MSL uniform struct and the C *UniformValues struct have different
 * layouts and cannot be made to match: the C typedefs are float[N], which is
 * 4-byte aligned, so members land at offsets MSL may not place a vector at.
 * So copy member by member, and element by element where the strides differ
 * (which they do for vec3: 12 bytes in C, 16 in MSL).
 */
static void upload_uniforms(uint8_t *dst, const uint8_t *src,
                            const MslUniformMember *members, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const MslUniformMember *m = &members[i];
        if (m->count == 0) {
            continue;
        }

        if (m->stride == m->src_stride) {
            memcpy(dst + m->offset, src + m->src_offset,
                   m->stride * m->count);
        } else {
            for (size_t e = 0; e < m->count; e++) {
                memcpy(dst + m->offset + e * m->stride,
                       src + m->src_offset + e * m->src_stride,
                       m->src_stride);
            }
        }
    }
}

static void bind_uniforms(PGRAPHState *pg, id<MTLRenderCommandEncoder> enc,
                          const ShaderState *state)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    /* All-zero locs means "every uniform is present", which is what the
     * generated MSL struct declares. */
    VshUniformLocs vsh_locs = { 0 };
    PshUniformLocs psh_locs = { 0 };
    VshUniformValues vsh_values;
    PshUniformValues psh_values;

    memset(&vsh_values, 0, sizeof(vsh_values));
    memset(&psh_values, 0, sizeof(psh_values));

    pgraph_glsl_set_vsh_uniform_values(pg, &state->vsh, vsh_locs, &vsh_values);
    pgraph_glsl_set_psh_uniform_values(pg, psh_locs, &psh_values);

    /*
     * texScale is not part of the shared uniform setup -- each backend fills
     * it from its own texture bindings, as GL and Vulkan do. It is the host
     * texels per guest texel, and the generated shader divides unnormalized
     * coordinates by textureSize/texScale. Leaving it zero made every linear
     * texture tile across the surface instead of covering it once.
     */
    for (int i = 0; i < NV2A_MAX_TEXTURES; i++) {
        psh_values.texScale[i] = r->texture_scale[i];
    }

    MslUniformMember vsh_members[VshUniform__COUNT];
    MslUniformMember psh_members[PshUniform__COUNT];

    bool skip_inline = !state->vsh.uniform_attrs;

    size_t vsh_size = pgraph_msl_uniform_layout(
        VshUniformInfo, VshUniform__COUNT,
        skip_inline ? VshUniform_inlineValue : -1, vsh_members);
    size_t psh_size = pgraph_msl_uniform_layout(
        PshUniformInfo, PshUniform__COUNT, -1, psh_members);

    uint8_t *vsh_buf = g_alloca(vsh_size);
    uint8_t *psh_buf = g_alloca(psh_size);
    memset(vsh_buf, 0, vsh_size);
    memset(psh_buf, 0, psh_size);

    upload_uniforms(vsh_buf, (const uint8_t *)&vsh_values, vsh_members,
                    VshUniform__COUNT);

    const char *uonly = getenv("XEMU_METAL_DEBUG_POS_TARGET");
    bool utarget_ok =
        !uonly || (r->color_binding &&
                   (unsigned long)r->color_binding->vram_addr ==
                       strtoul(uonly, NULL, 16));
    if (getenv("XEMU_METAL_DUMP_UNIFORMS") && utarget_ok) {
        static int n;
        if (n++ < 2) {
            fprintf(stderr, "  [ff=%d target=@%08lx binding_dim=%ux%u]\n",
                    state->vsh.is_fixed_function,
                    r->color_binding
                        ? (unsigned long)r->color_binding->vram_addr : 0UL,
                    pg->surface_binding_dim.width,
                    pg->surface_binding_dim.height);
            fprintf(stderr,
                    "uniforms: surfaceSize=(%.1f,%.1f) clipRange=(%g,%g,%g,%g)\n"
                    "  compositeMat c[0..3]:\n"
                    "    %8.3f %8.3f %8.3f %8.3f\n"
                    "    %8.3f %8.3f %8.3f %8.3f\n"
                    "    %8.3f %8.3f %8.3f %8.3f\n"
                    "    %8.3f %8.3f %8.3f %8.3f\n"
                    "  psh: clipRegion[0]=(%d,%d,%d,%d) surfaceScale=(%d,%d)"
                    " exclusive=%d alpha_test=%d\n",
                    vsh_values.surfaceSize[0][0], vsh_values.surfaceSize[0][1],
                    vsh_values.clipRange[0][0], vsh_values.clipRange[0][1],
                    vsh_values.clipRange[0][2], vsh_values.clipRange[0][3],
                    vsh_values.c[0][0], vsh_values.c[0][1],
                    vsh_values.c[0][2], vsh_values.c[0][3],
                    vsh_values.c[1][0], vsh_values.c[1][1],
                    vsh_values.c[1][2], vsh_values.c[1][3],
                    vsh_values.c[2][0], vsh_values.c[2][1],
                    vsh_values.c[2][2], vsh_values.c[2][3],
                    vsh_values.c[3][0], vsh_values.c[3][1],
                    vsh_values.c[3][2], vsh_values.c[3][3],
                    psh_values.clipRegion[0][0], psh_values.clipRegion[0][1],
                    psh_values.clipRegion[0][2], psh_values.clipRegion[0][3],
                    psh_values.surfaceScale[0][0],
                    psh_values.surfaceScale[0][1],
                    state->psh.window_clip_exclusive,
                    state->psh.alpha_test);
        }
    }
    upload_uniforms(psh_buf, (const uint8_t *)&psh_values, psh_members,
                    PshUniform__COUNT);

    /*
     * setVertexBytes:/setFragmentBytes: are limited to 4 KB. The vertex
     * uniform block is ~6 KB (192 float4 constants alone are 3 KB), so it
     * has to go through a real buffer or the binding is rejected and the
     * shader reads nothing.
     */
    if (r->vsh_uniform_buffer == nil ||
        r->vsh_uniform_buffer.length < vsh_size) {
        r->vsh_uniform_buffer =
            [r->device newBufferWithLength:vsh_size
                                   options:MTLResourceStorageModeShared];
    }
    memcpy(r->vsh_uniform_buffer.contents, vsh_buf, vsh_size);
    [enc setVertexBuffer:r->vsh_uniform_buffer
                  offset:0
                 atIndex:MSL_UNIFORM_BUFFER_INDEX];

    if (psh_size <= 4096) {
        [enc setFragmentBytes:psh_buf length:psh_size
                      atIndex:MSL_UNIFORM_BUFFER_INDEX];
    } else {
        if (r->psh_uniform_buffer == nil ||
            r->psh_uniform_buffer.length < psh_size) {
            r->psh_uniform_buffer =
                [r->device newBufferWithLength:psh_size
                                       options:MTLResourceStorageModeShared];
        }
        memcpy(r->psh_uniform_buffer.contents, psh_buf, psh_size);
        [enc setFragmentBuffer:r->psh_uniform_buffer
                        offset:0
                       atIndex:MSL_UNIFORM_BUFFER_INDEX];
    }
}

/* ------------------------------------------------------------------ */
/* Debug pipeline (bisect aid).                                        */
/* ------------------------------------------------------------------ */

/*
 * A shader pair that ignores every scrap of guest state and paints a large
 * fixed triangle. Under XEMU_METAL_DEBUG_SHADER it replaces the generated
 * pair, which splits the search space cleanly: if magenta appears, then
 * pipeline creation, the encoder, attachments, rasterization and readback
 * are all fine and the fault is in the generated shaders or their uniforms.
 * If nothing appears, the fault is in that plumbing instead.
 */
static const char *debug_shader_src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Out { float4 pos [[position]]; };\n"
    "vertex Out dbg_vs(uint vid [[vertex_id]]) {\n"
    "  float2 p[3] = { float2(-0.9, -0.9), float2(0.9, -0.9),\n"
    "                  float2(0.0,  0.9) };\n"
    "  Out o; o.pos = float4(p[vid % 3], 0.5, 1.0); return o;\n"
    "}\n"
    "fragment float4 dbg_fs() { return float4(1.0, 0.0, 1.0, 1.0); }\n";

/*
 * Generated vertex shader + constant-colour fragment shader. If shapes
 * appear, the transform and vertex fetch are correct and the fault is in the
 * generated fragment shader; if not, it is upstream of that.
 */
static id<MTLRenderPipelineState> get_hybrid_pipeline(PGRAPHMetalState *r,
                                                      MetalShaderBinding *sb,
                                                      MTLVertexDescriptor *vd)
{
    static id<MTLLibrary> lib;
    NSError *err = nil;

    if (lib == nil) {
        lib = [r->device newLibraryWithSource:@(debug_shader_src)
                                      options:nil
                                        error:&err];
        if (lib == nil) {
            return nil;
        }
    }

    MTLRenderPipelineDescriptor *pd =
        [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = sb->vsh_function;
    pd.fragmentFunction = [lib newFunctionWithName:@"dbg_fs"];
    pd.vertexDescriptor = vd;
    pd.colorAttachments[0].pixelFormat = r->color_binding->fmt.pixel_format;
    pd.depthAttachmentPixelFormat =
        r->zeta_binding ? r->zeta_binding->fmt.pixel_format
                        : MTLPixelFormatDepth32Float;
    if (r->zeta_binding && r->zeta_binding->fmt.stencil) {
        pd.stencilAttachmentPixelFormat =
            r->zeta_binding->fmt.pixel_format;
    }

    id<MTLRenderPipelineState> pso =
        [r->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (pso == nil) {
        static int once;
        if (!once++) {
            fprintf(stderr, "nv2a: metal: hybrid pipeline failed: %s\n",
                    [[err localizedDescription] UTF8String]);
        }
    }
    return pso;
}

static id<MTLRenderPipelineState> get_debug_pipeline(PGRAPHMetalState *r)
{
    static id<MTLRenderPipelineState> cached;
    if (cached != nil || r->color_binding == NULL) {
        return cached;
    }

    NSError *err = nil;
    id<MTLLibrary> lib =
        [r->device newLibraryWithSource:@(debug_shader_src)
                                options:nil
                                  error:&err];
    if (lib == nil) {
        fprintf(stderr, "nv2a: metal: debug shader failed: %s\n",
                [[err localizedDescription] UTF8String]);
        return nil;
    }

    MTLRenderPipelineDescriptor *pd =
        [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [lib newFunctionWithName:@"dbg_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"dbg_fs"];
    pd.colorAttachments[0].pixelFormat =
        r->color_binding->fmt.pixel_format;
    pd.depthAttachmentPixelFormat =
        r->zeta_binding ? r->zeta_binding->fmt.pixel_format
                        : MTLPixelFormatDepth32Float;

    cached = [r->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (cached == nil) {
        fprintf(stderr, "nv2a: metal: debug pipeline failed: %s\n",
                [[err localizedDescription] UTF8String]);
    }
    return cached;
}

/* ------------------------------------------------------------------ */
/* Pipeline state.                                                     */
/* ------------------------------------------------------------------ */

/*
 * Metal bakes the vertex descriptor into the pipeline state, so the descriptor
 * is part of the pipeline's identity and has to be part of the cache key.
 *
 * Keying on the pgraph attribute registers instead looks equivalent but is
 * not: the immediate-mode paths build a descriptor from the inline data
 * layout, which has no relationship to those registers. An inline_array draw
 * therefore hashed to whatever array-path pipeline last used the same
 * attribute formats, and Metal fetched its vertices with that pipeline's
 * strides -- reading zeroes. Hashing the descriptor itself makes the key
 * correct by construction, whoever built it.
 */
typedef struct MetalPipelineKey {
    const void *shader_binding;
    MTLPixelFormat color_format;
    MTLPixelFormat depth_format;
    uint32_t blend;
    uint32_t blend_color;
    uint32_t control_0;
    uint16_t compressed_attrs;
    uint16_t swizzle_attrs;
    struct {
        uint32_t format;
        uint32_t offset;
        uint32_t buffer_index;
    } attr[NV2A_VERTEXSHADER_ATTRIBUTES];
    struct {
        uint32_t stride;
        uint32_t step_function;
        uint32_t step_rate;
    } layout[METAL_VERTEX_BUFFER_BASE + NV2A_VERTEXSHADER_ATTRIBUTES];
} MetalPipelineKey;

static id<MTLRenderPipelineState> get_pipeline(NV2AState *d,
                                               MetalShaderBinding *sb,
                                               MTLVertexDescriptor *vd)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    MetalPipelineKey key;
    memset(&key, 0, sizeof(key));
    key.shader_binding = sb;
    key.color_format = r->color_binding ? r->color_binding->fmt.pixel_format
                                        : MTLPixelFormatInvalid;
    key.depth_format = r->zeta_binding ? r->zeta_binding->fmt.pixel_format
                                       : MTLPixelFormatDepth32Float;
    key.blend = pgraph_reg_r(pg, NV_PGRAPH_BLEND);
    key.blend_color = pgraph_reg_r(pg, NV_PGRAPH_BLENDCOLOR);
    key.control_0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
    key.compressed_attrs = pg->compressed_attrs;
    key.swizzle_attrs = pg->swizzle_attrs;
    for (int i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        MTLVertexAttributeDescriptor *a = vd.attributes[i];
        key.attr[i].format = (uint32_t)a.format;
        key.attr[i].offset = (uint32_t)a.offset;
        key.attr[i].buffer_index = (uint32_t)a.bufferIndex;
    }
    for (int i = 0; i < ARRAY_SIZE(key.layout); i++) {
        MTLVertexBufferLayoutDescriptor *l = vd.layouts[i];
        key.layout[i].stride = (uint32_t)l.stride;
        key.layout[i].step_function = (uint32_t)l.stepFunction;
        key.layout[i].step_rate = (uint32_t)l.stepRate;
    }

    GBytes *k = g_bytes_new(&key, sizeof(key));
    id<MTLRenderPipelineState> pso = (__bridge id<MTLRenderPipelineState>)
        g_hash_table_lookup(r->pipeline_cache, k);
    if (pso) {
        g_bytes_unref(k);
        return pso;
    }

    MTLRenderPipelineDescriptor *pd =
        [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = sb->vsh_function;
    pd.fragmentFunction = sb->psh_function;
    pd.vertexDescriptor = vd;

    if (r->color_binding) {
        MTLRenderPipelineColorAttachmentDescriptor *ca =
            pd.colorAttachments[0];
        ca.pixelFormat = r->color_binding->fmt.pixel_format;

        uint32_t control_0 = key.control_0;
        MTLColorWriteMask mask = MTLColorWriteMaskNone;
        if (control_0 & NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE) {
            mask |= MTLColorWriteMaskRed;
        }
        if (control_0 & NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE) {
            mask |= MTLColorWriteMaskGreen;
        }
        if (control_0 & NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE) {
            mask |= MTLColorWriteMaskBlue;
        }
        if (control_0 & NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE) {
            mask |= MTLColorWriteMaskAlpha;
        }
        ca.writeMask = mask;

        /* TODO: translate NV_PGRAPH_BLEND sfactor/dfactor/equation. Blending
         * off keeps first geometry legible; alpha-blended passes will look
         * wrong until this lands. */
        ca.blendingEnabled = NO;
    }

    if (r->zeta_binding) {
        pd.depthAttachmentPixelFormat = r->zeta_binding->fmt.pixel_format;
        if (r->zeta_binding->fmt.stencil) {
            pd.stencilAttachmentPixelFormat =
                r->zeta_binding->fmt.pixel_format;
        }
    } else {
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    }

    NSError *err = nil;
    pso = [r->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (pso == nil) {
        if (r->pipeline_failures < 4) {
            fprintf(stderr, "nv2a: metal: pipeline creation failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
        }
        r->pipeline_failures++;
        g_bytes_unref(k);
        return nil;
    }

    r->pipeline_count++;
    g_hash_table_insert(r->pipeline_cache, k,
                        (__bridge_retained void *)pso);
    return pso;
}

static id<MTLDepthStencilState> get_depth_stencil_state(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    uint32_t control_0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
    bool depth_test = control_0 & NV_PGRAPH_CONTROL_0_ZENABLE;
    bool depth_write = control_0 & NV_PGRAPH_CONTROL_0_ZWRITEENABLE;
    uint32_t func = GET_MASK(control_0, NV_PGRAPH_CONTROL_0_ZFUNC);

    static const MTLCompareFunction cmp_map[] = {
        MTLCompareFunctionNever,        MTLCompareFunctionLess,
        MTLCompareFunctionEqual,        MTLCompareFunctionLessEqual,
        MTLCompareFunctionGreater,      MTLCompareFunctionNotEqual,
        MTLCompareFunctionGreaterEqual, MTLCompareFunctionAlways,
    };

    MTLDepthStencilDescriptor *dsd = [[MTLDepthStencilDescriptor alloc] init];
    dsd.depthCompareFunction =
        (depth_test && func < ARRAY_SIZE(cmp_map)) ? cmp_map[func]
                                                   : MTLCompareFunctionAlways;
    dsd.depthWriteEnabled = depth_write && r->zeta_binding != NULL;

    /* Bisect aid: take depth out of the picture entirely. */
    if (getenv("XEMU_METAL_DEPTH_ALWAYS")) {
        dsd.depthCompareFunction = MTLCompareFunctionAlways;
    }

    return [r->device newDepthStencilStateWithDescriptor:dsd];
}

/* ------------------------------------------------------------------ */
/* Encoder.                                                            */
/* ------------------------------------------------------------------ */


/* ------------------------------------------------------------------ */
/* Primitive expansion.                                                */
/* ------------------------------------------------------------------ */

/*
 * Metal has no fan, quad or line-loop primitives. The NV2A uses them heavily
 * -- triangle fans alone were 38% of draws in a boot survey -- so they are
 * expanded into triangle/line lists on the CPU here.
 *
 * Winding is preserved: the expansions emit the same vertex order the
 * hardware rasterizes, so face culling behaves the same afterwards.
 */
static bool primitive_needs_expansion(unsigned int mode)
{
    switch (mode) {
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
    case PRIM_TYPE_QUADS:
    case PRIM_TYPE_QUAD_STRIP:
    case PRIM_TYPE_LINE_LOOP:
        return true;
    default:
        return false;
    }
}

/* Upper bound on expanded index count, for buffer sizing. */
static size_t expanded_index_count(unsigned int mode, size_t n)
{
    switch (mode) {
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
        return n >= 3 ? (n - 2) * 3 : 0;
    case PRIM_TYPE_QUADS:
        return (n / 4) * 6;
    case PRIM_TYPE_QUAD_STRIP:
        return n >= 4 ? ((n - 2) / 2) * 6 : 0;
    case PRIM_TYPE_LINE_LOOP:
        return n >= 2 ? n * 2 : 0;
    default:
        return n;
    }
}

/*
 * Expand `src` (or an implicit 0..n-1 sequence when src is NULL) into `dst`.
 * Returns the number of indices written.
 */
static size_t expand_primitive(unsigned int mode, const uint32_t *src,
                               size_t n, uint32_t *dst)
{
#define IDX(i) (src ? src[(i)] : (uint32_t)(i))
    size_t o = 0;

    switch (mode) {
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
        /* (0,1,2) (0,2,3) (0,3,4) ... */
        for (size_t i = 2; i < n; i++) {
            dst[o++] = IDX(0);
            dst[o++] = IDX(i - 1);
            dst[o++] = IDX(i);
        }
        break;

    case PRIM_TYPE_QUADS:
        /* Independent quads: (0,1,2) (0,2,3) per group of four. */
        for (size_t i = 0; i + 3 < n; i += 4) {
            dst[o++] = IDX(i);
            dst[o++] = IDX(i + 1);
            dst[o++] = IDX(i + 2);
            dst[o++] = IDX(i);
            dst[o++] = IDX(i + 2);
            dst[o++] = IDX(i + 3);
        }
        break;

    case PRIM_TYPE_QUAD_STRIP:
        /* Each pair of new vertices closes a quad with the previous pair. */
        for (size_t i = 0; i + 3 < n; i += 2) {
            dst[o++] = IDX(i);
            dst[o++] = IDX(i + 1);
            dst[o++] = IDX(i + 3);
            dst[o++] = IDX(i);
            dst[o++] = IDX(i + 3);
            dst[o++] = IDX(i + 2);
        }
        break;

    case PRIM_TYPE_LINE_LOOP:
        /* Line list, with the closing segment back to the first vertex. */
        for (size_t i = 0; i < n; i++) {
            dst[o++] = IDX(i);
            dst[o++] = IDX((i + 1) % n);
        }
        break;

    default:
        for (size_t i = 0; i < n; i++) {
            dst[o++] = IDX(i);
        }
        break;
    }

    return o;
#undef IDX
}

/* What the expanded stream draws as. */
static MTLPrimitiveType expanded_primitive_type(unsigned int mode)
{
    return (mode == PRIM_TYPE_LINE_LOOP) ? MTLPrimitiveTypeLine
                                         : MTLPrimitiveTypeTriangle;
}

static MTLPrimitiveType metal_primitive_type(unsigned int mode, bool *supported)
{
    *supported = true;
    switch (mode) {
    case PRIM_TYPE_POINTS:         return MTLPrimitiveTypePoint;
    case PRIM_TYPE_LINES:          return MTLPrimitiveTypeLine;
    case PRIM_TYPE_LINE_STRIP:     return MTLPrimitiveTypeLineStrip;
    case PRIM_TYPE_TRIANGLES:      return MTLPrimitiveTypeTriangle;
    case PRIM_TYPE_TRIANGLE_STRIP: return MTLPrimitiveTypeTriangleStrip;
    default:
        /* Anything else is expanded to a triangle/line list beforehand, so
         * reaching here means an unknown mode. */
        *supported = false;
        return MTLPrimitiveTypeTriangle;
    }
}

/* See scratch_depth in renderer.h for why this exists. */
static id<MTLTexture> get_scratch_depth(PGRAPHMetalState *r,
                                        NSUInteger w, NSUInteger h)
{
    if (r->scratch_depth != nil && r->scratch_depth.width >= w &&
        r->scratch_depth.height >= h) {
        return r->scratch_depth;
    }

    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                     width:MAX(w, 1)
                                    height:MAX(h, 1)
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;

    r->scratch_depth = [r->device newTextureWithDescriptor:td];
    return r->scratch_depth;
}

void pgraph_metal_flush_gpu(NV2AState *d, bool wait)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (r->encoder != nil) {
        [r->encoder endEncoding];
        r->encoder = nil;
        r->encoder_color = nil;
        r->encoder_depth = nil;
    }

    if (r->command_buffer == nil) {
        return;
    }

    [r->command_buffer commit];
    r->submits++;

    if (wait) {
        [r->command_buffer waitUntilCompleted];

        /*
         * A command buffer that errors out still completes as far as the CPU
         * is concerned, so without this a rejected pass is indistinguishable
         * from one that drew nothing.
         */
        if (r->command_buffer.status != MTLCommandBufferStatusCompleted) {
            r->cmdbuf_errors++;
            if (r->cmdbuf_errors < 5) {
                NSError *e = r->command_buffer.error;
                fprintf(stderr,
                        "nv2a: metal: command buffer status=%ld error=%s\n",
                        (long)r->command_buffer.status,
                        e ? [[e localizedDescription] UTF8String] : "(none)");
            }
        }
    }

    r->command_buffer = nil;
}

/*
 * Per-draw encoder state. Separate from starting a pass because these change
 * between draws that share one, and setting them is cheap where a new render
 * pass is not.
 */
static void apply_dynamic_state(NV2AState *d, id<MTLRenderCommandEncoder> enc)
{
    PGRAPHState *pg = &d->pgraph;

    unsigned int vp_w = pg->surface_binding_dim.width;
    unsigned int vp_h = pg->surface_binding_dim.height;
    pgraph_apply_scaling_factor(pg, &vp_w, &vp_h);
    [enc setViewport:(MTLViewport){ 0.0, 0.0, (double)vp_w, (double)vp_h,
                                    0.0, 1.0 }];

    unsigned int xmin = pg->surface_shape.clip_x;
    unsigned int ymin = pg->surface_shape.clip_y;
    unsigned int sw = pg->surface_shape.clip_width;
    unsigned int sh = pg->surface_shape.clip_height;
    pgraph_apply_anti_aliasing_factor(pg, &xmin, &ymin);
    pgraph_apply_anti_aliasing_factor(pg, &sw, &sh);
    pgraph_apply_scaling_factor(pg, &xmin, &ymin);
    pgraph_apply_scaling_factor(pg, &sw, &sh);

    /* Metal validates that the scissor lies inside the attachment. */
    sw = MIN(sw, vp_w > xmin ? vp_w - xmin : 0);
    sh = MIN(sh, vp_h > ymin ? vp_h - ymin : 0);
    if (getenv("XEMU_METAL_FULL_SCISSOR")) {
        [enc setScissorRect:(MTLScissorRect){ 0, 0, vp_w, vp_h }];
    } else if (sw && sh) {
        [enc setScissorRect:(MTLScissorRect){ xmin, ymin, sw, sh }];
    }

    /* Winding is reversed because clip-space y is inverted, matching GL. */
    [enc setFrontFacingWinding:(pgraph_reg_r(pg, NV_PGRAPH_SETUPRASTER) &
                                NV_PGRAPH_SETUPRASTER_FRONTFACE)
                                   ? MTLWindingClockwise
                                   : MTLWindingCounterClockwise];

    if (pgraph_reg_r(pg, NV_PGRAPH_SETUPRASTER) &
        NV_PGRAPH_SETUPRASTER_CULLENABLE) {
        uint32_t cull = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_SETUPRASTER),
                                 NV_PGRAPH_SETUPRASTER_CULLCTRL);
        MTLCullMode m = MTLCullModeNone;
        switch (cull) {
        case NV_PGRAPH_SETUPRASTER_CULLCTRL_FRONT: m = MTLCullModeFront; break;
        case NV_PGRAPH_SETUPRASTER_CULLCTRL_BACK:  m = MTLCullModeBack; break;
        default: break;
        }
        [enc setCullMode:getenv("XEMU_METAL_NO_CULL") ? MTLCullModeNone : m];
    } else {
        [enc setCullMode:MTLCullModeNone];
    }
}

static id<MTLRenderCommandEncoder> begin_encoder(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    id<MTLTexture> want_color =
        r->color_binding ? r->color_binding->texture : nil;
    id<MTLTexture> want_depth =
        r->zeta_binding ? r->zeta_binding->texture : nil;

    /*
     * Reuse the open pass when it already targets these attachments. Ending
     * and restarting a render pass per draw costs a full store and reload of
     * the render target on a tile GPU, which at a few thousand draws a frame
     * dominates everything else.
     */
    if (r->encoder != nil && r->encoder_color == want_color &&
        r->encoder_depth == want_depth) {
        apply_dynamic_state(d, r->encoder);
        return r->encoder;
    }

    if (r->encoder != nil) {
        [r->encoder endEncoding];
        r->encoder = nil;
    }

    MTLRenderPassDescriptor *rp =
        [MTLRenderPassDescriptor renderPassDescriptor];

    /* Load, not Clear: clears are issued separately and must survive. */
    if (r->color_binding) {
        rp.colorAttachments[0].texture = r->color_binding->texture;
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    if (r->zeta_binding) {
        rp.depthAttachment.texture = r->zeta_binding->texture;
        rp.depthAttachment.loadAction = MTLLoadActionLoad;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        if (r->zeta_binding->fmt.stencil) {
            rp.stencilAttachment.texture = r->zeta_binding->texture;
            rp.stencilAttachment.loadAction = MTLLoadActionLoad;
            rp.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    } else if (r->color_binding) {
        rp.depthAttachment.texture =
            get_scratch_depth(r, r->color_binding->texture.width,
                              r->color_binding->texture.height);
        rp.depthAttachment.loadAction = MTLLoadActionDontCare;
        rp.depthAttachment.storeAction = MTLStoreActionDontCare;
    }

    if (r->visibility_buffer == nil) {
        r->visibility_buffer =
            [r->device newBufferWithLength:METAL_VISIBILITY_SLOTS * 8
                                   options:MTLResourceStorageModeShared];
        memset(r->visibility_buffer.contents, 0, METAL_VISIBILITY_SLOTS * 8);
    }
    rp.visibilityResultBuffer = r->visibility_buffer;

    if (r->command_buffer == nil) {
        r->command_buffer = [r->queue commandBuffer];
    }
    id<MTLRenderCommandEncoder> enc =
        [r->command_buffer renderCommandEncoderWithDescriptor:rp];
    r->encoder_color = want_color;
    r->encoder_depth = want_depth;
    r->passes++;

    apply_dynamic_state(d, enc);

    if (getenv("XEMU_METAL_TARGET_COUNTS") && r->color_binding) {
        /* Totals per target. An earlier version sampled and I read the
         * sample index as a total, twice. Count everything. */
        static hwaddr addrs[16];
        static unsigned long counts[16];
        static int naddr;
        static unsigned long total;
        int idx = -1;
        for (int i = 0; i < naddr; i++) {
            if (addrs[i] == r->color_binding->vram_addr) {
                idx = i;
            }
        }
        if (idx < 0 && naddr < 16) {
            idx = naddr++;
            addrs[idx] = r->color_binding->vram_addr;
            counts[idx] = 0;
        }
        if (idx >= 0) {
            counts[idx]++;
        }
        if ((total++ % 1500) == 0) {
            fprintf(stderr, "target-counts (total %lu):", total);
            for (int i = 0; i < naddr; i++) {
                fprintf(stderr, " @%08lx=%lu", (unsigned long)addrs[i],
                        counts[i]);
            }
            fprintf(stderr, "\n");
        }
    }


    return enc;
}

void pgraph_metal_draw_begin(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    uint32_t control_0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
    bool mask_alpha = control_0 & NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE;
    bool mask_red = control_0 & NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE;
    bool mask_green = control_0 & NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE;
    bool mask_blue = control_0 & NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE;
    bool color_write = mask_alpha || mask_red || mask_green || mask_blue;
    bool depth_test = control_0 & NV_PGRAPH_CONTROL_0_ZENABLE;
    bool stencil_test = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_1) &
                        NV_PGRAPH_CONTROL_1_STENCIL_TEST_ENABLE;
    bool is_nop_draw = !(color_write || depth_test || stencil_test);

    pgraph_metal_surface_update(d, true, true, depth_test || stencil_test);

    if (is_nop_draw) {
        return;
    }

    if (!r->color_binding && !r->zeta_binding) {
        return;
    }

    MetalShaderBinding *sb = pgraph_metal_bind_shaders(pg);
    if (!sb->valid) {
        return;
    }

    /* Building the vertex descriptor also latches per-attribute buffer
     * offsets and the compressed/swizzle masks the shader state depends on,
     * so it has to happen before the pipeline lookup. */
    MTLVertexDescriptor *vd = pgraph_metal_build_vertex_descriptor(d);
    r->pending_vd = vd;

    /*
     * The pipeline is not created here. Which vertex descriptor is correct
     * depends on how the guest submits vertices -- array, inline buffer or
     * inline array -- and that is not settled until the BEGIN/END block
     * closes, so flush_draw creates and sets it.
     */
    bool debug_shader = getenv("XEMU_METAL_DEBUG_SHADER") != NULL;

    r->encoder = begin_encoder(d);
    [r->encoder setDepthStencilState:get_depth_stencil_state(d)];

    if (!debug_shader) {
        /* Textures first: binding them latches the per-stage scale that the
         * fragment uniforms carry. */
        pgraph_metal_bind_textures(d, r->encoder);
        bind_uniforms(pg, r->encoder, &sb->state);
        pgraph_metal_bind_vertex_buffers(d, r->encoder);
    }
}

void pgraph_metal_draw_end(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    uint32_t control_0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
    bool mask_alpha = control_0 & NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE;
    bool mask_red = control_0 & NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE;
    bool mask_green = control_0 & NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE;
    bool mask_blue = control_0 & NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE;
    bool color_write = mask_alpha || mask_red || mask_green || mask_blue;
    bool depth_test = control_0 & NV_PGRAPH_CONTROL_0_ZENABLE;
    bool stencil_test = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_1) &
                        NV_PGRAPH_CONTROL_1_STENCIL_TEST_ENABLE;

    if (!(color_write || depth_test || stencil_test)) {
        return;
    }

    pgraph_metal_flush_draw(d);

    if (r->encoder != nil) {
        r->encoder_draws++;

        /*
         * The pass stays open. It is closed and submitted when the
         * attachments change or when something needs the result -- see
         * pgraph_metal_flush_gpu. The diagnostics below do need the result,
         * so they force a flush of their own.
         */
        bool debug_readback = getenv("XEMU_METAL_DEBUG_POS") ||
                              getenv("XEMU_METAL_TRACE_SCANOUT") ||
                              getenv("XEMU_METAL_SYNC_EVERY_DRAW");
        if (debug_readback) {
            pgraph_metal_flush_gpu(d, true);
        }

        /*
         * Raw input against the clip position the vertex shader actually
         * produced. Everything up to the rasterizer has been measured; this
         * is the one stage whose output was otherwise only inferred.
         */
        const char *only = getenv("XEMU_METAL_DEBUG_POS_TARGET");
        bool pos_target_ok =
            !only || (r->color_binding &&
                      (unsigned long)r->color_binding->vram_addr ==
                          strtoul(only, NULL, 16));
        bool pos_prog_ok =
            !getenv("XEMU_METAL_DEBUG_POS_PROG") ||
            (r->shader_binding &&
             !r->shader_binding->state.vsh.is_fixed_function);
        if (getenv("XEMU_METAL_DEBUG_POS") && r->debug_pos_buffer &&
            pos_target_ok && pos_prog_ok) {
            static int shown;
            if (shown++ < 3) {
                unsigned int n = r->debug_pos_count;
                if (n > 12) {
                    n = 12;
                }
                const float *out = (const float *)r->debug_pos_buffer.contents;

                fprintf(stderr,
                        "  paths: elems=%u arrays=%u inl_arr=%u inl_buf=%u\n",
                        pg->inline_elements_length, pg->draw_arrays_length,
                        pg->inline_array_length, pg->inline_buffer_length);
                for (int a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES; a++) {
                    VertexAttribute *va = &pg->vertex_attributes[a];
                    if (!va->count && r->attr_is_constant[a] &&
                        !va->stride) {
                        continue;
                    }
                    fprintf(stderr,
                            "  attr%-2d fmt=0x%x size=%u count=%u stride=%u "
                            "off=%u dma=%d const=%d base=%#lx",
                            a, va->format, va->size, va->count, va->stride,
                            va->offset, va->dma_select, r->attr_is_constant[a],
                            (unsigned long)r->attr_buffer_offset[a]);
                    /* What the CPU sees at the address the GPU was pointed
                     * at. If this disagrees with what the shader fetched,
                     * the binding is wrong; if it agrees, the data is. */
                    if (!r->attr_is_constant[a] &&
                        r->attr_buffer_offset[a] + 32 <
                            memory_region_size(d->vram)) {
                        const float *m =
                            (const float *)(d->vram_ptr +
                                            r->attr_buffer_offset[a]);
                        fprintf(stderr, " mem[%.3f %.3f %.3f | %.3f %.3f %.3f]",
                                m[0], m[1], m[2], m[3], m[4], m[5]);
                    }
                    fprintf(stderr, "\n");
                }
                if (pg->inline_elements_length) {
                    fprintf(stderr, "  idx:");
                    for (unsigned e = 0;
                         e < MIN(8u, pg->inline_elements_length); e++) {
                        fprintf(stderr, " %u", pg->inline_elements[e]);
                    }
                    fprintf(stderr, "\n");
                }
                fprintf(stderr,
                        "vtx-dump: target @%08lx prim=%d verts=%u ff=%d "
                        "rect0=%d texScale0=%.2f\n",
                        r->color_binding
                            ? (unsigned long)r->color_binding->vram_addr
                            : 0UL,
                        pg->primitive_mode, r->debug_pos_count,
                        r->shader_binding
                            ? r->shader_binding->state.vsh.is_fixed_function
                            : -1,
                        r->shader_binding
                            ? r->shader_binding->state.psh.rect_tex[0]
                            : -1,
                        r->texture_scale[0]);
                for (unsigned int k = 0; k < n; k++) {
                    const float *p = out + k * 16;
                    const float *t = out + k * 16 + 4;
                    const float *a0 = out + k * 16 + 8;
                    const float *a9 = out + k * 16 + 12;
                    fprintf(stderr,
                            "   v%u v0(%8.2f %8.2f %8.2f %6.2f)"
                            " v9(%7.3f %7.3f) -> clip(%9.3f %9.3f %9.3f %9.3f)"
                            " oT0(%8.3f %8.3f)\n",
                            k, a0[0], a0[1], a0[2], a0[3], a9[0], a9[1],
                            p[0], p[1], p[2], p[3], t[0], t[1]);
                }
            }
        }

        /*
         * Every live colour surface, not just the scanout one. The scanout is
         * the end of a chain -- the frame is built in offscreen targets and
         * composited in -- so dumping only the last link cannot say which
         * link introduced a fault.
         */
        const char *sdump = getenv("XEMU_METAL_DUMP_SURFACES");
        if (sdump) {
            static unsigned long tick;
            static unsigned round;
            unsigned long t = tick++;
            if (t == 2000 || t == 8000 || t == 20000) {
                round++;
                /* Read back what the batched passes actually produced. */
                pgraph_metal_flush_gpu(d, true);
                MetalSurfaceBinding *s;
                QTAILQ_FOREACH (s, &r->surfaces, entry) {
                    if (!s->color || s->texture == nil ||
                        s->texture.storageMode != MTLStorageModeShared) {
                        continue;
                    }
                    unsigned tw = (unsigned)s->texture.width;
                    unsigned th = (unsigned)s->texture.height;
                    size_t sn = (size_t)tw * th;
                    uint32_t *sb = g_malloc(sn * 4);
                    [s->texture getBytes:sb
                             bytesPerRow:tw * 4
                              fromRegion:MTLRegionMake2D(0, 0, tw, th)
                             mipmapLevel:0];
                    char path[1024];
                    snprintf(path, sizeof(path), "%s%u_%08lx.raw", sdump,
                             round, (unsigned long)s->vram_addr);
                    FILE *sf = fopen(path, "wb");
                    if (sf) {
                        uint32_t hdr[2] = { tw, th };
                        fwrite(hdr, sizeof(hdr), 1, sf);
                        fwrite(sb, 4, sn, sf);
                        fclose(sf);
                        fprintf(stderr, "surface @%08lx %ux%u -> %s\n",
                                (unsigned long)s->vram_addr, tw, th, path);
                    }
                    g_free(sb);
                }
            }
        }

        if (getenv("XEMU_METAL_TRACE_SCANOUT") && r->color_binding &&
            r->color_binding->vram_addr == 0x032a4000 &&
            r->color_binding->texture.storageMode == MTLStorageModeShared) {
            unsigned tw = (unsigned)r->color_binding->texture.width;
            unsigned th = (unsigned)r->color_binding->texture.height;
            size_t n = (size_t)tw * th;
            uint32_t *buf = g_malloc(n * 4);
            [r->color_binding->texture getBytes:buf
                                    bytesPerRow:tw * 4
                                     fromRegion:MTLRegionMake2D(0, 0, tw, th)
                                    mipmapLevel:0];
            size_t nz = 0;
            for (size_t k = 0; k < n; k++) {
                if (buf[k]) { nz++; }
            }

            /* A coverage count says pixels were written, not that they are
             * the right pixels. Dump the raw target so the frame can be
             * looked at. */
            const char *dump = getenv("XEMU_METAL_DUMP_FRAME");
            if (dump) {
                static unsigned long frame, seq;
                if ((frame++ % 25) == 3 && seq < 12) {
                    char path[1024];
                    snprintf(path, sizeof(path), "%s%03lu.raw", dump, seq++);
                    FILE *f = fopen(path, "wb");
                    if (f) {
                        uint32_t hdr[2] = { tw, th };
                        fwrite(hdr, sizeof(hdr), 1, f);
                        fwrite(buf, 4, n, f);
                        fclose(f);
                        fprintf(stderr, "frame dumped: %ux%u -> %s\n", tw, th,
                                path);
                    }
                }
            }
            g_free(buf);
            static unsigned long tn;
            if ((tn++ % 20) == 0) {
                fprintf(stderr,
                        "scanout-draw: prim=%d inline_arr=%d vsize_words=%d "
                        "-> TEXTURE nonzero=%zu/%zu\n",
                        pg->primitive_mode, pg->inline_array_length,
                        pg->inline_array_length, nz, n);
            }
        }

    }

    if (getenv("XEMU_METAL_TRACE_STATE") && r->color_binding &&
        r->shader_binding) {
        const char *t = getenv("XEMU_METAL_TRACE_STATE");
        if ((unsigned long)r->color_binding->vram_addr ==
            strtoul(t, NULL, 16)) {
            uint32_t bl = pgraph_reg_r(pg, NV_PGRAPH_BLEND);
            uint32_t c0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
            fprintf(stderr,
                    "ST prim=%d ff=%d zpersp=%d blend_en=%d blend=%08x "
                    "ctrl0=%08x zwrite=%d ztest=%d zfunc=%d "
                    "cmask(a%d r%d g%d b%d) alphatest=%d\n",
                    pg->primitive_mode,
                    r->shader_binding->state.vsh.is_fixed_function,
                    r->shader_binding->state.psh.z_perspective,
                    !!(bl & NV_PGRAPH_BLEND_EN), bl, c0,
                    !!(c0 & NV_PGRAPH_CONTROL_0_ZWRITEENABLE),
                    !!(c0 & NV_PGRAPH_CONTROL_0_ZENABLE),
                    (int)GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0),
                                  NV_PGRAPH_CONTROL_0_ZFUNC),
                    !!(c0 & NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE),
                    !!(c0 & NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE),
                    !!(c0 & NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE),
                    !!(c0 & NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE),
                    r->shader_binding->state.psh.alpha_test);
        }
    }

    if (getenv("XEMU_DRAW_TRACE")) {
        static unsigned long dn;
        static unsigned long cap;
        if (!cap) {
            const char *c = getenv("XEMU_DRAW_TRACE_MAX");
            cap = c ? strtoul(c, NULL, 10) : 4000;
        }
        if (dn < cap) {
            fprintf(stderr, "D%lu %08lx p%d e%u a%u ia%u ib%u t%08lx\n", dn,
                    r->color_binding
                        ? (unsigned long)r->color_binding->vram_addr : 0UL,
                    pg->primitive_mode, pg->inline_elements_length,
                    pg->draw_arrays_length, pg->inline_array_length,
                    pg->inline_buffer_length,
                    (unsigned long)pgraph_get_texture_phys_addr(pg, 0));
        }
        dn++;
    }

    pg->draw_time++;
    if (r->color_binding && pgraph_color_write_enabled(pg)) {
        r->color_binding->draw_time = pg->draw_time;
    }
    if (r->zeta_binding && pgraph_zeta_write_enabled(pg)) {
        r->zeta_binding->draw_time = pg->draw_time;
    }

    pgraph_metal_set_surface_dirty(pg, color_write,
                                   depth_test || stencil_test);

    if (getenv("XEMU_METAL_TARGET_STATS") && r->color_binding) {
        static unsigned long tn;
        if ((tn++ % 4000) == 0) {
            fprintf(stderr,
                    "draw-target: color @%08lx %ux%u (tex %lux%lu) aa=%u "
                    "clip=%u,%u %ux%u tex=%p\n",
                    (unsigned long)r->color_binding->vram_addr,
                    r->color_binding->width, r->color_binding->height,
                    (unsigned long)r->color_binding->texture.width,
                    (unsigned long)r->color_binding->texture.height,
                    pg->surface_shape.anti_aliasing,
                    pg->surface_shape.clip_x, pg->surface_shape.clip_y,
                    pg->surface_shape.clip_width,
                    pg->surface_shape.clip_height,
                    (__bridge void *)r->color_binding->texture);
        }
    }

    if (getenv("XEMU_METAL_THROUGHPUT")) {
        static unsigned long last_draws;
        static int64_t last_ns;
        int64_t now = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
        if (last_ns == 0) {
            last_ns = now;
        } else if (now - last_ns >= 2 * 1000000000LL) {
            double secs = (double)(now - last_ns) / 1e9;
            fprintf(stderr,
                    "throughput: %.0f draws/s (%lu draws, %lu passes, "
                    "%lu submits)\n",
                    (double)(r->draws_issued - last_draws) / secs,
                    r->draws_issued, r->passes, r->submits);
            last_draws = r->draws_issued;
            last_ns = now;
        }
    }

    if (getenv("XEMU_METAL_DRAW_STATS")) {
        static unsigned long n;
        if ((n++ % 20000) == 0) {
            fprintf(stderr,
                    "draw-stats: ends=%lu issued=%lu expanded=%lu unsup_prim=%lu "
                    "unsup_draw=%lu pipelines=%u/%u tex=%lu/%lu prim=%d\n",
                    n, r->draws_issued, r->expanded_prims,
                    r->unsupported_prims,
                    r->unsupported_draws, r->pipeline_count,
                    r->pipeline_failures, r->texture_uploads,
                    r->texture_unsupported, pg->primitive_mode);
        }
    }
}

void pgraph_metal_flush_draw(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    if (r->encoder == nil) {
        return;
    }

    if (getenv("XEMU_METAL_DEBUG_SHADER")) {
        [r->encoder drawPrimitives:MTLPrimitiveTypeTriangle
                       vertexStart:0
                       vertexCount:3];
        r->draws_issued++;
        return;
    }

    MetalShaderBinding *sb = r->shader_binding;
    if (sb == NULL || !sb->valid) {
        return;
    }

    /*
     * Bisect aid: skip the Nth draw into a chosen render target.
     * XEMU_METAL_SKIP_TARGET=<hex addr> XEMU_METAL_SKIP_INDEX=<n>.
     */
    const char *skip_t = getenv("XEMU_METAL_SKIP_TARGET");
    if (skip_t && r->color_binding &&
        (unsigned long)r->color_binding->vram_addr ==
            strtoul(skip_t, NULL, 16)) {
        static int seq;
        const char *si = getenv("XEMU_METAL_SKIP_INDEX");
        int want = si ? atoi(si) : -1;
        int cur = seq++;

        /* Dump the shader belonging to one specific draw, rather than the
         * first few states compiled, which need not include it. */
        const char *dvs = getenv("XEMU_METAL_DUMP_DRAW_VSH");
        if (dvs && cur == want) {
            MString *msl = pgraph_metal_gen_vsh(&sb->state.vsh);
            char path[1024];
            snprintf(path, sizeof(path), "%s_draw%d.msl", dvs, cur);
            FILE *f = fopen(path, "w");
            if (f) { fputs(mstring_get_str(msl), f); fclose(f); }
            mstring_unref(msl);
            fprintf(stderr, "draw %d shader dumped: ff=%d -> %s\n", cur,
                    sb->state.vsh.is_fixed_function, path);
        }

        if (!dvs && cur == want) {
            return;
        }
    }

    bool debug_shader = getenv("XEMU_METAL_DEBUG_SHADER") != NULL;
    bool debug_fs = getenv("XEMU_METAL_DEBUG_FS") != NULL;

    /*
     * Build the vertex descriptor for the path actually taken, bind the
     * matching buffers, and only then create the pipeline -- see the note in
     * draw_begin for why the order matters.
     */
    MTLVertexDescriptor *vd;
    unsigned int inline_count = 0;

    if (pg->inline_buffer_length) {
        vd = [MTLVertexDescriptor vertexDescriptor];
        inline_count = pgraph_metal_bind_inline_buffer(d, r->encoder, vd);
    } else if (pg->inline_array_length) {
        vd = [MTLVertexDescriptor vertexDescriptor];
        inline_count = pgraph_metal_bind_inline_array(d, r->encoder, vd);
    } else {
        vd = pgraph_metal_build_vertex_descriptor(d);
        pgraph_metal_bind_vertex_buffers(d, r->encoder);
    }

    id<MTLRenderPipelineState> pso;
    if (debug_shader) {
        pso = get_debug_pipeline(r);
    } else if (debug_fs && r->color_binding) {
        pso = get_hybrid_pipeline(r, sb, vd);
    } else {
        pso = get_pipeline(d, sb, vd);
    }
    if (pso == nil) {
        return;
    }
    [r->encoder setRenderPipelineState:pso];

    /*
     * Occlusion query. The guest uses the z-pass count to drive effects --
     * the BIOS boot animation scales its lens flares by it -- so reporting a
     * constant zero, as this backend did until now, makes those effects
     * behave as if nothing were ever visible.
     */
    if (pg->zpass_pixel_count_enable &&
        r->visibility_slot < METAL_VISIBILITY_SLOTS) {
        [r->encoder setVisibilityResultMode:MTLVisibilityResultModeCounting
                                     offset:r->visibility_slot * 8];
        r->visibility_slot++;
    } else {
        [r->encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled
                                     offset:0];
    }

    if (getenv("XEMU_METAL_DEBUG_POS")) {
        /* Four float4 per vertex: clip position, texcoord 0, and the raw v0
         * and v9 the shader fetched. Sized for the full 16-bit index range
         * so the shader cannot write past the end. */
        const unsigned int bytes = 65536 * 64;
        if (r->debug_pos_buffer == nil) {
            r->debug_pos_buffer =
                [r->device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
        }
        memset(r->debug_pos_buffer.contents, 0, bytes);
        r->debug_pos_count = inline_count ? inline_count
                                          : pg->inline_elements_length;
        [r->encoder setVertexBuffer:r->debug_pos_buffer
                             offset:0
                            atIndex:MSL_DEBUG_POS_BUFFER_INDEX];
    }

    unsigned int mode = pg->primitive_mode;
    bool expand = primitive_needs_expansion(mode);

    const uint32_t *src = NULL;
    size_t src_count = 0;

    if (pg->inline_elements_length) {
        src = pg->inline_elements;
        src_count = pg->inline_elements_length;
    } else if (pg->draw_arrays_length) {
        if (!expand) {
            bool supported = false;
            MTLPrimitiveType prim = metal_primitive_type(mode, &supported);
            if (!supported) {
                r->unsupported_prims++;
                return;
            }
            for (int i = 0; i < pg->draw_arrays_length; i++) {
                [r->encoder drawPrimitives:prim
                               vertexStart:pg->draw_arrays_start[i]
                               vertexCount:pg->draw_arrays_count[i]];
            }
            r->draws_issued++;
            return;
        }
        /* Expanded draw_arrays: handled per range below via implicit
         * indices offset by the range start. */
    } else if (pg->inline_buffer_length || pg->inline_array_length) {
        /*
         * Immediate-mode vertices. Rare by count but not by importance: the
         * dashboard's final composite -- the draws that put the frame into
         * the scanout surface -- comes through here, which is why skipping
         * this path left a black screen while everything else rendered.
         */
        bool supported = false;
        MTLPrimitiveType prim = metal_primitive_type(mode, &supported);
        unsigned int n = inline_count;

        if (!n) {
            return;
        }

        if (expand) {
            size_t out_max = expanded_index_count(mode, n);
            if (!out_max) {
                return;
            }
            g_autofree uint32_t *idx = g_malloc(out_max * sizeof(uint32_t));
            size_t cnt = expand_primitive(mode, NULL, n, idx);
            id<MTLBuffer> ib =
                [r->device newBufferWithBytes:idx
                                       length:cnt * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
            [r->encoder drawIndexedPrimitives:expanded_primitive_type(mode)
                                   indexCount:cnt
                                    indexType:MTLIndexTypeUInt32
                                  indexBuffer:ib
                            indexBufferOffset:0];
            r->expanded_prims++;
        } else if (supported) {
            [r->encoder drawPrimitives:prim vertexStart:0 vertexCount:n];
        } else {
            r->unsupported_prims++;
            return;
        }

        r->draws_issued++;
        r->inline_draws++;
        return;
    } else {
        r->unsupported_draws++;
        return;
    }

    if (src) {
        size_t out_max = expanded_index_count(mode, src_count);
        if (out_max == 0) {
            return;
        }

        MTLPrimitiveType prim;
        size_t count;
        g_autofree uint32_t *expanded = NULL;

        if (expand) {
            expanded = g_malloc(out_max * sizeof(uint32_t));
            count = expand_primitive(mode, src, src_count, expanded);
            prim = expanded_primitive_type(mode);
            src = expanded;
            r->expanded_prims++;
        } else {
            bool supported = false;
            prim = metal_primitive_type(mode, &supported);
            if (!supported) {
                r->unsupported_prims++;
                return;
            }
            count = src_count;
        }

        id<MTLBuffer> ib =
            [r->device newBufferWithBytes:src
                                   length:count * sizeof(uint32_t)
                                  options:MTLResourceStorageModeShared];
        [r->encoder drawIndexedPrimitives:prim
                               indexCount:count
                                indexType:MTLIndexTypeUInt32
                              indexBuffer:ib
                        indexBufferOffset:0];
        r->draws_issued++;
        return;
    }

    /* Expanded draw_arrays ranges. */
    for (int i = 0; i < pg->draw_arrays_length; i++) {
        size_t start = pg->draw_arrays_start[i];
        size_t n = pg->draw_arrays_count[i];
        size_t out_max = expanded_index_count(mode, n);
        if (out_max == 0) {
            continue;
        }

        g_autofree uint32_t *expanded =
            g_malloc(out_max * sizeof(uint32_t));
        size_t count = expand_primitive(mode, NULL, n, expanded);
        for (size_t k = 0; k < count; k++) {
            expanded[k] += start;
        }

        id<MTLBuffer> ib =
            [r->device newBufferWithBytes:expanded
                                   length:count * sizeof(uint32_t)
                                  options:MTLResourceStorageModeShared];
        [r->encoder drawIndexedPrimitives:expanded_primitive_type(mode)
                               indexCount:count
                                indexType:MTLIndexTypeUInt32
                              indexBuffer:ib
                        indexBufferOffset:0];
        r->draws_issued++;
        r->expanded_prims++;
    }
}

uint64_t pgraph_metal_collect_zpass(NV2AState *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (r->visibility_slot == 0) {
        return r->zpass_pixel_count_result;
    }

    /* The counts are only in the buffer once the GPU has retired the pass. */
    pgraph_metal_flush_gpu(d, true);

    const uint64_t *v = (const uint64_t *)r->visibility_buffer.contents;
    for (unsigned int i = 0; i < r->visibility_slot; i++) {
        r->zpass_pixel_count_result += v[i];
    }
    memset(r->visibility_buffer.contents, 0, (size_t)r->visibility_slot * 8);
    r->visibility_slot = 0;

    return r->zpass_pixel_count_result;
}

void pgraph_metal_reset_zpass(NV2AState *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (r->visibility_buffer != nil && r->visibility_slot) {
        memset(r->visibility_buffer.contents, 0,
               (size_t)r->visibility_slot * 8);
    }
    r->visibility_slot = 0;
    r->zpass_pixel_count_result = 0;
}

void pgraph_metal_report_shader_stats(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    fprintf(stderr,
            "nv2a: metal: render passes %lu, submits %lu for %lu draws\n",
            r->passes, r->submits, r->draws_issued);
    fprintf(stderr,
            "nv2a: metal: shaders %u ok / %u failed, pipelines %u ok / %u "
            "failed, draws issued %lu, unsupported prims %lu draws %lu\n",
            r->shader_gen_successes, r->shader_gen_failures,
            r->pipeline_count, r->pipeline_failures, r->draws_issued,
            r->unsupported_prims, r->unsupported_draws);
}
