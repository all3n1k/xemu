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
}

void pgraph_metal_finalize_shader_cache(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

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

    pgraph_metal_bind_shaders(pg);

    /* TODO: pipeline state object, render command encoder, viewport/scissor,
     * blend/depth/stencil state, texture and uniform binding. */
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

    pg->draw_time++;
    if (r->color_binding && pgraph_color_write_enabled(pg)) {
        r->color_binding->draw_time = pg->draw_time;
    }
    if (r->zeta_binding && pgraph_zeta_write_enabled(pg)) {
        r->zeta_binding->draw_time = pg->draw_time;
    }

    pgraph_metal_set_surface_dirty(pg, color_write,
                                   depth_test || stencil_test);
}

void pgraph_metal_flush_draw(NV2AState *d)
{
    /* TODO: bind vertex attributes and dispatch. Indexed drawing
     * (inline_elements) covers 99.7% of draws and comes first. */
}

void pgraph_metal_report_shader_stats(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    fprintf(stderr,
            "nv2a: metal: shader cache: %u states compiled, %u failed\n",
            r->shader_gen_successes, r->shader_gen_failures);
}
