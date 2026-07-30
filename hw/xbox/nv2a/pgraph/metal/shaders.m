/*
 * Geforce NV2A PGRAPH Metal Renderer - shader generation
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

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "hw/xbox/nv2a/pgraph/glsl/msl.h"
#include "hw/xbox/nv2a/pgraph/glsl/vsh.h"
#include "hw/xbox/nv2a/pgraph/glsl/psh.h"
#include "renderer.h"
#include "shaders.h"

MString *pgraph_metal_gen_vsh(const VshState *state)
{
    GenVshGlslOptions opts = {
        .metal = true,
        .debug_pos = getenv("XEMU_METAL_DEBUG_POS") != NULL,
        .ubo_binding = MSL_UNIFORM_BUFFER_INDEX,
    };
    MString *msl = pgraph_glsl_gen_vsh(state, opts);

    /*
     * Both dialects come from one generator, so emitting the GLSL for the
     * same state alongside the MSL makes a translation bug a plain textual
     * diff rather than something to infer from pixels.
     */
    const char *dump = getenv("XEMU_METAL_DUMP_VSH");
    if (dump) {
        static int n;
        bool want = !getenv("XEMU_METAL_DUMP_VSH_FF") ||
                    state->is_fixed_function;
        if (want && n < 2) {
            GenVshGlslOptions g = { .vulkan = false, .ubo_binding = 0 };
            MString *glsl = pgraph_glsl_gen_vsh(state, g);
            char path[1024];
            snprintf(path, sizeof(path), "%s%d_ff%d.msl", dump, n,
                     state->is_fixed_function);
            FILE *f = fopen(path, "w");
            if (f) { fputs(mstring_get_str(msl), f); fclose(f); }
            snprintf(path, sizeof(path), "%s%d_ff%d.glsl", dump, n,
                     state->is_fixed_function);
            f = fopen(path, "w");
            if (f) { fputs(mstring_get_str(glsl), f); fclose(f); }
            mstring_unref(glsl);
            fprintf(stderr, "vsh dumped: ff=%d -> %s%d_ff%d.{msl,glsl}\n",
                    state->is_fixed_function, dump, n,
                    state->is_fixed_function);
            n++;
        }
    }

    return msl;
}

MString *pgraph_metal_gen_psh(const PshState *state)
{
    GenPshGlslOptions opts = {
        .metal = true,
        .ubo_binding = MSL_UNIFORM_BUFFER_INDEX,
    };
    return pgraph_glsl_gen_psh(state, opts);
}

id<MTLLibrary> pgraph_metal_compile_shader(id<MTLDevice> device,
                                           MString *source, Error **errp)
{
    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:mstring_get_str(source)];

    MTLCompileOptions *opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [device newLibraryWithSource:src
                                              options:opts
                                                error:&err];
    if (lib == nil) {
        error_setg(errp, "Metal shader compilation failed: %s",
                   err ? [[err localizedDescription] UTF8String] : "unknown");
    }

    return lib;
}

/* ------------------------------------------------------------------ */
/* Generator self-test.                                               */
/* ------------------------------------------------------------------ */

/*
 * The self-test exercises the MSL generator against hand-built shader states
 * without needing a draw path. It is not a substitute for running real
 * titles, which is what will cover the combinatorial space; it is here so a
 * generator regression shows up immediately at startup rather than as a black
 * screen much later.
 */

/* Bit positions of the vertex program token fields, matching field_mapping[]
 * in vsh-prog.c. Only the fields the self-test programs need are listed. */
static void vsh_token_set(uint32_t *token, int subtoken, int start_bit,
                          int bit_length, uint32_t value)
{
    uint32_t mask = (bit_length >= 32) ? 0xFFFFFFFF
                                       : ((1u << bit_length) - 1);
    token[subtoken] &= ~(mask << start_bit);
    token[subtoken] |= (value & mask) << start_bit;
}

/* Build "MOV oPos.xyzw, v0" followed by the end-of-program marker. */
static void build_mov_opos_program(ProgrammableVshState *prog)
{
    uint32_t *t = prog->program_data[0];

    memset(t, 0, VSH_TOKEN_SIZE * sizeof(uint32_t));

    vsh_token_set(t, 1, 21, 4, MAC_MOV);   /* FLD_MAC */
    vsh_token_set(t, 1, 9, 4, 0);          /* FLD_V: v0 */
    vsh_token_set(t, 1, 6, 2, SWIZZLE_X);  /* FLD_A_SWZ_X */
    vsh_token_set(t, 1, 4, 2, SWIZZLE_Y);  /* FLD_A_SWZ_Y */
    vsh_token_set(t, 1, 2, 2, SWIZZLE_Z);  /* FLD_A_SWZ_Z */
    vsh_token_set(t, 1, 0, 2, SWIZZLE_W);  /* FLD_A_SWZ_W */
    vsh_token_set(t, 2, 26, 2, PARAM_V);   /* FLD_A_MUX */
    /* decode_token() decodes input C unconditionally and input B whenever the
     * MAC opcode takes one, so both muxes must name a real register file even
     * for a single-operand MOV. */
    vsh_token_set(t, 2, 11, 2, PARAM_R);   /* FLD_B_MUX */
    vsh_token_set(t, 3, 28, 2, PARAM_R);   /* FLD_C_MUX */
    vsh_token_set(t, 3, 12, 4, 0xF);       /* FLD_OUT_O_MASK: xyzw */
    vsh_token_set(t, 3, 11, 1, OUTPUT_O);  /* FLD_OUT_ORB */
    vsh_token_set(t, 3, 3, 8, 0);          /* FLD_OUT_ADDRESS: oPos */
    vsh_token_set(t, 3, 2, 1, OMUX_MAC);   /* FLD_OUT_MUX */
    vsh_token_set(t, 3, 0, 1, 1);          /* FLD_FINAL */

    prog->program_length = 1;
}

static VshState selftest_vsh_state(int variant)
{
    VshState s;
    memset(&s, 0, sizeof(s));

    /* A zeroed state is already valid and minimal: fixed function, skinning
     * off, all texgen disabled, no lighting, no fog. */
    s.surface_scale_factor = 1;
    s.is_fixed_function = true;
    s.point_size = 1.0f;

    switch (variant) {
    case 0: /* minimal fixed function */
        break;

    case 1: /* fixed function, lighting of every type, fog, specular */
        s.fixed_function.lighting = true;
        s.fixed_function.normalization = true;
        s.fixed_function.local_eye = true;
        s.fixed_function.light[0] = LIGHT_INFINITE;
        s.fixed_function.light[1] = LIGHT_LOCAL;
        s.fixed_function.light[2] = LIGHT_SPOT;
        s.fixed_function.emission_src = MATERIAL_COLOR_SRC_DIFFUSE;
        s.fixed_function.ambient_src = MATERIAL_COLOR_SRC_SPECULAR;
        s.fixed_function.diffuse_src = MATERIAL_COLOR_SRC_MATERIAL;
        s.fixed_function.specular_src = MATERIAL_COLOR_SRC_DIFFUSE;
        s.fog_enable = true;
        s.fog_mode = FOG_MODE_EXP2;
        s.fixed_function.foggen = FOGGEN_RADIAL;
        s.specular_enable = true;
        s.separate_specular = true;
        s.ignore_specular_alpha = true;
        s.smooth_shading = true;
        break;

    case 2: /* texgen, texture matrices, skinning, point params */
        s.fixed_function.skinning = SKINNING_3WEIGHTS;
        for (int i = 0; i < 4; i++) {
            s.fixed_function.texture_matrix_enable[i] = true;
        }
        s.fixed_function.texgen[0][0] = TEXGEN_EYE_LINEAR;
        s.fixed_function.texgen[0][1] = TEXGEN_OBJECT_LINEAR;
        s.fixed_function.texgen[1][0] = TEXGEN_SPHERE_MAP;
        s.fixed_function.texgen[1][1] = TEXGEN_SPHERE_MAP;
        s.fixed_function.texgen[2][0] = TEXGEN_REFLECTION_MAP;
        s.fixed_function.texgen[2][2] = TEXGEN_NORMAL_MAP;
        s.point_params_enable = true;
        s.fog_enable = true;
        s.fog_mode = FOG_MODE_LINEAR_ABS;
        s.fixed_function.foggen = FOGGEN_PLANAR;
        break;

    case 3: /* programmable, trivial end-of-program only */
        s.is_fixed_function = false;
        memset(s.programmable.program_data[0], 0,
               VSH_TOKEN_SIZE * sizeof(uint32_t));
        vsh_token_set(s.programmable.program_data[0], 3, 0, 1, 1);
        s.programmable.program_length = 1;
        break;

    case 4: /* programmable with an instruction, fog and specular on */
        s.is_fixed_function = false;
        build_mov_opos_program(&s.programmable);
        s.fog_enable = true;
        s.fog_mode = FOG_MODE_EXP;
        s.specular_enable = true;
        break;

    case 5: /* compressed, swizzled and inline (uniform) attributes */
        s.is_fixed_function = false;
        build_mov_opos_program(&s.programmable);
        s.compressed_attrs = 1 << 1;
        s.swizzle_attrs = 1 << 3;
        s.uniform_attrs = (1 << 5) | (1 << 6);
        s.point_params_enable = true;
        break;

    default:
        g_assert_not_reached();
    }

    return s;
}

static PshState selftest_psh_state(int variant)
{
    PshState s;
    memset(&s, 0, sizeof(s));

    /* Zeroed: no combiner stages, no textures, no alpha test. */
    s.depth_format = DEPTH_FORMAT_D24;
    s.surface_zeta_format = NV097_SET_SURFACE_FORMAT_ZETA_Z24S8;

    switch (variant) {
    case 0: /* minimal */
        break;

    case 1: /* one texture stage, one combiner stage, final combiner */
        s.combiner_control = 1;
        s.shader_stage_program = PS_TEXTUREMODES_PROJECT2D;
        s.dim_tex[0] = 2;
        /* a=T0.rgb b=V0.rgb -> R0 */
        s.rgb_inputs[0] = (PS_REGISTER_T0 << 24) | (PS_REGISTER_V0 << 16);
        s.rgb_outputs[0] = PS_REGISTER_R0 << 4;
        s.alpha_inputs[0] = ((PS_REGISTER_T0 | PS_CHANNEL_ALPHA) << 24) |
                            ((PS_REGISTER_V0 | PS_CHANNEL_ALPHA) << 16);
        s.alpha_outputs[0] = PS_REGISTER_R0 << 4;
        /* final: rgb from R0, alpha from R0.a */
        s.final_inputs_0 = PS_REGISTER_R0;
        s.final_inputs_1 = (PS_REGISTER_R0 | PS_CHANNEL_ALPHA) << 8;
        s.alpha_test = true;
        s.alpha_func = ALPHA_FUNC_GEQUAL;
        s.smooth_shading = true;
        break;

    case 2: /* unnormalized (rect) texture: exercises the normN() path */
        s.combiner_control = 1;
        s.shader_stage_program = PS_TEXTUREMODES_PROJECT2D;
        s.dim_tex[0] = 2;
        s.rect_tex[0] = true;
        s.alphakill[0] = true;
        s.rgb_inputs[0] = (PS_REGISTER_T0 << 24) | (PS_REGISTER_V0 << 16);
        s.rgb_outputs[0] = PS_REGISTER_R0 << 4;
        s.final_inputs_0 = PS_REGISTER_R0;
        s.final_inputs_1 = (PS_REGISTER_R0 | PS_CHANNEL_ALPHA) << 8;
        s.z_perspective = true;
        s.window_clip_exclusive = true;
        break;

    case 3: /* point sprite, depth clipping, 16-bit depth */
        s.point_sprite = true;
        s.depth_clipping = true;
        s.depth_format = DEPTH_FORMAT_D16;
        s.surface_zeta_format = NV097_SET_SURFACE_FORMAT_ZETA_Z16;
        s.final_inputs_0 = PS_REGISTER_V0;
        break;

    case 4: /* color key + 3D texture */
        s.combiner_control = 1;
        s.shader_stage_program = PS_TEXTUREMODES_PROJECT2D;
        s.dim_tex[0] = 3;
        s.colorkey_mode[0] = COLOR_KEY_KILL_ALPHA;
        s.rgb_inputs[0] = (PS_REGISTER_T0 << 24) | (PS_REGISTER_V0 << 16);
        s.rgb_outputs[0] = PS_REGISTER_R0 << 4;
        s.final_inputs_0 = PS_REGISTER_R0;
        s.final_inputs_1 = (PS_REGISTER_R0 | PS_CHANNEL_ALPHA) << 8;
        break;

    default:
        g_assert_not_reached();
    }

    return s;
}

#define NUM_SELFTEST_VSH 6
#define NUM_SELFTEST_PSH 5

static int selftest_one(id<MTLDevice> device, const char *label,
                        MString *source, bool verbose)
{
    Error *err = NULL;
    id<MTLLibrary> lib = pgraph_metal_compile_shader(device, source, &err);

    if (lib == nil) {
        fprintf(stderr, "Metal shader selftest: %s FAILED\n%s\n", label,
                error_get_pretty(err));
        fprintf(stderr, "--- generated source ---\n%s\n-----------------------\n",
                mstring_get_str(source));
        error_free(err);
        return 1;
    }

    if (verbose) {
        fprintf(stderr, "Metal shader selftest: %s ok (%zu bytes)\n", label,
                strlen(mstring_get_str(source)));
    }

    return 0;
}

int pgraph_metal_shader_selftest(id<MTLDevice> device, bool verbose)
{
    int failures = 0;

    for (int i = 0; i < NUM_SELFTEST_VSH; i++) {
        VshState vsh = selftest_vsh_state(i);
        MString *src = pgraph_metal_gen_vsh(&vsh);
        g_autofree char *label = g_strdup_printf("vsh variant %d", i);
        failures += selftest_one(device, label, src, verbose);
        mstring_unref(src);
    }

    for (int i = 0; i < NUM_SELFTEST_PSH; i++) {
        PshState psh = selftest_psh_state(i);
        MString *src = pgraph_metal_gen_psh(&psh);
        g_autofree char *label = g_strdup_printf("psh variant %d", i);
        failures += selftest_one(device, label, src, verbose);
        mstring_unref(src);
    }

    if (failures) {
        fprintf(stderr, "Metal shader selftest: %d of %d states failed\n",
                failures, NUM_SELFTEST_VSH + NUM_SELFTEST_PSH);
    } else {
        fprintf(stderr, "Metal shader selftest: all %d states compiled\n",
                NUM_SELFTEST_VSH + NUM_SELFTEST_PSH);
    }

    return failures;
}
