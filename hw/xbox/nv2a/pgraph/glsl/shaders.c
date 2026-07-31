/*
 * Geforce NV2A PGRAPH GLSL Shader Generator
 *
 * Copyright (c) 2025 Matt Borgerson
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

#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "shaders.h"

ShaderState pgraph_glsl_get_shader_state(PGRAPHState *pg)
{
    pg->program_data_dirty = false; /* fixme */

    ShaderState state;

    // We will hash it, so make sure any padding is zeroed
    memset(&state, 0, sizeof(ShaderState));

    pgraph_glsl_set_vsh_state(pg, &state.vsh);
    pgraph_glsl_set_geom_state(pg, &state.geom);
    pgraph_glsl_set_psh_state(pg, &state.psh);

    /*
     * XEMU_DUMP_SHADER_STATE: the state feeding the generator, printed from
     * shared code so both backends emit it identically. The generated source
     * has been verified to be a faithful translation, so a rendering
     * difference between backends has to originate here or later.
     */
    if (getenv("XEMU_DUMP_SHADER_STATE")) {
        static int n;
        if (n < 24) {
            fprintf(stderr,
                    "SS%02d ff=%d light=%d l0=%d l1=%d norm=%d "
                    "fog=%d fogmode=%d foggen=%d spec=%d sepspec=%d "
                    "poly=%d/%d smooth=%d zpersp=%d "
                    "combiners=%d alphatest=%d texmat=%d%d%d%d\n",
                    n, state.vsh.is_fixed_function,
                    state.vsh.fixed_function.lighting,
                    state.vsh.fixed_function.light[0],
                    state.vsh.fixed_function.light[1],
                    state.vsh.fixed_function.normalization,
                    state.vsh.fog_enable, state.vsh.fog_mode,
                    state.vsh.fixed_function.foggen,
                    state.vsh.specular_enable, state.vsh.separate_specular,
                    state.geom.polygon_front_mode,
                    state.geom.polygon_back_mode,
                    state.vsh.smooth_shading, state.psh.z_perspective,
                    state.psh.combiner_control & 0xFF, state.psh.alpha_test,
                    state.vsh.fixed_function.texture_matrix_enable[0],
                    state.vsh.fixed_function.texture_matrix_enable[1],
                    state.vsh.fixed_function.texture_matrix_enable[2],
                    state.vsh.fixed_function.texture_matrix_enable[3]);
            n++;
        }
    }

    return state;
}

bool pgraph_glsl_check_shader_state_dirty(PGRAPHState *pg,
                                          const ShaderState *state)
{
    if (pg->program_data_dirty) {
        return true;
    }

    unsigned int regs[] = {
        NV_PGRAPH_COMBINECTL,      NV_PGRAPH_COMBINESPECFOG0,
        NV_PGRAPH_COMBINESPECFOG1, NV_PGRAPH_CONTROL_0,
        NV_PGRAPH_CONTROL_3,       NV_PGRAPH_CSV0_C,
        NV_PGRAPH_CSV0_D,          NV_PGRAPH_CSV1_A,
        NV_PGRAPH_CSV1_B,          NV_PGRAPH_POINTSIZE,
        NV_PGRAPH_SETUPRASTER,     NV_PGRAPH_SHADERCLIPMODE,
        NV_PGRAPH_SHADERCTL,       NV_PGRAPH_SHADERPROG,
        NV_PGRAPH_SHADOWCTL,       NV_PGRAPH_ZCOMPRESSOCCLUDE,
    };
    for (int i = 0; i < ARRAY_SIZE(regs); i++) {
        if (pgraph_is_reg_dirty(pg, regs[i])) {
            return true;
        }
    }

    int num_stages = pgraph_reg_r(pg, NV_PGRAPH_COMBINECTL) & 0xFF;
    for (int i = 0; i < num_stages; i++) {
        if (pgraph_is_reg_dirty(pg, NV_PGRAPH_COMBINEALPHAI0 + i * 4) ||
            pgraph_is_reg_dirty(pg, NV_PGRAPH_COMBINEALPHAO0 + i * 4) ||
            pgraph_is_reg_dirty(pg, NV_PGRAPH_COMBINECOLORI0 + i * 4) ||
            pgraph_is_reg_dirty(pg, NV_PGRAPH_COMBINECOLORO0 + i * 4)) {
            return true;
        }
    }

    if (pg->uniform_attrs != state->vsh.uniform_attrs ||
        pg->swizzle_attrs != state->vsh.swizzle_attrs ||
        pg->compressed_attrs != state->vsh.compressed_attrs ||
        pg->primitive_mode != state->geom.primitive_mode ||
        pg->surface_scale_factor != state->vsh.surface_scale_factor ||
        pg->surface_shape.zeta_format != state->psh.surface_zeta_format) {
        return true;
    }

    for (int i = 0; i < 4; i++) {
        if (pgraph_is_reg_dirty(pg, NV_PGRAPH_TEXCTL0_0 + i * 4) ||
            pgraph_is_reg_dirty(pg, NV_PGRAPH_TEXFILTER0 + i * 4) ||
            pgraph_is_reg_dirty(pg, NV_PGRAPH_TEXFMT0 + i * 4)) {
            return true;
        }

        if (pg->texture_matrix_enable[i] !=
            state->vsh.fixed_function.texture_matrix_enable[i]) {
            return true;
        }
    }

    return false;
}
