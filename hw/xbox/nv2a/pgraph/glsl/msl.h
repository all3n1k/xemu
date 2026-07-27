/*
 * Geforce NV2A PGRAPH Metal Shading Language dialect support
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
 * The NV2A shader generators (vsh.c, vsh-ff.c, vsh-prog.c, psh.c) emit
 * statement-level code that is very nearly dialect-neutral: register combiner
 * arithmetic, vertex program MAC/ILU macros, lighting and texgen math. Only
 * the *shell* around that code differs between GLSL and Metal Shading
 * Language: declarations, entry point signature, resource binding, and a
 * handful of builtin names.
 *
 * Rather than fork ~4,000 lines of NV2A semantics into a second emitter, the
 * generators take a `metal` dialect flag and delegate the shell to this file.
 * The statement-level code stays byte-identical between backends, which keeps
 * one source of truth for the hardware behavior.
 *
 * The compatibility prologue emitted here supplies MSL definitions for the
 * GLSL spellings the generators use: vec4/mat4 typedefs, lessThan(),
 * inversesqrt(), bit-cast helpers, and texture()/textureProj()/textureSize()
 * overloads reached through a `texSampN` -> `texN, smpN` macro pair.
 */

#ifndef HW_XBOX_NV2A_PGRAPH_GLSL_MSL_H
#define HW_XBOX_NV2A_PGRAPH_GLSL_MSL_H

#include "common.h"

/* Resource binding slots used by the generated MSL entry points. Buffer 0 is
 * the uniform struct; the vertex stage takes its attributes via [[stage_in]].
 * Textures and samplers are bound one per NV2A texture stage. */
#define MSL_UNIFORM_BUFFER_INDEX 0

/* Shared compatibility prologue: typedefs plus the GLSL builtin shims. */
const char *pgraph_msl_prologue(void);

/* Programmable vertex shader (MAC/ILU) macros and helper functions. These are
 * the MSL counterpart of the `vsh_header` string in vsh-prog.c. */
const char *pgraph_msl_vsh_prog_prologue(void);

/* Per-invocation mutable registers for the programmable vertex path. MSL has
 * no mutable program-scope variables, so what GLSL declares as globals must be
 * emitted as locals at the top of the entry point instead. */
const char *pgraph_msl_vsh_prog_locals(void);

/* Mutable NV2A output registers (oPos, oD0, ... oT3), likewise function-local
 * under MSL. */
const char *pgraph_msl_vsh_output_regs(void);

/*
 * Where each uniform member lands in the generated MSL struct, and where it
 * comes from in the C *UniformValues struct. The two layouts are not the
 * same and cannot be made the same (see msl.c), so the renderer copies
 * member by member -- and element by element where the strides differ, which
 * they do for vec3.
 */
typedef struct MslUniformMember {
    size_t offset;     /* byte offset within the MSL struct */
    size_t stride;     /* element stride in the MSL struct */
    size_t count;      /* element count */
    size_t src_offset; /* byte offset within the C *UniformValues struct */
    size_t src_stride; /* element stride in the C struct */
} MslUniformMember;

/* Compute the MSL layout for a uniform block; returns the struct size. */
size_t pgraph_msl_uniform_layout(const UniformInfo *info, size_t num_info,
                                 int skip_index, MslUniformMember *members);

/* Emit `struct <name> { ... };` describing a uniform block. Members use MSL's
 * natural alignment; pgraph_msl_uniform_layout() says where each one lands. */
void pgraph_msl_gen_uniform_struct(MString *out, const char *name,
                                   const UniformInfo *info, size_t num_info,
                                   int skip_index);

/* Emit locals aliasing each uniform member, so generated code can refer to
 * `c[3]` or `clipRange.y` unqualified exactly as it does in GLSL. */
void pgraph_msl_gen_uniform_locals(MString *out, const UniformInfo *info,
                                   size_t num_info, int skip_index);

/* Emit the interpolant struct shared by the vertex output and fragment input.
 * `is_vertex` adds [[position]] and [[point_size]]; the fragment side adds
 * [[point_coord]] instead. Both sides must agree on the [[user()]] names. */
void pgraph_msl_gen_vtx_struct(MString *out, const char *name, bool smooth,
                               bool is_vertex);

/* Declare the vtx* interpolants as locals, so generated code can assign them
 * by bare name; pack them into the output struct afterwards. */
void pgraph_msl_gen_vtx_out_locals(MString *out);
void pgraph_msl_gen_vtx_out_pack(MString *out);

/* Fragment side: initialize locals from the [[stage_in]] struct. */
void pgraph_msl_gen_vtx_in_locals(MString *out);

#endif
