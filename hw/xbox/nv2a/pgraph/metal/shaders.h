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

#ifndef XEMU_NV2A_PGRAPH_METAL_SHADERS_H
#define XEMU_NV2A_PGRAPH_METAL_SHADERS_H

#include "renderer.h"
#include "hw/xbox/nv2a/pgraph/glsl/shaders.h"

/* Entry point name used by both generated stages. */
#define PGRAPH_METAL_SHADER_ENTRY "main0"

/*
 * Generate Metal Shading Language for a shader state. Both return a new
 * MString owned by the caller.
 *
 * Note that the NV2A geometry shader stage has no Metal counterpart: Metal
 * has no geometry shaders, so primitive expansion has to happen on the draw
 * path instead (mirroring how the Vulkan backend handles quads). The
 * generated vertex shader writes the degenerate single-vertex values for
 * vtxPos0..2 and triMZ that the fragment shader expects, which is correct for
 * everything except the barycentric depth interpolation used by w-buffering.
 */
MString *pgraph_metal_gen_vsh(const VshState *state);
MString *pgraph_metal_gen_psh(const PshState *state);

/*
 * Compile MSL source into a library. Returns nil on failure and, when errp is
 * non-NULL, sets it to the compiler diagnostics.
 */
id<MTLLibrary> pgraph_metal_compile_shader(id<MTLDevice> device,
                                           MString *source,
                                           Error **errp);

/*
 * Generate and compile a spread of representative shader states, reporting
 * failures. Used to validate the generator without needing a full draw path.
 * Returns the number of states that failed to compile.
 */
int pgraph_metal_shader_selftest(id<MTLDevice> device, bool verbose);

#endif /* XEMU_NV2A_PGRAPH_METAL_SHADERS_H */
