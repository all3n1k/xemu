/*
 * Geforce NV2A PGRAPH Metal Renderer - textures
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

#ifndef XEMU_NV2A_PGRAPH_METAL_TEXTURE_H
#define XEMU_NV2A_PGRAPH_METAL_TEXTURE_H

#include "renderer.h"

void pgraph_metal_init_textures(PGRAPHState *pg);
void pgraph_metal_finalize_textures(PGRAPHState *pg);
void pgraph_metal_bind_textures(NV2AState *d, id<MTLRenderCommandEncoder> enc);

#endif /* XEMU_NV2A_PGRAPH_METAL_TEXTURE_H */
