/*
 * Geforce NV2A PGRAPH Metal Renderer - 2D image blit
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

#ifndef XEMU_NV2A_PGRAPH_METAL_BLIT_H
#define XEMU_NV2A_PGRAPH_METAL_BLIT_H

#include "renderer.h"

void pgraph_metal_image_blit(NV2AState *d);

#endif /* XEMU_NV2A_PGRAPH_METAL_BLIT_H */
