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

typedef struct MetalSurfaceBinding MetalSurfaceBinding;

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

} PGRAPHMetalState;

#endif /* XEMU_NV2A_PGRAPH_METAL_RENDERER_H */
