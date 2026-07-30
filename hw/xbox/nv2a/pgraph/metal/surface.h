/*
 * Geforce NV2A PGRAPH Metal Renderer - surface management
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

#ifndef XEMU_NV2A_PGRAPH_METAL_SURFACE_H
#define XEMU_NV2A_PGRAPH_METAL_SURFACE_H

#include "renderer.h"

/*
 * Xbox surface format -> Metal.
 *
 * `host_bytes_per_pixel` is the size of a texel in the MTLTexture, which is
 * not always the guest size: the guest's 16-bit formats have direct Metal
 * equivalents, but Z24S8 does not (Apple GPUs do not support
 * MTLPixelFormatDepth24Unorm_Stencil8), so it is widened to
 * Depth32Float_Stencil8 on the host.
 *
 * `guest_bytes_per_pixel` is what the guest expects in VRAM and therefore
 * what the download path must produce.
 */
typedef struct MetalSurfaceFormatInfo {
    unsigned int host_bytes_per_pixel;
    unsigned int guest_bytes_per_pixel;
    MTLPixelFormat pixel_format;
    bool depth;
    bool stencil;
} MetalSurfaceFormatInfo;

typedef struct MetalSurfaceBinding {
    QTAILQ_ENTRY(MetalSurfaceBinding) entry;
    MemAccessCallback *access_cb;

    hwaddr vram_addr;

    SurfaceShape shape;
    uintptr_t dma_addr;
    uintptr_t dma_len;
    bool color;
    bool swizzle;

    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    size_t size;

    bool cleared;
    int frame_time;
    int draw_time;
    bool draw_dirty;
    bool download_pending;
    bool upload_pending;

    id<MTLTexture> texture;
    MetalSurfaceFormatInfo fmt;
} MetalSurfaceBinding;

void pgraph_metal_reload_surface_scale_factor(PGRAPHState *pg);
void pgraph_metal_init_surfaces(PGRAPHState *pg);
void pgraph_metal_finalize_surfaces(PGRAPHState *pg);

void pgraph_metal_clear_surface(NV2AState *d, uint32_t parameter);
void pgraph_metal_surface_update(NV2AState *d, bool upload, bool color_write,
                                 bool zeta_write);
void pgraph_metal_surface_flush(NV2AState *d);
void pgraph_metal_unbind_surface(NV2AState *d, bool color);
void pgraph_metal_set_surface_dirty(PGRAPHState *pg, bool color, bool zeta);
void pgraph_metal_download_dirty_surfaces(NV2AState *d);
void pgraph_metal_process_pending_downloads(NV2AState *d);

MetalSurfaceBinding *pgraph_metal_surface_get(NV2AState *d, hwaddr addr);
MetalSurfaceBinding *pgraph_metal_surface_get_within(NV2AState *d, hwaddr addr);
void pgraph_metal_surface_invalidate(NV2AState *d, MetalSurfaceBinding *e);
void pgraph_metal_download_surfaces_overlapping(NV2AState *d, hwaddr addr,
                                                hwaddr len);

void pgraph_metal_surface_download_if_dirty(NV2AState *d,
                                            MetalSurfaceBinding *surface);
void pgraph_metal_upload_surface_data(NV2AState *d,
                                      MetalSurfaceBinding *surface,
                                      bool force);

#endif /* XEMU_NV2A_PGRAPH_METAL_SURFACE_H */
