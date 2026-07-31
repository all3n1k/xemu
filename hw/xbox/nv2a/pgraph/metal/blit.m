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

/*
 * The NV2A's 2D engine blit, which is how the dashboard gets a rendered
 * frame into the buffer the CRTC scans out. This was the missing link
 * behind the black screen: 3D geometry renders into offscreen surfaces,
 * and only this path moves it to the scanout address.
 *
 * Like the GL backend, the copy itself is done on the CPU in guest memory
 * rather than on the GPU. The source surface is pulled down first if it
 * holds rendered content, and the destination is marked for re-upload.
 */

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "renderer.h"
#include "surface.h"
#include "blit.h"

void pgraph_metal_image_blit(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    ContextSurfaces2DState *context_surfaces = &pg->context_surfaces_2d;
    ImageBlitState *image_blit = &pg->image_blit;

    pgraph_metal_surface_update(d, false, true, true);

    assert(context_surfaces->object_instance == image_blit->context_surfaces);

    unsigned int bytes_per_pixel;
    switch (context_surfaces->color_format) {
    case NV062_SET_COLOR_FORMAT_LE_Y8:
        bytes_per_pixel = 1;
        break;
    case NV062_SET_COLOR_FORMAT_LE_R5G6B5:
        bytes_per_pixel = 2;
        break;
    case NV062_SET_COLOR_FORMAT_LE_A8R8G8B8:
    case NV062_SET_COLOR_FORMAT_LE_X8R8G8B8:
    case NV062_SET_COLOR_FORMAT_LE_X8R8G8B8_Z8R8G8B8:
    case NV062_SET_COLOR_FORMAT_LE_Y32:
        bytes_per_pixel = 4;
        break;
    default:
        fprintf(stderr, "nv2a: metal: unknown blit surface format: 0x%x\n",
                context_surfaces->color_format);
        return;
    }

    hwaddr source_dma_len;
    uint8_t *source = (uint8_t *)nv_dma_map(
        d, context_surfaces->dma_image_source, &source_dma_len);
    assert(context_surfaces->source_offset < source_dma_len);
    source += context_surfaces->source_offset;
    hwaddr source_addr = source - d->vram_ptr;

    hwaddr dest_dma_len;
    uint8_t *dest = (uint8_t *)nv_dma_map(d, context_surfaces->dma_image_dest,
                                          &dest_dma_len);
    assert(context_surfaces->dest_offset < dest_dma_len);
    dest += context_surfaces->dest_offset;
    hwaddr dest_addr = dest - d->vram_ptr;

    if (getenv("XEMU_BLIT_TRACE")) {
        fprintf(stderr, "B src=%08lx dst=%08lx %ux%u bpp=%u\n",
                (unsigned long)source_addr, (unsigned long)dest_addr,
                image_blit->width, image_blit->height, bytes_per_pixel);
    }

    /* The source may still live only in a GPU texture. */
    MetalSurfaceBinding *surf_src = pgraph_metal_surface_get(d, source_addr);
    if (getenv("XEMU_BLIT_TRACE") && surf_src) {
        fprintf(stderr, "MT blit-src @%08lx %ux%u swizzle=%d draw_dirty=%d\n",
                (unsigned long)source_addr, surf_src->width, surf_src->height,
                surf_src->swizzle, surf_src->draw_dirty);
    }
    if (surf_src) {
        pgraph_metal_surface_download_if_dirty(d, surf_src);
    }

    /*
     * The blit source is where the guest actually built the image; dumping
     * the GPU texture here separates "rendered wrong" from "copied wrong".
     */
    const char *bd = getenv("XEMU_BLIT_DUMP");
    if (bd && surf_src && surf_src->texture != nil &&
        surf_src->texture.storageMode == MTLStorageModeShared) {
        static int bn;
        if (bn < 4) {
            unsigned tw = (unsigned)surf_src->texture.width;
            unsigned th = (unsigned)surf_src->texture.height;
            size_t n = (size_t)tw * th;
            uint32_t *buf = g_malloc(n * 4);
            [surf_src->texture getBytes:buf
                            bytesPerRow:tw * 4
                             fromRegion:MTLRegionMake2D(0, 0, tw, th)
                            mipmapLevel:0];
            char path[1024];
            snprintf(path, sizeof(path), "%s_src%08lx_%d.raw", bd,
                     (unsigned long)source_addr, bn++);
            FILE *fp = fopen(path, "wb");
            if (fp) {
                uint32_t hdr[2] = { tw, th };
                fwrite(hdr, sizeof(hdr), 1, fp);
                fwrite(buf, 4, n, fp);
                fclose(fp);
                fprintf(stderr,
                        "blit-src dumped: @%08lx tex %ux%u swizzle=%d "
                        "draw_dirty=%d -> %s\n",
                        (unsigned long)source_addr, tw, th,
                        surf_src->swizzle, surf_src->draw_dirty, path);
            }
            g_free(buf);
        }
    }

    hwaddr source_offset = image_blit->in_y * context_surfaces->source_pitch +
                           image_blit->in_x * bytes_per_pixel;
    hwaddr dest_offset = image_blit->out_y * context_surfaces->dest_pitch +
                         image_blit->out_x * bytes_per_pixel;

    size_t max_row_pixels =
        MIN(context_surfaces->source_pitch, context_surfaces->dest_pitch) /
        bytes_per_pixel;
    size_t row_pixels = MIN(max_row_pixels, image_blit->width);
    size_t row_bytes = row_pixels * bytes_per_pixel;

    uint8_t *source_row = source + source_offset;
    uint8_t *dest_row = dest + dest_offset;

    MetalSurfaceBinding *surf_dest = pgraph_metal_surface_get(d, dest_addr);
    if (surf_dest) {
        if (image_blit->height < surf_dest->height ||
            row_pixels < surf_dest->width) {
            pgraph_metal_surface_download_if_dirty(d, surf_dest);
        } else {
            /* A full replacement; any pending download is now moot. */
            surf_dest->download_pending = false;
            surf_dest->draw_dirty = false;
        }
        surf_dest->upload_pending = true;
        pg->draw_time++;
    }

    if (getenv("XEMU_METAL_BLIT_STATS")) {
        static unsigned long bn;
        if ((bn++ % 100) == 0) {
            fprintf(stderr,
                    "blit: src @%08lx -> dst @%08lx  %ux%u op=%d "
                    "src_pitch=%u dst_pitch=%u src_surf=%s dst_surf=%s\n",
                    (unsigned long)(source_addr + source_offset),
                    (unsigned long)(dest_addr + dest_offset),
                    image_blit->width, image_blit->height,
                    image_blit->operation,
                    context_surfaces->source_pitch,
                    context_surfaces->dest_pitch,
                    surf_src ? "yes" : "no", surf_dest ? "yes" : "no");
        }
    }

    if (image_blit->operation != NV09F_SET_OPERATION_SRCCOPY) {
        NV2A_UNIMPLEMENTED("Metal blit operation 0x%x",
                           image_blit->operation);
        return;
    }

    for (unsigned int y = 0; y < image_blit->height; y++) {
        memcpy(dest_row, source_row, row_bytes);
        source_row += context_surfaces->source_pitch;
        dest_row += context_surfaces->dest_pitch;
    }

    /* Let the display and texture paths see the new contents. */
    memory_region_set_client_dirty(d->vram, dest_addr + dest_offset,
                                   image_blit->height *
                                       context_surfaces->dest_pitch,
                                   DIRTY_MEMORY_VGA);
    memory_region_set_client_dirty(d->vram, dest_addr + dest_offset,
                                   image_blit->height *
                                       context_surfaces->dest_pitch,
                                   DIRTY_MEMORY_NV2A_TEX);

    {
        const char *gd = getenv("XEMU_GUESTMEM_DUMP");
        if (gd) {
            static int gn;
            if (gn < 4) {
                char gp[1024];
                snprintf(gp, sizeof(gp), "%s_%08lx_%d.raw", gd,
                         (unsigned long)dest_addr, gn++);
                FILE *gf = fopen(gp, "wb");
                if (gf) {
                    uint32_t hdr[2] = { image_blit->width,
                                        image_blit->height };
                    fwrite(hdr, sizeof(hdr), 1, gf);
                    for (unsigned int gy = 0; gy < image_blit->height; gy++) {
                        fwrite(d->vram_ptr + dest_addr +
                                   (size_t)gy * context_surfaces->dest_pitch,
                               bytes_per_pixel, image_blit->width, gf);
                    }
                    fclose(gf);
                    fprintf(stderr, "guestmem dumped: @%08lx %ux%u -> %s\n",
                            (unsigned long)dest_addr, image_blit->width,
                            image_blit->height, gp);
                }
            }
        }
    }
}
