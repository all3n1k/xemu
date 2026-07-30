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

/*
 * This mirrors the structure of gl/surface.c. The surface cache semantics
 * (address keying, compatibility checks, eviction, CPU access callbacks,
 * upload/download scheduling) are guest behavior and are deliberately kept
 * identical; only the resource creation and the transfer mechanics are
 * Metal-specific.
 *
 * Two Metal-specific choices are worth knowing about:
 *
 *  - Render targets are allocated MTLStorageModeShared. On Apple Silicon
 *    memory is unified, so a shared render target can be read back with
 *    getBytes: without a staging blit, which is what the download path uses.
 *  - Apple GPUs do not support MTLPixelFormatDepth24Unorm_Stencil8, so the
 *    guest's Z24S8 is widened to Depth32Float_Stencil8 on the host. Host and
 *    guest bytes-per-pixel therefore differ for zeta surfaces.
 */

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "hw/xbox/nv2a/pgraph/swizzle.h"
#include "ui/xemu-settings.h"
#include "renderer.h"
#include "surface.h"
#include "draw.h"

/* ------------------------------------------------------------------ */
/* Format maps.                                                        */
/* ------------------------------------------------------------------ */

static const MetalSurfaceFormatInfo kelvin_surface_color_format_metal_map[] = {
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_Z1R5G5B5] =
        { 2, 2, MTLPixelFormatBGR5A1Unorm, false, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5] =
        { 2, 2, MTLPixelFormatB5G6R5Unorm, false, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8] =
        { 4, 4, MTLPixelFormatBGRA8Unorm, false, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8] =
        { 4, 4, MTLPixelFormatBGRA8Unorm, false, false },

    // FIXME: Map channel color, as the GL backend also leaves unresolved.
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_B8] =
        { 1, 1, MTLPixelFormatR8Unorm, false, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_G8B8] =
        { 2, 2, MTLPixelFormatRG8Unorm, false, false },
};

/*
 * Depth formats. Metal has no 24-bit depth format on Apple GPUs, and no
 * fixed-point 16-bit depth+stencil, so both zeta formats are carried at
 * higher host precision than the guest uses. The download path is
 * responsible for narrowing back to the guest layout.
 */
static const MetalSurfaceFormatInfo kelvin_surface_zeta_format_metal_map[] = {
    [NV097_SET_SURFACE_FORMAT_ZETA_Z16] =
        { 4, 2, MTLPixelFormatDepth32Float, true, false },
    [NV097_SET_SURFACE_FORMAT_ZETA_Z24S8] =
        { 8, 4, MTLPixelFormatDepth32Float_Stencil8, true, true },
};

/* ------------------------------------------------------------------ */
/* Cache bookkeeping.                                                  */
/* ------------------------------------------------------------------ */

static void surface_download(NV2AState *d, MetalSurfaceBinding *surface,
                             bool force);

static bool check_surface_overlaps_range(const MetalSurfaceBinding *surface,
                                         hwaddr range_start, hwaddr range_len)
{
    hwaddr surface_end = surface->vram_addr + surface->size;
    hwaddr range_end = range_start + range_len;
    return !(surface->vram_addr >= range_end || range_start >= surface_end);
}

static void surface_access_callback(void *opaque, MemoryRegion *mr, hwaddr addr,
                                    hwaddr len, bool write)
{
    NV2AState *d = (NV2AState *)opaque;
    qemu_mutex_lock(&d->pgraph.lock);

    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    bool wait_for_downloads = false;

    MetalSurfaceBinding *surface;
    QTAILQ_FOREACH(surface, &r->surfaces, entry) {
        if (!check_surface_overlaps_range(surface, addr, len)) {
            continue;
        }

        if (surface->draw_dirty) {
            surface->download_pending = true;
            wait_for_downloads = true;
        }

        if (write) {
            surface->upload_pending = true;
        }
    }

    qemu_mutex_unlock(&d->pgraph.lock);

    if (wait_for_downloads) {
        qemu_mutex_lock(&d->pfifo.lock);
        qemu_event_reset(&r->downloads_complete);
        qatomic_set(&r->downloads_pending, true);
        pfifo_kick(d);
        qemu_mutex_unlock(&d->pfifo.lock);
        qemu_event_wait(&r->downloads_complete);
    }
}

static void register_cpu_access_callback(NV2AState *d,
                                         MetalSurfaceBinding *surface)
{
    if (tcg_enabled()) {
        if (surface->width && surface->height) {
            surface->access_cb = mem_access_callback_insert(
                qemu_get_cpu(0), d->vram, surface->vram_addr, surface->size,
                &surface_access_callback, d);
        } else {
            surface->access_cb = NULL;
        }
    }
}

static void unregister_cpu_access_callback(NV2AState *d,
                                           MetalSurfaceBinding const *surface)
{
    if (tcg_enabled()) {
        mem_access_callback_remove_by_ref(qemu_get_cpu(0), surface->access_cb);
    }
}

MetalSurfaceBinding *pgraph_metal_surface_get(NV2AState *d, hwaddr addr)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    MetalSurfaceBinding *surface;
    QTAILQ_FOREACH (surface, &r->surfaces, entry) {
        if (surface->vram_addr == addr) {
            return surface;
        }
    }

    return NULL;
}

MetalSurfaceBinding *pgraph_metal_surface_get_within(NV2AState *d, hwaddr addr)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    MetalSurfaceBinding *surface;
    QTAILQ_FOREACH (surface, &r->surfaces, entry) {
        if (addr >= surface->vram_addr &&
            addr < (surface->vram_addr + surface->size)) {
            return surface;
        }
    }

    return NULL;
}

void pgraph_metal_surface_invalidate(NV2AState *d,
                                     MetalSurfaceBinding *surface)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (surface == r->color_binding) {
        assert(d->pgraph.surface_color.buffer_dirty);
        pgraph_metal_unbind_surface(d, true);
    }
    if (surface == r->zeta_binding) {
        assert(d->pgraph.surface_zeta.buffer_dirty);
        pgraph_metal_unbind_surface(d, false);
    }

    unregister_cpu_access_callback(d, surface);

    /* The open pass may be targeting this texture. Submit it before the
     * binding goes away; Metal keeps the texture alive until the command
     * buffer retires, so there is no need to wait for it. */
    pgraph_metal_flush_gpu(d, false);

    surface->texture = nil;
    QTAILQ_REMOVE(&r->surfaces, surface, entry);
    g_free(surface);
}

static bool check_surfaces_overlap(const MetalSurfaceBinding *surface,
                                   const MetalSurfaceBinding *other)
{
    return check_surface_overlaps_range(surface, other->vram_addr, other->size);
}

static void invalidate_overlapping_surfaces(NV2AState *d,
                                            MetalSurfaceBinding *surface)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    MetalSurfaceBinding *other, *next;
    QTAILQ_FOREACH_SAFE(other, &r->surfaces, entry, next) {
        if (check_surfaces_overlap(surface, other)) {
            pgraph_metal_surface_download_if_dirty(d, other);
            pgraph_metal_surface_invalidate(d, other);
        }
    }
}

static MetalSurfaceBinding *surface_put(NV2AState *d, hwaddr addr,
                                        MetalSurfaceBinding *surface_in)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    assert(pgraph_metal_surface_get(d, addr) == NULL);

    invalidate_overlapping_surfaces(d, surface_in);

    MetalSurfaceBinding *surface_out = g_malloc(sizeof(MetalSurfaceBinding));
    *surface_out = *surface_in;

    register_cpu_access_callback(d, surface_out);

    QTAILQ_INSERT_TAIL(&r->surfaces, surface_out, entry);

    return surface_out;
}

static bool check_surface_compatibility(const MetalSurfaceBinding *s1,
                                        const MetalSurfaceBinding *s2,
                                        bool strict)
{
    bool compatible =
        (s1->color == s2->color) &&
        (s1->fmt.pixel_format == s2->fmt.pixel_format) &&
        (s1->pitch == s2->pitch) &&
        (s1->shape.clip_x <= s2->shape.clip_x) &&
        (s1->shape.clip_y <= s2->shape.clip_y);

    if (!strict) {
        return compatible && (s1->width >= s2->width) &&
               (s1->height >= s2->height);
    }

    return compatible && (s1->width == s2->width) &&
           (s1->height == s2->height);
}

/* ------------------------------------------------------------------ */
/* Surface creation.                                                   */
/* ------------------------------------------------------------------ */

static id<MTLTexture> create_surface_texture(PGRAPHMetalState *r,
                                             const MetalSurfaceFormatInfo *fmt,
                                             unsigned int width,
                                             unsigned int height)
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:fmt->pixel_format
                                     width:MAX(width, 1)
                                    height:MAX(height, 1)
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    /*
     * Shared storage keeps the target directly readable from the CPU, which
     * the download path depends on. Depth/stencil targets may refuse shared
     * storage depending on the GPU, so fall back to private for those and
     * accept that they cannot be read back until a staging blit exists.
     */
    desc.storageMode = MTLStorageModeShared;

    id<MTLTexture> tex = [r->device newTextureWithDescriptor:desc];
    if (tex == nil && fmt->depth) {
        desc.storageMode = MTLStorageModePrivate;
        tex = [r->device newTextureWithDescriptor:desc];
    }

    return tex;
}

static void surface_get_dimensions(PGRAPHState *pg, unsigned int *width,
                                   unsigned int *height)
{
    bool swizzle = (pg->surface_type == NV097_SET_SURFACE_FORMAT_TYPE_SWIZZLE);
    if (swizzle) {
        *width = 1 << pg->surface_shape.log_width;
        *height = 1 << pg->surface_shape.log_height;
    } else {
        *width = pg->surface_shape.clip_width;
        *height = pg->surface_shape.clip_height;
    }
}

static void populate_surface_binding_entry_sized(NV2AState *d, bool color,
                                                 unsigned int width,
                                                 unsigned int height,
                                                 MetalSurfaceBinding *entry)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    Surface *surface;
    hwaddr dma_address;
    MetalSurfaceFormatInfo fmt;

    if (color) {
        surface = &pg->surface_color;
        dma_address = pg->dma_color;
        assert(pg->surface_shape.color_format != 0);
        assert(pg->surface_shape.color_format <
               ARRAY_SIZE(kelvin_surface_color_format_metal_map));
        fmt = kelvin_surface_color_format_metal_map[pg->surface_shape
                                                        .color_format];
        if (fmt.host_bytes_per_pixel == 0) {
            fprintf(stderr, "nv2a: unimplemented color surface format 0x%x\n",
                    pg->surface_shape.color_format);
            abort();
        }
    } else {
        surface = &pg->surface_zeta;
        dma_address = pg->dma_zeta;
        assert(pg->surface_shape.zeta_format != 0);
        assert(pg->surface_shape.zeta_format <
               ARRAY_SIZE(kelvin_surface_zeta_format_metal_map));
        fmt = kelvin_surface_zeta_format_metal_map[pg->surface_shape
                                                       .zeta_format];
        if (fmt.host_bytes_per_pixel == 0) {
            fprintf(stderr, "nv2a: unimplemented zeta surface format 0x%x\n",
                    pg->surface_shape.zeta_format);
            abort();
        }
    }

    DMAObject dma = nv_dma_load(d, dma_address);
    /* There's a bunch of bugs that could cause us to hit this function at the
     * wrong time and get an invalid dma object. Check that it's sane. */
    assert(dma.dma_class == NV_DMA_IN_MEMORY_CLASS);
    assert(surface->offset <= dma.limit);
    assert(surface->offset + surface->pitch * height <= dma.limit + 1);
    assert(surface->pitch % fmt.guest_bytes_per_pixel == 0);
    assert((dma.address & ~0x07FFFFFF) == 0);

    entry->shape = (color || !r->color_binding) ? pg->surface_shape :
                                                  r->color_binding->shape;
    entry->texture = nil;
    entry->fmt = fmt;
    entry->color = color;
    entry->swizzle =
        (pg->surface_type == NV097_SET_SURFACE_FORMAT_TYPE_SWIZZLE);
    entry->vram_addr = dma.address + surface->offset;
    entry->width = width;
    entry->height = height;
    entry->pitch = surface->pitch;
    entry->size = height * MAX(surface->pitch,
                               width * fmt.guest_bytes_per_pixel);
    entry->upload_pending = true;
    entry->download_pending = false;
    entry->draw_dirty = false;
    entry->dma_addr = dma.address;
    entry->dma_len = dma.limit;
    entry->frame_time = pg->frame_time;
    entry->draw_time = pg->draw_time;
    entry->cleared = false;
}

static void populate_surface_binding_entry(NV2AState *d, bool color,
                                           MetalSurfaceBinding *entry)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    unsigned int width, height;

    if (color || !r->color_binding) {
        surface_get_dimensions(pg, &width, &height);
        pgraph_apply_anti_aliasing_factor(pg, &width, &height);

        /* Surface dimensions come from the clipping rectangle, so the surface
         * offset has to be included as well. */
        if (pg->surface_type != NV097_SET_SURFACE_FORMAT_TYPE_SWIZZLE) {
            width += pg->surface_shape.clip_x;
            height += pg->surface_shape.clip_y;
        }
    } else {
        width = r->color_binding->width;
        height = r->color_binding->height;
    }

    populate_surface_binding_entry_sized(d, color, width, height, entry);
}

/* ------------------------------------------------------------------ */
/* Download.                                                           */
/* ------------------------------------------------------------------ */

static void surface_copy_shrink_row(uint8_t *out, uint8_t *in,
                                    unsigned int width,
                                    unsigned int bytes_per_pixel,
                                    unsigned int factor)
{
    if (bytes_per_pixel == 4) {
        for (unsigned int x = 0; x < width; x++) {
            *(uint32_t *)out = *(uint32_t *)in;
            out += 4;
            in += 4 * factor;
        }
    } else if (bytes_per_pixel == 2) {
        for (unsigned int x = 0; x < width; x++) {
            *(uint16_t *)out = *(uint16_t *)in;
            out += 2;
            in += 2 * factor;
        }
    } else {
        for (unsigned int x = 0; x < width; x++) {
            memcpy(out, in, bytes_per_pixel);
            out += bytes_per_pixel;
            in += bytes_per_pixel * factor;
        }
    }
}

static void surface_download_to_buffer(NV2AState *d,
                                       MetalSurfaceBinding *surface,
                                       bool swizzle, bool downscale,
                                       uint8_t *pixels)
{
    PGRAPHState *pg = &d->pgraph;

    swizzle &= surface->swizzle;
    downscale &= (pg->surface_scale_factor != 1);

    if (!surface->width || !surface->height) {
        return;
    }

    if (surface->texture.storageMode != MTLStorageModeShared) {
        /* Private storage: no direct readback. Only depth targets can end up
         * here, and only on a GPU that refused shared depth. */
        NV2A_UNIMPLEMENTED("Readback of a private-storage surface");
        return;
    }

    if (!surface->color) {
        /* The host depth format is wider than the guest's (see the format
         * map), so a straight copy would corrupt guest memory. Narrowing is
         * not implemented yet; leaving guest memory untouched is the safer
         * failure. */
        NV2A_UNIMPLEMENTED("Zeta surface download");
        return;
    }

    unsigned int scale = pg->surface_scale_factor;
    unsigned int bpp = surface->fmt.host_bytes_per_pixel;

    uint8_t *read_buf = pixels;
    uint8_t *swizzle_buf = pixels;

    if (swizzle) {
        assert(scale == 1 || downscale);
        swizzle_buf = (uint8_t *)g_malloc(surface->size);
        read_buf = swizzle_buf;
    }

    if (downscale) {
        pg->scale_buf = (uint8_t *)g_realloc(
            pg->scale_buf, scale * scale * surface->size);
        read_buf = pg->scale_buf;
    }

    /* Unified memory: a shared render target is readable without staging. */
    unsigned int rw = scale * surface->width;
    unsigned int rh = scale * surface->height;
    if (rw == 0 || rh == 0 || rw > surface->texture.width ||
        rh > surface->texture.height) {
        fprintf(stderr,
                "nv2a: metal: skipping download of %ux%u region from %lux%lu "
                "texture\n",
                rw, rh, (unsigned long)surface->texture.width,
                (unsigned long)surface->texture.height);
        return;
    }
    /* Draws accumulate in an open render pass; the target is not readable
     * until that pass has been submitted and has finished. */
    pgraph_metal_flush_gpu(d, true);

    MTLRegion region = MTLRegionMake2D(0, 0, rw, rh);
    [surface->texture getBytes:read_buf
                   bytesPerRow:scale * surface->pitch
                    fromRegion:region
                   mipmapLevel:0];

    /* FIXME: Replace with a hardware-accelerated downscale. */
    if (downscale) {
        assert(surface->pitch >= (surface->width * bpp));
        uint8_t *out = swizzle_buf, *in = pg->scale_buf;
        for (unsigned int y = 0; y < surface->height; y++) {
            surface_copy_shrink_row(out, in, surface->width, bpp, scale);
            in += surface->pitch * scale * scale;
            out += surface->pitch;
        }
    }

    if (swizzle) {
        swizzle_rect(swizzle_buf, surface->width, surface->height, pixels,
                     surface->pitch, bpp);
        g_free(swizzle_buf);
    }
}

static void surface_download(NV2AState *d, MetalSurfaceBinding *surface,
                             bool force)
{
    if (!(surface->download_pending || force) || !surface->width ||
        !surface->height) {
        return;
    }

    nv2a_profile_inc_counter(NV2A_PROF_SURF_DOWNLOAD);

    surface_download_to_buffer(d, surface, true, true,
                               d->vram_ptr + surface->vram_addr);

    memory_region_set_client_dirty(d->vram, surface->vram_addr,
                                   surface->pitch * surface->height,
                                   DIRTY_MEMORY_VGA);
    memory_region_set_client_dirty(d->vram, surface->vram_addr,
                                   surface->pitch * surface->height,
                                   DIRTY_MEMORY_NV2A_TEX);

    if (getenv("XEMU_METAL_FB_STATS")) {
        static int n;
        if ((n++ % 30) == 0) {
            const uint32_t *px = (const uint32_t *)(d->vram_ptr +
                                                    surface->vram_addr);
            size_t count = (size_t)surface->width * surface->height;
            size_t nonblack = 0, distinct_hint = 0;
            uint32_t first = count ? px[0] : 0;
            for (size_t i = 0; i < count; i++) {
                if ((px[i] & 0x00FFFFFF) != 0) {
                    nonblack++;
                }
                if (px[i] != first) {
                    distinct_hint = 1;
                }
            }
            /* Alpha is masked out above, so a surface legitimately full of
             * opaque black reads as 0% -- report the raw first pixel too. */
            fprintf(stderr,
                    "fb-stats #%d: %ux%u @%08lx  nonblack=%zu/%zu (%.1f%%)  "
                    "varied=%s first=%08x\n",
                    n, surface->width, surface->height,
                    (unsigned long)surface->vram_addr, nonblack, count,
                    count ? 100.0 * nonblack / count : 0.0,
                    distinct_hint ? "yes" : "no", first);
        }
    }

    surface->download_pending = false;
    surface->draw_dirty = false;
}

/*
 * Write back any rendered surface whose memory overlaps [addr, addr+len).
 *
 * The CPU-access callback that would normally catch this is registered only
 * under TCG, and Apple Silicon runs under HVF, so a guest that renders into a
 * surface and then samples that memory as a texture would otherwise read
 * whatever was in RAM before the draw. Measured: the BIOS logo texture came
 * back differing from the GL backend in 602007 of 4194312 source bytes for
 * exactly this reason.
 */
void pgraph_metal_download_surfaces_overlapping(NV2AState *d, hwaddr addr,
                                                hwaddr len)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (getenv("XEMU_METAL_TRACE_TEXSRC") && len > 1000000) {
        static int n;
        if (n++ < 3) {
            fprintf(stderr, "texsrc: texture @%08lx len=%lu; live surfaces:",
                    (unsigned long)addr, (unsigned long)len);
            MetalSurfaceBinding *q;
            QTAILQ_FOREACH (q, &r->surfaces, entry) {
                fprintf(stderr, " @%08lx(%ux%u%s%s)",
                        (unsigned long)q->vram_addr, q->width, q->height,
                        q->color ? "c" : "z", q->draw_dirty ? ",dirty" : "");
            }
            fprintf(stderr, "\n");
        }
    }

    MetalSurfaceBinding *surface;
    QTAILQ_FOREACH (surface, &r->surfaces, entry) {
        if (surface->color && surface->draw_dirty &&
            check_surface_overlaps_range(surface, addr, len)) {
            surface_download(d, surface, false);
            surface->draw_dirty = false;
        }
    }
}

void pgraph_metal_surface_download_if_dirty(NV2AState *d,
                                            MetalSurfaceBinding *surface)
{
    if (surface->draw_dirty) {
        surface_download(d, surface, true);
    }
}

void pgraph_metal_process_pending_downloads(NV2AState *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    MetalSurfaceBinding *surface;
    QTAILQ_FOREACH(surface, &r->surfaces, entry) {
        surface_download(d, surface, false);
    }

    qatomic_set(&r->downloads_pending, false);
    qemu_event_set(&r->downloads_complete);
}

void pgraph_metal_download_dirty_surfaces(NV2AState *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    MetalSurfaceBinding *surface;
    QTAILQ_FOREACH(surface, &r->surfaces, entry) {
        pgraph_metal_surface_download_if_dirty(d, surface);
    }

    qatomic_set(&r->download_dirty_surfaces_pending, false);
    qemu_event_set(&r->dirty_surfaces_download_complete);
}


/* ------------------------------------------------------------------ */
/* Upload.                                                             */
/* ------------------------------------------------------------------ */

/*
 * Guest memory -> render target.
 *
 * This is the other half of the lazy surface synchronization the GL and
 * Vulkan backends do, and it was missing here. Anything that writes a
 * surface's memory with the CPU -- most importantly the 2D blit that moves a
 * rendered frame to the scanout buffer -- marks the destination
 * upload_pending, and without this the GPU texture never sees those pixels.
 *
 * Two layout differences have to be undone: the NV2A's swizzled ordering,
 * and a guest pitch that is wider than the texture row (the surface may be a
 * window into a larger buffer). Getting either wrong shows up as diagonally
 * skewed streaks rather than as a blank screen.
 */
void pgraph_metal_upload_surface_data(NV2AState *d,
                                      MetalSurfaceBinding *surface, bool force)
{
    if (!(surface->upload_pending || force)) {
        return;
    }

    PGRAPHState *pg = &d->pgraph;

    surface->upload_pending = false;
    surface->draw_time = pg->draw_time;

    if (!surface->width || !surface->height || surface->texture == nil) {
        return;
    }

    if (!surface->color) {
        /* The host depth format is wider than the guest's, so a raw copy
         * would be wrong; see the format map. */
        return;
    }

    unsigned int bpp = surface->fmt.host_bytes_per_pixel;
    uint8_t *base = d->vram_ptr + surface->vram_addr;

    g_autofree uint8_t *unswizzled = NULL;
    uint8_t *buf = base;

    if (surface->swizzle) {
        unswizzled = g_malloc(surface->size);
        unswizzle_rect(base, surface->width, surface->height, unswizzled,
                       surface->pitch, bpp);
        buf = unswizzled;
    }

    /* replaceRegion: takes a source stride, so a wider guest pitch is fine
     * as long as it is passed through rather than assumed to be width*bpp. */
    unsigned int src_pitch = surface->pitch;
    if (src_pitch < surface->width * bpp) {
        /* Malformed; refuse rather than read past the row. */
        return;
    }

    unsigned int w = MIN(surface->width, (unsigned int)surface->texture.width);
    unsigned int h = MIN(surface->height,
                         (unsigned int)surface->texture.height);
    if (!w || !h) {
        return;
    }

    if (pg->surface_scale_factor != 1) {
        /* The texture is larger than the guest surface; a straight upload
         * would only cover a corner of it. Scaled upload is not implemented,
         * so leave the target alone rather than corrupt it. */
        return;
    }

    /* A CPU write into a texture the open pass may still be rendering to. */
    pgraph_metal_flush_gpu(d, true);

    [surface->texture replaceRegion:MTLRegionMake2D(0, 0, w, h)
                        mipmapLevel:0
                          withBytes:buf
                        bytesPerRow:src_pitch];
}

/* ------------------------------------------------------------------ */
/* Binding.                                                            */
/* ------------------------------------------------------------------ */

void pgraph_metal_unbind_surface(NV2AState *d, bool color)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (color) {
        r->color_binding = NULL;
    } else {
        r->zeta_binding = NULL;
    }
}

static void update_surface_part(NV2AState *d, bool upload, bool color)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    MetalSurfaceBinding entry;
    populate_surface_binding_entry(d, color, &entry);

    Surface *surface = color ? &pg->surface_color : &pg->surface_zeta;

    bool mem_dirty = !tcg_enabled() && memory_region_test_and_clear_dirty(
                                           d->vram, entry.vram_addr, entry.size,
                                           DIRTY_MEMORY_NV2A);

    if (upload && (surface->buffer_dirty || mem_dirty)) {
        pgraph_metal_unbind_surface(d, color);

        MetalSurfaceBinding *found = pgraph_metal_surface_get(d,
                                                              entry.vram_addr);
        if (found != NULL) {
            MetalSurfaceBinding *other =
                (color ? r->zeta_binding : r->color_binding);
            if (found == other) {
                NV2A_UNIMPLEMENTED("Same color & zeta surface offset");
                pgraph_metal_unbind_surface(d, !color);
            }
        }

        bool should_create = true;

        if (found != NULL) {
            bool is_compatible =
                check_surface_compatibility(found, &entry, false);

            assert(!(entry.swizzle && pg->clearing));

            if (found->swizzle != entry.swizzle) {
                /* Clears should only be done on linear surfaces. Allow (1) a
                 * surface marked swizzled to be cleared, assuming the whole
                 * surface is destined to be cleared, and (2) a fully cleared
                 * linear surface to be marked swizzled. Match size strictly to
                 * avoid pathological cases. */
                is_compatible &= (pg->clearing || found->cleared) &&
                                 check_surface_compatibility(found, &entry,
                                                             true);
            }

            if (is_compatible && color &&
                !check_surface_compatibility(found, &entry, true)) {
                MetalSurfaceBinding zeta_entry;
                populate_surface_binding_entry_sized(
                    d, !color, found->width, found->height, &zeta_entry);
                hwaddr color_end = found->vram_addr + found->size;
                hwaddr zeta_end = zeta_entry.vram_addr + zeta_entry.size;
                is_compatible &= found->vram_addr >= zeta_end ||
                                 zeta_entry.vram_addr >= color_end;
            }

            if (is_compatible && !color && r->color_binding) {
                is_compatible &= (found->width == r->color_binding->width) &&
                                 (found->height == r->color_binding->height);
            }

            if (is_compatible) {
                pg->surface_binding_dim.width = found->width;
                pg->surface_binding_dim.clip_x = found->shape.clip_x;
                pg->surface_binding_dim.clip_width = found->shape.clip_width;
                pg->surface_binding_dim.height = found->height;
                pg->surface_binding_dim.clip_y = found->shape.clip_y;
                pg->surface_binding_dim.clip_height = found->shape.clip_height;
                found->upload_pending |= mem_dirty;
                pg->surface_zeta.buffer_dirty |= color;
                should_create = false;
            } else {
                pgraph_metal_surface_download_if_dirty(d, found);
                pgraph_metal_surface_invalidate(d, found);
            }
        }

        if (should_create) {
            unsigned int width = entry.width ? entry.width : 1;
            unsigned int height = entry.height ? entry.height : 1;
            pgraph_apply_scaling_factor(pg, &width, &height);

            entry.texture = create_surface_texture(r, &entry.fmt, width,
                                                   height);
            if (entry.texture == nil) {
                fprintf(stderr,
                        "nv2a: Metal surface allocation failed "
                        "(%s %ux%u fmt %lu)\n",
                        color ? "color" : "zeta", width, height,
                        (unsigned long)entry.fmt.pixel_format);
                abort();
            }

            found = surface_put(d, entry.vram_addr, &entry);

            pg->surface_binding_dim.width = entry.width;
            pg->surface_binding_dim.clip_x = entry.shape.clip_x;
            pg->surface_binding_dim.clip_width = entry.shape.clip_width;
            pg->surface_binding_dim.height = entry.height;
            pg->surface_binding_dim.clip_y = entry.shape.clip_y;
            pg->surface_binding_dim.clip_height = entry.shape.clip_height;

            if (color && r->zeta_binding &&
                (r->zeta_binding->width != entry.width ||
                 r->zeta_binding->height != entry.height)) {
                pg->surface_zeta.buffer_dirty = true;
            }
        }

        if (color) {
            r->color_binding = found;
        } else {
            r->zeta_binding = found;
        }

        /* If guest memory is newer than the texture, pull it in before the
         * next draw reads or blends against it. */
        pgraph_metal_upload_surface_data(d, found, false);

        surface->buffer_dirty = false;
    }

    if (!upload && surface->draw_dirty) {
        if (!tcg_enabled()) {
            /* FIXME: Cannot monitor for reads/writes; flush now */
            surface_download(d, color ? r->color_binding : r->zeta_binding,
                             true);
        }

        surface->write_enabled_cache = false;
        surface->draw_dirty = false;
    }
}

static bool framebuffer_dirty(PGRAPHState *pg)
{
    bool shape_changed = memcmp(&pg->surface_shape, &pg->last_surface_shape,
                                sizeof(SurfaceShape)) != 0;
    if (!shape_changed || (!pg->surface_shape.color_format &&
                           !pg->surface_shape.zeta_format)) {
        return false;
    }
    return true;
}

void pgraph_metal_surface_update(NV2AState *d, bool upload, bool color_write,
                                 bool zeta_write)
{
    PGRAPHState *pg = &d->pgraph;

    pg->surface_shape.z_format =
        GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_SETUPRASTER),
                 NV_PGRAPH_SETUPRASTER_Z_FORMAT);

    /* The write-enable gating is what keeps a surface with no configured
     * format from reaching binding creation. */
    color_write =
        color_write && (pg->clearing || pgraph_color_write_enabled(pg));
    zeta_write = zeta_write && (pg->clearing || pgraph_zeta_write_enabled(pg));

    if (upload) {
        if (framebuffer_dirty(pg)) {
            memcpy(&pg->last_surface_shape, &pg->surface_shape,
                   sizeof(SurfaceShape));
            pg->surface_color.buffer_dirty = true;
            pg->surface_zeta.buffer_dirty = true;
        }

        if (pg->surface_color.buffer_dirty) {
            pgraph_metal_unbind_surface(d, true);
        }

        if (color_write) {
            update_surface_part(d, true, true);
        }

        if (pg->surface_zeta.buffer_dirty) {
            pgraph_metal_unbind_surface(d, false);
        }

        if (zeta_write) {
            update_surface_part(d, true, false);
        }
    } else {
        if ((color_write || pg->surface_color.write_enabled_cache) &&
            pg->surface_color.draw_dirty) {
            update_surface_part(d, false, true);
        }
        if ((zeta_write || pg->surface_zeta.write_enabled_cache) &&
            pg->surface_zeta.draw_dirty) {
            update_surface_part(d, false, false);
        }
    }
}

void pgraph_metal_set_surface_dirty(PGRAPHState *pg, bool color, bool zeta)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    /* FIXME: Does this apply to CLEARs too? (carried over from the GL path) */
    color = color && pgraph_color_write_enabled(pg);
    zeta = zeta && pgraph_zeta_write_enabled(pg);
    pg->surface_color.draw_dirty |= color;
    pg->surface_zeta.draw_dirty |= zeta;

    if (r->color_binding) {
        r->color_binding->draw_dirty |= color;
        r->color_binding->frame_time = pg->frame_time;
        r->color_binding->cleared = false;
    }

    if (r->zeta_binding) {
        r->zeta_binding->draw_dirty |= zeta;
        r->zeta_binding->frame_time = pg->frame_time;
        r->zeta_binding->cleared = false;
    }
}

/* ------------------------------------------------------------------ */
/* Clear.                                                              */
/* ------------------------------------------------------------------ */

/*
 * Fill a rectangle of a shared-storage color target on the CPU.
 *
 * Metal expresses a clear as a render pass load action, which always covers
 * the whole attachment. The NV2A can clear a sub-rectangle, and until the
 * draw path exists there is no scissored quad to fall back on, so partial
 * clears are done directly against the shared texture. This is slow but
 * correct, and it goes away once draw.m can issue a scissored clear quad.
 */
static void clear_color_region_cpu(MetalSurfaceBinding *surface,
                                   MTLClearColor c, unsigned int x,
                                   unsigned int y, unsigned int w,
                                   unsigned int h)
{
    unsigned int bpp = surface->fmt.host_bytes_per_pixel;

    /* Clamp to the allocated texture: the clear rect comes from guest
     * registers and is not guaranteed to lie inside it. */
    if (x >= surface->texture.width || y >= surface->texture.height) {
        return;
    }
    w = MIN(w, (unsigned int)surface->texture.width - x);
    h = MIN(h, (unsigned int)surface->texture.height - y);
    if (w == 0 || h == 0) {
        return;
    }

    size_t row_bytes = (size_t)w * bpp;
    uint8_t *row = g_malloc(row_bytes);

    /* Only the formats in the color map can reach here. */
    switch (surface->fmt.pixel_format) {
    case MTLPixelFormatBGRA8Unorm: {
        uint8_t b = (uint8_t)(CLAMP(c.blue, 0.0, 1.0) * 255.0 + 0.5);
        uint8_t g = (uint8_t)(CLAMP(c.green, 0.0, 1.0) * 255.0 + 0.5);
        uint8_t rr = (uint8_t)(CLAMP(c.red, 0.0, 1.0) * 255.0 + 0.5);
        uint8_t a = (uint8_t)(CLAMP(c.alpha, 0.0, 1.0) * 255.0 + 0.5);
        uint32_t px = b | (g << 8) | (rr << 16) | ((uint32_t)a << 24);
        for (unsigned int i = 0; i < w; i++) {
            ((uint32_t *)row)[i] = px;
        }
        break;
    }
    case MTLPixelFormatB5G6R5Unorm: {
        uint16_t px = (uint16_t)((((unsigned)(CLAMP(c.red, 0.0, 1.0) * 31.0 + 0.5)) << 11) |
                                 (((unsigned)(CLAMP(c.green, 0.0, 1.0) * 63.0 + 0.5)) << 5) |
                                 ((unsigned)(CLAMP(c.blue, 0.0, 1.0) * 31.0 + 0.5)));
        for (unsigned int i = 0; i < w; i++) {
            ((uint16_t *)row)[i] = px;
        }
        break;
    }
    case MTLPixelFormatBGR5A1Unorm: {
        uint16_t px = (uint16_t)((((unsigned)(CLAMP(c.alpha, 0.0, 1.0) + 0.5)) << 15) |
                                 (((unsigned)(CLAMP(c.red, 0.0, 1.0) * 31.0 + 0.5)) << 10) |
                                 (((unsigned)(CLAMP(c.green, 0.0, 1.0) * 31.0 + 0.5)) << 5) |
                                 ((unsigned)(CLAMP(c.blue, 0.0, 1.0) * 31.0 + 0.5)));
        for (unsigned int i = 0; i < w; i++) {
            ((uint16_t *)row)[i] = px;
        }
        break;
    }
    default:
        /* R8 / RG8 channel formats are already flagged unimplemented in the
         * format map comment; fill with zero rather than guess. */
        memset(row, 0, row_bytes);
        break;
    }

    for (unsigned int i = 0; i < h; i++) {
        [surface->texture replaceRegion:MTLRegionMake2D(x, y + i, w, 1)
                            mipmapLevel:0
                              withBytes:row
                            bytesPerRow:row_bytes];
    }

    g_free(row);
}

void pgraph_metal_clear_surface(NV2AState *d, uint32_t parameter)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    pg->clearing = true;

    bool write_color = (parameter & NV097_CLEAR_SURFACE_COLOR);
    bool write_zeta =
        (parameter & (NV097_CLEAR_SURFACE_Z | NV097_CLEAR_SURFACE_STENCIL));

    float rgba[4] = { 0, 0, 0, 0 };
    float clear_depth = 1.0f;
    int clear_stencil = 0;

    if (write_zeta) {
        pgraph_get_clear_depth_stencil_value(pg, &clear_depth, &clear_stencil);
    }
    if (write_color) {
        pgraph_get_clear_color(pg, rgba);
    }

    pgraph_metal_surface_update(d, true, write_color, write_zeta);

    /* FIXME: Needs confirmation (mirrors the GL backend) */
    unsigned int xmin = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTX),
                                 NV_PGRAPH_CLEARRECTX_XMIN);
    unsigned int xmax = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTX),
                                 NV_PGRAPH_CLEARRECTX_XMAX);
    unsigned int ymin = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTY),
                                 NV_PGRAPH_CLEARRECTY_YMIN);
    unsigned int ymax = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTY),
                                 NV_PGRAPH_CLEARRECTY_YMAX);

    unsigned int scissor_width = xmax - xmin + 1;
    unsigned int scissor_height = ymax - ymin + 1;
    pgraph_apply_anti_aliasing_factor(pg, &xmin, &ymin);
    pgraph_apply_anti_aliasing_factor(pg, &scissor_width, &scissor_height);

    bool full_clear = !xmin && !ymin &&
                      scissor_width >= pg->surface_binding_dim.width &&
                      scissor_height >= pg->surface_binding_dim.height;

    pgraph_apply_scaling_factor(pg, &xmin, &ymin);
    pgraph_apply_scaling_factor(pg, &scissor_width, &scissor_height);

    /* Colour writes are masked per channel by the guest. A load-action clear
     * cannot express a channel mask, so fall back to the CPU path when the
     * mask is partial. */
    bool full_color_mask = (parameter & NV097_CLEAR_SURFACE_R) &&
                           (parameter & NV097_CLEAR_SURFACE_G) &&
                           (parameter & NV097_CLEAR_SURFACE_B) &&
                           (parameter & NV097_CLEAR_SURFACE_A);

    MTLClearColor clear_color =
        MTLClearColorMake(rgba[0], rgba[1], rgba[2], rgba[3]);

    bool use_render_pass = full_clear &&
                           (!write_color || full_color_mask);

    if (use_render_pass) {
        MTLRenderPassDescriptor *pass =
            [MTLRenderPassDescriptor renderPassDescriptor];
        bool any = false;

        if (write_color && r->color_binding) {
            pass.colorAttachments[0].texture = r->color_binding->texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.colorAttachments[0].clearColor = clear_color;
            any = true;
        }

        if (write_zeta && r->zeta_binding) {
            if (parameter & NV097_CLEAR_SURFACE_Z) {
                pass.depthAttachment.texture = r->zeta_binding->texture;
                pass.depthAttachment.loadAction = MTLLoadActionClear;
                pass.depthAttachment.storeAction = MTLStoreActionStore;
                pass.depthAttachment.clearDepth = clear_depth;
                any = true;
            }
            if ((parameter & NV097_CLEAR_SURFACE_STENCIL) &&
                r->zeta_binding->fmt.stencil) {
                pass.stencilAttachment.texture = r->zeta_binding->texture;
                pass.stencilAttachment.loadAction = MTLLoadActionClear;
                pass.stencilAttachment.storeAction = MTLStoreActionStore;
                pass.stencilAttachment.clearStencil = clear_stencil;
                any = true;
            }
        }

        if (any) {
            /* Ordering: the open pass holds draws recorded before this clear,
             * and they must be submitted first or the clear would be applied
             * underneath them. */
            pgraph_metal_flush_gpu(d, false);

            /* An empty render pass is exactly a clear: the load action does
             * the work and there is nothing to draw. */
            id<MTLCommandBuffer> cmd = [r->queue commandBuffer];
            id<MTLRenderCommandEncoder> enc =
                [cmd renderCommandEncoderWithDescriptor:pass];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status != MTLCommandBufferStatusCompleted) {
                static int once;
                if (!once++) {
                    fprintf(stderr,
                            "nv2a: metal: clear command buffer status=%ld "
                            "error=%s\n", (long)cmd.status,
                            cmd.error ?
                                [[cmd.error localizedDescription] UTF8String] :
                                "(none)");
                }
            }

        }
    } else {
        if (write_color && r->color_binding) {
            unsigned int w = MIN(scissor_width,
                                 (unsigned int)r->color_binding->texture.width -
                                     MIN(xmin, (unsigned int)r->color_binding
                                                   ->texture.width));
            unsigned int h = MIN(scissor_height,
                                 (unsigned int)r->color_binding->texture.height -
                                     MIN(ymin, (unsigned int)r->color_binding
                                                   ->texture.height));
            if (w && h) {
                clear_color_region_cpu(r->color_binding, clear_color, xmin,
                                       ymin, w, h);
            }
        }
        if (write_zeta && r->zeta_binding) {
            NV2A_UNIMPLEMENTED("Partial or masked zeta clear");
        }
    }

    pgraph_metal_set_surface_dirty(pg, write_color, write_zeta);

    /* Ordering matters: set_surface_dirty clears these, so the cleared flag
     * has to be applied after it, as in the GL path. */
    if (r->color_binding) {
        r->color_binding->cleared = full_clear && write_color;
    }
    if (r->zeta_binding) {
        r->zeta_binding->cleared = full_clear && write_zeta;
    }

    pg->clearing = false;
}

/* ------------------------------------------------------------------ */
/* Lifecycle.                                                          */
/* ------------------------------------------------------------------ */

void pgraph_metal_reload_surface_scale_factor(PGRAPHState *pg)
{
    int factor = g_config.display.quality.surface_scale;
    pg->surface_scale_factor = factor < 1 ? 1 : factor;
}

void pgraph_metal_init_surfaces(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    /* Without this the scale factor stays 0 and every render target is
     * allocated degenerate. */
    pgraph_metal_reload_surface_scale_factor(pg);

    QTAILQ_INIT(&r->surfaces);
    r->color_binding = NULL;
    r->zeta_binding = NULL;
    r->downloads_pending = false;
    qemu_event_init(&r->downloads_complete, false);
    qemu_event_init(&r->dirty_surfaces_download_complete, false);
}

static void flush_surfaces(NV2AState *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    MetalSurfaceBinding *surface, *next;
    QTAILQ_FOREACH_SAFE(surface, &r->surfaces, entry, next) {
        pgraph_metal_surface_download_if_dirty(d, surface);
        pgraph_metal_surface_invalidate(d, surface);
    }
}

void pgraph_metal_surface_flush(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;

    pgraph_metal_flush_gpu(d, true);

    /* Clearing bindings forces the next surface_update to re-resolve them. */
    pgraph_metal_unbind_surface(d, true);
    pgraph_metal_unbind_surface(d, false);

    flush_surfaces(d);

    pg->surface_color.buffer_dirty = true;
    pg->surface_zeta.buffer_dirty = true;
}

void pgraph_metal_finalize_surfaces(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    MetalSurfaceBinding *surface, *next;
    QTAILQ_FOREACH_SAFE(surface, &r->surfaces, entry, next) {
        surface->texture = nil;
        QTAILQ_REMOVE(&r->surfaces, surface, entry);
        g_free(surface);
    }

    r->color_binding = NULL;
    r->zeta_binding = NULL;
}
