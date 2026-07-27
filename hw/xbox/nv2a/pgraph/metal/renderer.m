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

/*
 * Stage A: Skeleton.
 *
 * This file currently implements only the PGRAPHRenderer ops-table with stubs,
 * plus real Metal device/queue initialization. The goal of this stage is to
 * get the backend registered, selectable in the UI, and compiling cleanly
 * alongside the existing OpenGL and Vulkan backends without breaking them.
 *
 * Subsequent stages will fill in:
 *   - shaders.c    (GLSL -> MSL translation and pipeline state caching)
 *   - surface.c    (MTLTexture render-target management)
 *   - texture.c    (Xbox texture -> MTLTexture upload)
 *   - draw.c       (render command encoder + vertex dispatch)
 *   - display.c    (CAMetalLayer presentation path)
 *   - vertex.c, blit.c, reports.c
 */

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "renderer.h"
#include "shaders.h"
#include "surface.h"
#include "draw.h"

/* ------------------------------------------------------------------ */
/* Forward declarations of the (stub, for now) ops.                    */
/* ------------------------------------------------------------------ */

static void pgraph_metal_early_context_init(void);
static void pgraph_metal_init(NV2AState *d, Error **errp);
static void pgraph_metal_finalize(NV2AState *d);
static void pgraph_metal_clear_report_value(NV2AState *d);
static void pgraph_metal_flip_stall(NV2AState *d);
static void pgraph_metal_get_report(NV2AState *d, uint32_t parameter);
static void pgraph_metal_image_blit(NV2AState *d);
static void pgraph_metal_pre_savevm_trigger(NV2AState *d);
static void pgraph_metal_pre_savevm_wait(NV2AState *d);
static void pgraph_metal_pre_shutdown_trigger(NV2AState *d);
static void pgraph_metal_pre_shutdown_wait(NV2AState *d);
static void pgraph_metal_process_pending(NV2AState *d);
static void pgraph_metal_process_pending_reports(NV2AState *d);
static void pgraph_metal_set_surface_scale_factor(NV2AState *d,
                                                  unsigned int scale);
static unsigned int pgraph_metal_get_surface_scale_factor(NV2AState *d);
static int pgraph_metal_get_framebuffer_surface(NV2AState *d);
static GPUProperties *pgraph_metal_get_gpu_properties(void);

/* ------------------------------------------------------------------ */
/* GPU properties.                                                     */
/* ------------------------------------------------------------------ */

/*
 * Unlike the GL probe, Metal does not expose GPU-vendor geometry-shader
 * quirks, so we return all-zero windings. The Metal shader generator will
 * have to assume a canonical winding and emit any corrective rotation in
 * shader source itself, which will be addressed in a later stage.
 */
static GPUProperties pgraph_metal_gpu_properties;

static GPUProperties *pgraph_metal_get_gpu_properties(void)
{
    return &pgraph_metal_gpu_properties;
}

/* ------------------------------------------------------------------ */
/* Device / queue initialization.                                       */
/* ------------------------------------------------------------------ */

static void pgraph_metal_early_context_init(void)
{
    /*
     * The GL backend uses this hook to create its offscreen GL contexts on
     * the main thread before the worker thread spins up. Metal does not
     * require an explicit per-thread context object: a single MTLDevice can
     * be shared across threads, and command encoders are created on demand
     * from command buffers off the shared queue. So nothing to do here for
     * now. We may create the MTLDevice eagerly in the future to validate
     * Metal availability at startup.
     */
}

static void pgraph_metal_init(NV2AState *d, Error **errp)
{
    PGRAPHState *pg = &d->pgraph;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        error_setg(errp, "Metal: MTLCreateSystemDefaultDevice returned nil");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        error_setg(errp, "Metal: newCommandQueue returned nil");
        return;
    }

    PGRAPHMetalState *r = g_new0(PGRAPHMetalState, 1);
    r->device = device;
    r->queue = queue;
    r->display_texture = nil;
    r->initialized = true;

    pg->metal_renderer_state = r;

    pgraph_metal_init_surfaces(pg);
    pgraph_metal_init_shader_cache(pg);

    fprintf(stderr, "Metal renderer initialized: %s\n",
            [[device name] UTF8String]);

    /* Stage B: the draw path does not exist yet, so nothing would otherwise
     * exercise the MSL generator. Compile a spread of representative shader
     * states now so generator regressions surface here rather than as a
     * blank screen once Stage C lands. */
    pgraph_metal_shader_selftest(device, false);
}

static void pgraph_metal_finalize(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;
    if (!r) {
        return;
    }

    pgraph_metal_report_shader_stats(pg);
    pgraph_metal_finalize_shader_cache(pg);
    pgraph_metal_finalize_surfaces(pg);

    /* Release Objective-C retained objects. */
    r->display_texture = nil;
    r->queue = nil;
    r->device = nil;

    g_free(r);
    pg->metal_renderer_state = NULL;
}

/* ------------------------------------------------------------------ */
/* Ops-table stubs.                                                    */
/* ------------------------------------------------------------------ */

static void pgraph_metal_clear_report_value(NV2AState *d)
{
    /* TODO Stage B: end any in-flight occlusion query. */
}


static void pgraph_metal_flip_stall(NV2AState *d)
{
    /*
     * The GL backend implements this as glFinish() to ensure the previous
     * output buffer is no longer in flight before the VSYNC flip completes.
     * The Metal equivalent is to wait on the most recent command buffer's
     * completion handler; we have nothing to wait on in the stub, but we
     * still drive the sync logic so the surface flip state machine in
     * pgraph.c sees the expected "complete" state.
     */
    qatomic_set(&d->pgraph.sync_pending, false);
    qemu_event_set(&d->pgraph.sync_complete);
}

static void pgraph_metal_get_report(NV2AState *d, uint32_t parameter)
{
    /* Z-pass pixel count report. No queries in flight yet -> 0. */
    pgraph_write_zpass_pixel_cnt_report(d, parameter, 0);
}

static void pgraph_metal_image_blit(NV2AState *d)
{
    /* TODO Stage B: use MTLBlitCommandEncoder. */
}

static void pgraph_metal_pre_savevm_trigger(NV2AState *d)
{
    /* TODO Stage B: drain GPU work + download dirty surfaces to VRAM. */
}

static void pgraph_metal_pre_savevm_wait(NV2AState *d)
{
    /* No work to wait on in the stub. */
}

static void pgraph_metal_pre_shutdown_trigger(NV2AState *d)
{
    /* TODO Stage B: write shader cache to disk. */
}

static void pgraph_metal_pre_shutdown_wait(NV2AState *d)
{
    /* No work to wait on in the stub. */
}

static void pgraph_metal_process_pending(NV2AState *d)
{
    /*
     * Mirror the null renderer's implementation: satisfy any pending sync
     * or flush requests so the rest of the pgraph state machine isn't
     * blocked.
     */
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (qatomic_read(&r->downloads_pending) ||
        qatomic_read(&r->download_dirty_surfaces_pending) ||
        qatomic_read(&d->pgraph.sync_pending) ||
        qatomic_read(&d->pgraph.flush_pending)) {
        qemu_mutex_unlock(&d->pfifo.lock);
        qemu_mutex_lock(&d->pgraph.lock);
        if (qatomic_read(&r->downloads_pending)) {
            pgraph_metal_process_pending_downloads(d);
        }
        if (qatomic_read(&r->download_dirty_surfaces_pending)) {
            pgraph_metal_download_dirty_surfaces(d);
        }
        if (qatomic_read(&d->pgraph.sync_pending)) {
            qatomic_set(&d->pgraph.sync_pending, false);
            qemu_event_set(&d->pgraph.sync_complete);
        }
        if (qatomic_read(&d->pgraph.flush_pending)) {
            qatomic_set(&d->pgraph.flush_pending, false);
            qemu_event_set(&d->pgraph.flush_complete);
        }
        qemu_mutex_unlock(&d->pgraph.lock);
        qemu_mutex_lock(&d->pfifo.lock);
    }
}

static void pgraph_metal_process_pending_reports(NV2AState *d)
{
    /* No zpass queries in flight in the stub. */
}


static void pgraph_metal_set_surface_scale_factor(NV2AState *d,
                                                  unsigned int scale)
{
    if (scale < 1) {
        scale = 1;
    }
    d->pgraph.surface_scale_factor = scale;
}

static unsigned int pgraph_metal_get_surface_scale_factor(NV2AState *d)
{
    return d->pgraph.surface_scale_factor;
}

static int pgraph_metal_get_framebuffer_surface(NV2AState *d)
{
    /*
     * Stage A stub: returns 0 to indicate "no accelerated framebuffer
     * available". This forces the UI layer to fall back to its existing
     * VGA-path that uploads from the d->vga buffer, so we still get a
     * visible (slow) display while verifying the rest of the wiring.
     *
     * In Stage C this will be replaced with a CAMetalLayer-backed display
     * path that presents an actual MTLTexture.
     */
    return 0;
}

/* ------------------------------------------------------------------ */
/* Renderer registration.                                              */
/* ------------------------------------------------------------------ */

static PGRAPHRenderer pgraph_metal_renderer = {
    .type = CONFIG_DISPLAY_RENDERER_METAL,
    .name = "Metal",
    .ops = {
        .early_context_init        = pgraph_metal_early_context_init,
        .init                      = pgraph_metal_init,
        .finalize                  = pgraph_metal_finalize,
        .clear_report_value        = pgraph_metal_clear_report_value,
        .clear_surface             = pgraph_metal_clear_surface,
        .draw_begin                = pgraph_metal_draw_begin,
        .draw_end                  = pgraph_metal_draw_end,
        .flip_stall                = pgraph_metal_flip_stall,
        .flush_draw                = pgraph_metal_flush_draw,
        .get_report                = pgraph_metal_get_report,
        .image_blit                = pgraph_metal_image_blit,
        .pre_savevm_trigger        = pgraph_metal_pre_savevm_trigger,
        .pre_savevm_wait            = pgraph_metal_pre_savevm_wait,
        .pre_shutdown_trigger       = pgraph_metal_pre_shutdown_trigger,
        .pre_shutdown_wait          = pgraph_metal_pre_shutdown_wait,
        .process_pending            = pgraph_metal_process_pending,
        .process_pending_reports    = pgraph_metal_process_pending_reports,
        .surface_update             = pgraph_metal_surface_update,
        .surface_flush              = pgraph_metal_surface_flush,
        .set_surface_scale_factor   = pgraph_metal_set_surface_scale_factor,
        .get_surface_scale_factor   = pgraph_metal_get_surface_scale_factor,
        .get_framebuffer_surface    = pgraph_metal_get_framebuffer_surface,
        .get_gpu_properties         = pgraph_metal_get_gpu_properties,
    },
};

static void __attribute__((constructor)) register_renderer(void)
{
    pgraph_renderer_register(&pgraph_metal_renderer);
}
