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

/*
 * Texture upload and binding.
 *
 * Most of the format grind is already shared: pgraph_convert_texture_data()
 * handles palettes and the awkward channel orders, and unswizzle_rect()
 * undoes the NV2A's swizzled layout. What is left here is choosing an
 * MTLPixelFormat, staging level 0, and building the sampler.
 *
 * Scope: 2D textures, level 0 only. Mipmaps, cubemaps, 3D and the
 * compressed formats fall back to a white 1x1 texture so the geometry using
 * them still shows (white is the multiplicative identity through the
 * combiners) rather than turning black and hiding the fact that anything
 * rendered at all. Unmapped formats are counted and reported once each.
 */

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/nv2a/pgraph/pgraph.h"
#include "hw/xbox/nv2a/pgraph/texture.h"
#include "hw/xbox/nv2a/pgraph/swizzle.h"
#include "qemu/fast-hash.h"
#include "renderer.h"
#include "texture.h"
#include "surface.h"

typedef struct MetalTextureFormat {
    MTLPixelFormat pixel_format;
    bool           supported;
    bool           bgra_swizzle; /* needs component reorder via swizzle */
} MetalTextureFormat;

/*
 * Xbox texture format -> Metal. Only the formats a boot actually exercises
 * are mapped so far; anything else falls back and is reported.
 *
 * The NV2A's *R8G8B8A8-style names describe memory order, so most of the
 * 32-bit formats are Metal's BGRA8. pgraph_convert_texture_data() normalizes
 * several of the odd ones before they reach here.
 */
static MetalTextureFormat metal_texture_format(unsigned int fmt)
{
#define MAP(nv, mtl) case nv: return (MetalTextureFormat){ mtl, true, false }
    switch (fmt) {
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A8R8G8B8,      MTLPixelFormatBGRA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_X8R8G8B8,      MTLPixelFormatBGRA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A8R8G8B8, MTLPixelFormatBGRA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_X8R8G8B8, MTLPixelFormatBGRA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_B8G8R8A8,      MTLPixelFormatRGBA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_B8G8R8A8, MTLPixelFormatRGBA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_R8G8B8A8,      MTLPixelFormatRGBA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_R8G8B8A8, MTLPixelFormatRGBA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A8B8G8R8,      MTLPixelFormatRGBA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A8B8G8R8, MTLPixelFormatRGBA8Unorm);

    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_R5G6B5,        MTLPixelFormatB5G6R5Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_R5G6B5,  MTLPixelFormatB5G6R5Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A1R5G5B5,      MTLPixelFormatBGR5A1Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A1R5G5B5, MTLPixelFormatBGR5A1Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_X1R5G5B5,      MTLPixelFormatBGR5A1Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_X1R5G5B5, MTLPixelFormatBGR5A1Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A4R4G4B4,      MTLPixelFormatABGR4Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A4R4G4B4, MTLPixelFormatABGR4Unorm);

    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A8,            MTLPixelFormatA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A8,      MTLPixelFormatA8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_Y8,            MTLPixelFormatR8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_Y8,      MTLPixelFormatR8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A8Y8,          MTLPixelFormatRG8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A8Y8,    MTLPixelFormatRG8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_G8B8,          MTLPixelFormatRG8Unorm);
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_G8B8,    MTLPixelFormatRG8Unorm);

    /* Palettized: pgraph_convert_texture_data() expands this to 32-bit. */
    MAP(NV097_SET_TEXTURE_FORMAT_COLOR_SZ_I8_A8R8G8B8,   MTLPixelFormatBGRA8Unorm);

    default:
        return (MetalTextureFormat){ MTLPixelFormatInvalid, false, false };
    }
#undef MAP
}

static MTLSamplerAddressMode address_mode(unsigned int nv_mode)
{
    switch (nv_mode) {
    case NV_PGRAPH_TEXADDRESS0_ADDRU_WRAP:
        return MTLSamplerAddressModeRepeat;
    case NV_PGRAPH_TEXADDRESS0_ADDRU_MIRROR:
        return MTLSamplerAddressModeMirrorRepeat;
    case NV_PGRAPH_TEXADDRESS0_ADDRU_CLAMP_TO_EDGE:
    case NV_PGRAPH_TEXADDRESS0_ADDRU_CLAMP_OGL:
        return MTLSamplerAddressModeClampToEdge;
    case NV_PGRAPH_TEXADDRESS0_ADDRU_BORDER:
        return MTLSamplerAddressModeClampToBorderColor;
    default:
        return MTLSamplerAddressModeRepeat;
    }
}

void pgraph_metal_init_textures(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    r->texture_cache = g_hash_table_new_full(
        g_bytes_hash, g_bytes_equal, (GDestroyNotify)g_bytes_unref,
        (GDestroyNotify)CFRelease);
    r->sampler_cache = g_hash_table_new_full(
        g_bytes_hash, g_bytes_equal, (GDestroyNotify)g_bytes_unref,
        (GDestroyNotify)CFRelease);

    /*
     * White fallbacks, one per texture type.
     *
     * The type matters as much as the contents: Metal rejects an entire draw
     * if a bound texture's type does not match what the shader declared, and
     * does so silently unless API validation is enabled. Binding a 2D
     * fallback where the generated shader declared a cubemap therefore
     * discards the draw rather than producing a wrongly-shaded one, which is
     * indistinguishable from rendering nothing.
     *
     * White rather than black so a stage whose format is unimplemented still
     * lets geometry through the combiners.
     */
    uint32_t white = 0xFFFFFFFF;

    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:1
                                    height:1
                                 mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    r->white_texture = [r->device newTextureWithDescriptor:td];
    [r->white_texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                        mipmapLevel:0
                          withBytes:&white
                        bytesPerRow:4];

    MTLTextureDescriptor *cd = [MTLTextureDescriptor
        textureCubeDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                        size:1
                                   mipmapped:NO];
    cd.storageMode = MTLStorageModeShared;
    r->white_texture_cube = [r->device newTextureWithDescriptor:cd];
    for (int face = 0; face < 6; face++) {
        [r->white_texture_cube replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                                 mipmapLevel:0
                                       slice:face
                                   withBytes:&white
                                 bytesPerRow:4
                               bytesPerImage:4];
    }

    MTLTextureDescriptor *dd = [[MTLTextureDescriptor alloc] init];
    dd.textureType = MTLTextureType3D;
    dd.pixelFormat = MTLPixelFormatBGRA8Unorm;
    dd.width = 1; dd.height = 1; dd.depth = 1;
    dd.storageMode = MTLStorageModeShared;
    r->white_texture_3d = [r->device newTextureWithDescriptor:dd];
    [r->white_texture_3d replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1)
                           mipmapLevel:0
                                 slice:0
                             withBytes:&white
                           bytesPerRow:4
                         bytesPerImage:4];
}

void pgraph_metal_finalize_textures(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    if (r->texture_cache) {
        g_hash_table_destroy(r->texture_cache);
        r->texture_cache = NULL;
    }
    if (r->sampler_cache) {
        g_hash_table_destroy(r->sampler_cache);
        r->sampler_cache = NULL;
    }
    r->white_texture = nil;
    r->white_texture_cube = nil;
    r->white_texture_3d = nil;
}

typedef struct MetalTextureKey {
    TextureShape shape;
    hwaddr       vram_addr;
    hwaddr       palette_addr;
    uint64_t     data_hash;
} MetalTextureKey;

static id<MTLTexture> upload_texture(NV2AState *d, int i,
                                     const TextureShape *s)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    /*
     * Surface-as-texture.
     *
     * A render target's pixels live only in its MTLTexture; its guest memory
     * is never written unless something explicitly downloads it. So when a
     * texture stage points at the memory of a live render surface, reading
     * that memory yields zeros and the sample comes back black.
     *
     * This is exactly how the dashboard composites: it renders the scene into
     * offscreen surfaces, then issues a small number of draws into the
     * scanout surface that sample those surfaces as textures. Without this
     * the final composite samples nothing and the screen stays black, even
     * though every earlier stage rendered correctly.
     *
     * The GL backend solves this with pgraph_gl_render_surface_to_texture.
     * Binding the surface's texture directly is the Metal equivalent and is
     * cheaper, at the cost of not handling format or scale mismatches yet.
     */
    hwaddr tex_addr = pgraph_get_texture_phys_addr(pg, i);
    MetalSurfaceBinding *surf = pgraph_metal_surface_get(d, tex_addr);
    if (surf && surf->color && surf->texture != nil) {
        r->surface_as_texture++;
        return surf->texture;
    }

    MetalTextureFormat mf = metal_texture_format(s->color_format);
    if (!mf.supported || s->dimensionality != 2 || s->cubemap) {
        r->texture_unsupported++;
        if (!r->reported_format[s->color_format & 0x3F]) {
            r->reported_format[s->color_format & 0x3F] = true;
            fprintf(stderr,
                    "nv2a: metal: unsupported texture format 0x%x "
                    "(dim %u cubemap %d) -> white\n",
                    s->color_format, s->dimensionality, s->cubemap);
        }
        return r->white_texture;
    }

    BasicColorFormatInfo f =
        kelvin_color_format_info_map[s->color_format];
    if (f.bytes_per_pixel == 0) {
        return r->white_texture;
    }

    hwaddr texture_addr = pgraph_get_texture_phys_addr(pg, i);
    size_t palette_len = 0;
    hwaddr palette_addr =
        pgraph_get_texture_palette_phys_addr_length(pg, i, &palette_len);

    const uint8_t *texture_data = d->vram_ptr + texture_addr;
    const uint8_t *palette_data =
        palette_len ? d->vram_ptr + palette_addr : NULL;

    unsigned int width = s->width, height = s->height;
    if (!width || !height) {
        return r->white_texture;
    }

    /* Cache on address + shape; contents are re-checked by the caller via
     * pg->texture_dirty. */
    MetalTextureKey key;
    memset(&key, 0, sizeof(key));
    key.shape = *s;
    key.vram_addr = texture_addr;
    key.palette_addr = palette_addr;
    key.data_hash = fast_hash(texture_data,
                              MIN(pgraph_get_texture_length(pg, (TextureShape *)s),
                                  (size_t)height * width * f.bytes_per_pixel));

    GBytes *k = g_bytes_new(&key, sizeof(key));
    id<MTLTexture> cached =
        (__bridge id<MTLTexture>)g_hash_table_lookup(r->texture_cache, k);
    if (cached) {
        g_bytes_unref(k);
        return cached;
    }

    unsigned int pitch = f.linear ? s->pitch : width * f.bytes_per_pixel;

    g_autofree uint8_t *unswizzled = NULL;
    const uint8_t *src = texture_data;

    if (!f.linear) {
        unswizzled = g_malloc((size_t)height * pitch);
        unswizzle_rect(texture_data, width, height, unswizzled, pitch,
                       f.bytes_per_pixel);
        src = unswizzled;
    }

    g_autofree uint8_t *converted = pgraph_convert_texture_data(
        *s, src, palette_data, width, height, 1, pitch, 0, NULL);
    if (converted) {
        src = converted;
        pitch = width * 4; /* conversion expands to 32-bit */
    }

    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:mf.pixel_format
                                     width:width
                                    height:height
                                 mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;

    id<MTLTexture> tex = [r->device newTextureWithDescriptor:td];
    if (tex == nil) {
        g_bytes_unref(k);
        return r->white_texture;
    }

    [tex replaceRegion:MTLRegionMake2D(0, 0, width, height)
           mipmapLevel:0
             withBytes:src
           bytesPerRow:pitch];

    r->texture_uploads++;
    g_hash_table_insert(r->texture_cache, k, (__bridge_retained void *)tex);
    return tex;
}

static id<MTLSamplerState> get_sampler(PGRAPHMetalState *r, uint32_t filter,
                                       uint32_t address)
{
    uint32_t key_data[2] = { filter, address };
    GBytes *k = g_bytes_new(key_data, sizeof(key_data));

    id<MTLSamplerState> cached =
        (__bridge id<MTLSamplerState>)g_hash_table_lookup(r->sampler_cache, k);
    if (cached) {
        g_bytes_unref(k);
        return cached;
    }

    unsigned int min_f = GET_MASK(filter, NV_PGRAPH_TEXFILTER0_MIN);
    unsigned int mag_f = GET_MASK(filter, NV_PGRAPH_TEXFILTER0_MAG);

    MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
    /* NEAREST variants are the odd values below the LINEAR ones; anything
     * with mipmap filtering still samples level 0 for now. */
    sd.minFilter = (min_f == NV_PGRAPH_TEXFILTER0_MIN_BOX_LOD0)
                       ? MTLSamplerMinMagFilterNearest
                       : MTLSamplerMinMagFilterLinear;
    /* MAG has only box (nearest) and tent (linear); box is the lower value. */
    sd.magFilter = (mag_f == 1) ? MTLSamplerMinMagFilterNearest
                                : MTLSamplerMinMagFilterLinear;
    sd.sAddressMode =
        address_mode(GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRU));
    sd.tAddressMode =
        address_mode(GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRV));
    sd.rAddressMode =
        address_mode(GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRP));

    id<MTLSamplerState> smp = [r->device newSamplerStateWithDescriptor:sd];
    g_hash_table_insert(r->sampler_cache, k, (__bridge_retained void *)smp);
    return smp;
}

void pgraph_metal_bind_textures(NV2AState *d, id<MTLRenderCommandEncoder> enc)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    for (int i = 0; i < NV2A_MAX_TEXTURES; i++) {
        uint32_t ctl_0 = pgraph_reg_r(pg, NV_PGRAPH_TEXCTL0_0 + i * 4);
        bool enabled = pgraph_is_texture_stage_active(pg, i) &&
                       (ctl_0 & NV_PGRAPH_TEXCTL0_0_ENABLE);

        /*
         * The generated shader picks its sampler type from the same state
         * (see get_sampler_type in psh.c), so derive the expected type the
         * same way and make sure whatever is bound matches it. Metal discards
         * an entire draw on a type mismatch, silently unless validation is
         * on, so this has to agree with the shader exactly.
         *
         * Only inspect the shape for an enabled stage: pgraph_get_texture_shape
         * asserts on the unconfigured state a disabled stage carries.
         */
        id<MTLTexture> tex = r->white_texture;

        if (enabled) {
            TextureShape s = pgraph_get_texture_shape(pg, i);

            id<MTLTexture> fallback = r->white_texture;
            MTLTextureType want = MTLTextureType2D;
            if (s.cubemap) {
                fallback = r->white_texture_cube;
                want = MTLTextureTypeCube;
            } else if (s.dimensionality == 3) {
                fallback = r->white_texture_3d;
                want = MTLTextureType3D;
            }

            tex = upload_texture(d, i, &s);
            if (tex == nil || tex.textureType != want) {
                if (tex != nil && tex != r->white_texture) {
                    r->texture_type_mismatch++;
                }
                tex = fallback;
            }
        }

        uint32_t filter = pgraph_reg_r(pg, NV_PGRAPH_TEXFILTER0 + i * 4);
        uint32_t address = pgraph_reg_r(pg, NV_PGRAPH_TEXADDRESS0 + i * 4);

        [enc setFragmentTexture:tex atIndex:i];
        [enc setFragmentSamplerState:get_sampler(r, filter, address)
                             atIndex:i];
    }
}
