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

#include "qemu/osdep.h"
#include "msl.h"

/*
 * MSL spellings for the uniform element types. Two tables are needed:
 *
 *  - `struct` types use packed vectors, giving the same tight layout as the
 *    C `vecN` typedefs in common.h (float[N]). That lets the renderer upload
 *    a *UniformValues struct without a separate std140-style repack.
 *  - `value` types are the natural aligned vectors, used when copying a
 *    scalar member into a local.
 *
 * Both are indexed by enum UniformElementType, so they must stay in the same
 * order as UNIFORM_ELEMENT_TYPE_X in common.h.
 */
static const char *msl_uniform_struct_type[] = {
    "float",   /* float */
    "int",     /* int   */
    "int2",    /* ivec2 */
    "int4",    /* ivec4 */
    "float2x2", /* mat2 */
    "uint",    /* uint  */
    "packed_float2", /* vec2 */
    "packed_float3", /* vec3 */
    "float4",  /* vec4  */
};

static const char *msl_uniform_value_type[] = {
    "float", "int", "int2", "int4", "float2x2", "uint",
    "float2", "float3", "float4",
};

/*
 * Interpolants passed from the vertex stage to the fragment stage. This must
 * mirror the attribute table in pgraph_glsl_get_vtx_header() (common.c): the
 * generated statement code refers to these by bare name in both stages.
 */
static const struct {
    const char *type;
    const char *name;
    bool flat;        /* always flat, regardless of shade mode */
    bool shade_mode;  /* flat unless smooth shading is enabled */
} msl_vtx_attrs[] = {
    { "float4", "vtxD0",   false, true  },
    { "float4", "vtxD1",   false, true  },
    { "float4", "vtxB0",   false, true  },
    { "float4", "vtxB1",   false, true  },
    { "float",  "vtxFog",  false, false },
    { "float4", "vtxT0",   false, false },
    { "float4", "vtxT1",   false, false },
    { "float4", "vtxT2",   false, false },
    { "float4", "vtxT3",   false, false },
    { "float4", "vtxPos0", true,  false },
    { "float4", "vtxPos1", true,  false },
    { "float4", "vtxPos2", true,  false },
    { "float",  "triMZ",   true,  false },
};

const char *pgraph_msl_prologue(void)
{
    // clang-format off
    return
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "typedef float2 vec2;\n"
        "typedef float3 vec3;\n"
        "typedef float4 vec4;\n"
        "typedef int2 ivec2;\n"
        "typedef int3 ivec3;\n"
        "typedef int4 ivec4;\n"
        "typedef uint2 uvec2;\n"
        "typedef uint3 uvec3;\n"
        "typedef uint4 uvec4;\n"
        "typedef bool2 bvec2;\n"
        "typedef bool3 bvec3;\n"
        "typedef bool4 bvec4;\n"
        "typedef float2x2 mat2;\n"
        "typedef float3x3 mat3;\n"
        "typedef float4x4 mat4;\n"
        "\n"
        /* GLSL statement spellings with no MSL equivalent. `precise` is a
         * no-op: Metal does not reassociate the fma() sequences that use it. */
        "#define discard discard_fragment()\n"
        "#define precise\n"
        "#define NV2A_INFINITY as_type<float>(0x7F800000u)\n"
        "#define FLOAT_MAX as_type<float>(0x7F7FFFFFu)\n"
        "\n"
        "inline uint floatBitsToUint(float f) { return as_type<uint>(f); }\n"
        "inline float uintBitsToFloat(uint u) { return as_type<float>(u); }\n"
        "inline int bitfieldExtract(int v, int off, int bits) {\n"
        "  return extract_bits(v, off, bits);\n"
        "}\n"
        "inline float inversesqrt(float x) { return rsqrt(x); }\n"
        "\n"
        /* GLSL relational builtins return boolean vectors. */
        "inline bool2 lessThan(float2 a, float2 b) { return a < b; }\n"
        "inline bool3 lessThan(float3 a, float3 b) { return a < b; }\n"
        "inline bool4 lessThan(float4 a, float4 b) { return a < b; }\n"
        "inline bool2 greaterThanEqual(float2 a, float2 b) { return a >= b; }\n"
        "inline bool3 greaterThanEqual(float3 a, float3 b) { return a >= b; }\n"
        "inline bool4 greaterThanEqual(float4 a, float4 b) { return a >= b; }\n"
        "\n"
        /* mix() selecting on a boolean vector is GLSL's select(). */
        "inline float2 mix(float2 a, float2 b, bool2 c) { return select(a, b, c); }\n"
        "inline float3 mix(float3 a, float3 b, bool3 c) { return select(a, b, c); }\n"
        "inline float4 mix(float4 a, float4 b, bool4 c) { return select(a, b, c); }\n"
        "\n"
        /* Texture access. The generators emit `texture(texSampN, coord)`; the
         * entry point defines `texSampN` as the comma pair `texN, smpN`, so
         * these three-argument overloads are what the call resolves to. */
        "inline float4 texture(texture2d<float> t, sampler s, float2 c) {\n"
        "  return t.sample(s, c);\n"
        "}\n"
        "inline float4 texture(texture3d<float> t, sampler s, float3 c) {\n"
        "  return t.sample(s, c);\n"
        "}\n"
        "inline float4 texture(texturecube<float> t, sampler s, float3 c) {\n"
        "  return t.sample(s, c);\n"
        "}\n"
        "inline float4 textureProj(texture2d<float> t, sampler s, float3 c) {\n"
        "  return t.sample(s, c.xy / c.z);\n"
        "}\n"
        "inline float4 textureProj(texture2d<float> t, sampler s, float4 c) {\n"
        "  return t.sample(s, c.xy / c.w);\n"
        "}\n"
        "inline float4 textureProj(texture3d<float> t, sampler s, float4 c) {\n"
        "  return t.sample(s, c.xyz / c.w);\n"
        "}\n"
        "inline float4 textureProj(texturecube<float> t, sampler s, float4 c) {\n"
        "  return t.sample(s, c.xyz / c.w);\n"
        "}\n"
        "inline float2 textureSize(texture2d<float> t, sampler s, int lod) {\n"
        "  return float2(t.get_width(lod), t.get_height(lod));\n"
        "}\n"
        "inline float3 textureSize(texture3d<float> t, sampler s, int lod) {\n"
        "  return float3(t.get_width(lod), t.get_height(lod), t.get_depth(lod));\n"
        "}\n"
        "\n"
        /* Unnormalized (rect) texture coordinate remap. The generators emit
         * `normN(coord)`; the entry point defines that as a call to these with
         * the stage's texture and scale bound in. */
        "inline float2 msl_norm(float2 c, texture2d<float> t, float scale) {\n"
        "  return c / (float2(t.get_width(0), t.get_height(0)) / scale);\n"
        "}\n"
        "inline float3 msl_norm(float3 c, texture2d<float> t, float scale) {\n"
        "  return float3(msl_norm(c.xy, t, scale), c.z);\n"
        "}\n"
        "inline float4 msl_norm(float4 c, texture2d<float> t, float scale) {\n"
        "  return float4(msl_norm(c.xy, t, scale), 0.0, c.w);\n"
        "}\n"
        "\n"
        /* Shared numeric helpers (counterpart of the GLSL header in vsh.c). */
        "inline float4 NaNToOne(float4 src) {\n"
        "  return mix(src, float4(1.0), isnan(src));\n"
        "}\n"
        "inline float4 NaNToValue(float4 src, float replacement) {\n"
        "  return mix(src, float4(replacement), isnan(src));\n"
        "}\n"
        /* Clamp to range [2^(-64), 2^64] or [-2^64, -2^(-64)]. */
        "inline float clampAwayZeroInf(float t) {\n"
        "  if (t > 0.0 || floatBitsToUint(t) == 0) {\n"
        "    t = clamp(t, uintBitsToFloat(0x1F800000), uintBitsToFloat(0x5F800000));\n"
        "  } else {\n"
        "    t = clamp(t, uintBitsToFloat(0xDF800000), uintBitsToFloat(0x9F800000));\n"
        "  }\n"
        "  return t;\n"
        "}\n"
        /* The NV2A rasterizer has a 4 bit fixed-point fractional part and
         * converts by truncating rather than flooring. */
        "inline float2 roundScreenCoords(float2 pos) {\n"
        "  return trunc(pos * 16.0f) / 16.0f;\n"
        "}\n"
        "inline float4 decompress_11_11_10(int cmp) {\n"
        "  float x = float(bitfieldExtract(cmp, 0,  11)) / 1023.0;\n"
        "  float y = float(bitfieldExtract(cmp, 11, 11)) / 1023.0;\n"
        "  float z = float(bitfieldExtract(cmp, 22, 10)) / 511.0;\n"
        "  return float4(x, y, z, 1);\n"
        "}\n"
        "\n";
    // clang-format on
}

const char *pgraph_msl_vsh_prog_prologue(void)
{
    /*
     * MSL port of the `vsh_header` string in vsh-prog.c. The macro spellings
     * are identical, so the instruction stream decoded by decode_token() is
     * emitted unchanged for both dialects; only these definitions differ.
     */
    // clang-format off
    return
        "/* Converts the input to vec4, pads with last component */\n"
        "inline vec4 _in(float v) { return vec4(v); }\n"
        "inline vec4 _in(vec2 v) { return v.xyyy; }\n"
        "inline vec4 _in(vec3 v) { return v.xyzz; }\n"
        "inline vec4 _in(vec4 v) { return v.xyzw; }\n"
        "\n"
        "#define MOV(dest, mask, src) dest.mask = _MOV(_in(src)).mask\n"
        "inline vec4 _MOV(vec4 src)\n"
        "{\n"
        "  return src;\n"
        "}\n"
        "\n"
        "#define MUL(dest, mask, src0, src1) dest.mask = _MUL(_in(src0), _in(src1)).mask\n"
        "inline vec4 _MUL(vec4 src0, vec4 src1)\n"
        "{\n"
        /* Per-component comparisons guarantee the HW behavior that anything
         * multiplied by zero is zero, which a plain multiply (or mix()) does
         * not give for inf/NaN operands. */
        "  vec4 zero_components = sign(NaNToOne(src0)) * sign(NaNToOne(src1));\n"
        "  vec4 ret = src0 * src1;\n"
        "  if (zero_components.x == 0.0) { ret.x = 0.0; }\n"
        "  if (zero_components.y == 0.0) { ret.y = 0.0; }\n"
        "  if (zero_components.z == 0.0) { ret.z = 0.0; }\n"
        "  if (zero_components.w == 0.0) { ret.w = 0.0; }\n"
        "  return ret;\n"
        "}\n"
        "\n"
        "#define ADD(dest, mask, src0, src1) dest.mask = _ADD(_in(src0), _in(src1)).mask\n"
        "inline vec4 _ADD(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return src0 + src1;\n"
        "}\n"
        "\n"
        "#define MAD(dest, mask, src0, src1, src2) dest.mask = _MAD(_in(src0), _in(src1), _in(src2)).mask\n"
        "inline vec4 _MAD(vec4 src0, vec4 src1, vec4 src2)\n"
        "{\n"
        "  return _MUL(src0, src1) + src2;\n"
        "}\n"
        "\n"
        "#define DP3(dest, mask, src0, src1) dest.mask = _DP3(_in(src0), _in(src1)).mask\n"
        "inline vec4 _DP3(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return vec4(dot(src0.xyz, src1.xyz));\n"
        "}\n"
        "\n"
        "#define DPH(dest, mask, src0, src1) dest.mask = _DPH(_in(src0), _in(src1)).mask\n"
        "inline vec4 _DPH(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return vec4(dot(vec4(src0.xyz, 1.0), src1));\n"
        "}\n"
        "\n"
        "#define DP4(dest, mask, src0, src1) dest.mask = _DP4(_in(src0), _in(src1)).mask\n"
        "inline vec4 _DP4(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return vec4(dot(src0, src1));\n"
        "}\n"
        "\n"
        "#define DST(dest, mask, src0, src1) dest.mask = _DST(_in(src0), _in(src1)).mask\n"
        "inline vec4 _DST(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return vec4(1.0,\n"
        "              src0.y * src1.y,\n"
        "              src0.z,\n"
        "              src1.w);\n"
        "}\n"
        "\n"
        "#define MIN(dest, mask, src0, src1) dest.mask = _MIN(_in(src0), _in(src1)).mask\n"
        "inline vec4 _MIN(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return min(src0, src1);\n"
        "}\n"
        "\n"
        "#define MAX(dest, mask, src0, src1) dest.mask = _MAX(_in(src0), _in(src1)).mask\n"
        "inline vec4 _MAX(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return max(src0, src1);\n"
        "}\n"
        "\n"
        "#define SLT(dest, mask, src0, src1) dest.mask = _SLT(_in(src0), _in(src1)).mask\n"
        "inline vec4 _SLT(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return vec4(lessThan(src0, src1));\n"
        "}\n"
        "\n"
        "#define ARL(dest, src) dest = _ARL(_in(src).x)\n"
        "inline int _ARL(float src)\n"
        "{\n"
        /* The Xbox GPU specifies rounding where Metal does not, so a bias is
         * needed: we want to floor 16.99.. to 17, not 16. The error comes from
         * vertex attributes normalized from a byte, where 17/255 may land
         * either side depending on the host GPU. */
        "  return int(floor(src + 0.001));\n"
        "}\n"
        "\n"
        "#define SGE(dest, mask, src0, src1) dest.mask = _SGE(_in(src0), _in(src1)).mask\n"
        "inline vec4 _SGE(vec4 src0, vec4 src1)\n"
        "{\n"
        "  return vec4(greaterThanEqual(src0, src1));\n"
        "}\n"
        "\n"
        "#define RCP(dest, mask, src) dest.mask = _RCP(_in(src).x).mask\n"
        "inline vec4 _RCP(float src)\n"
        "{\n"
        "  return vec4(1.0 / src);\n"
        "}\n"
        "\n"
        "#define RCC(dest, mask, src) dest.mask = _RCC(_in(src).x).mask\n"
        "inline vec4 _RCC(float src)\n"
        "{\n"
        "  float t = clampAwayZeroInf(1.0 / src);\n"
        "  return vec4(t);\n"
        "}\n"
        "\n"
        "#define RSQ(dest, mask, src) dest.mask = _RSQ(_in(src).x).mask\n"
        "inline vec4 _RSQ(float src)\n"
        "{\n"
        "  if (src == 0.0) { return vec4(NV2A_INFINITY); }\n"
        "  if (isinf(src)) { return vec4(0.0); }\n"
        "  return vec4(inversesqrt(abs(src)));\n"
        "}\n"
        "\n"
        "#define EXP(dest, mask, src) dest.mask = _EXP(_in(src).x).mask\n"
        "inline vec4 _EXP(float src)\n"
        "{\n"
        "  vec4 result;\n"
        "  result.x = exp2(floor(src));\n"
        "  result.y = src - floor(src);\n"
        "  result.z = exp2(src);\n"
        "  result.w = 1.0;\n"
        "  return result;\n"
        "}\n"
        "\n"
        "#define LOG(dest, mask, src) dest.mask = _LOG(_in(src).x).mask\n"
        "inline vec4 _LOG(float src)\n"
        "{\n"
        "  float tmp = abs(src);\n"
        "  if (tmp == 0.0) { return vec4(-NV2A_INFINITY, 1.0f, -NV2A_INFINITY, 1.0f); }\n"
        "  vec4 result;\n"
        "  result.x = floor(log2(tmp));\n"
        "  result.y = tmp / exp2(floor(log2(tmp)));\n"
        "  result.z = log2(tmp);\n"
        "  result.w = 1.0;\n"
        "  return result;\n"
        "}\n"
        "\n"
        "#define LIT(dest, mask, src) dest.mask = _LIT(_in(src)).mask\n"
        "inline vec4 _LIT(vec4 src)\n"
        "{\n"
        "  vec4 s = src;\n"
        "  float epsilon = 1.0 / 256.0;\n"
        "  s.w = clamp(s.w, -(128.0 - epsilon), 128.0 - epsilon);\n"
        "  s.x = max(s.x, 0.0);\n"
        "  s.y = max(s.y, 0.0);\n"
        "  vec4 t = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "  t.y = s.x;\n"
        "  t.z = (s.x > 0.0) ? exp2(s.w * log2(s.y)) : 0.0;\n"
        "  return t;\n"
        "}\n"
        "\n";
    // clang-format on
}

const char *pgraph_msl_vsh_prog_locals(void)
{
    // clang-format off
    return
        "  int A0 = 0;\n"
        "  vec4 R0 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R1 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R2 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R3 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R4 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R5 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R6 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R7 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R8 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R9 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R10 = vec4(0.0,0.0,0.0,0.0);\n"
        "  vec4 R11 = vec4(0.0,0.0,0.0,0.0);\n"
        /* Used to emulate concurrency of paired MAC+ILU instructions */
        "  vec4 _temp_vec = vec4(0.0);\n"
        "  int _temp_addr = 0;\n"
        "\n";
    // clang-format on
}

const char *pgraph_msl_vsh_output_regs(void)
{
    // clang-format off
    return
        "  vec4 oPos = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oD0 = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oD1 = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oB0 = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oB1 = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oPts = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oFog = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oT0 = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oT1 = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oT2 = vec4(0.0,0.0,0.0,1.0);\n"
        "  vec4 oT3 = vec4(0.0,0.0,0.0,1.0);\n"
        "\n";
    // clang-format on
}

void pgraph_msl_gen_uniform_struct(MString *out, const char *name,
                                   const UniformInfo *info, size_t num_info,
                                   int skip_index)
{
    mstring_append_fmt(out, "struct %s {\n", name);
    for (size_t i = 0; i < num_info; i++) {
        if ((int)i == skip_index) {
            continue;
        }
        const char *type_str = msl_uniform_struct_type[info[i].type];
        if (info[i].count == 1) {
            mstring_append_fmt(out, "  %s %s;\n", type_str, info[i].name);
        } else {
            mstring_append_fmt(out, "  %s %s[%zd];\n", type_str, info[i].name,
                               info[i].count);
        }
    }
    mstring_append(out, "};\n\n");
}

void pgraph_msl_gen_uniform_locals(MString *out, const UniformInfo *info,
                                   size_t num_info, int skip_index)
{
    for (size_t i = 0; i < num_info; i++) {
        if ((int)i == skip_index) {
            continue;
        }
        if (info[i].count == 1) {
            /* Scalars and single vectors are copied by value, converting away
             * from the packed storage type. */
            mstring_append_fmt(out, "  %s %s = U.%s;\n",
                               msl_uniform_value_type[info[i].type],
                               info[i].name, info[i].name);
        } else {
            /* Arrays stay in the constant address space and are reached
             * through a pointer, so `name[i]` indexes them as in GLSL. */
            mstring_append_fmt(out, "  constant %s *%s = U.%s;\n",
                               msl_uniform_struct_type[info[i].type],
                               info[i].name, info[i].name);
        }
    }
    mstring_append(out, "\n");
}

void pgraph_msl_gen_vtx_struct(MString *out, const char *name, bool smooth,
                               bool is_vertex)
{
    mstring_append_fmt(out, "struct %s {\n", name);
    mstring_append(out, "  float4 nv2a_position [[position]];\n");
    if (is_vertex) {
        mstring_append(out, "  float nv2a_pointSize [[point_size]];\n");
    } else {
        mstring_append(out, "  float2 nv2a_pointCoord [[point_coord]];\n");
    }

    for (int i = 0; i < ARRAY_SIZE(msl_vtx_attrs); i++) {
        bool flat = msl_vtx_attrs[i].flat ||
                    (msl_vtx_attrs[i].shade_mode && !smooth);
        mstring_append_fmt(out, "  %s %s [[user(%s)]]%s;\n",
                           msl_vtx_attrs[i].type, msl_vtx_attrs[i].name,
                           msl_vtx_attrs[i].name, flat ? " [[flat]]" : "");
    }
    mstring_append(out, "};\n\n");
}

void pgraph_msl_gen_vtx_out_locals(MString *out)
{
    for (int i = 0; i < ARRAY_SIZE(msl_vtx_attrs); i++) {
        mstring_append_fmt(out, "  %s %s = %s(0.0);\n", msl_vtx_attrs[i].type,
                           msl_vtx_attrs[i].name, msl_vtx_attrs[i].type);
    }
    mstring_append(out, "\n");
}

void pgraph_msl_gen_vtx_out_pack(MString *out)
{
    for (int i = 0; i < ARRAY_SIZE(msl_vtx_attrs); i++) {
        mstring_append_fmt(out, "  out.%s = %s;\n", msl_vtx_attrs[i].name,
                           msl_vtx_attrs[i].name);
    }
}

void pgraph_msl_gen_vtx_in_locals(MString *out)
{
    for (int i = 0; i < ARRAY_SIZE(msl_vtx_attrs); i++) {
        mstring_append_fmt(out, "  %s %s = in.%s;\n", msl_vtx_attrs[i].type,
                           msl_vtx_attrs[i].name, msl_vtx_attrs[i].name);
    }
    mstring_append(out, "\n");
}
