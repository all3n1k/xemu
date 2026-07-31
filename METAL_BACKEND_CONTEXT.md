# Native Metal backend for xemu — findings

Branch `metal-backend` on `github.com/all3n1k/xemu`. Written 2026-07-31,
replacing the running handoff notes with a summary.

## Verdict first

The BIOS boot animation renders. **Games do not** — Burnout 3 reaches its
first load screen with correct audio and a black picture, and runs roughly
60-120x too slowly.

That gap is not a bug to find. Games depend on features this backend never
implemented, chiefly DXT texture compression, which the boot animation
happens not to use at all. That is why the animation got as far as it did
and a game gets nowhere.

**Recommendation: stop here.** Not because the approach failed, but because
what remains is weeks of work whose destination is a fork upstream has
already declined, while a working alternative ships today. See "Why stop".

## What works

- MSL generation from the shared NV2A shader generators, via a `metal`
  dialect flag alongside the existing GLSL/Vulkan ones. 25/25 shader states
  from a real boot compile; 11/11 in the self-test.
- Surfaces: creation, clear, upload, download, eviction with writeback,
  format mapping, zero-copy VRAM aliasing.
- Vertex: all three submission paths (arrays, inline buffer, inline array),
  primitive expansion for fans/quads/line loops, zero-copy attribute fetch.
- Textures: 30 of 42 formats, cubemaps, samplers, surface-as-texture.
- Draws: pipeline cache, batched render passes, depth/stencil, blend, cull,
  polygon fill mode, occlusion queries.
- Metal API validation clean across a full boot.

## Twelve fixes, in the order found

1. **Texture type mismatch** — shader declared a cubemap, a 2D texture was
   bound. Metal discards the entire draw, silently, and the command buffer
   still reports success. Found by `MTL_DEBUG_LAYER=1` in one run, after
   eight rounds of guessing from pixel counts.
2. **Pipeline built before the submission path was known** — Metal bakes the
   vertex descriptor into the pipeline at creation; it was being mutated
   afterwards.
3. **Pipeline cache keyed on the wrong state** — keyed on the attribute
   registers, which for immediate-mode draws describe a different layout
   than the descriptor actually used. Every `inline_array` draw got an
   array-path pipeline and fetched vertices with the wrong strides.
4. **Surface swizzle not re-stamped** — GL re-stamps the flag on every
   upload; this backend set it once at creation. A surface the guest
   switched from linear to swizzled kept being written back linear, and
   reading it as a swizzled texture unswizzled linear bytes into a tiled
   scramble. This was the "pattern spray" on screen.
5. **Render-pass batching** — one command buffer plus a full GPU stall per
   draw. 2895 -> 23068 draws/s; 21838 render passes -> 660.
6. **Uniform cross-draw aliasing** — batched draws all read the last draw's
   uniforms.
7. **Const-attribute cross-draw aliasing** — same, for constant vertex
   attributes.
8. **Occlusion queries** unimplemented.
9. **Surface eviction** dropped rendered content instead of writing it back.
10. **Depth interpolation**, plus a scratch-depth target for pipelines that
    write depth with no depth attachment (Metal rejects those; GL discards).
11. **Cubemap textures** — rejected outright, substituted with flat white.
12. **Render-to-texture vertical flip** — Metal writes row 0 as the top of a
    rendered image, GL as the bottom, so a surface read back through memory
    and sampled as a texture comes out mirrored. Must be scoped to
    render-to-texture downloads only; applying it to the scanout inverts the
    whole screen.

## Verified correct — do not re-investigate

Every stage of the shader pipeline was checked against the GL path and is a
faithful translation:

| Stage | Method | Result |
|---|---|---|
| Vertex codegen | MSL vs GLSL textual diff | 3 expected lines |
| Fragment codegen | same | only the intended depth substitution |
| MAC/ILU opcodes | same | identical |
| Vertex uniforms | computed layout dump | vec3 arrays repacked 12->16 correctly |
| Fragment uniforms | same | every member correctly aligned |
| Shader state | shared-code dump, both backends | byte-identical over 24 states |
| One whole program | hand-evaluated against GPU output | matches |

## Still broken

1. **DXT/compressed textures: 0 of 3 mapped.** Games use these for nearly
   every texture. Single largest reason games render black.
2. **3D textures unimplemented; mipmaps level 0 only.**
3. **12 of 42 texture formats unmapped.**
4. **Display path round-trips through guest RAM.** The UI composites with
   OpenGL (`ui/xemu.c:825` takes a `GLuint`), so every frame is downloaded
   to guest memory to reach the screen. This is the 60-120x slowdown.
   Fixing it means IOSurface sharing — "Stage D", never started.
5. **Orb glow missing in the boot animation.** Combiners, shader state,
   uniform layouts, uniform values and textures all verified correct, and
   the shading is still absent. No hypothesis remains.
6. **Wireframe machinery.** Confirmed *not* polygon mode — the guest asks
   for fill everywhere.
7. **The flip fix is incomplete.** Games still show flipped UI, so there is
   at least one render-to-texture path beyond the blit.

## Why stop

- **Upstream will not take this.** The xemu maintainer has stated that
  platform-specific backends mean maintenance overhead they do not want,
  that MoltenVK is the preferred path, and that large LLM-generated patches
  have been rejected before. That applies regardless of quality.
- **A working alternative exists now.** The MoltenVK forks
  (`CosmicSnow/xemu-for-macos`, `MichaelJSr/xemu-macos`) run games at 45-60
  fps on Apple silicon today.
- **The remaining distance is weeks**, and only then would one game run
  well.

If the goal is contributing to Mac xemu rather than owning a backend, those
forks have narrow, named bugs — a wrapping/offset shift, Steel Battalion
menu textures, a render-target size limit above 2x — which are small,
reviewable, and welcome upstream.

## Method notes — the reusable part

**Direct observation beat every metric.** Four separate times, watching the
window produced a correction the tooling could not: the pattern spray, the
flip, the wireframe, the flat orbs. Several overturned conclusions that had
been stated confidently.

**Two instruments were themselves wrong** and produced false conclusions: an
fb-stats counter masked alpha, so opaque black read as "0% non-black"; and a
draw counter printed a sample index that was read as a total, twice.
Validate the instrument before trusting a surprising measurement.

**Byte-diffing texture dumps cannot see a mirror** — a flipped image and a
corrupt one give similar difference counts — **and cannot separate a
rendering fault from two runs being at different animation moments.** That
confound invalidated three findings. The comparisons that did work read
*guest memory* at a known address, or used a spatial metric (centroid)
rather than a byte count.

**`MTL_DEBUG_LAYER=1` first, always.** It named a real bug in one run that
eight rounds of black-box measurement had missed.

## Instrumentation — env-gated, off by default

| Variable | Purpose |
|---|---|
| `MTL_DEBUG_LAYER=1` | Metal's own API validation. Run first. |
| `XEMU_METAL_DEBUG_POS` | Vertex shader writes clip position, texcoords and fetched attributes to a buffer, dumped per vertex, with the CPU-side view of the same memory alongside. |
| `XEMU_METAL_DUMP_SURFACES=<pfx>` | Every live colour surface to disk. |
| `XEMU_METAL_FILMSTRIP=<pfx>` | One file per displayed frame, sampled. |
| `XEMU_METAL_DUMP_VSH` / `_PSH=<pfx>` | Generated MSL and GLSL for the same state, for diffing. |
| `XEMU_DUMP_SHADER_STATE` | The state feeding the generator, from shared code so both backends match. |
| `XEMU_METAL_DUMP_LAYOUT` | Computed MSL uniform offsets and strides vs the C struct. |
| `XEMU_METAL_THROUGHPUT` | draws/s, passes, submits. |
| `XEMU_METAL_SYNC_EVERY_DRAW` | Restores per-draw sync, for A/B. |
| `XEMU_METAL_NO_FLIP`, `_FORCE_FILL`, `_NO_CULL`, `_FULL_SCISSOR`, `_DEBUG_SHADER`, `_DEBUG_FS` | Bisect switches. |

Dumps are raw `[u32 width][u32 height][BGRA8 rows]`.

## Build and run

```
ninja -C build qemu-system-i386
DYLD_LIBRARY_PATH=./macos-libs/arm64/opt/local/lib ./build/qemu-system-i386
```

One invariant this backend adds: **anything that reads a surface texture
from the CPU, or destroys one, must call `pgraph_metal_flush_gpu()` first.**
Draws accumulate in an open render pass until then.
