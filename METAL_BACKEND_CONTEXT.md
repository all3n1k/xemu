# Metal Backend for xemu — Context Handoff (updated 2026-07-29)

## Project
Fork xemu-project/xemu → Apple Silicon native Metal render backend.
- **Fork**: `https://github.com/all3n1k/xemu`
- **Branch**: `metal-backend`
- **Upstream**: `https://github.com/xemu-project/xemu`
- **Working dir**: `/Users/neo/projects/xemu-metal`

## Environment
- **macOS 26.5 Tahoe, Apple M5**
- **Build**: `cd /Users/neo/projects/xemu-metal && export PATH="/opt/homebrew/bin:$PATH" && ninja -C build qemu-system-i386`
- **Run**: `DYLD_LIBRARY_PATH=./macos-libs/arm64/opt/local/lib ./build/qemu-system-i386`
- **Build system**: meson + ninja via Homebrew. Deps: SDL3, libepoxy, ninja, meson, pkgconf, cmake.
- **Metal libraries**: system frameworks, linked in root `meson.build`.
- **NOTE**: the `metal` CLI compiler is NOT installed (`xcode-select -p` →
  CommandLineTools, no full Xcode). This does not matter: shaders are compiled
  at runtime with `newLibraryWithSource:`, which uses the system
  MTLCompilerService and needs no Xcode.

### Testing shaders without booting the emulator
A ~40-line ObjC harness that runs a `.metal` file through
`newLibraryWithSource:` and prints diagnostics is the fastest iteration loop
for dialect questions:
```
clang -fobjc-arc -framework Metal -framework Foundation -o mslcheck mslcheck.m
./mslcheck shader.metal
```
Recreate it if needed; it is not checked in (it lives outside the repo).

## Stage A — Scaffolding (DONE, `c513bfe46f`)
- All 16 PGRAPHRenderer ops registered as stubs
- Real `MTLDevice` + `MTLCommandQueue` init
- "Metal" option in the UI renderer combo (`main-menu.cc:743`)
- `get_framebuffer_surface` returns 0 → falls back to the slow VGA path

## Stage B — Shader generation (DONE, `78952c96af`)

### Design decision (deviates from the original plan — read this first)
The original plan called for a from-scratch MSL emitter duplicating the
generators. That was reconsidered after reading them: the statement-level code
they emit (combiner arithmetic, MAC/ILU vertex program, lighting, texgen) is
**already dialect-neutral**. Only the shell differs — declarations, entry point
signature, resource binding, ~10 builtin spellings.

So the generators take a `metal` flag in `GenVshGlslOptions` /
`GenPshGlslOptions` and delegate the shell to `glsl/msl.c`. This keeps one
source of truth for NV2A semantics instead of a second 4,000-line emitter that
would drift from upstream. Every Metal branch is gated; GL/Vulkan output is
unchanged.

### Files
| File | What |
|------|------|
| `glsl/msl.h` / `msl.c` | **New.** MSL prologue (typedefs + GLSL builtin shims) and MSL-specific declaration emitters (uniform struct, [[stage_in]] struct, interpolant struct) |
| `glsl/vsh.c` / `vsh.h` | `metal` option; MSL branches for uniforms, attributes, entry point, output packing |
| `glsl/vsh-prog.c` / `.h` | `metal` param: register file becomes function-local, macros come from the prologue. Instruction stream unchanged. |
| `glsl/psh.c` / `psh.h` | `metal` option; MSL preflight, entry point signature with texture/sampler params, builtin macros |
| `glsl/vsh-ff.c` | **Untouched** — emits only #defines and dialect-neutral statements |
| `metal/shaders.h` / `shaders.m` | **New.** `pgraph_metal_gen_vsh/psh()`, `pgraph_metal_compile_shader()`, self-test |
| `metal/renderer.m` | Calls the self-test at init |

### Three things MSL cannot spell the same way
1. **No mutable program-scope variables.** The NV2A output registers (oPos,
   oD0..oT3) and the vertex program register file (R0..R11, A0, _temp_*) are
   emitted as entry point locals instead of globals.
2. **Textures/samplers are entry point parameters, not globals.** `texSampN` is
   a macro expanding to the pair `texN, smpN`, which resolves to three-argument
   `texture()`/`textureProj()`/`textureSize()` overloads in the prologue.
   `normN()` is likewise a macro binding the stage's texture into `msl_norm()`.
3. **The fixed-function attribute aliases collide.** `vsh-ff.c` emits
   `#define position v0` and `#define pointSize v6` at file scope, which
   rewrite `out.position` / `out.pointSize`. Hence the `nv2a_` prefix on the
   builtin struct members (`nv2a_position`, `nv2a_pointSize`,
   `nv2a_pointCoord`). **Do not rename these back.**

### Statements adjusted to be valid in both dialects (identity for GLSL)
- `.st` / `.p` stpq swizzles → `.xy` / `.z` (MSL has no stpq set) — BUMPENVMAP_LUM
- Integer literals in float contexts → `0.0` etc. (MSL does not implicitly convert)
- `gl_FragCoord.xy / surfaceScale` → `/ vec2(surfaceScale)` (int2 → float2)

### Verified
- Self-test at init generates + compiles 11 representative shader states
  (6 vertex incl. FF lighting/texgen/skinning and programmable variants,
  5 pixel incl. combiners, rect textures, point sprite, color key):
  **all 11 compile on Apple M5.**
- OpenGL and Vulkan re-run afterwards: zero shader errors, so the shared
  generators are genuinely unchanged for them.

### Resource binding contract (Stage C must match)
- Entry point name: `main0` (both stages), `PGRAPH_METAL_SHADER_ENTRY`
- Uniform struct: `[[buffer(0)]]`, `MSL_UNIFORM_BUFFER_INDEX`
- Textures `[[texture(N)]]`, samplers `[[sampler(N)]]`, N = NV2A stage 0..3
- Interpolants matched by `[[user(name)]]`, names identical in both stages

## Stage C — Surface + Texture + Draw (IN PROGRESS)

### C.1 Render targets, clear, readback (DONE, `d2083f3e73`)
`metal/surface.h` + `surface.m`, mirroring `gl/surface.c`. The cache semantics
(address keying, compatibility checks, eviction, CPU access callbacks,
upload/download scheduling) are guest behaviour and are kept structurally
identical to the GL backend on purpose, so the two stay comparable when
debugging. Only resource creation and transfers are Metal-specific.

Metal-specific choices:
- Render targets are `MTLStorageModeShared`. Unified memory means a shared
  target is directly readable with `getBytes:`, so download needs no staging
  blit. Depth targets fall back to `Private` if a GPU refuses shared, in which
  case readback is skipped rather than producing garbage.
- Apple GPUs have **no `MTLPixelFormatDepth24Unorm_Stencil8`**, so guest Z24S8
  is widened to `Depth32Float_Stencil8`. Host and guest bytes-per-pixel differ
  for zeta — the format table carries both, and any zeta readback must narrow.
- A full clear is a render pass load action with no draws (cheapest form Metal
  offers). Partial or channel-masked clears cannot be expressed that way and
  fall back to a CPU fill on the shared texture. **Replace this with a
  scissored clear quad once draw.m exists.**

Verified on M5 against the real BIOS: requested clear rgba(0,0,0,1) reads back
as BGRA `00 00 00 ff` at 1280x480 (640x480 with 2x1 AA); surfaces download into
guest VRAM; clean boot with no asserts or Metal validation errors; OpenGL
re-run clean afterwards.

Two bugs worth remembering, both found by running rather than by reading:
1. `surface_scale_factor` starts at 0. GL initialises it in
   `pgraph_gl_init_surfaces` via `reload_surface_scale_factor`; missing that
   made every render target degenerate (1x1) and tripped an AGX
   `width > 0` assertion. `pgraph_metal_reload_surface_scale_factor` now does
   the same.
2. `pgraph_metal_surface_update` must keep GL's write-enable gating,
   `framebuffer_dirty` shape-change handling and unbind-on-dirty. A simplified
   version reached binding creation with `color_format == 0` and asserted.

### C.2 Shader cache on live state (DONE, `2e58e93a77`)
`metal/draw.h` + `draw.m`. ShaderState in → compiled `MTLLibrary` +
`MTLFunction` pair out, keyed on the state bytes (`pgraph_glsl_get_shader_state`
zeroes the struct specifically so it can be hashed that way). Failures are
cached too — a state that does not compile will not start compiling later, and
re-running the generator every draw is expensive.

**This closed the biggest open risk from Stage B.** Until now the MSL generator
had only run against hand-built states in the self-test, which could only ever
test what was already thought of. Against a live BIOS boot it compiles
**25 distinct real states, 0 failures**, including programmable vertex shaders
with 6-stage pixel combiners.

#### Draw path survey — measured, do not re-derive
One instrumented run over a real boot (120k draws / 60s) gave the vertex
submission mix:

| path | count | share |
|---|---|---|
| `inline_elements` | 119,581 | **99.7%** |
| `draw_arrays` | 314 | 0.3% |
| `inline_array` | 105 | 0.1% |
| `inline_buffer` | 0 | — |

So **implement indexed drawing first** (`drawIndexedPrimitives` with a uint32
index buffer); the other three paths can wait. Primitive mode observed:
triangle strips.

Note: instrument `draw_end`, not `ops.flush_draw` — the latter is only called
from `pgraph_expand_draw_arrays`'s squash case and sees almost no traffic.

### C.3 Vertex attributes (DONE, `b40f218b37`)
`metal/vertex.h` + `vertex.m`.

**Zero-copy VRAM aliasing.** GL keeps a buffer object mirroring guest VRAM and
re-uploads whichever region each draw touches. On Apple Silicon that copy is
unnecessary: one `MTLBuffer` from `newBufferWithBytesNoCopy:` aliases the whole
guest VRAM mapping, attributes bind as offsets into it, guest writes are
visible to the GPU immediately. **Verified on M5 — alignment requirements hold,
the copy fallback never triggers.** Keep the fallback anyway for other hosts.

Each attribute gets its own buffer slot (`METAL_VERTEX_BUFFER_BASE` = 1, so
1..16; slot 0 is the uniform struct).

Two mismatches with Metal's vertex model, already handled:
- **Constant attributes.** The guest can supply an attribute as one value
  rather than an array (GL: `glVertexAttrib4fv`). Metal has no per-attribute
  constant, so those slots point at a scratch buffer with
  `MTLVertexStepFunctionConstant`.
- **UB_D3D is BGRA in memory** and Metal has no BGRA vertex format. Read as
  RGBA and swizzled in the shader — the generator's existing `swizzle_attrs`
  path already emits exactly that.

Shader coverage against a live boot: **50 distinct real states, 0 failures**,
reaching 8-stage pixel combiners and the fixed-function vertex path.

### C.4 Draw dispatch (DONE, `873ae7d966`) — draws issue, screen still black

Pipeline cache, encoder, uniform upload, indexed draw. **12,321 draws issued
over a boot, 39 pipeline states, 0 creation failures.** Framebuffer is still
black, so the bug is between vertex transform and raster.

#### The uniform layout assumption was WRONG — now fixed
Carried since Stage B, checked before building on it, and it did not hold. The
MSL struct cannot be byte-identical to the C `*UniformValues` struct: the C
typedefs are `float[N]` (4-byte aligned), so members land where MSL may not
place a vector — `VshUniformValues::inlineValue` at 3096 needs 16-byte
alignment, `PshUniformValues::bumpMat` at 4 needs 8. Six vertex and five pixel
members affected; `vec3` arrays also stride 12 vs 16.

Packed types fix alignment but break arithmetic (`float2x2` must multiply as a
matrix). So `pgraph_msl_uniform_layout()` computes the real MSL layout and the
renderer repacks member-by-member, element-by-element where strides differ.
**That function is the only place that knows the rule — keep it that way.**

#### Two Metal/NV2A mismatches, handled
- The fragment shader always writes `gl_FragDepth` (w-buffering emulation
  needs it). Metal rejects a depth-writing pipeline with no depth attachment;
  GL just discards. Unbound-zeta draws get a scratch depth target rather than
  changing the shader.
- `get_framebuffer_surface` returning 0 was **not sufficient**. The UI falls
  back to uploading the guest framebuffer, but nothing copied the rendered
  target back into guest RAM, so it showed a stale buffer. It now forces the
  scanned-out surface down synchronously first.

#### Ruled out already (don't re-investigate)
- Shader generation — 50 real states compile, 0 failures
- Pipeline creation — 39 built, 0 failures
- Draw submission — 12,321 issued
- Readback — 991 downloads of the right 640x480 surface, no Metal errors
- Clears — verified landing in C.1

### C.5 Primitive expansion + bisect (DONE, `a30c68e813`)

**Expansion.** Metal has no fan/quad/quad-strip/line-loop primitive. All are
now expanded to triangle/line lists on the CPU, preserving vertex order so
culling still behaves. Unsupported prims **7,664 → 0**; draws issued
**12,321 → 19,985** (99.9% of draw_ends). The remaining 16 are
inline_array/inline_buffer (0.1%), still unstaged.

**Bisect result — half the system is now eliminated.**
- depth-always + cull-none + full-scissor, all at once: **no change**. Not
  depth, not culling, not scissor.
- Hardcoded shader ignoring all guest state: **40.5% of the framebuffer
  non-black**, exactly that triangle's coverage.

So pipeline creation, encoder, attachments, rasterization and readback into
guest RAM **all work — Metal renders real pixels.** The fault is confined to
the generated shaders or the uniform values fed to them.

**Bug found and fixed on the way (not the cause):** the vertex uniform block
was bound with `setVertexBytes:`, which caps at **4 KB**. The block is ~6 KB
(192 float4 constants = 3 KB alone), so that binding could never have been
valid. Now goes through a real MTLBuffer. Geometry still not visible with it
fixed, so keep looking.

#### C.6 — the black screen: SOLVED (`04f4509a6f`)

The boot animation renders. Three real bugs, all the same species: Metal
binds something at a different *time* or with different *identity* than GL
does, and fails silently when it disagrees.

1. **Texture type mismatch** (`a615ee4012`). The shader declared a cubemap,
   a 2D texture was bound. Metal discards the entire draw on a type
   mismatch — silently; the command buffer still reports success. Found in
   one run with `MTL_DEBUG_LAYER=1` after eight rounds of guessing from
   pixel counts. Fixed with per-type white fallbacks and a type check at
   bind time.

2. **Pipeline created before the submission path was known** (`f3cf749653`).
   Metal bakes the vertex descriptor into the pipeline at creation. It was
   built in `draw_begin` and the descriptor was then mutated in
   `flush_draw`. Necessary fix, but not sufficient on its own — see 3.

3. **Pipeline cache keyed on the wrong state** (`04f4509a6f`). The key used
   `pg->vertex_attributes[]` (format/stride/count). For array draws that
   matches the descriptor because the descriptor is built from those
   registers. For the immediate-mode paths it does not: the descriptor comes
   from the inline data layout while the registers still hold whatever the
   last array draw left behind. Every `inline_array` draw therefore got back
   an array-path pipeline and Metal fetched its vertices with that
   pipeline's strides. The key is now derived from the descriptor itself, so
   it is correct by construction whoever built it.

The signature of 3, from `XEMU_METAL_DEBUG_POS` — three different inputs,
one output:

```
in(  -0.53   -0.53) -> clip(-2^64 -2^64 0 +2^64)  ndc(-1 -1 0)
in(2559.47   -0.53) -> clip(-2^64 -2^64 0 +2^64)  ndc(-1 -1 0)
in(  -0.53 1919.47) -> clip(-2^64 -2^64 0 +2^64)  ndc(-1 -1 0)
```

A degenerate triangle rasterizes nothing, which is indistinguishable from a
draw that never happened. `2^64` is the ceiling of `clampAwayZeroInf`: the
shader saw `v0 = (0,0,0,0)` and took the reciprocal of a zero `w`. After the
fix the same draw gives `clip(-1 -1 0 1) / (7 -1 0 1) / (-1 7 0 1)` and the
scanout goes from 0/307200 covered to 307200/307200.

#### Performance (`10f65ea163`)

Draws batch into one render pass while the attachments hold. Measured over
the boot animation, same binary, switched with `XEMU_METAL_SYNC_EVERY_DRAW`:

```
per-draw sync    2895 draws/s    21838 draws / 21838 passes / 21838 submits
batched         23068 draws/s   127186 draws /   660 passes /   659 submits
```

The pass closes only when the attachments change or when something needs the
result: surface readback, a CPU write into a surface texture, a clear (it
builds its own pass and would otherwise land underneath recorded draws),
binding teardown, surface flush. Only the readback cases wait.

**If you add anything that touches a surface texture from the CPU, or that
destroys one, it must call `pgraph_metal_flush_gpu()` first.** That is the
one invariant this design adds.

#### Instrumentation — use these before theorising

Every one of the three bugs above was found by direct readback and none by
inference from pixel counts. All are env-gated and off by default.

- `MTL_DEBUG_LAYER=1` — Metal's own API validation. Run it first, always.
- `XEMU_METAL_DEBUG_POS` — the generated vertex shader writes clip position,
  texcoord 0, and the raw `v0`/`v9` it fetched into a buffer, dumped per
  vertex. Distinguishes a bad transform from a bad fetch, and prints the
  CPU-side view of the same memory alongside so a binding error shows up as
  a disagreement.
- `XEMU_METAL_DUMP_SURFACES=<prefix>` — every live colour surface to disk.
  The scanout is the end of a chain; dumping only it cannot say which link
  broke.
- `XEMU_METAL_DUMP_TEX=<prefix>` — uploaded textures, to tell a wrong image
  apart from wrong coordinates.
- `XEMU_METAL_THROUGHPUT` — draws/s, passes, submits.
- `XEMU_METAL_SYNC_EVERY_DRAW` — restores the old per-draw sync for A/B.
- Bisect switches: `XEMU_METAL_DEBUG_SHADER` (hardcoded shader),
  `XEMU_METAL_DEBUG_FS` (real VS + solid-colour FS), `XEMU_METAL_NO_CULL`,
  `XEMU_METAL_FULL_SCISSOR`, `XEMU_METAL_DRAW_STATS`.

The dumps are raw `[u32 width][u32 height][BGRA8 rows]`; there is a
throwaway converter in the session scratch, ~30 lines of stdlib Python.

#### Wrong turns worth not repeating

- Anti-aliasing, the 2D blit, missing texture upload, a per-surface bug, and
  "the guest only issues one draw" were all asserted with too much
  confidence and all wrong.
- Two instruments were themselves broken and produced false conclusions: an
  fb-stats counter masked alpha so opaque black read as 0% non-black, and a
  draw counter printed a sample index that got read as a total, twice.
  **Validate the instrument before trusting a surprising measurement.**
- `texScale` was genuinely never set by this backend (GL and Vulkan both
  set it from their own texture bindings) and is now fixed — but it changed
  nothing visually. Do not credit it for anything.
- "attribute 0 is fetched wrong" was a misreading: the strip's first ring is
  degenerate at a pole, so identical positions there are correct. Look at
  more than the first four vertices.

### C.4 original plan (superseded, kept for reference)
Everything else is in place; this is what puts geometry on screen.
- `MTLRenderPipelineState` cache. Key: shader binding + colour/depth
  attachment pixel formats + blend regs + write masks + the vertex
  descriptor. Build with `vsh_function`/`psh_function` from the shader cache
  and `pgraph_metal_build_vertex_descriptor()`.
- Render command encoder on the bound surfaces. Load action must be **Load**,
  not Clear (clears are separate and already work), store action Store.
- Viewport + scissor from `surface_binding_dim` and `surface_shape.clip_*`,
  both scaled — see `pgraph_gl_draw_begin` for the exact arithmetic.
- Depth/stencil via `MTLDepthStencilState` from `NV_PGRAPH_CONTROL_0/1/2`.
- Uniform upload: fill `VshUniformValues`/`PshUniformValues` with
  `pgraph_glsl_set_{vsh,psh}_uniform_values` (pass a locs array of all-zero,
  i.e. "every uniform present"), then bind at slot 0. **This is where the
  unverified packed-struct layout assumption gets tested** — if geometry comes
  out garbled, check that first.
- Dispatch: `drawIndexedPrimitives` with a uint32 index buffer from
  `pg->inline_elements` (99.7% of draws — see the survey above), primitive
  type from `pg->primitive_mode`.
- `texture.m` — Xbox texture formats → MTLTexture upload, sampler state. Not
  needed for first geometry; untextured triangles are enough to prove the path.

Until dispatch lands the screen stays black: clears work, shaders compile, but
nothing is drawn. That is the expected intermediate state, which is why C.1 and
C.2 were verified by reading pixels back and by compile counts rather than by
looking at the window. **A black window is not evidence of failure at this
stage, and an animated one means you are running OpenGL.**

### Known gaps carried forward

Current state: the BIOS boot animation renders correctly at ~23k draws/s
with Metal API validation clean and 25/25 shader states compiling. No game
has been tried yet — see the ordering note at the end.

1. **Texture formats.** Two fall back to a white texture over a boot:
   `0x30` (`LU_IMAGE_DEPTH_Y16_FIXED`) and format `0x7` when it is a
   cubemap. Cubemaps, 3D textures and mipmaps are all unimplemented in
   `upload_texture` — level 0, 2D only. The white fallback is per-type so
   the draw is not discarded, but the shading is wrong wherever it hits.
2. **Zeta download unimplemented** — needs narrowing from the widened host
   depth format (Apple GPUs have no `Depth24Unorm_Stencil8`, so Z24S8
   becomes `Depth32Float_Stencil8`). Reports NV2A_UNIMPLEMENTED rather than
   corrupting guest memory.
3. **Surface-as-texture ignores format mismatch.** A stage pointing at a
   live render target binds that target's texture directly. The scale is now
   handled (it feeds `texScale`); a differing pixel format is not.
4. **Partial/masked clears go through the CPU** — correct but slow.
5. **No geometry shader stage.** vtxPos0/1/2 and triMZ get degenerate
   per-vertex values, so w-buffering's barycentric depth interpolation is
   still wrong.
6. **Shader cache is per-run.** No disk cache, so every boot recompiles.
   Cheap to add, and 25 states is not much, but a game will ask for more.
7. **The display path still round-trips through guest RAM.** See Stage D —
   this is the other half of the performance story and is untouched.

Resolved since the last handoff: the vertex descriptor does populate all 16
attributes; the uniform layout is verified and repacked member-by-member via
`pgraph_msl_uniform_layout()` rather than memcpy'd; primitive expansion
(fans, quads, line loops) is implemented on the draw path.

## Open bug: the BIOS logo texture (2026-07-30)

The boot animation runs end to end and reaches the logo phase. The
wordmark renders. The X emblem renders as a regular grid of small
repeated glyphs.

Reproduce with `XEMU_METAL_FILMSTRIP=<prefix>` and
`XEMU_GL_FILMSTRIP=<prefix>` (same capture, both backends, one file per
displayed frame) plus `XEMU_METAL_DUMP_TEX` / `XEMU_GL_DUMP_TEX` (each
texture dumped twice: `.src` = raw guest bytes before unswizzling,
`.raw` = decoded, under a content-derived filename so the two backends'
files pair up).

What is established, by byte comparison rather than inference:

  - The 1024x1024 swizzled logo texture decodes differently from GL in
    602007 of 4194312 bytes.
  - Its **raw source bytes** differ by exactly the same 602007. The
    decode is not at fault.
  - The 16x16, 128x128 and 256x256 swizzled textures decode
    byte-identical to GL.
  - Two Metal runs are byte-identical to each other. Deterministic.
  - Both versions are substantially populated (1.29 MB vs 1.36 MB of
    non-zero bytes) and differ in both directions, so it is two
    different images, not a partial write or a timing snapshot.
  - The texture lives at guest 0x01c54000, spans 4 MB, and is not backed
    by any live surface. The nearest are a 1024x1024 zeta at 0x02054000
    (exactly where the texture range ends) and a 1024x1024 colour
    surface at 0x02454000.

Disproven, each by measurement, each after being asserted with too much
confidence first:

  1. Bad unswizzle. Ruled out: other swizzled textures are identical and
     the source bytes differ by the same count as the decoded ones.
  2. Missing surface writeback. A fix was written (and kept, since the
     gap is real -- see below) and changed the differing byte count not
     at all.
  3. The guest steering the texture off an occlusion query result.
     Queries were implemented (also real, also kept) and the count again
     did not move.

**Caveat on the comparison itself, unresolved:** the two backends' dumps
are paired by `widthxheight_format` filename. Metal's guest address is
logged (0x01c54000); GL's log prints a host pointer, which differs
between runs, so it was never confirmed that both backends dumped the
*same* texture rather than two different 1024x1024 fmt-0x6 textures.
Confirming this is the first thing to do -- it could invalidate the
602007 figure entirely. `upload_gl_texture()` does not have `NV2AState`
in scope; the offset is available in the caller around gl/texture.c:343.

Next after that: diff the draw sequence between backends -- which draws
happen, in what order, into which surface. That has never been run and
would catch a whole class of causes at once.

Two fixes landed while chasing this, neither of which fixed it, both
worth keeping because GL has the equivalent behaviour:

  - **Surface writeback before texture sampling.** The CPU-access
    callback that downloads a rendered surface when the guest touches
    its memory registers only under TCG, and Apple Silicon runs under
    HVF, so it never exists.
  - **Occlusion queries.** `get_report` returned a constant zero and
    `clear_report_value` was empty, so every query answered "nothing
    visible". Now counts into a Metal visibility result buffer, one slot
    per draw, summed when the guest reads the report.

Also still open: Metal runs the animation noticeably slower than GL in
wall-clock terms, so the two backends' frame N are not the same
animation moment. And user-reported, unverified: white streaks at the
very start, and glow orbs that move wrongly. The orbs were the
motivation for implementing occlusion queries; whether that fixed them
has not been checked.

## What to do next, in order

1. **Finish correctness on the BIOS.** Texture formats and the gaps above.
   Mechanical, and the instrumentation is in place.
2. **Then Stage D (display).** The UI composites with OpenGL
   (`ui/xemu.c:825` takes a `GLuint`), so Metal output currently reaches the
   screen only by downloading to guest RAM. IOSurface sharing removes that
   round trip. This is the remaining half of the performance work; the
   draw-submission half is done.
3. **Only then try a game.** Not before — debugging correctness through a
   round-tripped display is miserable, and the per-frame download will
   dominate any measurement you take.

Worth being honest about the payoff: xemu already runs Vulkan through
MoltenVK on Apple silicon. Native Metal avoids that translation layer; it
does not enable anything otherwise impossible.

## Stage D — Display (LAST)

**Read this before planning Stage D.** `ui/xemu.c:825` does
`GLuint tex = nv2a_get_framebuffer_surface();` — the xemu UI composites with
**OpenGL**. A Metal backend therefore cannot hand its MTLTexture to the UI
directly. Options:
  - Share via `IOSurface` (`CAMetalLayer`/`MTLTexture` backed by an IOSurface,
    imported into GL through `CGLTexImageIOSurface2D`). Keeps the existing UI.
  - Port the UI compositor to Metal. Much larger change.
  - Status quo: `get_framebuffer_surface` returns 0 and the UI falls back to
    uploading the guest VRAM framebuffer (`xb_surface_gl_create_texture`).
    Correct but costs a full round trip through guest memory every frame.

That fallback is why C.1's readback path is load-bearing rather than optional.

## Notes
- `objc_args: ['-fobjc-arc']` is REQUIRED in meson for `.m` files
- Test config used during Stage B: a portable-mode `build/xemu.toml` (xemu uses
  it when a `xemu.toml` sits next to the binary) with `renderer = 'METAL'`, so
  the user's real config is untouched. It was deleted after testing.
- Local BIOS/HDD paths live in `~/Library/Application Support/xemu/xemu/xemu.toml`

## Workflow
```bash
cd /Users/neo/projects/xemu-metal
export PATH="/opt/homebrew/bin:$PATH"
ninja -C build qemu-system-i386
DYLD_LIBRARY_PATH=./macos-libs/arm64/opt/local/lib ./build/qemu-system-i386
git add -A && git commit -m "metal: message" && git push origin metal-backend
```
