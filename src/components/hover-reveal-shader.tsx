"use client"

import { ShaderLabComposition, type ShaderLabConfig } from "@basementstudio/shader-lab"
import { useGLTF } from "@react-three/drei"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Suspense, useEffect, useMemo, useRef, useState } from "react"
import * as THREE from "three"
import {
  dot,
  float,
  max as tslMax,
  min as tslMin,
  mix,
  positionLocal,
  smoothstep,
  texture as tslTexture,
  uv,
  vec2,
  vec3,
  vec4,
  viewportUV,
} from "three/tsl"
import { MeshBasicNodeMaterial, WebGPURenderer } from "three/webgpu"

const ACCUM_SIZE = 512
const ACCUM_RT_OPTIONS = {
  depthBuffer: false,
  generateMipmaps: false,
  magFilter: THREE.NearestFilter,
  minFilter: THREE.NearestFilter,
  type: THREE.HalfFloatType,
  format: THREE.RGBAFormat,
}
const LUMINANCE_WEIGHTS = vec3(0.2126, 0.7152, 0.0722)
// WebGLRenderTarget/RenderTarget textures come out vertically flipped
// relative to normal image/canvas textures in this renderer, so re-sampling
// one in a later pass needs this correction (canvas/image textures do not).
const renderTargetUv = vec2(uv().x, float(1).sub(uv().y))
// Screen-space viewport UV instead of the consuming mesh's own UV — for
// painting onto a 3D model's surface where mesh UV doesn't correspond to
// screen position. Empirically, viewportUV's Y already matches the render
// target's storage orientation (unlike a mesh's own UV), so no extra flip
// is needed here — adding one inverted the vertical tracking.
const viewportMaskUv = viewportUV
// The fluid composition's own WebGPU context can take a while to spin up
// (it's a second, independent WebGPU context on the page). Until it has,
// its canvas can read back as blank/garbage rather than true black — and
// since the mask latches forever, sampling it too early would permanently
// bake in a false "fully revealed" state. So instead of guessing a fixed
// delay, actually sample the canvas each frame and only start accumulating
// once it reads back sane (near-black), with a generous timeout fallback.
const WARMUP_MIN_MS = 500
const WARMUP_TIMEOUT_MS = 30000
const WARMUP_BRIGHTNESS_THRESHOLD = 40 // out of 255
// Reveal threshold: how strongly an area must be painted (raw latch value)
// before it snaps to fully showing the next layer. Sits well above the
// fluid sim's own ambient baseline noise so noise alone never latches.
const REVEAL_LOW = 0.35
const REVEAL_HIGH = 0.55
// Both masks use the same additive-creep MECHANISM (see useLatchedMask's
// creepGain) at the same RATE, so layer1->layer2 and layer2->layer3 take
// the same amount of dragging and feel like the identical gesture once
// triggered -- a single ~half-second stroke is enough to finish either
// one. What keeps layer3 from finishing inside the same stroke that
// reveals layer2 is not a slower rate here; it's that mask2's gate
// (useStrokeSnapshotTexture) only reflects mask1's progress from BEFORE
// the current stroke began, so layer3 structurally requires a separate,
// later stroke no matter how fast either mask accumulates.
const CREEP_GAIN = 0.02

function isCanvasSettled(
  canvas: HTMLCanvasElement,
  probeCtx: CanvasRenderingContext2D
): boolean {
  try {
    probeCtx.drawImage(canvas, 0, 0, 8, 8)
    const { data } = probeCtx.getImageData(0, 0, 8, 8)
    let sum = 0
    for (let i = 0; i < data.length; i += 4) {
      sum += data[i] + data[i + 1] + data[i + 2]
    }
    return sum / ((data.length / 4) * 3) < WARMUP_BRIGHTNESS_THRESHOLD
  } catch {
    return false
  }
}

// Renders `max(previousLatchedMask, currentFluidMask)` into a ping-ponged
// render target every frame, so once an area is revealed by the fluid it
// never fades back — only the union of everywhere it has ever painted.
// With optional `creepGain`, this instead creeps up additively every frame
// the live fluid is bright here (capped at 1), which is what actually
// makes "paint more to reveal more" possible: max() alone only ever
// remembers a pixel's single highest momentary hit, so once one stroke's
// peak brightness has touched a pixel, repeating that same stroke is a
// no-op for it -- there's nothing further for max() to raise it to.
function useLatchedMask(
  fluidCanvas: HTMLCanvasElement | null,
  options?: {
    // UV to sample the resulting mask with when consumed by an outer
    // material. Defaults to the render-target-corrected mesh UV (fine for
    // a full-screen plane, where mesh UV ~= screen space). A 3D model's own
    // UV doesn't correspond to screen space, so callers painting onto a
    // rotatable mesh should pass a screen-space (viewport) UV instead, so
    // the paint follows the cursor rather than the model's UV layout.
    outputUv?: ReturnType<typeof vec2>
    // If set, switches this mask from an instant max()-latch to an
    // additive creep: each frame's live fluid brightness nudges the latch
    // up by (brightness * creepGain), capped at 1, instead of snapping
    // straight to whatever that frame's brightness is. Requires several
    // frames of sustained painting over the same spot to fully reveal,
    // rather than one touch -- used for mask2 so layer3 always needs
    // meaningfully more painting than layer2 needed, at the same pixel.
    creepGain?: number
    // Gates this mask's accumulation by another mask's current latch
    // texture, read live every frame (a ref, never a one-time snapshot).
    // Painting only counts here once the gate mask has already crossed
    // REVEAL_HIGH at this pixel -- this is what makes mask2's trigger
    // sequential rather than simultaneous: dragging over still-foggy
    // layer1 drives mask1 only, and that same drag over an area mask1
    // hasn't revealed yet contributes nothing to mask2 until it has.
    gateRef?: { current: THREE.Texture | null }
  }
) {
  const { gl: rawGl } = useThree()
  const gl = rawGl as unknown as WebGPURenderer
  const [ready, setReady] = useState(false)
  const latchRef = useRef<THREE.Texture | null>(null)

  const stateRef = useRef<{
    read: THREE.WebGLRenderTarget
    write: THREE.WebGLRenderTarget
    scene: THREE.Scene
    camera: THREE.OrthographicCamera
    fluidTexture: THREE.CanvasTexture
    prevNode: ReturnType<typeof tslTexture>
    outputNode: ReturnType<typeof tslTexture>
    gateNode: ReturnType<typeof tslTexture> | null
    fluidCanvas: HTMLCanvasElement
    probeCtx: CanvasRenderingContext2D
    startedAt: number
    armed: boolean
  } | null>(null)

  useEffect(() => {
    if (!fluidCanvas) return

    const rtA = new THREE.WebGLRenderTarget(
      ACCUM_SIZE,
      ACCUM_SIZE,
      ACCUM_RT_OPTIONS
    )
    const rtB = new THREE.WebGLRenderTarget(
      ACCUM_SIZE,
      ACCUM_SIZE,
      ACCUM_RT_OPTIONS
    )

    const fluidTexture = new THREE.CanvasTexture(fluidCanvas)
    const fluidNode = tslTexture(fluidTexture, uv())
    const prevNode = tslTexture(rtA.texture, renderTargetUv)
    const fluidLuminance = dot(fluidNode.rgb, LUMINANCE_WEIGHTS)
    const prevLuminance = dot(prevNode.rgb, LUMINANCE_WEIGHTS)
    // Backed by this mask's own still-cleared rtA until the gate ref hands
    // over a real frame -- reads as "not revealed yet", the correct
    // default, so there's no premature creep before the gate mask exists.
    const gateNode = options?.gateRef
      ? tslTexture(rtA.texture, renderTargetUv)
      : null
    const gate = gateNode
      ? smoothstep(REVEAL_LOW, REVEAL_HIGH, dot(gateNode.rgb, LUMINANCE_WEIGHTS))
      : null
    const gatedFluidLuminance = gate ? fluidLuminance.mul(gate) : fluidLuminance
    const latched = options?.creepGain
      ? tslMin(prevLuminance.add(gatedFluidLuminance.mul(options.creepGain)), float(1))
      : tslMax(prevLuminance, fluidLuminance)

    const mat = new MeshBasicNodeMaterial()
    mat.colorNode = vec4(vec3(latched), 1)

    const scene = new THREE.Scene()
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat)
    mesh.frustumCulled = false
    scene.add(mesh)

    const outputNode = tslTexture(rtB.texture, options?.outputUv ?? renderTargetUv)

    const probeCanvas = document.createElement("canvas")
    probeCanvas.width = 8
    probeCanvas.height = 8
    const probeCtx = probeCanvas.getContext("2d", {
      willReadFrequently: true,
    })
    if (!probeCtx) return

    gl.setRenderTarget(rtA)
    gl.clear()
    gl.setRenderTarget(rtB)
    gl.clear()
    gl.setRenderTarget(null)

    stateRef.current = {
      read: rtA,
      write: rtB,
      scene,
      camera,
      fluidTexture,
      prevNode,
      outputNode,
      gateNode,
      fluidCanvas,
      probeCtx,
      startedAt: Date.now(),
      armed: false,
    }
    setReady(true)

    return () => {
      setReady(false)
      stateRef.current = null
      rtA.dispose()
      rtB.dispose()
      fluidTexture.dispose()
      mat.dispose()
      mesh.geometry.dispose()
    }
  }, [fluidCanvas, gl])

  useFrame(() => {
    const state = stateRef.current
    if (!state) return

    state.fluidTexture.needsUpdate = true

    if (!state.armed) {
      const elapsed = Date.now() - state.startedAt
      const settled =
        elapsed >= WARMUP_MIN_MS &&
        isCanvasSettled(state.fluidCanvas, state.probeCtx)
      const timedOut = elapsed >= WARMUP_TIMEOUT_MS
      if (settled || timedOut) {
        state.armed = true
      } else {
        gl.setRenderTarget(state.read)
        gl.clear()
        gl.setRenderTarget(state.write)
        gl.clear()
        gl.setRenderTarget(null)
        state.outputNode.value = state.write.texture
        return
      }
    }

    state.prevNode.value = state.read.texture
    if (state.gateNode && options?.gateRef?.current) {
      state.gateNode.value = options.gateRef.current
    }

    gl.setRenderTarget(state.write)
    gl.render(state.scene, state.camera)
    gl.setRenderTarget(null)

    state.outputNode.value = state.write.texture
    latchRef.current = state.write.texture

    const nextRead = state.write
    const nextWrite = state.read
    state.read = nextRead
    state.write = nextWrite
  })

  return {
    node: ready ? stateRef.current?.outputNode ?? null : null,
    latchRef,
  }
}

// Fluid layer only — its live canvas is sampled as a luminance mask,
// never rendered on screen itself (opacity: 0, but still receives
// real pointer events since it sits on top of the R3F canvas).
const fluidMaskConfig: ShaderLabConfig = {
  layers: [
    {
      blendMode: "normal",
      compositeMode: "filter",
      maskConfig: { invert: false, mode: "multiply", source: "luminance" },
      hue: 0,
      id: "fluid-mask",
      kind: "source",
      name: "Fluid Mask",
      opacity: 1,
      params: {
        simRes: 64,
        dyeRes: 512,
        iterations: 20,
        densityDissipation: 4.2,
        velocityDissipation: 18,
        pressureDissipation: 0,
        curlStrength: 1.5,
        // Bigger, bolder splats: wider footprint per stroke (radius) and
        // more force behind it, plus a brightness bump so a single pass
        // reliably clears REVEAL_HIGH instead of landing near the threshold
        // and reading as a faint smudge.
        radius: 0.75,
        splatForce: 6000,
        autoSplats: false,
        brightness: 2.2,
        colorMode: "monochrome",
        monoDark: "#000000",
        monoLight: "#ffffff",
        duotoneDark: "#101010",
        duotoneLight: "#f3f3ef",
      },
      saturation: 1,
      type: "fluid",
      visible: true,
    },
  ],
  timeline: { duration: 8, loop: true, tracks: [] },
}

function useImageTexture(src: string) {
  const [texture, setTexture] = useState<THREE.Texture | null>(null)

  useEffect(() => {
    let cancelled = false
    const loader = new THREE.TextureLoader()
    loader.load(src, (tex) => {
      if (cancelled) return
      tex.colorSpace = THREE.SRGBColorSpace
      tex.needsUpdate = true
      setTexture(tex)
    })
    return () => {
      cancelled = true
    }
  }, [src])

  return texture
}

const MODEL_URL = "/models/vinyl_record.glb"
// Names of the glTF materials on the two meshes that make up the record:
// the grooved playing surface, and the small paper label at the center.
const RECORD_SURFACE_MATERIAL_NAME = "tracks"
const RECORD_LABEL_MATERIAL_NAME = "label"
// The label mesh's own UV coordinates are degenerate in this model (a
// single point), so its texture is mapped from local position instead.
// Measured from the GLB: the label mesh's local radius (in the same units
// as the track mesh, whose local radius is 1). LABEL_IMAGE_RADIUS picks
// which UV radius lands on the mesh's outer edge — smaller values zoom
// in tighter so the label artwork fully covers the mesh with no gap of
// surrounding disc pattern showing through, at the cost of slightly
// cropping whatever sits at the very edge of the label graphic.
const LABEL_LOCAL_RADIUS = 0.0507
const LABEL_IMAGE_RADIUS = 0.15
const ROTATE_SPEED = 0.006
const ROTATE_PITCH_LIMIT = Math.PI / 3

type TouchGestureCallbacks = {
  onRotateMove?: (dx: number, dy: number) => void
  onPaintStart?: () => void
}

// Right-drag to rotate the model, implemented by hand rather than via
// OrbitControls: drei's OrbitControls captures the pointer on its
// domElement, which — since it has to share the same element the fluid
// composition listens on for hover-to-paint — silently breaks the fluid's
// own pointer listeners on that element. Plain listeners here coexist fine.
//
// `touchGestureRef.current.onRotateMove` is wired up here so the touch-
// gesture controller (two-finger drag) can drive the exact same rotation
// math a mouse right-drag uses, since touch has no button to gate on.
function useManualOrbit(
  domElement: HTMLCanvasElement | null,
  groupRef: React.RefObject<THREE.Group | null>,
  touchGestureRef: React.RefObject<TouchGestureCallbacks>
) {
  useEffect(() => {
    if (!domElement) return
    let dragging = false
    let lastX = 0
    let lastY = 0

    const applyDelta = (dx: number, dy: number) => {
      if (!groupRef.current) return
      groupRef.current.rotation.y += dx * ROTATE_SPEED
      groupRef.current.rotation.x = THREE.MathUtils.clamp(
        groupRef.current.rotation.x + dy * ROTATE_SPEED,
        -ROTATE_PITCH_LIMIT,
        ROTATE_PITCH_LIMIT
      )
    }

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 2) return
      dragging = true
      lastX = e.clientX
      lastY = e.clientY
    }
    const onPointerMove = (e: PointerEvent) => {
      if (!dragging) return
      const dx = e.clientX - lastX
      const dy = e.clientY - lastY
      lastX = e.clientX
      lastY = e.clientY
      applyDelta(dx, dy)
    }
    const onPointerUp = () => {
      dragging = false
    }
    const onContextMenu = (e: MouseEvent) => {
      e.preventDefault()
    }

    domElement.addEventListener("pointerdown", onPointerDown)
    window.addEventListener("pointermove", onPointerMove)
    window.addEventListener("pointerup", onPointerUp)
    domElement.addEventListener("contextmenu", onContextMenu)
    touchGestureRef.current.onRotateMove = applyDelta

    return () => {
      domElement.removeEventListener("pointerdown", onPointerDown)
      window.removeEventListener("pointermove", onPointerMove)
      window.removeEventListener("pointerup", onPointerUp)
      domElement.removeEventListener("contextmenu", onContextMenu)
      touchGestureRef.current.onRotateMove = undefined
    }
  }, [domElement, groupRef, touchGestureRef])
}

// Renders a true, static, point-in-time GPU copy of `sourceRef`'s current
// texture into its own dedicated render target, only at the start of each
// new left-click stroke (pointerdown) -- unlike a plain JS reference to
// `sourceRef.current`, this copy's destination buffer is owned solely by
// this hook and nothing else ever writes into it between pointerdowns, so
// it genuinely stays frozen. (A first attempt just aliased `sourceRef`'s
// current texture object; that broke, because that object is one of only
// two ping-ponged render targets `sourceRef`'s own owner keeps rendering
// into every frame regardless of pointer state -- "freezing" a reference
// to a buffer someone else keeps overwriting doesn't freeze its pixels.)
// The returned ref's `.current` is a single stable Texture object (this
// hook's own render target's texture) for its whole lifetime -- only its
// GPU contents change, at pointerdown -- so a gate built from it only
// ever reflects "what was already achieved before this stroke began".
function useStrokeSnapshotTexture(
  domElement: HTMLCanvasElement | null,
  sourceRef: { current: THREE.Texture | null },
  touchGestureRef: React.RefObject<TouchGestureCallbacks>
) {
  const { gl: rawGl } = useThree()
  const gl = rawGl as unknown as WebGPURenderer
  const snapshotRef = useRef<THREE.Texture | null>(null)

  const stateRef = useRef<{
    rt: THREE.WebGLRenderTarget
    scene: THREE.Scene
    camera: THREE.OrthographicCamera
    sourceNode: ReturnType<typeof tslTexture>
    pending: boolean
  } | null>(null)

  useEffect(() => {
    const rt = new THREE.WebGLRenderTarget(ACCUM_SIZE, ACCUM_SIZE, ACCUM_RT_OPTIONS)

    // Reads as "nothing revealed" until the first real pointerdown copy --
    // same safe default as gateNode's own pre-gate fallback further down.
    const blackTex = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1)
    blackTex.needsUpdate = true

    const sourceNode = tslTexture(blackTex, renderTargetUv)
    const mat = new MeshBasicNodeMaterial()
    mat.colorNode = vec4(sourceNode.rgb, 1)

    const scene = new THREE.Scene()
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat)
    mesh.frustumCulled = false
    scene.add(mesh)

    gl.setRenderTarget(rt)
    gl.clear()
    gl.setRenderTarget(null)

    stateRef.current = { rt, scene, camera, sourceNode, pending: false }
    snapshotRef.current = rt.texture

    return () => {
      stateRef.current = null
      snapshotRef.current = null
      rt.dispose()
      mat.dispose()
      mesh.geometry.dispose()
      blackTex.dispose()
    }
  }, [gl])

  useEffect(() => {
    if (!domElement) return
    const onPointerDown = (e: PointerEvent) => {
      // Touch paint-start is driven exclusively by the touch-gesture
      // controller's onPaintStart callback below -- a second finger
      // landing also satisfies button===0 and would otherwise re-mark a
      // stroke as pending mid-rotate.
      if (e.pointerType === "touch") return
      if (e.button !== 0) return
      if (stateRef.current) stateRef.current.pending = true
    }
    domElement.addEventListener("pointerdown", onPointerDown)
    touchGestureRef.current.onPaintStart = () => {
      if (stateRef.current) stateRef.current.pending = true
    }
    return () => {
      domElement.removeEventListener("pointerdown", onPointerDown)
      touchGestureRef.current.onPaintStart = undefined
    }
  }, [domElement, touchGestureRef])

  useFrame(() => {
    const state = stateRef.current
    if (!state || !state.pending) return
    if (!sourceRef.current) return // mask1 hasn't produced a frame yet; retry
    state.pending = false
    state.sourceNode.value = sourceRef.current
    gl.setRenderTarget(state.rt)
    gl.render(state.scene, state.camera)
    gl.setRenderTarget(null)
  })

  return snapshotRef
}

// Touch-only gesture controller: one finger paints (scroll is locked via
// `touch-none` on the outer container, see HoverRevealShader), a second
// finger switches to rotate. Mouse/pen are untouched -- they keep using the
// button-gated listeners in useManualOrbit/useStrokeSnapshotTexture above.
// Plain function (not a hook) so this stays textually close to its
// navbardano port (vinyl-reveal.js's createTouchGestureController), which
// this file is the original source of.
//
// Two listener scopes, same split as useManualOrbit's mouse path:
//  - `wrapperEl`, capture phase: decides whether an event may reach the
//    fluid-paint canvas at all (must run before FluidPass's own listener,
//    which is bound directly on that canvas).
//  - `window`: tracks finger positions/deltas so a finger that drags
//    outside the tile's bounds is still tracked, matching mouse behavior.
function createTouchGestureController(
  wrapperEl: HTMLElement,
  canvasEl: HTMLCanvasElement | null,
  { onPaintStart, onRotateMove }: TouchGestureCallbacks
) {
  const touches = new Map<number, { x: number; y: number }>()
  let mode: "idle" | "paint" | "rotate" = "idle"
  let primaryId: number | null = null

  function onWrapperPointerDownCapture(event: PointerEvent) {
    // Mouse pointerdown is left alone here, matching the existing
    // blockHoverPaint rationale: a button is already down when it fires,
    // so it's exactly a drag's first sample, not a hover.
    if (event.pointerType !== "touch") return
    const wasEmpty = touches.size === 0
    touches.set(event.pointerId, { x: event.clientX, y: event.clientY })
    if (wasEmpty) {
      primaryId = event.pointerId
      mode = "paint"
      onPaintStart?.()
      // First finger's own touchdown is a legitimate first stroke sample --
      // let it through.
      return
    }
    if (touches.size === 2) {
      mode = "rotate"
    }
    // Any touchdown beyond the first (the 2nd finger landing, or a stray
    // 3rd+) must never reach the fluid canvas as a paint sample.
    event.stopPropagation()
  }

  function onWrapperPointerMoveCapture(event: PointerEvent) {
    if (event.pointerType !== "touch") {
      if (event.buttons === 0) event.stopPropagation()
      return
    }
    if (mode === "rotate") event.stopPropagation()
  }

  function endTouch(pointerId: number) {
    if (!touches.has(pointerId)) return
    const wasRotating = mode === "rotate"
    touches.delete(pointerId)

    if (touches.size === 0) {
      mode = "idle"
      primaryId = null
      return
    }

    if (wasRotating) {
      // Dropping back to one finger: resume painting immediately with the
      // remaining finger. FluidPass tracks a single shared
      // lastPointerX/lastPointerY, reset to null only on `pointerleave` --
      // since every touch event was blocked from it during rotate, that
      // state is stale (last real value came from the primary finger,
      // before the 2nd finger landed). A synthetic pointerleave nulls it,
      // so the remaining finger's next real pointermove just re-anchors
      // position instead of diffing against stale coordinates and
      // splatting a spurious jump.
      canvasEl?.dispatchEvent(
        new PointerEvent("pointerleave", { bubbles: false, cancelable: false })
      )
      const remaining = touches.keys().next().value as number
      primaryId = remaining
      mode = "paint"
    }
  }

  function onWindowPointerMove(event: PointerEvent) {
    if (event.pointerType !== "touch") return
    const last = touches.get(event.pointerId)
    if (!last) return
    const dx = event.clientX - last.x
    const dy = event.clientY - last.y
    touches.set(event.pointerId, { x: event.clientX, y: event.clientY })
    if (mode === "rotate" && event.pointerId === primaryId) {
      onRotateMove?.(dx, dy)
    }
  }

  function onWindowPointerUp(event: PointerEvent) {
    if (event.pointerType !== "touch") return
    endTouch(event.pointerId)
  }

  // Capture phase on window too, and not just for symmetry with the
  // wrapper listeners: window precedes the wrapper in the capture
  // traversal (window -> document -> ... -> wrapperEl -> target), so a
  // capture-phase listener here always sees the event before
  // onWrapperPointerMoveCapture's stopPropagation (during rotate) can cut
  // the dispatch short. A bubble-phase listener here would never fire in
  // that case, since stopPropagation during capture halts the entire
  // dispatch, including the later bubble phase back up to window.
  wrapperEl.addEventListener("pointerdown", onWrapperPointerDownCapture, true)
  wrapperEl.addEventListener("pointermove", onWrapperPointerMoveCapture, true)
  window.addEventListener("pointermove", onWindowPointerMove, true)
  window.addEventListener("pointerup", onWindowPointerUp, true)
  window.addEventListener("pointercancel", onWindowPointerUp, true)

  return function dispose() {
    wrapperEl.removeEventListener("pointerdown", onWrapperPointerDownCapture, true)
    wrapperEl.removeEventListener("pointermove", onWrapperPointerMoveCapture, true)
    window.removeEventListener("pointermove", onWindowPointerMove, true)
    window.removeEventListener("pointerup", onWindowPointerUp, true)
    window.removeEventListener("pointercancel", onWindowPointerUp, true)
  }
}

function VinylRecord({
  layer1Src,
  layer2Src,
  layer3Src,
  maskCanvas,
  touchGestureRef,
}: {
  layer1Src: string
  layer2Src: string
  layer3Src: string
  maskCanvas: HTMLCanvasElement | null
  touchGestureRef: React.RefObject<TouchGestureCallbacks>
}) {
  const layer1Texture = useImageTexture(layer1Src)
  const layer2Texture = useImageTexture(layer2Src)
  const layer3Texture = useImageTexture(layer3Src)

  // mask1 paints layer1 -> layer2 live, every frame, for as long as the
  // current stroke drags across still-foggy area. mask2 paints layer2 ->
  // layer3, but its gate is a true GPU snapshot of mask1 taken at the
  // moment the current stroke began (useStrokeSnapshotTexture) rather
  // than mask1's live value -- so this is a genuine second, separate
  // trigger: dragging layer1 all the way to a fully-revealed layer2
  // within one continuous stroke still can't unlock mask2 mid-stroke,
  // since the snapshot mask2 is gated on was copied before that progress
  // happened. Only the next stroke (a fresh pointerdown) re-snapshots,
  // now seeing what the previous stroke actually finished. Both masks use
  // the exact same additive creep mechanism at the exact same rate
  // (CREEP_GAIN) -- so once triggered, either reveal takes the same
  // amount of dragging and feels like the identical gesture. It's the
  // stroke-snapshot gate alone, not a speed difference, that makes the
  // second reveal a deliberate separate act. The revealOpacity2 multiply
  // further down is a second, belt-and-suspenders guarantee on top of the
  // gate -- layer3 still can't render ahead of layer2 even at the
  // smoothstep edge where the gate is only fractional.
  const mask1 = useLatchedMask(maskCanvas, {
    outputUv: viewportMaskUv,
    creepGain: CREEP_GAIN,
  })
  const mask2GateRef = useStrokeSnapshotTexture(maskCanvas, mask1.latchRef, touchGestureRef)
  const mask2 = useLatchedMask(maskCanvas, {
    outputUv: viewportMaskUv,
    creepGain: CREEP_GAIN,
    gateRef: mask2GateRef,
  })

  const { scene } = useGLTF(MODEL_URL)

  const buildRevealMaterial = (uvNode: ReturnType<typeof vec2>) => {
    if (
      !layer1Texture ||
      !layer2Texture ||
      !layer3Texture ||
      !mask1.node ||
      !mask2.node
    )
      return null

    const mat = new MeshBasicNodeMaterial()
    const layer1Node = tslTexture(layer1Texture, uvNode)
    const layer2Node = tslTexture(layer2Texture, uvNode)
    const layer3Node = tslTexture(layer3Texture, uvNode)
    const revealOpacity1 = smoothstep(REVEAL_LOW, REVEAL_HIGH, mask1.node.r)
    const revealOpacity2raw = smoothstep(REVEAL_LOW, REVEAL_HIGH, mask2.node.r)
    // Hard guarantee that layer3 can only be as visible here as layer1
    // already is, on top of the stroke-snapshot gate -- this is what
    // makes the ordering unconditional rather than just the usual case.
    // Since x*x < x for any x in (0,1), gating revealOpacity2 by
    // revealOpacity1 also naturally keeps layer3 lagging visibly behind
    // layer2 throughout the reveal, not just clamped at the very end.
    const revealOpacity2 = revealOpacity2raw.mul(revealOpacity1)
    const stage1 = mix(layer1Node.rgb, layer2Node.rgb, revealOpacity1)
    const stage2 = mix(stage1, layer3Node.rgb, revealOpacity2)
    mat.colorNode = vec4(stage2, 1)
    return mat
  }

  const surfaceMaterial = useMemo(
    () => buildRevealMaterial(uv()),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [layer1Texture, layer2Texture, layer3Texture, mask1.node, mask2.node]
  )

  // The label's own UV attribute is degenerate, so build its texture
  // coordinate from local vertex position instead, scaled to land on the
  // same central patch of the composite image that the label artwork
  // actually occupies.
  const labelMaterial = useMemo(
    () =>
      buildRevealMaterial(
        vec2(
          positionLocal.x,
          positionLocal.y.mul(-1)
        )
          .div(LABEL_LOCAL_RADIUS)
          .mul(LABEL_IMAGE_RADIUS)
          .add(0.5)
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [layer1Texture, layer2Texture, layer3Texture, mask1.node, mask2.node]
  )

  const preparedScene = useMemo(() => {
    const cloned = scene.clone(true)
    cloned.traverse((child) => {
      const mesh = child as THREE.Mesh
      if (!mesh.isMesh || !mesh.material) return
      const name = (mesh.material as THREE.Material).name
      if (name === RECORD_SURFACE_MATERIAL_NAME && surfaceMaterial) {
        mesh.material = surfaceMaterial
      } else if (name === RECORD_LABEL_MATERIAL_NAME && labelMaterial) {
        mesh.material = labelMaterial
      }
    })
    return cloned
  }, [scene, surfaceMaterial, labelMaterial])

  const { camera } = useThree()
  const groupRef = useRef<THREE.Group>(null)
  useManualOrbit(maskCanvas, groupRef, touchGestureRef)

  useEffect(() => {
    const box = new THREE.Box3().setFromObject(preparedScene)
    const size = box.getSize(new THREE.Vector3())
    const center = box.getCenter(new THREE.Vector3())
    const maxDim = Math.max(size.x, size.y, size.z) || 1
    const perspective = camera as THREE.PerspectiveCamera
    const fovRad = (perspective.fov * Math.PI) / 180
    const distance = (maxDim / 2 / Math.tan(fovRad / 2)) * 1.5

    // The model is re-centered at the world origin below, so the camera
    // orbits/looks around origin rather than the model's original offset.
    camera.position.set(0, maxDim * 0.18, distance)
    camera.near = distance / 100
    camera.far = distance * 10
    camera.lookAt(0, 0, 0)
    perspective.updateProjectionMatrix()

    if (groupRef.current) {
      groupRef.current.position.set(-center.x, -center.y, -center.z)
    }
  }, [preparedScene, camera])

  if (!surfaceMaterial) return null

  return (
    <group ref={groupRef}>
      <primitive object={preparedScene} />
    </group>
  )
}
useGLTF.preload(MODEL_URL)

export function HoverRevealShader({
  layer1Src,
  layer2Src,
  layer3Src,
}: {
  layer1Src: string
  layer2Src: string
  layer3Src: string
}) {
  const fluidWrapperRef = useRef<HTMLDivElement>(null)
  const [maskCanvas, setMaskCanvas] = useState<HTMLCanvasElement | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const touchGestureRef = useRef<TouchGestureCallbacks>({})

  useEffect(() => {
    const el = fluidWrapperRef.current?.querySelector("canvas") ?? null
    setMaskCanvas(el)
  }, [])

  // The fluid library's own FluidPass (node_modules/@basementstudio/
  // shader-lab) splats on every pointermove/pointerdown it sees, regardless
  // of button state -- there's no config for "only while dragging".
  // createTouchGestureController owns the wrapper-capture interception that
  // gates this (mouse hover-paint gating + touch 1-finger-paint/2-finger-
  // rotate disambiguation), replacing the old standalone blockHoverPaint.
  useEffect(() => {
    const wrapper = fluidWrapperRef.current
    if (!wrapper) return
    const dispose = createTouchGestureController(wrapper, maskCanvas, {
      onRotateMove: (dx, dy) => touchGestureRef.current.onRotateMove?.(dx, dy),
      onPaintStart: () => touchGestureRef.current.onPaintStart?.(),
    })
    return dispose
  }, [maskCanvas])

  return (
    <div className="relative aspect-square w-full touch-none overscroll-none overflow-hidden rounded-lg border border-neutral-800 bg-black">
      <div
        ref={fluidWrapperRef}
        className="absolute inset-0 z-10 opacity-0"
        aria-hidden
      >
        <ShaderLabComposition
          config={fluidMaskConfig}
          onRuntimeError={setError}
        />
      </div>

      {maskCanvas && (
        <Canvas
          className="absolute inset-0"
          onCreated={() => setReady(true)}
          gl={async (props) => {
            const renderer = new WebGPURenderer(
              props as ConstructorParameters<typeof WebGPURenderer>[0]
            )
            await renderer.init()
            return renderer
          }}
          camera={{ position: [0, 0.9, 2.4], fov: 35 }}
        >
          <ambientLight intensity={0.7} />
          <directionalLight position={[2, 3, 4]} intensity={1.4} />
          <directionalLight position={[-2, -1, -3]} intensity={0.4} />
          <Suspense fallback={null}>
            <VinylRecord
              layer1Src={layer1Src}
              layer2Src={layer2Src}
              layer3Src={layer3Src}
              maskCanvas={maskCanvas}
              touchGestureRef={touchGestureRef}
            />
          </Suspense>
        </Canvas>
      )}

      {!ready && !error && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-xs text-neutral-600">
          Loading shader…
        </div>
      )}

      {error && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-red-950/80 p-4 text-center text-sm text-red-300">
          {error}
        </div>
      )}
    </div>
  )
}
