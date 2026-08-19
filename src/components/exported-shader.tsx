"use client"

import { ShaderLabComposition, type ShaderLabConfig } from "@basementstudio/shader-lab"

const config: ShaderLabConfig = {
  layers: [
    {
      blendMode: "normal",
      compositeMode: "filter",
      maskConfig: {
        invert: false,
        mode: "multiply",
        source: "luminance",
      },
      hue: 0,
      id: "9d1ca270-4f13-4746-8c94-94c467b47466",
      kind: "effect",
      name: "Progressive Blur",
      opacity: 1,
      params: {
        angle: 0,
        start: 0,
        end: 0,
        strength: 60,
        samples: 32,
      },
      saturation: 1,
      type: "smear",
      visible: true,
    },
    {
      blendMode: "normal",
      compositeMode: "mask",
      maskConfig: {
        invert: false,
        mode: "multiply",
        source: "luminance",
      },
      hue: 0,
      id: "d7e08b71-e934-43ee-84db-9383699b07df",
      kind: "source",
      name: "Fluid",
      opacity: 1,
      params: {
        simRes: 192,
        dyeRes: 1024,
        iterations: 20,
        densityDissipation: 6.57,
        velocityDissipation: 1.97,
        pressureDissipation: 0,
        curlStrength: 30,
        radius: 2,
        splatForce: 2900,
        autoSplats: false,
        brightness: 1.6,
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
    {
      blendMode: "normal",
      compositeMode: "filter",
      maskConfig: {
        invert: false,
        mode: "multiply",
        source: "luminance",
      },
      hue: 0,
      id: "526a451e-3065-4841-80b0-a3909a5e5c3c",
      kind: "source",
      name: "Image",
      opacity: 1,
      params: {
        fitMode: "cover",
        scale: 1,
        offset: [0, 0],
        svgRasterResolution: "2048",
      },
      saturation: 1,
      type: "image",
      visible: true,
      asset: {
        fileName: "Gemini_Generated_Image_ku5z95ku5z95ku5z.png",
        kind: "image",
        src: "/replace/image/Gemini_Generated_Image_ku5z95ku5z95ku5z.png",
      },
    },
  ],
  timeline: {
    duration: 8,
    loop: true,
    tracks: [],
  },
}

export function ExportedShader({
  onRuntimeError,
}: {
  onRuntimeError?: (message: string | null) => void
}) {
  return (
    <ShaderLabComposition config={config} onRuntimeError={onRuntimeError} />
  )
}
