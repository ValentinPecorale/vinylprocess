"use client"

import { HoverRevealShader } from "@/components/hover-reveal-shader"

export default function Home() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <header className="border-b border-neutral-800 px-6 py-4">
        <h1 className="text-sm font-medium tracking-wide text-neutral-400">
          Shader Lab Playground
        </h1>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-4 px-6 py-10">
        <p className="text-sm text-neutral-500">
          Left-click and drag to paint the record and reveal the second
          image, then keep going to unlock the third once the second is
          fully revealed. Right-click and drag to rotate.
        </p>
        <HoverRevealShader
          layer1Src="/images/reveal-layer1.png"
          layer2Src="/images/reveal-layer2.png"
          layer3Src="/images/reveal-layer3.png"
        />
      </main>
    </div>
  )
}
