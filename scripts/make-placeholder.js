const fs = require("fs")
const path = require("path")
const zlib = require("zlib")

function crc32(buf) {
  let c
  const table = crc32.table || (crc32.table = (() => {
    const t = new Uint32Array(256)
    for (let n = 0; n < 256; n++) {
      c = n
      for (let k = 0; k < 8; k++) {
        c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
      }
      t[n] = c >>> 0
    }
    return t
  })())
  let crc = 0xffffffff
  for (let i = 0; i < buf.length; i++) {
    crc = table[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ 0xffffffff) >>> 0
}

function chunk(type, data) {
  const typeBuf = Buffer.from(type, "ascii")
  const lenBuf = Buffer.alloc(4)
  lenBuf.writeUInt32BE(data.length, 0)
  const crcBuf = Buffer.alloc(4)
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0)
  return Buffer.concat([lenBuf, typeBuf, data, crcBuf])
}

function makePng(width, height, pixelFn) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])

  const ihdrData = Buffer.alloc(13)
  ihdrData.writeUInt32BE(width, 0)
  ihdrData.writeUInt32BE(height, 4)
  ihdrData[8] = 8 // bit depth
  ihdrData[9] = 2 // color type RGB
  ihdrData[10] = 0
  ihdrData[11] = 0
  ihdrData[12] = 0
  const ihdr = chunk("IHDR", ihdrData)

  const raw = Buffer.alloc((width * 3 + 1) * height)
  let offset = 0
  for (let y = 0; y < height; y++) {
    raw[offset++] = 0 // filter type none
    for (let x = 0; x < width; x++) {
      const [r, g, b] = pixelFn(x, y, width, height)
      raw[offset++] = r
      raw[offset++] = g
      raw[offset++] = b
    }
  }
  const idatData = zlib.deflateSync(raw, { level: 9 })
  const idat = chunk("IDAT", idatData)
  const iend = chunk("IEND", Buffer.alloc(0))

  return Buffer.concat([sig, ihdr, idat, iend])
}

function lerp(a, b, t) {
  return a + (b - a) * t
}

// Soft radial gradient portrait-ish placeholder, warm center to dark edges.
const png = makePng(1024, 1024, (x, y, w, h) => {
  const cx = w / 2
  const cy = h / 2
  const dx = (x - cx) / (w / 2)
  const dy = (y - cy) / (h / 2)
  const dist = Math.min(1, Math.sqrt(dx * dx + dy * dy))

  const inner = [235, 200, 150]
  const outer = [20, 18, 24]

  const r = Math.round(lerp(inner[0], outer[0], dist))
  const g = Math.round(lerp(inner[1], outer[1], dist))
  const b = Math.round(lerp(inner[2], outer[2], dist))
  return [r, g, b]
})

const outPath = path.join(
  __dirname,
  "..",
  "public",
  "replace",
  "image",
  "Gemini_Generated_Image_ku5z95ku5z95ku5z.png"
)
fs.writeFileSync(outPath, png)
console.log("wrote", outPath, png.length, "bytes")
