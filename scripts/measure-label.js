const fs = require("fs")
const zlib = require("zlib")

function decodePNG(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error("not a png")
  let offset = 8
  let width, height, bitDepth, colorType
  const idatChunks = []
  while (offset < buf.length) {
    const length = buf.readUInt32BE(offset)
    const type = buf.toString("ascii", offset + 4, offset + 8)
    const data = buf.slice(offset + 8, offset + 8 + length)
    if (type === "IHDR") {
      width = data.readUInt32BE(0)
      height = data.readUInt32BE(4)
      bitDepth = data[8]
      colorType = data[9]
    } else if (type === "IDAT") {
      idatChunks.push(data)
    }
    offset += 8 + length + 4
  }
  const raw = zlib.inflateSync(Buffer.concat(idatChunks))
  const channels = colorType === 6 ? 4 : colorType === 2 ? 3 : 1
  const bpp = (channels * bitDepth) / 8
  const stride = width * bpp
  const pixels = Buffer.alloc(height * stride)
  let rawOffset = 0
  for (let y = 0; y < height; y++) {
    const filterType = raw[rawOffset]
    rawOffset += 1
    const rowStart = y * stride
    for (let x = 0; x < stride; x++) {
      const raw_x = raw[rawOffset + x]
      const a = x >= bpp ? pixels[rowStart + x - bpp] : 0
      const b = y > 0 ? pixels[rowStart - stride + x] : 0
      const c = y > 0 && x >= bpp ? pixels[rowStart - stride + x - bpp] : 0
      let value
      if (filterType === 0) value = raw_x
      else if (filterType === 1) value = (raw_x + a) & 0xff
      else if (filterType === 2) value = (raw_x + b) & 0xff
      else if (filterType === 3) value = (raw_x + Math.floor((a + b) / 2)) & 0xff
      else if (filterType === 4) {
        const p = a + b - c
        const pa = Math.abs(p - a)
        const pb = Math.abs(p - b)
        const pc = Math.abs(p - c)
        const pr = pa <= pb && pa <= pc ? a : pb <= pc ? b : c
        value = (raw_x + pr) & 0xff
      } else value = raw_x
      pixels[rowStart + x] = value
    }
    rawOffset += stride
  }
  return { width, height, channels, pixels }
}

const img = decodePNG(fs.readFileSync(process.argv[2]))
const { width, height, channels, pixels } = img
const cy = Math.floor(height / 2)

function getPixel(x, y) {
  const idx = (y * width + x) * channels
  return {
    r: pixels[idx],
    g: pixels[idx + 1],
    b: pixels[idx + 2],
    a: channels === 4 ? pixels[idx + 3] : 255,
  }
}

// Scan horizontal center line, find disc outer edge (alpha transition) and label edge (color becomes near-white/flat)
let discLeft = -1, discRight = -1
for (let x = 0; x < width; x++) {
  const p = getPixel(x, cy)
  if (p.a > 128) { discLeft = x; break }
}
for (let x = width - 1; x >= 0; x--) {
  const p = getPixel(x, cy)
  if (p.a > 128) { discRight = x; break }
}
const discCenterX = (discLeft + discRight) / 2
const discRadius = (discRight - discLeft) / 2

// Find label edges using a column-averaged sample (a window of rows around
// cy) so thin text strokes/spindle hole don't create false transitions.
function isLabelish(p) {
  const max = Math.max(p.r, p.g, p.b)
  const min = Math.min(p.r, p.g, p.b)
  const sat = max === 0 ? 0 : (max - min) / max
  return max > 170 && sat < 0.3
}
function columnLabelFraction(x) {
  let count = 0
  let total = 0
  for (let dy = -40; dy <= 40; dy += 4) {
    const y = cy + dy
    if (y < 0 || y >= height) continue
    total++
    if (isLabelish(getPixel(x, y))) count++
  }
  return count / total
}

let labelLeft = -1
for (let x = Math.floor(discCenterX); x >= discLeft; x--) {
  if (columnLabelFraction(x) < 0.6) { labelLeft = x + 1; break }
}
let labelRight = -1
for (let x = Math.floor(discCenterX); x <= discRight; x++) {
  if (columnLabelFraction(x) < 0.6) { labelRight = x - 1; break }
}

const labelRadius = (labelRight - labelLeft) / 2
const labelCenterX = (labelLeft + labelRight) / 2

console.log(JSON.stringify({
  width, height,
  discLeft, discRight, discCenterX, discRadius,
  labelLeft, labelRight, labelCenterX, labelRadius,
  labelToDiscRadiusRatio: labelRadius / discRadius,
}, null, 2))
