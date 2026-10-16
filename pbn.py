"""pbn.py - turn a photo into a printable paint-by-numbers PDF.

Pipeline
  1. shrink + denoise the photo
  2. k-means colour quantisation in CIELAB (perceptual) -> N paint colours
  3. smooth the label map and dissolve regions too small to paint
  4. upscale the label map to print resolution with smooth boundaries
  5. trace region outlines, place numbers at the most "open" point of each region
  6. render a PDF: page 1 = line art with numbers, page 2 = colour key + reference

CLI:  python pbn.py photo.jpg -o out.pdf --colors 16 --detail medium
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from reportlab.lib.pagesizes import A4, landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2
from skimage import color as skcolor
from skimage import measure, segmentation, transform

try:  # iPhone HEIC photos (optional dependency)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

# work_long   : processing resolution (long side, px)
# min_radius  : smallest inscribed radius a region may have at work resolution (px)
# median      : median-filter size used to denoise before quantising
# sigma       : label-smoothing strength at work resolution
DETAIL_PRESETS = {
    "low": dict(work_long=640, min_radius=7.0, median=7, sigma=1.6),
    "medium": dict(work_long=900, min_radius=5.5, median=5, sigma=1.2),
    "high": dict(work_long=1200, min_radius=4.5, median=3, sigma=0.9),
}
HI_LONG = 2400  # resolution of the printed line art (long side, px)
PAPER = {"letter": letter, "a4": A4}
MARGIN = 0.4 * inch
FOOTER = 0.3 * inch
LINE_GRAY = 110  # 0=black .. 255=white, colour of the outlines
NUMBER_GRAY = 0.45  # reportlab gray level for the numbers
MAX_FONT_PT = 13.0
MIN_FONT_PT = 4.0

FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


@dataclass
class Result:
    outline: Image.Image  # hi-res "L" image, white background, gray outlines
    numbered: Image.Image  # outline with numbers drawn in (preview)
    colored: Image.Image  # quantised reference image, hi-res RGB
    palette: list[tuple[int, int, int]]  # palette[i] is the colour for number i+1
    numbers: list[tuple[float, float, float, int]]  # (x, y, radius, number) in hi-res px
    paper: str
    labels: list[str] | None = None  # text printed for number i+1 (kit pot numbers); None = "1".."N"
    names: list[str] | None = None  # paint names from the kit, parallel to palette
    faces: int = 0  # faces found by find_faces (human + cat)

    def label(self, num: int) -> str:
        return self.labels[num - 1] if self.labels else str(num)

    @property
    def size(self) -> tuple[int, int]:
        return self.outline.size


# --------------------------------------------------------------------------- helpers
def load_image(src) -> Image.Image:
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _fit(img: Image.Image, long_side: int) -> Image.Image:
    w, h = img.size
    s = long_side / max(w, h)
    if s >= 1:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def page_geometry(size: tuple[int, int], paper: str):
    """Return (page_w, page_h, scale, x0, y0) for drawing an image of `size` on the page."""
    pw, ph = PAPER[paper]
    W, H = size
    if W > H:
        pw, ph = landscape((pw, ph))
    avail_w, avail_h = pw - 2 * MARGIN, ph - 2 * MARGIN - FOOTER
    s = min(avail_w / W, avail_h / H)
    dw, dh = W * s, H * s
    x0 = (pw - dw) / 2
    y0 = MARGIN + FOOTER + (avail_h - dh) / 2
    return pw, ph, s, x0, y0


def font_size_pt(radius_px: float, scale: float) -> float:
    """Font size so that a 2-digit number fits inside a circle of `radius_px` (hi-res px)."""
    return min(MAX_FONT_PT, 1.25 * radius_px * scale)


# --------------------------------------------------------------------------- core steps
DETAIL_WEIGHT = 12.0  # how much more a detail-mask pixel counts when choosing / keeping colours


YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
_YUNET: object | None | bool = None


def _yunet():
    """OpenCV's YuNet face detector, loaded once. False if OpenCV or the model is missing."""
    global _YUNET
    if _YUNET is None:
        try:
            import cv2  # optional dependency: face detection only

            _YUNET = cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", (320, 320), 0.6, 0.3, 50) if YUNET_MODEL.is_file() else False
        except (ImportError, AttributeError, Exception):  # noqa: BLE001 - any failure just disables faces
            _YUNET = False
    return _YUNET


def find_faces(rgb: np.ndarray) -> list[dict]:
    """Human faces in the working-resolution image, with eye positions.

    Returns [{"kind": "human", "box": (x, y, w, h), "eyes": [(x, y, w, h), (x, y, w, h)]}].
    YuNet gives the two eye centres as landmarks; each becomes a box sized to the face.
    Empty list when OpenCV or the model is missing. Pets are not detected by this model;
    their eyes rely on the contrast-based detail mask."""
    det = _yunet()
    if not det:
        return []
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, found = det.detect(bgr)
    faces: list[dict] = []
    for f in [] if found is None else found:
        x, y, fw, fh = (int(v) for v in f[:4])
        if fw < 0.04 * max(w, h):
            continue  # too small to paint an eye at 8x10 anyway
        ew, eh = max(6, int(fw * 0.26)), max(4, int(fw * 0.16))
        eyes = [(int(f[i]) - ew // 2, int(f[i + 1]) - eh // 2, ew, eh) for i in (4, 6)]
        faces.append(dict(kind="human", box=(x, y, fw, fh), eyes=eyes, score=float(f[14])))
    return faces


def _face_eye_mask(shape, faces: list[dict], dark: np.ndarray, bright: np.ndarray) -> np.ndarray:
    """Force the eye area of every detected face into the detail mask.

    Detected eye boxes (padded a little) go in outright. Otherwise the eye band of the face
    (20-55 % of its height for humans, 30-60 % for cats) is searched with half the normal
    top-hat thresholds; if even that finds nothing, the band itself is protected."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for f in faces:
        x, y, w, h = f["box"]
        if f["eyes"]:
            for ex, ey, ew, eh in f["eyes"]:
                px, py = int(ew * 0.25), int(eh * 0.25)
                m[max(0, ey - py) : min(H, ey + eh + py), max(0, ex - px) : min(W, ex + ew + px)] = True
            continue
        b0, b1 = (0.20, 0.55) if f["kind"] == "human" else (0.30, 0.60)
        ys, xs = slice(y + int(h * b0), min(H, y + int(h * b1))), slice(x + int(w * 0.1), min(W, x + int(w * 0.9)))
        band = (dark[ys, xs] > 11) | (bright[ys, xs] > 14)
        band = ndi.binary_opening(band, iterations=1)
        if band.any():
            m[ys, xs] |= ndi.binary_dilation(band, iterations=2)
        else:
            m[ys, xs] = True
    return m


def detail_mask(rgb: np.ndarray, long_side: int, faces: list[dict] | None = None) -> np.ndarray:
    """Where the small, high-contrast features live: eyes, pupils, catchlights, nostrils, mouths.

    Top-hat transforms on lightness pick out dark or bright features smaller than ~6% of the
    picture; we then protect those areas from the smoothing and dissolving that would
    otherwise blur an eye into the fur or skin around it. Detected faces (see find_faces)
    get their eye area added regardless of contrast."""
    L = skcolor.rgb2lab(rgb.astype(np.float32) / 255.0)[..., 0]
    long_side = max(L.shape)  # the picture may be smaller than the preset asks for
    size = max(7, int(round(0.03 * long_side)) | 1)
    dark = ndi.grey_closing(L, size=(size, size)) - L  # dark features on a lighter ground
    bright = L - ndi.grey_opening(L, size=(size, size))  # bright features on a darker ground
    m = (dark > 22) | (bright > 28)
    m = ndi.binary_opening(m, iterations=1)  # drop single-pixel noise
    # keep only compact blobs: a pupil, an eye-white, a nostril. Fur, tweed and text merge into
    # big sprawling components; dust is tiny. Both are left alone.
    comp, n = measure.label(m, connectivity=2, return_num=True)
    if n:
        area = np.bincount(comp.ravel(), minlength=n + 1)
        lo, hi = (0.004 * long_side) ** 2, (0.09 * long_side) ** 2
        keep = (area >= lo) & (area <= hi)
        keep[0] = False
        m = keep[comp]
    m = ndi.binary_dilation(m, iterations=max(2, size // 4))
    if faces:
        m |= _face_eye_mask(m.shape, faces, dark, bright)
    return m


def _sample_lab(lab: np.ndarray, mask: np.ndarray | None, seed: int) -> np.ndarray:
    """Pixels to run k-means on: a uniform sample plus an extra helping of detail pixels so
    that eyes (tiny by pixel count) still earn their own colours."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lab), size=min(len(lab), 50_000), replace=False)
    if mask is not None and mask.any():
        d = np.flatnonzero(mask.ravel())
        idx = np.concatenate([idx, rng.choice(d, size=min(len(d), 15_000), replace=False)])
    return lab[idx]


def quantize(rgb: np.ndarray, k: int, seed: int = 0, mask: np.ndarray | None = None):
    """K-means in CIELAB. Returns label map (H, W) with values in 0..k-1."""
    lab = skcolor.rgb2lab(rgb.astype(np.float32) / 255.0).reshape(-1, 3).astype(np.float32)
    cents, _ = kmeans2(_sample_lab(lab, mask, seed), k, minit="++", seed=seed, iter=30)
    return _assign(lab, cents).reshape(rgb.shape[:2])


def load_paints(path) -> list[dict]:
    """Read a paint kit file: {"name": ..., "paints": [{"number": 1, "name": "...", "hex": "#RRGGBB"}, ...]}.

    Returns the paint list with an added "rgb" tuple per entry, in file order."""
    data = json.loads(Path(path).read_text())
    paints = data["paints"] if isinstance(data, dict) else data
    out = []
    for i, p in enumerate(paints):
        h = p["hex"].lstrip("#")
        rgb = tuple(int(h[j : j + 2], 16) for j in (0, 2, 4))
        out.append(dict(number=p.get("number", i + 1), name=p.get("name", ""), hex="#" + h.upper(), rgb=rgb))
    return out


def quantize_to_paints(rgb: np.ndarray, paints: list[dict], k: int, seed: int = 0, mask: np.ndarray | None = None):
    """Like quantize(), but every pixel ends up on one of at most k real paints from the kit.

    Every pixel first goes to its nearest pot in CIELAB (so each gets its true best match), then
    pots are retired one at a time, least-used first, until at most k remain; a retired pot's
    pixels move to their nearest surviving pot. Pixels in the detail mask (eyes etc.) count
    DETAIL_WEIGHT times, so the few pixels of a pupil or eye-white keep their pot.
    Returns (labels, chosen) where labels index into `chosen`, a list of paint dicts."""
    k = max(1, min(k, len(paints)))
    lab = skcolor.rgb2lab(rgb.astype(np.float32) / 255.0).reshape(-1, 3).astype(np.float32)
    plab = skcolor.rgb2lab(np.array([p["rgb"] for p in paints], dtype=np.float32)[None] / 255.0)[0]
    plab = plab.astype(np.float32)
    # Decide which pots to keep on a sample (all detail pixels + a uniform draw), by retiring the
    # pot whose loss adds the least total colour error; then assign every pixel to the survivors.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lab), size=min(len(lab), 120_000), replace=False)
    w = np.ones(len(idx), dtype=np.float32)
    if mask is not None and mask.any():
        d = np.flatnonzero(mask.ravel())
        idx = np.concatenate([idx, d])
        w = np.concatenate([w, np.full(len(d), DETAIL_WEIGHT, dtype=np.float32)])
    S = lab[idx]
    D = np.sqrt(np.maximum(((S[:, None, :] - plab[None, :, :]) ** 2).sum(2), 0))  # (n, pots)
    alive = np.ones(len(paints), dtype=bool)
    while alive.sum() > k:
        Da = np.where(alive[None, :], D, np.inf)
        best = Da.argmin(1)
        part = np.partition(Da, 1, axis=1)
        extra = (part[:, 1] - part[:, 0]) * w  # error added if this pixel loses its best pot
        cost = np.bincount(best, weights=extra, minlength=len(paints))
        cost[~alive] = np.inf
        alive[int(cost.argmin())] = False
    survivors = np.flatnonzero(alive)
    pot = survivors[_assign(lab, plab[survivors])]
    used = np.bincount(pot, minlength=len(paints)) > 0
    survivors = np.flatnonzero(alive & used)
    remap = np.full(len(paints), -1, dtype=np.int32)
    remap[survivors] = np.arange(len(survivors))
    chosen = [paints[i] for i in survivors]
    return remap[pot].reshape(rgb.shape[:2]), chosen


def _assign(pts: np.ndarray, cents: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(len(pts), dtype=np.int32)
    c2 = (cents**2).sum(1)
    for i in range(0, len(pts), chunk):
        p = pts[i : i + chunk]
        d = c2[None, :] - 2.0 * (p @ cents.T)  # |p|^2 is constant per row, skip it
        out[i : i + chunk] = d.argmin(1)
    return out


def smooth_labels(
    labels: np.ndarray, k: int, sigma: float = 0.0, out_shape=None, detail: np.ndarray | None = None
) -> np.ndarray:
    """Soft majority vote: blur (and optionally upscale) each class mask, take the argmax.

    This rounds off jagged pixel edges and gives smooth, paintable boundaries. Inside the
    `detail` mask (same shape as the output) the blur is much weaker so eyes keep their shape.
    """
    if detail is not None and detail.any() and sigma > 0:
        coarse = smooth_labels(labels, k, sigma, out_shape)
        fine = smooth_labels(labels, k, sigma * 0.35, out_shape)
        return np.where(detail, fine, coarse)
    best = None
    out = None
    for i in range(k):
        m = (labels == i).astype(np.float32)
        if not m.any():
            continue
        if out_shape is not None and tuple(out_shape) != m.shape:
            m = transform.resize(m, out_shape, order=1, anti_aliasing=False, preserve_range=True)
            m = m.astype(np.float32)
        if sigma > 0:
            m = ndi.gaussian_filter(m, sigma)
        if best is None:
            best = m
            out = np.full(m.shape, i, dtype=np.int32)
        else:
            better = m > best
            out[better] = i
            np.maximum(best, m, out=best)
    return out


def _boundaries(comp: np.ndarray) -> np.ndarray:
    """Pixels on a region boundary. The picture edge counts as a boundary too, so
    numbers never land on the edge and thin strips along the edge get dissolved."""
    b = segmentation.find_boundaries(comp, mode="inner")
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    return b


def dissolve_small_regions(
    labels: np.ndarray, min_radius: float, min_area: float, max_passes: int = 5, protect: np.ndarray | None = None
):
    """Merge regions that are too small or too thin to paint into their neighbours.

    Regions that lie mostly inside `protect` (the detail mask) are allowed to be half the
    radius / a quarter of the area, so pupils, catchlights and eye-whites survive."""
    lab = labels.copy()
    for _ in range(max_passes):
        comp, n = measure.label(lab, connectivity=1, background=-1, return_num=True)
        if n <= 1:
            break
        idx = np.arange(1, n + 1)
        area = np.bincount(comp.ravel(), minlength=n + 1)[1:]
        edt = ndi.distance_transform_edt(~_boundaries(comp))
        radius = np.asarray(ndi.maximum(edt, comp, idx)) + 0.5
        r_min = np.full(n, min_radius)
        a_min = np.full(n, min_area)
        if protect is not None:
            inside = np.bincount(comp.ravel(), weights=protect.ravel(), minlength=n + 1)[1:] / area
            r_min[inside > 0.5] *= 0.5
            a_min[inside > 0.5] *= 0.25
        small = (radius < r_min) | (area < a_min)
        small[area.argmax()] = False  # never dissolve the largest region
        if not small.any():
            break
        kill = small[comp - 1]
        _, (iy, ix) = ndi.distance_transform_edt(kill, return_indices=True)
        lab = lab[iy, ix]  # every killed pixel takes the colour of its nearest survivor
    return lab


def compact_palette(labels: np.ndarray, rgb: np.ndarray, paints: list[dict] | None = None):
    """Renumber the labels actually in use. Returns (labels, palette, paints_used).

    Free palette: colours are the mean of their pixels, ordered lightest first.
    Kit palette (`paints` given, parallel to label values): colours are the real paint colours,
    ordered by kit pot number so the printed numbers match the pots."""
    used = np.unique(labels)
    if paints is not None:
        order = sorted(range(len(used)), key=lambda i: paints[used[i]]["number"])
        cols = np.array([paints[u]["rgb"] for u in used], dtype=float)
        kept = [paints[used[oi]] for oi in order]
    else:
        flat = rgb.reshape(-1, 3)
        cols = np.array([flat[(labels == u).ravel()].mean(0) for u in used])
        L = skcolor.rgb2lab(cols[None] / 255.0)[0, :, 0]
        order = np.argsort(-L)
        kept = None
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for new, oi in enumerate(order):
        remap[used[oi]] = new
    palette = [tuple(int(round(c)) for c in cols[oi]) for oi in order]
    return remap[labels], palette, kept


def place_numbers(hi: np.ndarray, min_r: float, spacing: float):
    """One number per region at its most open point; big regions get extra numbers."""
    comp, _ = measure.label(hi, connectivity=1, background=-1, return_num=True)
    edt = ndi.distance_transform_edt(~_boundaries(comp)) + 0.5
    out = []
    for ci, sl in enumerate(ndi.find_objects(comp), start=1):
        if sl is None:
            continue
        m = comp[sl] == ci
        e = np.where(m, edt[sl], 0.0)
        num = int(hi[sl][m][0]) + 1
        oy, ox = sl[0].start, sl[1].start
        yy, xx = np.ogrid[: e.shape[0], : e.shape[1]]
        threshold = min_r
        while True:
            flat = int(e.argmax())
            r = float(e.flat[flat])
            if r < threshold:
                break
            y, x = divmod(flat, e.shape[1])
            out.append((x + ox + 0.5, y + oy + 0.5, r, num))
            e[(yy - y) ** 2 + (xx - x) ** 2 < spacing**2] = 0.0
            threshold = max(min_r, 1.5 * min_r)  # be pickier about extra numbers
    return out


def _font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size_px)
            except OSError:
                continue
    return ImageFont.load_default(size=size_px)


# --------------------------------------------------------------------------- pipeline
def process(
    src, colors: int = 16, detail: str = "medium", paper: str = "letter", seed: int = 0, paints: list[dict] | None = None
) -> Result:
    """`paints` (from load_paints) restricts the palette to real kit colours; numbers then match the pots."""
    colors = int(max(4, min(len(paints) if paints else 30, colors)))
    preset = DETAIL_PRESETS.get(detail, DETAIL_PRESETS["medium"])
    paper = paper if paper in PAPER else "letter"

    # 1. shrink + denoise (lighter denoise where the fine detail is: eyes, noses, mouths)
    img = _fit(load_image(src), preset["work_long"])
    raw = np.asarray(img)
    faces = find_faces(raw)
    detail = detail_mask(raw, preset["work_long"], faces)
    rgb = np.asarray(img.filter(ImageFilter.MedianFilter(preset["median"])))
    if preset["median"] > 3:
        rgb = np.where(detail[..., None], np.asarray(img.filter(ImageFilter.MedianFilter(3))), rgb)
    H, W = rgb.shape[:2]

    # 2. quantise, 3. smooth + dissolve at work resolution
    if paints:
        labels, chosen = quantize_to_paints(rgb, paints, colors, seed, mask=detail)
        colors = len(chosen)
    else:
        labels, chosen = quantize(rgb, colors, seed, mask=detail), None
    labels = smooth_labels(labels, colors, sigma=preset["sigma"], detail=detail)
    labels = dissolve_small_regions(labels, preset["min_radius"], 3.0 * preset["min_radius"] ** 2, protect=detail)
    labels, palette, kit = compact_palette(labels, rgb, chosen)

    # 4. upscale to print resolution with smooth boundaries, clean up again
    scale = HI_LONG / max(W, H)
    hi_shape = (round(H * scale), round(W * scale))
    hi_detail = transform.resize(detail.astype(np.float32), hi_shape, order=0, preserve_range=True) > 0.5
    hi = smooth_labels(labels, len(palette), sigma=0.6 * scale, out_shape=hi_shape, detail=hi_detail)
    hi = dissolve_small_regions(
        hi, preset["min_radius"] * scale * 0.9, 3.0 * (preset["min_radius"] * scale) ** 2, 2, protect=hi_detail
    )
    used = np.unique(hi)
    if len(used) != len(palette):  # a colour vanished during cleanup; renumber
        remap = np.zeros(len(palette), dtype=np.int32)
        remap[used] = np.arange(len(used))
        hi = remap[hi]
        palette = [palette[u] for u in used]
        if kit:
            kit = [kit[u] for u in used]
    HH, WW = hi.shape
    labels_txt = [str(p["number"]) for p in kit] if kit else None
    names = [p["name"] for p in kit] if kit else None

    # 5. outlines + numbers
    _, _, s, _, _ = page_geometry((WW, HH), paper)
    line_px = max(2, round(0.011 * inch / s))  # ~0.28 mm lines on paper
    edges = segmentation.find_boundaries(hi, mode="thick")
    if line_px > 2:
        edges = ndi.binary_dilation(edges, iterations=(line_px - 2 + 1) // 2)
    outline_arr = np.where(edges, LINE_GRAY, 255).astype(np.uint8)
    outline = Image.fromarray(outline_arr, "L")

    min_r = MIN_FONT_PT / (1.25 * s)  # smallest region that can hold a readable number
    spacing = 1.6 * inch / s  # extra numbers roughly every 1.6" in big areas
    numbers = place_numbers(hi, min_r, spacing)

    numbered = outline.convert("RGB")
    draw = ImageDraw.Draw(numbered)
    cache: dict[int, ImageFont.ImageFont] = {}
    for x, y, r, num in numbers:
        fs_px = int(round(font_size_pt(r, s) / s))
        if fs_px < 3:
            continue
        f = cache.setdefault(fs_px, _font(fs_px))
        txt = labels_txt[num - 1] if labels_txt else str(num)
        draw.text((x, y), txt, fill=(115, 115, 115), font=f, anchor="mm")

    colored = Image.fromarray(np.asarray(palette, dtype=np.uint8)[hi], "RGB")
    return Result(outline, numbered, colored, palette, numbers, paper, labels_txt, names, len(faces))


# --------------------------------------------------------------------------- PDF
def _hex(c):
    return "#%02X%02X%02X" % tuple(c)


def render_pdf(res: Result, path, title: str = "Paint by Numbers"):
    W, H = res.size
    pw, ph, s, x0, y0 = page_geometry((W, H), res.paper)
    dw, dh = W * s, H * s
    c = canvas.Canvas(str(path), pagesize=(pw, ph))
    c.setTitle(title)

    # ---- page 1: line art with numbers
    c.drawImage(ImageReader(res.outline), x0, y0, dw, dh)
    c.setStrokeGray(0.6)
    c.setLineWidth(0.5)
    c.rect(x0, y0, dw, dh)
    c.setFillGray(NUMBER_GRAY)
    for x, y, r, num in res.numbers:
        fs = font_size_pt(r, s)
        if fs < MIN_FONT_PT:
            continue
        c.setFont("Helvetica", fs)
        c.drawCentredString(x0 + x * s, y0 + (H - y) * s - fs * 0.35, res.label(num))
    c.setFillGray(0.5)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN, MARGIN, f"{title}  -  {len(res.palette)} colours  -  colour key on page 2")
    c.drawRightString(pw - MARGIN, MARGIN, dt.date.today().isoformat())
    c.showPage()

    # ---- page 2: colour key + reference image
    c.setFillGray(0.1)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN, ph - MARGIN - 14, "Colour key")
    n = len(res.palette)
    cols = 6 if pw > ph else 4
    rows = math.ceil(n / cols)
    sw = 0.42 * inch
    cell_w = (pw - 2 * MARGIN) / cols
    cell_h = sw + 0.45 * inch  # swatch + two text lines + gap
    top = ph - MARGIN - 30
    for i, col in enumerate(res.palette):
        r_, c_ = divmod(i, cols)
        x = MARGIN + c_ * cell_w
        y = top - (r_ + 1) * cell_h + 0.3 * inch
        c.setFillColorRGB(*(v / 255 for v in col))
        c.setStrokeGray(0.5)
        c.rect(x, y, sw, sw, fill=1, stroke=1)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        c.setFillGray(0.0 if lum > 140 else 1.0)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(x + sw / 2, y + sw / 2 - 5, res.label(i + 1))
        c.setFillGray(0.3)
        c.setFont("Helvetica", 7.5)
        if res.names:
            c.drawString(x, y - 10, (res.names[i] or "")[:22])
            c.drawString(x, y - 19, f"pot {res.label(i + 1)}  {_hex(col)}")
        else:
            c.drawString(x, y - 10, _hex(col))
            c.drawString(x, y - 19, "rgb(%d, %d, %d)" % col)
    key_bottom = top - rows * cell_h - 6
    # reference image fills what is left
    avail_w, avail_h = pw - 2 * MARGIN, key_bottom - MARGIN - FOOTER - 20
    if avail_h > 1 * inch:
        rs = min(avail_w / W, avail_h / H)
        rw, rh = W * rs, H * rs
        rx = (pw - rw) / 2
        ry = MARGIN + FOOTER + (avail_h - rh) / 2
        c.setFillGray(0.3)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN, ry + rh + 6, "Finished reference")
        c.drawImage(ImageReader(res.colored), rx, ry, rw, rh)
        c.setStrokeGray(0.6)
        c.rect(rx, ry, rw, rh)
    c.setFillGray(0.5)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN, MARGIN, title)
    c.showPage()
    c.save()


def save_outputs(res: Result, out_dir, title: str = "Paint by Numbers", preview_long: int = 1200) -> dict:
    """Write pdf + pngs + meta.json into out_dir. Returns the meta dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_pdf(res, out_dir / "pbn.pdf", title)
    _fit(res.numbered, preview_long).save(out_dir / "preview.png", optimize=True)
    _fit(res.colored, preview_long).save(out_dir / "colored.png", optimize=True)
    res.numbered.save(out_dir / "numbered.png", optimize=True)
    meta = {
        "palette": [list(c) for c in res.palette],
        "hex": [_hex(c) for c in res.palette],
        "labels": res.labels or [str(i + 1) for i in range(len(res.palette))],
        "names": res.names,
        "faces": res.faces,
        "regions": len(res.numbers),
        "size": list(res.size),
        "paper": res.paper,
    }
    (out_dir / "result.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Photo -> paint-by-numbers PDF")
    ap.add_argument("photo")
    ap.add_argument("-o", "--out", default="pbn_out", help="output directory")
    ap.add_argument("-c", "--colors", type=int, default=16)
    ap.add_argument("-d", "--detail", choices=DETAIL_PRESETS, default="medium")
    ap.add_argument("-p", "--paper", choices=PAPER, default="letter")
    ap.add_argument("-k", "--kit", help="paint kit JSON (e.g. paints.json): use only these real paint colours")
    a = ap.parse_args()
    import time

    t = time.time()
    res = process(a.photo, a.colors, a.detail, a.paper, paints=load_paints(a.kit) if a.kit else None)
    meta = save_outputs(res, a.out, title=f"Paint by Numbers - {Path(a.photo).stem}")
    print(f"done in {time.time() - t:.1f}s: {len(res.palette)} colours, {meta['regions']} numbers -> {a.out}/pbn.pdf")


if __name__ == "__main__":
    main()
