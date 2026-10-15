# Design: Paint by Numbers kiosk

A home-made version of a "paint your photo by number" kit. A phone scans a QR code,
uploads a photo, previews the result, and a two-page kit prints on the local printer:
page 1 is the numbered line art, page 2 is the colour key. The numbers on the page are
the pot numbers of a real, store-bought paint set, so the painter never mixes colours.

This document explains what was built and why. `README.md` covers how to run it.

## Goals

1. Zero-install for guests: a phone browser and the Wi-Fi, nothing else.
2. Output that is actually paintable with a brush: no slivers, no regions smaller than
   the brush, readable numbers.
3. Colours that exist as physical paint: every region maps to one pot in the kit.
4. Faces and pets must keep their eyes. A portrait without eyes is not a portrait.

## System overview

```
phone ──QR──▶ Flask (app.py) ──thread──▶ pbn.process() ──▶ jobs/<id>/{pbn.pdf, *.png}
                 │                                              │
   kiosk page ◀──┘ polls /api/jobs                 lpr -P printer ◀── /api/jobs/<id>/print
```

- **app.py** is a small Flask server. It serves three pages (phone upload, result/print,
  kiosk) and a JSON API. Each upload becomes a job folder under `jobs/` with the original,
  previews, the PDF and a `job.json`; jobs are reloaded on restart. Processing runs in a
  daemon thread per job so the request returns immediately and the phone polls.
- **pbn.py** is the image pipeline and PDF renderer. It has no Flask dependency and doubles
  as a CLI, which is how most of the tuning was done.
- **paints.json** describes the paint kit: pot number, a descriptive name, and a hex colour.
- **templates/** are three Jinja pages sharing one stylesheet. No build step, no JS framework.

## The pipeline (`pbn.process`)

Each step exists to fix a specific failure seen on real photos.

### 1. Shrink and denoise
The photo is fitted to a working resolution (640 to 1200 px long side, by detail preset)
and median-filtered. Working small makes k-means and the morphology fast; the median filter
removes sensor noise and JPEG blocks that would otherwise become hundreds of tiny regions.

### 2. Detail mask (eyes)
Before anything is blurred, a *detail mask* marks the small, compact, high-contrast
features: pupils, catchlights, eye-whites, nostrils, mouth corners.

- Lightness (CIELAB L) goes through a black top-hat (dark features on a lighter ground)
  and a white top-hat (bright on dark) with a window of about 3 % of the picture's long
  side, so only features smaller than that respond.
- The thresholded response is split into connected blobs; blobs are kept only if their area
  is between (0.4 % of long side)² and (9 % of long side)². Fur, tweed and lettering merge into
  sprawling blobs and are dropped; dust is too small and is dropped.
- The surviving blobs are dilated so the protection extends a little past the feature.

The mask then changes four later steps: the median filter is lighter inside it, label
smoothing is about a third as strong, regions may be half the normal size before they are
dissolved, and each mask pixel counts 12× when colours are chosen. The effect is that an eye,
which is a few hundred pixels in a million-pixel photo, keeps its own colours and its shape.

### 3. Colour quantisation
Two modes.

**Free palette.** k-means in CIELAB on a 50 000-pixel sample plus up to 15 000 extra
pixels drawn from the detail mask. CIELAB distances track perceived colour difference,
which matters because the painter judges the result by eye. Clusters become the palette.

**Paint kit.** Every pixel goes to its nearest pot of the 48 in CIELAB. Then pots are
retired one at a time until the requested count remains: on a weighted sample, the pot whose
removal adds the least total colour error (sum over its pixels of second-best minus best
distance) is retired first. Finally all pixels are assigned to the survivors. This replaced an
earlier "cluster claims nearest pot" scheme that collapsed a 16-colour request to 8 on a
greyscale photo, because several clusters would claim pots that no pixel actually preferred.

### 4. Smooth and dissolve at working resolution
Label maps straight out of quantisation are jagged. `smooth_labels` blurs each colour's mask
with a Gaussian and takes the per-pixel argmax (a soft majority vote), which rounds the
boundaries without shifting them much. `dissolve_small_regions` then measures every
connected region's largest inscribed circle (via a distance transform) and dissolves any
region whose circle is smaller than the preset's minimum radius or whose area is too small;
dissolved pixels take the label of their nearest surviving pixel. Both steps respect the
detail mask as described above. Up to five passes run because dissolving can create new
small regions.

### 5. Upscale to print resolution
The label map is upscaled to 2400 px on the long side using the same soft-vote trick
(resize each mask bilinearly, blur, argmax), so boundaries become smooth curves rather than
pixel staircases. A second, gentler dissolve pass removes anything the upscale created.

### 6. Numbers
Outlines come from `find_boundaries`, dilated to about 0.28 mm on paper. Each region gets
a number at the point of its largest inscribed circle (the most open spot), and regions
larger than about 1.6 inches get extra numbers at that spacing so the painter never has to
hunt. Font size scales with the circle so a two-digit number always fits; regions too small
for a 4 pt number keep their outline but get no number.

With a kit loaded the printed number is the pot number, not the palette index; the colour
key on page 2 lists pot number, name and hex.

### 7. PDF
ReportLab draws the outline image as a bitmap and the numbers as vector text, so numbers
stay crisp at any printer resolution. Page 2 is the colour key with a small finished
reference. `lpr` sends the PDF to the CUPS default printer with fit-to-page.

## The paint kit

`paints.json` is the Paintable Pictures *Paint Your Photo by Number, Pets Edition*
(48 pots, Amazon B0CJT74PPP). The maker does not publish the colours and the web tool keeps
its palette server-side, so the values were sampled from the maker's back-of-box artwork:
the 48 pots were located with a Hough circle transform on the full-resolution box image and
each pot's colour ring was sampled with a median. They are therefore close but not
spectrophotometer-accurate; the file is meant to be hand-corrected against real paint.
Any kit can be substituted by supplying a file of the same shape.

## Decisions and trade-offs

- **CIELAB everywhere.** RGB distance is a poor proxy for how different two paints look.
  Every nearest-colour decision in the pipeline is made in CIELAB.
- **Numbers before colours.** The dissolve thresholds are derived from what a brush can paint
  and what a number needs to be legible, not from image statistics. Detail presets only move
  those thresholds.
- **Contrast-based eye detection first.** It needs no model, works on any species, and
  catches nostrils and mouths for free. Its weakness is low-contrast eyes (a dark eye on dark
  fur). A face detector was added on top for the cases where contrast alone is not enough;
  see the next section.
- **Threads, not a queue.** One process, one thread per job, a lock around the job dict.
  Enough for a kiosk with one printer.
- **No database.** `job.json` per folder is the record; the folder is the artefact.

## Face detection

Contrast finds most eyes, but not all: a black Labrador's eyes, or a face in deep shadow,
can fall under the top-hat threshold. OpenCV's bundled Haar cascades (human frontal face
with an eye cascade inside each face, plus the frontal cat-face cascade) now run on the
working-resolution image. For every detected face, the eye band is added to the detail
mask outright: the detected eye boxes when the eye cascade fires, otherwise the band at
20 % to 55 % of the face height for humans and 30 % to 60 % for cats. Inside a face, the
top-hat thresholds are also halved so faint features are still picked up. Dogs have no
bundled cascade and still rely on the contrast mask alone.

## Known limits

- Faces smaller than about 5 % of the picture cannot be painted at 8 × 10 regardless of
  settings; the eye would be smaller than a brush tip.
- The kit colours are sampled from printed artwork, not measured paint.
- Very textured subjects (tweed, long fur) lose detail by design; that is what makes them
  paintable.
