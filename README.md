# Paint by Numbers kiosk

Scan a QR code with your phone, pick a photo, preview the result, tap **Print**.
Out comes a two-page paint-by-numbers kit: page 1 is the numbered line art,
page 2 is the colour key plus a small finished reference.

## Run it

```bash
./run.sh
```

That creates a virtualenv on first run, starts the server on port 8765, and opens
the kiosk page (big QR code + recent jobs) in your browser. Phones must be on the
same Wi-Fi network as this computer. macOS may ask once to allow incoming
connections for Python; say yes.

Options are environment variables:

| Variable         | Default            | Meaning                                              |
|------------------|--------------------|------------------------------------------------------|
| `PBN_PORT`       | `8765`             | HTTP port                                            |
| `PBN_PAPER`      | `letter`           | `letter` or `a4`                                     |
| `PBN_PRINTER`    | system default     | CUPS printer name (see `lpstat -p`)                  |
| `PBN_NO_PRINT`   | unset              | `1` disables printing (demo mode, PDF download only) |
| `PBN_NO_BROWSER` | unset              | `1` skips auto-opening the kiosk page                |
| `PBN_PAINTS`     | `paints.json`      | Paint-kit file; every painting uses only these real paint colours and prints the kit's pot numbers. Set to `""` for free colours |

Example: `PBN_PRINTER=Canon_TS8000_series PBN_PAPER=a4 ./run.sh`

### Optional: face detection

Human eyes are protected by a face detector when OpenCV is installed:

```bash
.venv/bin/pip install -r requirements-faces.txt
```

Without it the pipeline still runs and eyes are found by contrast alone. The detector is
OpenCV's YuNet model (`models/face_detection_yunet_2023mar.onnx`, bundled, Apache-2.0 from the
OpenCV model zoo). On Python 3.14 pip currently compiles OpenCV from source, which takes a long
time; on 3.12/3.13 it is a quick wheel install.

## Command line (no phone needed)

```bash
.venv/bin/python pbn.py photo.jpg -o out --colors 16 --detail medium
```

Writes `out/pbn.pdf`, `out/preview.png`, `out/colored.png`, `out/numbered.png`.

Add `--kit paints.json` to restrict the result to the kit's paints.

## Paint kit

`paints.json` describes the real paints on hand: Paintable Pictures *Paint Your Photo by Number, Pets Edition* (48 numbered pots). With a kit loaded, the pipeline still runs k-means to find the photo's dominant colours, but each cluster then claims the closest unclaimed pot (in CIELAB) and every pixel is reassigned to its nearest chosen pot. The printed numbers are the pot numbers, so region 17 is painted with pot 17. The colour key on page 2 lists pot number, a descriptive name and the hex value.

The pot colours were sampled from the maker's back-of-box artwork, not measured from wet paint, so expect small differences from the real pots. Edit the hex values in `paints.json` if you colour-match them by hand, or point `PBN_PAINTS` at a different kit file (same shape: `{"name": ..., "paints": [{"number": 1, "name": "...", "hex": "#RRGGBB"}, ...]}`).

## How it works

1. Shrink and median-filter the photo to kill noise.
   A *detail mask* is built first: top-hat filters on lightness find small, compact, high-contrast
   blobs (pupils, catchlights, eye-whites, nostrils). Inside the mask the median filter is lighter,
   label smoothing is weaker, regions may be half the usual size before being dissolved, and the
   mask's pixels count 12x when colours are chosen, so eyes keep their own colours instead of
   melting into the surrounding fur or skin.
   If OpenCV is installed, YuNet face detection runs first and the two eye positions of every
   human face are added to the mask outright, so eyes are kept even when they are low-contrast.
2. K-means cluster the pixels in CIELAB (perceptual colour space) into N colours. With a paint kit,
   every pixel instead goes to its nearest pot, then pots are retired one at a time (whichever adds
   the least colour error) until N remain.
3. Smooth the label map with a soft majority vote, then dissolve any region that is
   too small or too thin to paint (measured by its largest inscribed circle).
   Dissolved pixels take the colour of their nearest surviving neighbour.
4. Upscale to print resolution (2400 px long side) using the same soft-vote trick so
   boundaries come out as smooth curves instead of pixel staircases.
5. Trace boundaries, place a number at the most open point of every region (plus
   extra numbers every ~1.6 inches in large areas), scale the font to fit.
6. Render the PDF with ReportLab: numbers are vector text, so they print crisp.

Without a kit, colours are numbered lightest to darkest, which is the usual painting order. With a kit, numbers are the kit's pot numbers.

## Files

- `app.py` – Flask server: upload page for phones, job queue, kiosk page, `lpr` printing
- `pbn.py` – the image pipeline and PDF renderer (also a CLI)
- `templates/` – phone upload page, result/print page, kiosk page
- `paints.json` – the real paint kit (pot numbers + colours)
- `jobs/` – one folder per upload (original, previews, PDF, job.json)
