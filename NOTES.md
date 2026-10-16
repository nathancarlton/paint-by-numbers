# Working notes

Things learned while building this that are not obvious from the code. `README.md` says how
to run it, `DESIGN.md` says how it works; this file is the scar tissue.

## Build log (October 2026)

1. **Kiosk upload button.** The phone page could already upload; the kiosk page could not.
   Added a direct upload with colour and detail selectors so the computer running the kiosk
   can make a painting without a phone.
2. **Real paint kit.** Switched from free k-means colours to the 48 pots of the Paintable
   Pictures Pets Edition kit. Printed numbers are now pot numbers.
3. **Eyes.** The first results blurred eyes into fur and skin. Added the detail mask
   (contrast-based), then an error-based pot retirement for the kit mode, then a face
   detector so human eyes are protected regardless of contrast.
4. **Repo, design doc, this file.**

## Where the kit colours came from

The maker (Paintable Pictures) publishes no colour values anywhere: not on Amazon, not on
their shop, not in the FAQ, and their web tool keeps the palette server-side. The values in
`paints.json` were sampled from the back-of-box artwork on their shop site: a Hough circle
transform located the 48 pots on the full-resolution image and the median of each pot's
colour ring was taken. Treat them as close, not exact. If a pot looks wrong against real
paint, edit its hex in `paints.json`; do not re-derive the whole palette.

The box art is the May 2025 packaging (web tool SKU `PBNPET4801`). A `PBNPET4802` SKU
exists in the tool and may be a newer revision of the kit.

## OpenCV on Python 3.14

There is no prebuilt wheel, so `pip install opencv-python-headless` compiles from source
(about 40 minutes). The resulting build imports but has no `CascadeClassifier` and ships no
Haar cascade files, which is why face detection uses YuNet rather than cascades. YuNet is
human-only. Dead ends tried and rejected:

- scikit-image's Viola-Jones `Cascade` with OpenCV's LBP cat-face cascade: minutes per
  image even at 320 px.
- Downloading models from `raw.githubusercontent.com/opencv/opencv_zoo` returns a 131-byte
  Git LFS pointer. Use `media.githubusercontent.com/media/opencv/opencv_zoo/...`.

OpenCV stays optional (`requirements-faces.txt`). Without it everything runs; eyes rely on
the contrast mask alone.

## Running and testing

- Flask runs with debug off, so templates and modules are cached. After editing anything,
  kill the server and restart it. Kill by PID (`lsof -iTCP:8765 -sTCP:LISTEN -t | xargs kill`).
- The LAN address changes between networks, so the QR code differs per run. The startup
  banner prints the current one.
- Test uploads made with the Flask test client land in `jobs/` and appear on the kiosk.
  Delete those folders afterwards. `jobs/` is gitignored because it holds people's photos.
- `pbn.py` is a CLI: `.venv/bin/python pbn.py photo.jpg -o out --colors 24 --kit paints.json`.
  Most tuning was done this way, comparing `colored.png` and `preview.png` side by side.

## Pushing

The remote is `github.com/nathancarlton/paint-by-numbers`. The `gh` CLI on this machine has
two accounts; pushes need the `nathancarlton` one (`gh auth switch --user nathancarlton`),
otherwise GitHub answers 403.

## Open ideas

- Pet face detection (YuNet is human-only). A small animal-face model would let cat and dog
  eyes be forced into the mask the way human eyes are now.
- Measure the real pots with a colour-calibrated photo and replace the sampled values.
- The group-photo case (faces about 60 px wide at working resolution) is beyond what an
  8 x 10 can hold; a crop-to-face option on the phone page would help more than any pipeline
  change.
