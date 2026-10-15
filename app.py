"""app.py - scan a QR code on your phone, pick a photo, get a paint-by-numbers kit printed.

Run:   python app.py          (or ./run.sh)
Env:   PBN_PORT=8765  PBN_PAPER=letter|a4  PBN_PRINTER=<cups name>  PBN_NO_PRINT=1  PBN_NO_BROWSER=1
       PBN_PAINTS=paints.json   (real paint-kit colours; set to "" to go back to free colours)
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import socket
import subprocess
import threading
import uuid
import webbrowser
from pathlib import Path

import qrcode
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

import pbn

BASE = Path(__file__).resolve().parent
JOBS_DIR = BASE / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

PORT = int(os.environ.get("PBN_PORT", "8765"))
PAPER = os.environ.get("PBN_PAPER", "letter").lower()
PRINTER = os.environ.get("PBN_PRINTER") or None
NO_PRINT = os.environ.get("PBN_NO_PRINT") == "1"
PAINTS_FILE = os.environ.get("PBN_PAINTS", "paints.json")
PAINTS: list[dict] = []
KIT_NAME = ""
if PAINTS_FILE and (BASE / PAINTS_FILE).is_file():
    PAINTS = pbn.load_paints(BASE / PAINTS_FILE)
    KIT_NAME = json.loads((BASE / PAINTS_FILE).read_text()).get("name", PAINTS_FILE)
COLOR_CHOICES = [c for c in [8, 12, 16, 20, 24, 32, 48] if not PAINTS or c <= len(PAINTS)] if PAINTS else [8, 12, 16, 20, 24]
DEFAULT_COLORS = 24 if PAINTS else 16
DETAIL_CHOICES = list(pbn.DETAIL_PRESETS)
SERVED_FILES = {"preview.png", "colored.png", "numbered.png", "pbn.pdf"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


# --------------------------------------------------------------------------- helpers
def lan_ip() -> str:
    """The address phones on the same Wi-Fi can reach us at."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packets are sent; just picks the outbound interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def base_url() -> str:
    return f"http://{lan_ip()}:{PORT}"


def default_printer() -> str | None:
    if PRINTER:
        return PRINTER
    try:
        out = subprocess.run(["lpstat", "-d"], capture_output=True, text=True, timeout=5).stdout
        if ":" in out:
            return out.split(":", 1)[1].strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def send_to_printer(pdf: Path) -> str:
    printer = default_printer()
    cmd = ["lpr"]
    if printer:
        cmd += ["-P", printer]
    media = "A4" if PAPER == "a4" else "Letter"
    cmd += ["-o", f"media={media}", "-o", "fit-to-page", "-o", "sides=one-sided", "-#", "1", str(pdf)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip() or "lpr failed")
    return printer or "default printer"


def _save(job: dict) -> None:
    (JOBS_DIR / job["id"] / "job.json").write_text(json.dumps(job, indent=2))


def _load_jobs() -> None:
    for d in JOBS_DIR.iterdir():
        f = d / "job.json"
        if not f.is_file():
            continue
        try:
            job = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if job.get("status") == "processing":
            job.update(status="error", error="server restarted during processing")
        _jobs[job["id"]] = job


def _worker(job_id: str, src: str) -> None:
    job = _jobs[job_id]
    try:
        res = pbn.process(src, job["colors"], job["detail"], PAPER, paints=PAINTS or None)
        meta = pbn.save_outputs(res, JOBS_DIR / job_id, title=f"Paint by Numbers  #{job_id[:6]}")
        with _lock:
            job.update(status="done", palette=meta["hex"], labels=meta["labels"], names=meta["names"],
                       regions=meta["regions"], n_colors=len(meta["hex"]))
            _save(job)
    except Exception as e:  # noqa: BLE001 - surface anything to the phone
        app.logger.exception("job %s failed", job_id)
        with _lock:
            job.update(status="error", error=f"{type(e).__name__}: {e}")
            _save(job)


def _get(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        abort(404)
    return job


# --------------------------------------------------------------------------- pages
@app.get("/")
def index():
    return render_template("index.html", colors=COLOR_CHOICES, default_colors=DEFAULT_COLORS, details=DETAIL_CHOICES, kit=KIT_NAME)


@app.get("/job/<job_id>")
def job_page(job_id):
    return render_template("job.html", job=_get(job_id), no_print=NO_PRINT)


@app.get("/kiosk")
def kiosk():
    return render_template(
        "kiosk.html", url=base_url(), printer=default_printer(), paper=PAPER, no_print=NO_PRINT,
        colors=COLOR_CHOICES, default_colors=DEFAULT_COLORS, details=DETAIL_CHOICES, kit=KIT_NAME, n_paints=len(PAINTS),
    )


@app.get("/qr.png")
def qr_png():
    img = qrcode.make(base_url(), box_size=12, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", max_age=0)


@app.get("/jobs/<job_id>/<name>")
def job_file(job_id, name):
    if name not in SERVED_FILES:
        abort(404)
    return send_from_directory(JOBS_DIR / _get(job_id)["id"], name, max_age=0)


# --------------------------------------------------------------------------- api
@app.post("/api/jobs")
def create_job():
    f = request.files.get("photo")
    if f is None or not f.filename:
        return jsonify(error="No photo received"), 400
    try:
        colors = int(request.form.get("colors", DEFAULT_COLORS))
    except ValueError:
        colors = DEFAULT_COLORS
    colors = colors if 4 <= colors <= max(COLOR_CHOICES) else DEFAULT_COLORS
    detail = request.form.get("detail", "medium")
    detail = detail if detail in DETAIL_CHOICES else "medium"

    job_id = uuid.uuid4().hex[:10]
    d = JOBS_DIR / job_id
    d.mkdir()
    ext = Path(f.filename).suffix.lower()[:8] or ".img"
    src = d / f"original{ext}"
    f.save(src)
    job = dict(
        id=job_id,
        status="processing",
        created=dt.datetime.now().isoformat(timespec="seconds"),
        colors=colors,
        detail=detail,
        prints=0,
    )
    with _lock:
        _jobs[job_id] = job
        _save(job)
    threading.Thread(target=_worker, args=(job_id, str(src)), daemon=True).start()
    return jsonify(id=job_id)


@app.get("/api/jobs")
def list_jobs():
    jobs = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)[:18]
    return jsonify(jobs=jobs)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    return jsonify(_get(job_id))


@app.post("/api/jobs/<job_id>/print")
def print_job(job_id):
    job = _get(job_id)
    if job["status"] != "done":
        return jsonify(error="Not ready yet"), 409
    if NO_PRINT:
        return jsonify(error="Printing is disabled on this server (PBN_NO_PRINT=1)"), 403
    try:
        printer = send_to_printer(JOBS_DIR / job_id / "pbn.pdf")
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        return jsonify(error=str(e)), 500
    with _lock:
        job["prints"] = job.get("prints", 0) + 1
        job["last_print"] = dt.datetime.now().isoformat(timespec="seconds")
        _save(job)
    return jsonify(ok=True, printer=printer)


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    _load_jobs()
    url = base_url()
    printer = default_printer() or "none found (set PBN_PRINTER)"
    kit = KIT_NAME if PAINTS else "none - free colours (set PBN_PAINTS=paints.json)"
    print(f"\n  Phone URL : {url}\n  Kiosk/QR  : {url}/kiosk\n  Printer   : {printer}{'  [printing disabled]' if NO_PRINT else ''}\n  Paper     : {PAPER}\n  Paint kit : {kit}\n")
    if os.environ.get("PBN_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"{url}/kiosk")).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
