"""
Render the story's share image, public/og/story.png (1200×630), from the hero film.

Needs the dev server (`npm run dev`, default http://localhost:5173) or pass a base URL:
    node scripts/py.mjs scripts/render_og.py [http://localhost:5181]
Renders the story's first viewport at device scale 2 with reduced motion (the final frame, regions of interest measured),
hides the install line for the card only, then resizes to 1200×630 with Pillow if available, else sips.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public/og/story.png"
base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5173"

with tempfile.TemporaryDirectory() as d:
    raw = Path(d) / "og-2x.png"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1200, "height": 630}, device_scale_factor=2,
                                   reduced_motion="reduce").new_page()
        page.goto(f"{base}/story/")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2500)
        page.add_style_tag(content=".hero-meta{display:none!important}")
        page.wait_for_timeout(400)
        page.screenshot(path=str(raw))
        browser.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.open(raw).resize((1200, 630), Image.LANCZOS).save(OUT, optimize=True)
    except ImportError:
        if not shutil.which("sips"):
            raise SystemExit("install Pillow (pip install pillow) to resize the image")
        subprocess.run(["sips", "-z", "630", "1200", str(raw), "--out", str(OUT)], check=True, capture_output=True)
print(f"wrote {OUT.relative_to(ROOT)}")
