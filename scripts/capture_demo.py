#!/usr/bin/env python3
"""Capture Streamlit console screenshots + GIF for demo media."""

from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
URL = "http://127.0.0.1:8502"


def wait_ready(page, timeout_ms: int = 60000) -> None:
    page.goto(URL, wait_until="networkidle", timeout=timeout_ms)
    page.get_by_text("NFL Fourth-Down EV Lab").first.wait_for(timeout=timeout_ms)
    # Plotly / recommendation settle
    time.sleep(2.5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        wait_ready(page)

        default_path = OUT / "console-default.png"
        page.screenshot(path=str(default_path), full_page=False)
        frames.append(default_path)
        print(f"wrote {default_path}")

        # S2 goal line
        page.get_by_role("button", name="S2", exact=True).click()
        time.sleep(3)
        goal_path = OUT / "console-goal-line.png"
        page.screenshot(path=str(goal_path), full_page=False)
        frames.append(goal_path)
        print(f"wrote {goal_path}")

        # S1 own territory
        page.get_by_role("button", name="S1", exact=True).click()
        time.sleep(3)
        own_path = OUT / "console-own-territory.png"
        page.screenshot(path=str(own_path), full_page=False)
        frames.append(own_path)
        print(f"wrote {own_path}")

        # Extra frames for GIF: tweak yards to go slider if visible
        page.get_by_role("button", name="S3", exact=True).click()
        time.sleep(2.5)
        mid = OUT / "_frame_s3.png"
        page.screenshot(path=str(mid), full_page=False)
        frames.append(mid)

        page.get_by_role("button", name="S5", exact=True).click()
        time.sleep(2.5)
        late = OUT / "_frame_s5.png"
        page.screenshot(path=str(late), full_page=False)
        frames.append(late)

        browser.close()

    # Build GIF
    try:
        from PIL import Image

        imgs = [Image.open(f).convert("RGB") for f in frames]
        gif_path = OUT / "demo.gif"
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=1200,
            loop=0,
        )
        print(f"wrote {gif_path}")
    finally:
        for f in OUT.glob("_frame_*.png"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
