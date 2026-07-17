"""Add a page-boundary clipPath to the topography group in the flattened SVG.

When Inkscape locks a layer it rasterizes it to an off-screen pixel buffer
sized to the path bounding box, not the page.  Topography paths extend far
outside the page, so the buffer can be many times larger than the viewport.
The rounding of that oversized buffer back to screen pixels introduces a
visible sub-pixel affine error (appears as slight shift + scale/rotation) that
differs between locked and unlocked rendering.

Fix: add a <clipPath> scoped to the InkScape page rectangle and apply it to
the topography parent group.  The raster buffer is then bounded to page size,
reducing the rounding error to at most ±1 px / page_width ≈ imperceptible.

Input:  OLD_WORLD_ATLAS_flattened.svg
Output: OLD_WORLD_ATLAS_flattened_clipPath_corrected.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

from lxml import etree

NS_SVG = "http://www.w3.org/2000/svg"
NS_INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
NS_SODIPODI = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0"

CLIP_PATH_ID = "topo-page-clip"
TOPO_GROUP_ID = "g82-8"

DEFAULT_INPUT = Path(
    r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps"
    r"\OLD_WORLD_ATLAS_flattened.svg"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps"
    r"\OLD_WORLD_ATLAS_flattened_clipPath_corrected.svg"
)


def get_page_rect(root: etree._Element) -> tuple[float, float, float, float]:
    """Return (x, y, width, height) of the Inkscape page rectangle."""
    ns = {"sodipodi": NS_SODIPODI, "inkscape": NS_INKSCAPE}
    # Try <inkscape:page> in <sodipodi:namedview>
    page = root.find(
        f".//{{{NS_INKSCAPE}}}page",
    )
    if page is not None:
        x = float(page.get("x", "0"))
        y = float(page.get("y", "0"))
        w = float(page.get("width", "0"))
        h = float(page.get("height", "0"))
        if w > 0 and h > 0:
            print(f"  Page rect from <inkscape:page>: x={x} y={y} w={w} h={h}")
            return x, y, w, h

    # Fallback: parse viewBox
    vb = root.get("viewBox", "")
    parts = vb.split()
    if len(parts) == 4:
        x, y, w, h = map(float, parts)
        print(f"  Page rect from viewBox: x={x} y={y} w={w} h={h}")
        return x, y, w, h

    raise ValueError("Could not determine page dimensions from SVG")


def add_page_clippath(source: Path, dest: Path) -> None:
    print(f"Parsing {source} ({source.stat().st_size / 1_048_576:.1f} MB)…")
    parser = etree.XMLParser(remove_blank_text=False, recover=True, huge_tree=True)
    tree = etree.parse(str(source), parser)
    root = tree.getroot()

    # --- 1. Get page dimensions ---
    x, y, w, h = get_page_rect(root)

    # --- 2. Find or create the first <defs> element ---
    defs = root.find(f"{{{NS_SVG}}}defs")
    if defs is None:
        defs = etree.SubElement(root, f"{{{NS_SVG}}}defs", id="defs-added")
        root.insert(0, defs)
        print("  Created new <defs> element")

    # Guard: don't add a duplicate
    existing = defs.find(f".//{{{NS_SVG}}}clipPath[@id='{CLIP_PATH_ID}']")
    if existing is not None:
        print(f"  clipPath '{CLIP_PATH_ID}' already exists — skipping insertion")
    else:
        clip_path_elem = etree.SubElement(
            defs,
            f"{{{NS_SVG}}}clipPath",
            id=CLIP_PATH_ID,
        )
        etree.SubElement(
            clip_path_elem,
            f"{{{NS_SVG}}}rect",
            x=str(x),
            y=str(y),
            width=str(w),
            height=str(h),
        )
        print(f"  Added <clipPath id='{CLIP_PATH_ID}'> with rect {x},{y} {w}×{h}")

    # --- 3. Find the topography group and apply clip-path ---
    topo_group = root.find(f".//*[@id='{TOPO_GROUP_ID}']")
    if topo_group is None:
        print(f"ERROR: topography group '{TOPO_GROUP_ID}' not found", file=sys.stderr)
        sys.exit(1)

    existing_clip = topo_group.get("clip-path")
    if existing_clip:
        print(f"  Group '{TOPO_GROUP_ID}' already has clip-path='{existing_clip}' — overwriting")

    topo_group.set("clip-path", f"url(#{CLIP_PATH_ID})")
    print(f"  Set clip-path on group '{TOPO_GROUP_ID}'")

    # --- 4. Write output ---
    print(f"Writing {dest}…")
    tree.write(
        str(dest),
        xml_declaration=True,
        encoding="UTF-8",
        standalone=False,
        pretty_print=False,
    )

    size_in = source.stat().st_size / 1_048_576
    size_out = dest.stat().st_size / 1_048_576
    print(f"\n=== Done ===")
    print(f"  Input : {size_in:.1f} MB")
    print(f"  Output: {size_out:.1f} MB  →  {dest}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add page-boundary clipPath to topography group to fix "
                    "Inkscape locked-layer rasterization distortion."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    add_page_clippath(args.input, args.output)
