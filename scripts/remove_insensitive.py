"""Remove all sodipodi:insensitive="true" attributes from the SVG.

These are session-state flags Inkscape writes when you lock a layer before
saving.  They are baked into the file and cause a serious rendering artefact:

  When a sub-layer is locked (or unlocked) inside a parent that is ALSO saved
  as insensitive, Inkscape re-rasterizes the entire parent context.  In this
  file layer41 'The World' is saved as insensitive and contains rect3131 which
  is 20 566 × 10 283 SVG units (4.3× the page width).  The locked-layer
  raster buffer must cover this entire extent, and the round-trip
      SVG coords -> huge raster buffer -> back to SVG coords
  introduces a non-uniform scale error that visually appears as a
  shift-with-rotation when the topography '0' layer (g70-2) is
  locked/unlocked.

Fix: strip sodipodi:insensitive="true" from all elements.

The user can still lock/unlock layers interactively during editing sessions;
Inkscape re-adds the attribute temporarily.  It will no longer be persisted
back into the canonical file between sessions by this script.

Input:  OLD_WORLD_ATLAS_flattened.svg
Output: OLD_WORLD_ATLAS_flattened_unlocked.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

from lxml import etree

NS_SODIPODI = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
NS_INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
INSENSITIVE_ATTR = f"{{{NS_SODIPODI}}}insensitive"
INKSCAPE_LABEL_ATTR = f"{{{NS_INKSCAPE}}}label"
INKSCAPE_GROUPMODE_ATTR = f"{{{NS_INKSCAPE}}}groupmode"

LOCKED_LAYER_LABELS = {
    "0",
    "100",
    "250",
    "500",
    "750",
    "1000",
    "1500",
    "2000",
    "2500",
    "3000",
    "4000",
    "5000",
}

DEFAULT_INPUT = Path(
    r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps"
    r"\OLD_WORLD_ATLAS.svg"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps"
    r"\OLD_WORLD_ATLAS_flattened_unlocked.svg"
)


def remove_insensitive(source: Path, dest: Path) -> None:
    print(f"Parsing {source} ({source.stat().st_size / 1_048_576:.1f} MB)…")
    parser = etree.XMLParser(remove_blank_text=False, recover=True, huge_tree=True)
    tree = etree.parse(str(source), parser)
    root = tree.getroot()

    removed = 0
    restored = 0
    for elem in root.iter():
        if INSENSITIVE_ATTR in elem.attrib:
            # Confirm the value is "true" before removing (there could be "false")
            if elem.attrib[INSENSITIVE_ATTR] == "true":
                id_ = elem.get("id", "?")
                label = elem.get(INKSCAPE_LABEL_ATTR, "")
                print(f"  Removing insensitive from [{id_}] '{label}'")
                del elem.attrib[INSENSITIVE_ATTR]
                removed += 1
            else:
                # value is something other than "true" — leave it
                pass

    print(f"\nRemoved sodipodi:insensitive from {removed} elements")

    # Restore default lock state for topography layers "0" and "100".
    for elem in root.iter():
        if etree.QName(elem.tag).localname != "g":
            continue
        if elem.get(INKSCAPE_GROUPMODE_ATTR) != "layer":
            continue
        label = elem.get(INKSCAPE_LABEL_ATTR, "")
        if label in LOCKED_LAYER_LABELS:
            elem.set(INSENSITIVE_ATTR, "true")
            id_ = elem.get("id", "?")
            print(f"  Restoring insensitive on [{id_}] '{label}'")
            restored += 1

    print(f"Restored sodipodi:insensitive on {restored} target layers")

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
        description="Remove sodipodi:insensitive='true' from all layers to prevent "
                    "oversized locked-layer rasterization artefacts."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    remove_insensitive(args.input, args.output)
