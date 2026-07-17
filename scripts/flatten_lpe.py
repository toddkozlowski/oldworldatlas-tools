"""Flatten Inkscape Live Path Effects (LPEs) in an SVG file.

Option B repair for OLD_WORLD_ATLAS.svg:

1. For every element that has both `d` (rendered output) and
   `inkscape:original-d` (LPE input), promote `d` as authoritative and
   remove `inkscape:original-d` and `inkscape:path-effect` from the element.

2. Remove all <inkscape:path-effect> definitions from <defs> (now orphaned).

3. Remove all `inkscape:transform-center-x` and `inkscape:transform-center-y`
   attributes from every element (stale after the 125% horizontal stretch;
   Inkscape recalculates these on first selection).

The existing `d` attribute on each path already contains the correct,
fully-stretched geometry as output by bake_in_transforms.py.  The `simplify`,
`powerstroke`, and `bspline` LPEs stored in `inkscape:original-d` still carry
the old, unstretched coordinates.  Inkscape re-evaluates LPEs on document
mutation events, causing affected layers to "jump" back to pre-stretch widths.

Output is written to a sibling file with `_flattened` appended to the stem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lxml import etree

# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------
NS_SVG = "http://www.w3.org/2000/svg"
NS_INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
NS_SODIPODI = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0"

INKSCAPE_ORIGINAL_D = f"{{{NS_INKSCAPE}}}original-d"
INKSCAPE_PATH_EFFECT = f"{{{NS_INKSCAPE}}}path-effect"
INKSCAPE_PATH_EFFECT_TAG = f"{{{NS_INKSCAPE}}}path-effect"  # element tag in defs
INKSCAPE_TC_X = f"{{{NS_INKSCAPE}}}transform-center-x"
INKSCAPE_TC_Y = f"{{{NS_INKSCAPE}}}transform-center-y"

DEFAULT_INPUT = Path(
    r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS.svg"
)


def build_output_path(source: Path) -> Path:
    return source.with_name(source.stem + "_flattened" + source.suffix)


def flatten_lpe(source: Path, dest: Path, verbose: bool = True) -> None:
    print(f"Parsing {source} ({source.stat().st_size / 1_048_576:.1f} MB)…")

    # Use lxml for namespace-aware, high-performance parsing.
    # recover=True tolerates minor XML issues that can appear in complex Inkscape files.
    parser = etree.XMLParser(remove_blank_text=False, recover=True, huge_tree=True)
    tree = etree.parse(str(source), parser)
    root = tree.getroot()

    stats = {
        "original_d_removed": 0,
        "path_effect_attr_removed": 0,
        "path_effect_def_removed": 0,
        "transform_center_removed": 0,
        "elements_cleaned": 0,
    }

    # ------------------------------------------------------------------
    # Pass 1: walk every element in the document
    # ------------------------------------------------------------------
    print("Pass 1: stripping inkscape:original-d, inkscape:path-effect attrs,")
    print("        and inkscape:transform-center-x/y from all elements…")

    for elem in root.iter():
        changed = False

        # Remove inkscape:original-d — the unstretched LPE input data
        if INKSCAPE_ORIGINAL_D in elem.attrib:
            del elem.attrib[INKSCAPE_ORIGINAL_D]
            stats["original_d_removed"] += 1
            changed = True

        # Remove the back-reference to the LPE definition
        if INKSCAPE_PATH_EFFECT in elem.attrib:
            del elem.attrib[INKSCAPE_PATH_EFFECT]
            stats["path_effect_attr_removed"] += 1
            changed = True

        # Remove stale rotation-center hints
        if INKSCAPE_TC_X in elem.attrib:
            del elem.attrib[INKSCAPE_TC_X]
            stats["transform_center_removed"] += 1
            changed = True
        if INKSCAPE_TC_Y in elem.attrib:
            del elem.attrib[INKSCAPE_TC_Y]
            stats["transform_center_removed"] += 1
            changed = True

        if changed:
            stats["elements_cleaned"] += 1

    # ------------------------------------------------------------------
    # Pass 2: remove <inkscape:path-effect> elements from <defs>
    # ------------------------------------------------------------------
    print("Pass 2: removing <inkscape:path-effect> definitions from <defs>…")

    # Collect all defs elements (there may be more than one <defs> block)
    defs_elements = root.findall(f".//{{{NS_SVG}}}defs") + root.findall(f"{{{NS_SVG}}}defs")
    # Deduplicate while preserving order
    seen = set()
    unique_defs = []
    for d in defs_elements:
        if id(d) not in seen:
            seen.add(id(d))
            unique_defs.append(d)

    for defs in unique_defs:
        to_remove = [
            child for child in defs
            if child.tag == INKSCAPE_PATH_EFFECT_TAG
        ]
        for child in to_remove:
            defs.remove(child)
            stats["path_effect_def_removed"] += 1

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    print(f"Writing output to {dest}…")
    tree.write(
        str(dest),
        xml_declaration=True,
        encoding="UTF-8",
        standalone=False,
        pretty_print=False,   # preserve original whitespace fidelity
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    size_in = source.stat().st_size / 1_048_576
    size_out = dest.stat().st_size / 1_048_576
    print("\n=== Flatten LPE complete ===")
    print(f"  inkscape:original-d removed       : {stats['original_d_removed']:,}")
    print(f"  inkscape:path-effect attr removed  : {stats['path_effect_attr_removed']:,}")
    print(f"  inkscape:path-effect defs removed  : {stats['path_effect_def_removed']:,}")
    print(f"  transform-center attrs removed     : {stats['transform_center_removed']:,}")
    print(f"  Elements modified                  : {stats['elements_cleaned']:,}")
    print(f"  Input size                         : {size_in:.1f} MB")
    print(f"  Output size                        : {size_out:.1f} MB")
    print(f"  Output file                        : {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten Inkscape LPEs: promote rendered `d` as authoritative, "
                    "strip inkscape:original-d, inkscape:path-effect, and "
                    "inkscape:transform-center-x/y attributes."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source SVG file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output SVG file (default: <input>_flattened.svg)",
    )
    args = parser.parse_args()

    source: Path = args.input
    if not source.exists():
        print(f"ERROR: input file not found: {source}", file=sys.stderr)
        sys.exit(1)

    dest: Path = args.output or build_output_path(source)

    flatten_lpe(source, dest)


if __name__ == "__main__":
    main()
