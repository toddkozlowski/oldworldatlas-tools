"""
Diagnostic script to report SVG coordinates in Inkscape user-display units
for settlements in the Settlements layer.

Specifically targets: Altdorf, Middenheim, Nuln, Talabecland

Uses the affine transformation derived from reference points to convert SVG file
coordinates to Inkscape GUI display coordinates (which have y-axis flipped upward).
"""

import re
from pathlib import Path
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SVG_PATH = (
    Path(__file__).parent.parent.parent
    / "oldworldatlas-maps"
    / "OLD_WORLD_ATLAS.svg"
)

TARGETS = {"Altdorf", "Middenheim", "Nuln", "Talabecland"}

NS = {
    "svg": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
}

# Affine transformation parameters: converts SVG file coords to Inkscape GUI coords
# Derived from calibration with Altdorf (0.000, 51.460) and Middenheim (1.172, 55.312)
TRANSFORM_X_SCALE = 0.01429896
TRANSFORM_X_OFFSET = -5.86684920
TRANSFORM_Y_SCALE = -0.01761478
TRANSFORM_Y_OFFSET = 58.55155173


# ---------------------------------------------------------------------------
# Transformation Helpers
# ---------------------------------------------------------------------------

def svg_to_gui(x: float, y: float) -> tuple[float, float]:
    """
    Convert SVG file coordinates to Inkscape GUI display coordinates.
    
    Args:
        x: X coordinate in SVG file space
        y: Y coordinate in SVG file space
        
    Returns:
        (x_gui, y_gui) in Inkscape display space
    """
    x_gui = TRANSFORM_X_SCALE * x + TRANSFORM_X_OFFSET
    y_gui = TRANSFORM_Y_SCALE * y + TRANSFORM_Y_OFFSET
    return (x_gui, y_gui)


def get_text_content(elem) -> str | None:
    """Return the visible text of a <text> element (via tspan children or direct text)."""
    parts = []
    for tspan in elem.findall(f"{{{NS['svg']}}}tspan"):
        if tspan.text:
            parts.append(tspan.text.strip())
    if not parts and elem.text:
        parts.append(elem.text.strip())
    return " ".join(parts) if parts else None


def parse_translate(transform: str) -> tuple[float, float]:
    """Extract (tx, ty) from a translate() transform string; returns (0, 0) if absent."""
    if not transform:
        return (0.0, 0.0)
    m = re.search(r"translate\s*\(\s*([-\d.eE+]+)\s*[,\s]\s*([-\d.eE+]+)\s*\)", transform)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (0.0, 0.0)


def find_settlements_layer(root) -> ET.Element | None:
    """Locate the top-level 'Settlements' layer group."""
    for g in root.findall(f".//{{{NS['svg']}}}g"):
        if g.get(f"{{{NS['inkscape']}}}label") == "Settlements":
            return g
    return None


def collect_text_elements(parent: ET.Element, accumulated_tx: float = 0.0, accumulated_ty: float = 0.0):
    """
    Walk the element tree, accumulating translate transforms from ancestor groups,
    and yield (elem, effective_tx, effective_ty) for every <text> element found.
    """
    for child in parent:
        tag = child.tag
        if tag == f"{{{NS['svg']}}}g":
            # Accumulate this group's translate on top of the parent's
            tx, ty = parse_translate(child.get("transform", ""))
            yield from collect_text_elements(child, accumulated_tx + tx, accumulated_ty + ty)
        elif tag == f"{{{NS['svg']}}}text":
            yield child, accumulated_tx, accumulated_ty


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not SVG_PATH.exists():
        print(f"ERROR: SVG not found at {SVG_PATH}")
        return

    print(f"Parsing SVG: {SVG_PATH}\n")
    tree = ET.parse(SVG_PATH)
    root = tree.getroot()

    # Report viewBox so we can see the unit/scale context
    svg_elem = root if root.tag == f"{{{NS['svg']}}}svg" else root.find(f"{{{NS['svg']}}}svg")
    if svg_elem is None:
        svg_elem = root
    vb = svg_elem.get("viewBox", "<not set>")
    width = svg_elem.get("width", "<not set>")
    height = svg_elem.get("height", "<not set>")
    print(f"SVG root  width={width}  height={height}  viewBox={vb}\n")

    # Report transformation parameters
    print(f"Coordinate Transformation (SVG file → Inkscape GUI display):")
    print(f"  x_gui = {TRANSFORM_X_SCALE:.8f} × x_file + {TRANSFORM_X_OFFSET:.8f}")
    print(f"  y_gui = {TRANSFORM_Y_SCALE:.8f} × y_file + {TRANSFORM_Y_OFFSET:.8f}")
    print(f"  (Note: negative y-scale reflects y-axis flip)\n")

    # Locate Settlements layer
    settlements_layer = find_settlements_layer(root)
    if settlements_layer is None:
        print("ERROR: 'Settlements' layer not found in SVG.")
        return

    print(f"Found Settlements layer (id={settlements_layer.get('id', '?')})\n")
    print(f"{'Settlement':<20}  {'SVG File Coords':<25}  {'Inkscape GUI Coords':<25}")
    print(f"{'':20}  {'(after transform)':<25}  {'(expected display)':<25}")
    print("-" * 80)

    found = {}

    for text_elem, parent_tx, parent_ty in collect_text_elements(settlements_layer):
        name = get_text_content(text_elem)
        if name not in TARGETS:
            continue

        # Raw coordinates from the element's own attributes
        raw_x = float(text_elem.get("x") or 0.0)
        raw_y = float(text_elem.get("y") or 0.0)

        # Element-level transform (some Inkscape elements carry their own translate)
        elem_transform = text_elem.get("transform", "")
        elem_tx, elem_ty = parse_translate(elem_transform)

        # Effective position in SVG file space = raw + elem transform + accumulated parent translates
        file_x = raw_x + elem_tx + parent_tx
        file_y = raw_y + elem_ty + parent_ty

        # Convert to Inkscape GUI display coordinates
        gui_x, gui_y = svg_to_gui(file_x, file_y)

        print(
            f"{name:<20}  ({file_x:>10.3f}, {file_y:>10.3f})  "
            f"→  ({gui_x:>10.3f}, {gui_y:>10.3f})"
        )
        found[name] = (file_x, file_y, gui_x, gui_y)

    missing = TARGETS - found.keys()
    if missing:
        print(f"\nNot found in Settlements layer: {', '.join(sorted(missing))}")

    print(f"\n{'='*80}")
    print(f"Summary – Inkscape GUI user-unit coordinates (as displayed):")
    for name, (file_x, file_y, gui_x, gui_y) in sorted(found.items()):
        print(f"  {name:<20}  x={gui_x:>8.3f},  y={gui_y:>8.3f}")


if __name__ == "__main__":
    main()
