"""generate_tiles_ratmode.py — Generate OpenLayers-compatible TMS PNG tiles from
the Old World Atlas "ratmode" SVG (Skaven Under-Empire).

This is a parallel counterpart to generate_tiles.py: same tile-generation
pipeline, geometry, and CLI, but reading from a separate source SVG
(OLD_WORLD_ATLAS_RATMODE.svg) with a different set of visible layers, and
writing to a separate output tileset. It does not read, write, or otherwise
touch the main SVG or the main "map_tiles" output in any way.

Coordinate-input approach:
    The tile region is supplied explicitly as origin + size in *display* units
    (the same units shown in Inkscape's UI, configured by this project to map
    to geospatial coordinates):
        origin: (x, y)
        width:  w
        height: h
    The region is always rectangular and all tile export areas are derived by
    linearly subdividing those supplied bounds.

    The ratmode SVG shares the exact same viewBox/physical size and
    calibration as the main SVG (same geospatial region), so the same default
    region and calibration constants apply here.

Tile convention (TMS, matching OpenLayers TileGrid with y=0 at south):
    On disk: <output_root>/<tileset_name>/<z>/<x>/<y>.png

    z = 1 ->   2 x   2 tiles
    z = 2 ->   4 x   4 tiles
    z = 8 -> 256 x 256 tiles

    Each tile is always 256 x 256 pixels.

Layer visibility:
    Only the layers listed in VISIBLE_LAYERS are rendered ('topography',
    'Marshes', 'Under-Empire', 'Under-Empire Paths'). Every other top-level
    layer -- including 'Settlements', 'Points of Interest', 'Lakes',
    'Rivers', and 'Skavendom_settlements' -- is hidden with display:none.
    The "Frame (Old World)" rect is also hidden in the temporary SVG so it
    does not appear in the rendered tiles. The original SVG is never
    modified.

Usage:
    python generate_tiles_ratmode.py [--tileset NAME] [--min-zoom Z] [--max-zoom Z]
                             [--block-tiles N] [--inkscape PATH]
                             [--background COLOR] [--origin X Y]
                             [--width W] [--height H] [--dry-run] [--verbose]

    Example:
        python generate_tiles_ratmode.py -o -25 32 -w 50 -h 37.5

If no region arguments are provided, the script explicitly prompts for:
    - origin x
    - origin y
    - width
    - height

Display-coordinate convention used by this script:
    - origin (x, y) is the SOUTH-WEST (bottom-left) corner of the region
    - x increases east/right
    - y increases north/up

These display coordinates are converted to absolute SVG coordinates using the
same calibration used by process_map_svg.py before any export-area math.

    --background   CSS colour for tile background (e.g. white, #ffffff).
                   Defaults to transparent.
    --dry-run      Print every Inkscape command without executing anything.
                   Use to verify extents before a long render run.
    --block-tiles  Tiles per block edge for chunk rendering (default 32).
                   Reduce to 16 if Inkscape runs out of memory at high zoom.

OpenLayers TileGrid config (printed at runtime):
    The exact resolutions and extent required by map-manager.js are printed
    under "=== VIEWER CONFIG ===" before tile generation begins.
"""

import argparse
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required.  Install it with:  pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SVG_PATH = (
    Path(__file__).parent.parent.parent / "oldworldatlas-maps" / "OLD_WORLD_ATLAS_RATMODE.svg"
)
OUTPUT_ROOT = Path(
    r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-repository"
)

# ---------------------------------------------------------------------------
# Default region extent in SVG user units
# ---------------------------------------------------------------------------
#
# The ratmode SVG shares the exact same viewBox and physical document size
# as the main atlas SVG (same geospatial region), so the same default region
# and calibration constants apply.
# ---------------------------------------------------------------------------

DEFAULT_ORIGIN_X: float = -19.0
DEFAULT_ORIGIN_Y: float = 32.0
DEFAULT_REGION_WIDTH: float = 37.0
DEFAULT_REGION_HEIGHT: float = 37.0

# Display/SVG conversion constants (must match scripts/process_map_svg.py).
INKSCAPE_VB_X: float = 428.530
INKSCAPE_SCALE: float = 0.017504
INKSCAPE_C_Y: float = 3347.85

# ---------------------------------------------------------------------------
# Zoom levels and tile geometry
# ---------------------------------------------------------------------------

MIN_ZOOM:  int = 1    # lowest-resolution level, 2x2 tiles
MAX_ZOOM:  int = 8    # highest-resolution level, 256x256 tiles
TILE_SIZE: int = 256  # pixels per tile (square)

# Tiles per block edge for chunk rendering.  32 tiles x 256 px = 8192 px.
# Reduce to 16 if Inkscape runs out of memory at high zoom levels.
DEFAULT_BLOCK_TILES: int = 32

# ---------------------------------------------------------------------------
# Layer visibility
# ---------------------------------------------------------------------------
#
# These top-level layer labels (inkscape:label) are made visible in the
# temporary SVG.  Every other top-level layer -- including 'Settlements',
# 'Points of Interest', 'Lakes', 'Rivers', and 'Skavendom_settlements' -- is
# hidden with display:none.
# ---------------------------------------------------------------------------

VISIBLE_LAYERS: frozenset = frozenset({
    "topography",          # elevation colour scale
    "Marshes",
    "Under-Empire",
    "Under-Empire Paths",
})

# ---------------------------------------------------------------------------
# XML namespaces
# ---------------------------------------------------------------------------

NS_SVG      = "http://www.w3.org/2000/svg"
NS_INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
NS_SODIPODI = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
NS_XLINK    = "http://www.w3.org/1999/xlink"

INKSCAPE_LABEL = f"{{{NS_INKSCAPE}}}label"
SVG_G          = f"{{{NS_SVG}}}g"
SVG_RECT       = f"{{{NS_SVG}}}rect"

# ---------------------------------------------------------------------------
# SVG transform helpers
# ---------------------------------------------------------------------------


def _parse_svg_transform(
    transform_str: str,
) -> Tuple[float, float, float, float, float, float]:
    """
    Parse an SVG transform attribute into a 2D affine matrix (a, b, c, d, e, f).

    Handles matrix, translate, rotate, scale and chained transforms.
    Returns the identity matrix for empty strings.
    """
    if not transform_str:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    result: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    # Find every function(...) token in order
    for fn, args_str in re.findall(r"(\w+)\(([^)]*)\)", transform_str):
        args = [float(v) for v in re.split(r"[,\s]+", args_str.strip()) if v]
        tf: Tuple[float, float, float, float, float, float]

        if fn == "matrix" and len(args) == 6:
            tf = tuple(args)  # type: ignore[assignment]

        elif fn == "translate":
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0.0
            tf = (1.0, 0.0, 0.0, 1.0, tx, ty)

        elif fn == "rotate":
            a = math.radians(args[0])
            ca, sa = math.cos(a), math.sin(a)
            if len(args) == 3:
                cx, cy = args[1], args[2]
                tf = (ca, sa, -sa, ca, cx - cx * ca + cy * sa, cy - cx * sa - cy * ca)
            else:
                tf = (ca, sa, -sa, ca, 0.0, 0.0)

        elif fn == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            tf = (sx, 0.0, 0.0, sy, 0.0, 0.0)

        elif fn == "skewX":
            tf = (1.0, 0.0, math.tan(math.radians(args[0])), 1.0, 0.0, 0.0)

        elif fn == "skewY":
            tf = (1.0, math.tan(math.radians(args[0])), 0.0, 1.0, 0.0, 0.0)

        else:
            logger.debug("Unrecognised SVG transform function '%s' — skipping.", fn)
            continue

        result = _compose(result, tf)

    return result


def _compose(
    parent: Tuple[float, float, float, float, float, float],
    child:  Tuple[float, float, float, float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    """Compose two affine transforms: result = parent composed with child."""
    a1, b1, c1, d1, e1, f1 = parent
    a2, b2, c2, d2, e2, f2 = child
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply(
    t: Tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> Tuple[float, float]:
    """Apply affine transform matrix (a, b, c, d, e, f) to point (x, y)."""
    a, b, c, d, e, f = t
    return (a * x + c * y + e, b * x + d * y + f)


# ---------------------------------------------------------------------------
# Region handling
# ---------------------------------------------------------------------------


def prompt_float(prompt: str) -> float:
    """Prompt until a valid float is entered."""
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def resolve_region_inputs(args: argparse.Namespace) -> Tuple[float, float, float, float]:
    """Resolve region origin/size from CLI args; prompt explicitly when needed."""
    if args.origin is None and args.width is None and args.height is None:
        print("No region arguments supplied. Please enter the tile export region.")
        print("Coordinates are display units: origin is bottom-left; Y increases upward.")
        origin_x = prompt_float("Origin X: ")
        origin_y = prompt_float("Origin Y: ")
        width = prompt_float("Width: ")
        height = prompt_float("Height: ")
    else:
        origin_x = args.origin[0] if args.origin is not None else prompt_float("Origin X: ")
        origin_y = args.origin[1] if args.origin is not None else prompt_float("Origin Y: ")
        width = args.width if args.width is not None else prompt_float("Width: ")
        height = args.height if args.height is not None else prompt_float("Height: ")

    if width <= 0.0 or height <= 0.0:
        raise ValueError("Width and height must both be > 0.")

    return origin_x, origin_y, width, height


def resolve_max_zoom(args: argparse.Namespace) -> int:
    """Resolve max zoom from CLI or prompt with default 8 if omitted."""
    if args.max_zoom is not None:
        return args.max_zoom

    raw = input(f"Maximum zoom level [{MAX_ZOOM}]: ").strip()
    if not raw:
        return MAX_ZOOM
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("Maximum zoom must be an integer.") from exc


def display_region_to_svg_bounds(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
) -> Tuple[float, float, float, float]:
    """Convert display-unit region (origin SW, Y up) to SVG bounds (x0,y0,x1,y1).

    Returns bounds in absolute SVG coordinates where:
      x0 = west,  x1 = east
      y0 = north, y1 = south
    """
    west = origin_x
    east = origin_x + width
    south = origin_y
    north = origin_y + height

    svg_west = (west / INKSCAPE_SCALE) + INKSCAPE_VB_X
    svg_east = (east / INKSCAPE_SCALE) + INKSCAPE_VB_X
    svg_north = INKSCAPE_C_Y - (north / INKSCAPE_SCALE)
    svg_south = INKSCAPE_C_Y - (south / INKSCAPE_SCALE)

    x0 = min(svg_west, svg_east)
    x1 = max(svg_west, svg_east)
    y0 = min(svg_north, svg_south)
    y1 = max(svg_north, svg_south)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# SVG user-unit → Inkscape physical-pixel coordinate conversion
# ---------------------------------------------------------------------------


def read_doc_scale(svg_path: Path) -> Tuple[float, float, float, float]:
    """
    Read the SVG document's physical dimensions and viewBox to compute the
    scale needed to convert SVG user coordinates to the physical CSS-pixel
    coordinates required by Inkscape's --export-area parameter.

    Inkscape's --export-area uses a document-space coordinate system where:
      - The origin is the top-left of the viewBox (not the SVG user-unit origin)
      - Y increases downward (same direction as SVG)
      - Units are CSS pixels (96 dpi)

    Returns (scale_x, scale_y, offset_x, offset_y) such that:
        inkscape_x = svg_x * scale_x + offset_x
        inkscape_y = svg_y * scale_y + offset_y

    Raises RuntimeError if the SVG has no viewBox or width/height attributes.
    """
    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    viewbox_str = root.get("viewBox", "").strip()
    if not viewbox_str:
        raise RuntimeError(
            f"SVG '{svg_path}' has no viewBox attribute; "
            "cannot compute SVG-to-Inkscape coordinate scale."
        )
    vb_x, vb_y, vb_w, vb_h = map(float, viewbox_str.split())

    def _parse_length(s: str, fallback: float) -> float:
        """Convert a CSS length string to physical pixels at 96 dpi."""
        if not s:
            return fallback
        s = s.strip()
        if s.endswith("in"):
            return float(s[:-2]) * 96.0
        if s.endswith("mm"):
            return float(s[:-2]) * (96.0 / 25.4)
        if s.endswith("cm"):
            return float(s[:-2]) * (96.0 / 2.54)
        if s.endswith("pt"):
            return float(s[:-2]) * (96.0 / 72.0)
        if s.endswith("px"):
            return float(s[:-2])
        return float(s)  # bare number assumed to be px

    doc_w_px = _parse_length(root.get("width", ""), vb_w)
    doc_h_px = _parse_length(root.get("height", ""), vb_h)

    scale_x = doc_w_px / vb_w
    scale_y = doc_h_px / vb_h
    offset_x = -vb_x * scale_x
    offset_y = -vb_y * scale_y

    logger.info(
        "Document: %.0f x %.0f px  |  viewBox: (%.3f, %.3f) + %.3f x %.3f uu"
        "  |  scale: %.6f px/uu",
        doc_w_px, doc_h_px, vb_x, vb_y, vb_w, vb_h, scale_x,
    )
    return scale_x, scale_y, offset_x, offset_y


def svg_frame_to_inkscape(
    frame: Tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> Tuple[float, float, float, float]:
    """
    Convert a frame bounding box from SVG user units to Inkscape CSS-pixel
    coordinates suitable for --export-area.
    """
    x0, y0, x1, y1 = frame
    return (
        x0 * scale_x + offset_x,
        y0 * scale_y + offset_y,
        x1 * scale_x + offset_x,
        y1 * scale_y + offset_y,
    )


# ---------------------------------------------------------------------------
# Export-area computation -- gdal2tiles-style linear subdivision of the frame
# ---------------------------------------------------------------------------


def block_export_area(
    frame: Tuple[float, float, float, float],
    tiles_x: int,
    tiles_y: int,
    col_start: int,
    tms_row_bottom: int,
    block_cols: int,
    block_rows: int,
) -> Tuple[float, float, float, float]:
    """
    Return the Inkscape --export-area (x0:y0:x1:y1) for a rectangular block
    of tiles by linearly subdividing the frame bounding box.

    This is the gdal2tiles approach: the frame is divided into tiles_x x tiles_y
    equal-sized cells.  Because every zoom level subdivides the SAME frame
    boundaries, tiles at adjacent zoom levels align perfectly.

    TMS convention: tms_row_bottom=0 is the SOUTH/bottom row.
    Y increases downward, so the south edge has the LARGER y value.

    Parameters
    ----------
    frame          : (x0, y0, x1, y1) in Inkscape CSS-pixel coordinates,
                     as returned by svg_frame_to_inkscape().
                     y0 < y1:  y0 is the NORTH/top edge, y1 is the SOUTH/bottom.
    tiles_x        : number of tile columns at this zoom
    tiles_y        : number of tile rows at this zoom
    col_start      : westernmost tile column (0 = left/west edge of frame)
    tms_row_bottom : southernmost TMS row in the block (0 = south edge of frame)
    block_cols     : number of tile columns in this block
    block_rows     : number of tile rows in this block

    Returns
    -------
    (x0, y0, x1, y1) in Inkscape CSS-pixel coordinates, ready for
    --export-area.  y0 < y1 (north edge first).
    """
    fx0, fy0, fx1, fy1 = frame

    tile_w = (fx1 - fx0) / tiles_x
    tile_h = (fy1 - fy0) / tiles_y

    x0 = fx0 + col_start * tile_w
    x1 = fx0 + (col_start + block_cols) * tile_w

    # TMS row 0 = south = bottom of the frame (fy1).
    # TMS row tiles_y-1 = north = top of the frame (fy0).
    # North edge of this block (top of rendered image, smaller y):
    y0 = fy0 + (tiles_y - (tms_row_bottom + block_rows)) * tile_h
    # South edge of this block (bottom of rendered image, larger y):
    y1 = fy0 + (tiles_y - tms_row_bottom) * tile_h

    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Viewer config output
# ---------------------------------------------------------------------------


def viewer_config_string(
    min_zoom: int,
    max_zoom: int,
    tileset: str,
    extent: Tuple[float, float, float, float],
) -> str:
    """
    Return an OpenLayers TileGrid configuration snippet for map-manager.js
    that exactly matches the tile grid produced by this script.
    """
    x_min, y_min, x_max, y_max = extent
    x_span = x_max - x_min
    y_span = y_max - y_min

    def _dims(z: int) -> Tuple[int, int]:
        tx = 2 ** z
        ty = max(1, int(round(tx * (y_span / x_span))))
        return tx, ty

    min_tx, min_ty = _dims(min_zoom)
    max_tx, max_ty = _dims(max_zoom)

    resolutions = [x_span / (TILE_SIZE * (2 ** z)) for z in range(min_zoom, max_zoom + 1)]
    res_strs = ",\n            ".join(
        f"{r:.10f}  // z={z}"
        for z, r in zip(range(min_zoom, max_zoom + 1), resolutions)
    )
    return (
        "\n=== VIEWER CONFIG (paste into map-manager.js createTileLayer) ===\n"
        "\n"
        "// Tile extent in your supplied region units (Inkscape document units)\n"
        f"const tileExtent = [{x_min}, {y_min}, {x_max}, {y_max}];\n"
        "\n"
        "new ol.layer.Tile({\n"
        f"    title: 'Map Tiles ({tileset})',\n"
        "    source: new ol.source.TileImage({\n"
        "        tileGrid: new ol.tilegrid.TileGrid({\n"
        "            extent:      tileExtent,\n"
        f"            origin:      [{x_min}, {y_min}],  // SW corner\n"
        "            resolutions: [\n"
        f"            {res_strs}\n"
        "            ],\n"
        f"            tileSize: [{TILE_SIZE}, {TILE_SIZE}]\n"
        "        }),\n"
        "        tileUrlFunction: function(tileCoord) {\n"
        "            const z = tileCoord[0];\n"
        "            const x = tileCoord[1];\n"
        "            const y = -1 - tileCoord[2];  // TMS: y=0 at south\n"
        f"            return `YOUR_BASE_URL/{tileset}/${{z}}/${{x}}/${{y}}.png`;\n"
        "        }\n"
        "    })\n"
        "})\n"
        "\n"
        f"// z={min_zoom} = lowest res ({min_tx}x{min_ty} tiles), "
        f"z={max_zoom} = highest res ({max_tx}x{max_ty} tiles).\n"
        f"// Tile extent: x [{x_min}, {x_max}], y [{y_min}, {y_max}]\n"
        f"// Region span: {x_span} x {y_span} units\n"
        "// Non-square regions are supported; keep this extent and resolutions synchronized.\n"
        "================================================================="
    )


# ---------------------------------------------------------------------------
# Inkscape detection
# ---------------------------------------------------------------------------


def find_inkscape(override: Optional[str] = None) -> Path:
    """
    Locate the Inkscape executable.

    Checks in order:
        1. --inkscape command-line override
        2. 'inkscape' on the system PATH
        3. Common Windows installation directories
    """
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"Inkscape not found at override path: {override}")

    on_path = shutil.which("inkscape")
    if on_path:
        return Path(on_path)

    candidates = [
        Path(r"C:\Program Files\Inkscape\bin\inkscape.exe"),
        Path(r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Inkscape not found.  Install Inkscape or pass --inkscape <path>."
    )


# ---------------------------------------------------------------------------
# SVG layer visibility management
# ---------------------------------------------------------------------------


def _set_css_display(style: str, value: str) -> str:
    """Replace or insert 'display:<value>' in a CSS inline-style string."""
    parts = [p.strip() for p in style.split(";") if p.strip()]
    new_parts: List[str] = []
    found = False
    for part in parts:
        if part.lower().startswith("display"):
            new_parts.append(f"display:{value}")
            found = True
        else:
            new_parts.append(part)
    if not found:
        new_parts.insert(0, f"display:{value}")
    return ";".join(new_parts)


def set_layer_visibility(root: ET.Element, visible_labels: frozenset) -> None:
    """
    Walk direct children of *root* and set display:inline on layers whose
    inkscape:label is in *visible_labels*; set display:none on all others.

    Only top-level <g> elements (Inkscape layers) are touched.
    """
    for elem in root:
        if elem.tag != SVG_G:
            continue
        label = elem.get(INKSCAPE_LABEL, "")
        desired = "inline" if label in visible_labels else "none"
        elem.set("style", _set_css_display(elem.get("style", ""), desired))


def hide_frame_element(root: ET.Element) -> None:
    """
    Hide the historical frame rectangle so it does not appear in tiles.
    Searches the entire element tree and sets display:none on the element.
    """
    def _walk(elem: ET.Element) -> bool:
        label = elem.get(INKSCAPE_LABEL, "")
        eid   = elem.get("id", "")
        if label == "Frame (Old World)" or eid == "Frame (Old World)":
            elem.set("style", _set_css_display(elem.get("style", ""), "none"))
            logger.debug("Frame element hidden (id=%s, label=%s)", eid, label)
            return True
        for child in elem:
            if _walk(child):
                return True
        return False

    if not _walk(root):
        logger.warning(
            "Could not find historical frame rectangle to hide in temporary SVG; "
            "it may appear in rendered tiles.",
        )


def create_temp_svg(visible_labels: frozenset) -> Path:
    """
    Write a temporary copy of SVG_PATH with:
      - Layer visibility set (only VISIBLE_LAYERS shown).
      - 'Frame (Old World)' rectangle hidden so it does not appear in tiles.

    The original SVG is never modified.  Returns the path of the temp file;
    the caller is responsible for deleting it.
    """
    ET.register_namespace("",          NS_SVG)
    ET.register_namespace("svg",       NS_SVG)
    ET.register_namespace("inkscape",  NS_INKSCAPE)
    ET.register_namespace("sodipodi",  NS_SODIPODI)
    ET.register_namespace("xlink",     NS_XLINK)

    tree = ET.parse(str(SVG_PATH))
    root = tree.getroot()

    set_layer_visibility(root, visible_labels)
    hide_frame_element(root)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".svg", prefix="owa_ratmode_tiles_")
    os.close(tmp_fd)
    tree.write(tmp_path, encoding="unicode", xml_declaration=True)
    logger.debug("Temporary SVG: %s", tmp_path)
    return Path(tmp_path)


# ---------------------------------------------------------------------------
# Inkscape rendering
# ---------------------------------------------------------------------------


def render_block(
    inkscape:    Path,
    svg_path:    Path,
    export_area: Tuple[float, float, float, float],
    px_width:    int,
    px_height:   int,
    out_png:     Path,
    background:  str = "",
) -> bool:
    """
    Call Inkscape to render a rectangular SVG area to a PNG.

    Parameters
    ----------
    inkscape     : path to the inkscape executable
    svg_path     : source SVG (temporary modified copy)
    export_area  : (x0, y0, x1, y1) in SVG user coordinates for --export-area
    px_width     : output image width in pixels
    px_height    : output image height in pixels
    out_png      : destination PNG path
    background   : CSS colour for --export-background (empty = transparent)

    Returns True on success.
    """
    x0, y0, x1, y1 = export_area
    cmd = [
        str(inkscape),
        str(svg_path),
        f"--export-area={x0}:{y0}:{x1}:{y1}",
        f"--export-width={px_width}",
        f"--export-height={px_height}",
        f"--export-filename={out_png}",
    ]
    if background:
        cmd.append(f"--export-background={background}")

    logger.debug("Inkscape cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(
            "Inkscape exit code %d.\nSTDERR: %s",
            result.returncode,
            result.stderr[:1000],
        )
        return False
    if not out_png.exists():
        logger.error("Inkscape succeeded but output PNG not found: %s", out_png)
        return False
    return True


# ---------------------------------------------------------------------------
# Tile splitting
# ---------------------------------------------------------------------------


def split_block(
    block_png:      Path,
    n_cols:         int,
    n_rows:         int,
    out_dir:        Path,
    z:              int,
    col_start:      int,
    tms_row_bottom: int,
) -> int:
    """
    Split a rendered block PNG into TILE_SIZE x TILE_SIZE tiles and save them.

    The block image covers n_cols x n_rows tiles.
    Image row 0 (top) = NORTH edge = highest TMS y in the block.
    Image row (n_rows-1) (bottom) = SOUTH edge = tms_row_bottom.

    Output paths: <out_dir>/<z>/<x>/<y_tms>.png

    Returns the number of tiles written.
    """
    try:
        img = Image.open(block_png)
    except Exception as exc:
        logger.error("Cannot open block image %s: %s", block_png, exc)
        return 0

    expected_w = n_cols * TILE_SIZE
    expected_h = n_rows * TILE_SIZE
    if img.size != (expected_w, expected_h):
        logger.warning(
            "Block %s has size %s, expected %dx%d -- resampling.",
            block_png.name, img.size, expected_w, expected_h,
        )
        img = img.resize((expected_w, expected_h), Image.LANCZOS)

    count = 0
    tms_row_top = tms_row_bottom + n_rows - 1

    for row_idx in range(n_rows):
        # row_idx=0 (image top) -> northernmost tile -> highest TMS y
        y_tms = tms_row_top - row_idx
        for col_idx in range(n_cols):
            x_tile = col_start + col_idx
            left   = col_idx * TILE_SIZE
            top    = row_idx * TILE_SIZE
            tile   = img.crop((left, top, left + TILE_SIZE, top + TILE_SIZE))

            tile_dir = out_dir / str(z) / str(x_tile)
            tile_dir.mkdir(parents=True, exist_ok=True)
            tile.save(tile_dir / f"{y_tms}.png")
            count += 1

    return count


# ---------------------------------------------------------------------------
# Per-zoom-level generation
# ---------------------------------------------------------------------------


def generate_zoom_level(
    z:           int,
    frame:       Tuple[float, float, float, float],
    region_extent: Tuple[float, float, float, float],
    inkscape:    Path,
    svg_path:    Path,
    out_dir:     Path,
    block_tiles: int = DEFAULT_BLOCK_TILES,
    background:  str = "",
    dry_run:     bool = False,
) -> int:
    """
    Render and slice all tiles for zoom level *z*.

    Parameters
    ----------
    frame : SVG bounding box of the export region in Inkscape px.

    Returns the total number of tiles written (0 in dry-run mode).
    """
    region_x_min, region_y_min, region_x_max, region_y_max = region_extent
    region_span_x = region_x_max - region_x_min
    region_span_y = region_y_max - region_y_min

    # Define zoom by X density (2^z columns), then derive Y rows from aspect ratio.
    tiles_x = 2 ** z
    tiles_y = max(1, int(round(tiles_x * (region_span_y / region_span_x))))

    block_cols_base = min(block_tiles, tiles_x)
    block_rows_base = min(block_tiles, tiles_y)
    block_cols_count = math.ceil(tiles_x / block_cols_base)
    block_rows_count = math.ceil(tiles_y / block_rows_base)

    total_tiles  = 0
    total_blocks = block_cols_count * block_rows_count
    block_num    = 0

    logger.info(
        "Zoom %d: %dx%d tiles  |  max block=%dx%d tiles  |  %dx%d blocks = %d Inkscape calls",
        z, tiles_x, tiles_y,
        block_cols_base, block_rows_base,
        block_cols_count, block_rows_count,
        total_blocks,
    )

    tile_units_x = region_span_x / tiles_x
    tile_units_y = region_span_y / tiles_y

    with tempfile.TemporaryDirectory(prefix="owa_ratmode_blocks_") as tmpdir:
        for block_row in range(block_rows_count):
            # block_row=0 is the northernmost block row
            for block_col in range(block_cols_count):
                block_num += 1

                col_start = block_col * block_cols_base
                cols_this_block = min(block_cols_base, tiles_x - col_start)

                # block_row=0 -> northernmost -> highest TMS rows
                tms_row_top = tiles_y - 1 - block_row * block_rows_base
                rows_this_block = min(block_rows_base, tms_row_top + 1)
                tms_row_bottom = tms_row_top - rows_this_block + 1

                area = block_export_area(
                    frame,
                    tiles_x,
                    tiles_y,
                    col_start,
                    tms_row_bottom,
                    cols_this_block,
                    rows_this_block,
                )

                px_block_w = cols_this_block * TILE_SIZE
                px_block_h = rows_this_block * TILE_SIZE

                # Human-readable geographic bounds for logging
                x_lo = region_x_min + col_start * tile_units_x
                x_hi = region_x_min + (col_start + cols_this_block) * tile_units_x
                y_lo = region_y_min + tms_row_bottom * tile_units_y
                y_hi = region_y_min + (tms_row_bottom + rows_this_block) * tile_units_y

                logger.info(
                    "  [%d/%d] z=%d block(row=%d,col=%d) tms_y=%d..%d "
                    "region=(%.2f..%.2f X, %.2f..%.2f Y) "
                    "svg=(%.1f:%.1f:%.1f:%.1f) %dx%d px",
                    block_num, total_blocks,
                    z, block_row, block_col,
                    tms_row_bottom, tms_row_top,
                    x_lo, x_hi, y_lo, y_hi,
                    area[0], area[1], area[2], area[3],
                    px_block_w, px_block_h,
                )

                if dry_run:
                    logger.info(
                        "  [DRY RUN] tiles %d/%d/%d..%d  through  %d/%d/%d..%d",
                        z, col_start, tms_row_bottom, tms_row_top,
                        z, col_start + cols_this_block - 1, tms_row_bottom, tms_row_top,
                    )
                    continue

                block_png = Path(tmpdir) / f"z{z}_br{block_row}_bc{block_col}.png"

                if not render_block(
                    inkscape, svg_path, area,
                    px_block_w, px_block_h,
                    block_png, background,
                ):
                    logger.error("  Block failed -- skipping.")
                    continue

                n_written = split_block(
                    block_png, cols_this_block, rows_this_block,
                    out_dir, z, col_start, tms_row_bottom,
                )
                total_tiles += n_written
                logger.info("  %d tiles written", n_written)

    return total_tiles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Generate OpenLayers TMS PNG tiles from the Old World Atlas "
            "'ratmode' (Skaven Under-Empire) SVG.\n"
            "Output: <output_root>/<tileset>/<z>/<x>/<y>.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    p.add_argument(
        "--help", action="help",
        help="Show this help message and exit.",
    )
    p.add_argument(
        "-o", "--origin", nargs=2, type=float, metavar=("X", "Y"),
        help=(
            "Tile-region origin in display units (x y), e.g. -o -25 32. "
            "If omitted together with width/height, you will be prompted."
        ),
    )
    p.add_argument(
        "-w", "--width", type=float,
        help="Tile-region width in display units. Prompted if omitted.",
    )
    p.add_argument(
        "-h", "--height", type=float,
        help="Tile-region height in display units. Prompted if omitted.",
    )
    p.add_argument(
        "--tileset", default="ratmode_tiles",
        help="Tileset directory name under the repository root (default: ratmode_tiles)",
    )
    p.add_argument(
        "--min-zoom", type=int, default=MIN_ZOOM,
        help=f"First zoom level to generate (default: {MIN_ZOOM})",
    )
    p.add_argument(
        "--max-zoom", type=int,
        help=(
            f"Last zoom level to generate. If omitted, prompted at runtime "
            f"(default if blank: {MAX_ZOOM})."
        ),
    )
    p.add_argument(
        "--block-tiles", type=int, default=DEFAULT_BLOCK_TILES,
        help=(
            f"Tiles per block edge for chunk rendering (default: {DEFAULT_BLOCK_TILES}). "
            "Reduce to 16 if Inkscape runs out of memory at high zoom levels."
        ),
    )
    p.add_argument(
        "--inkscape",
        help="Explicit path to Inkscape.  Auto-detected if omitted.",
    )
    p.add_argument(
        "--background", default="",
        help=(
            "CSS colour for the tile background (e.g. 'white' or '#ffffff').  "
            "Defaults to transparent."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print every Inkscape command and resulting tile paths without "
            "rendering anything.  Use to verify extents before a long render run."
        ),
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Validate inputs ---------------------------------------------------
    if not SVG_PATH.exists():
        logger.error("Source SVG not found: %s", SVG_PATH)
        sys.exit(1)

    try:
        args.max_zoom = resolve_max_zoom(args)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if args.min_zoom < 1 or args.max_zoom > 8 or args.min_zoom > args.max_zoom:
        logger.error("Zoom levels must satisfy 1 <= min-zoom <= max-zoom <= 8.")
        sys.exit(1)

    try:
        region_origin_x, region_origin_y, region_w, region_h = resolve_region_inputs(args)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    region_extent = (
        region_origin_x,
        region_origin_y,
        region_origin_x + region_w,
        region_origin_y + region_h,
    )
    logger.info(
        "Region (display units): origin=(%.3f, %.3f) size=(%.3f x %.3f)",
        region_origin_x, region_origin_y, region_w, region_h,
    )

    inkscape = find_inkscape(args.inkscape)
    logger.info("Inkscape: %s", inkscape)

    out_dir = OUTPUT_ROOT / args.tileset
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # -- Build frame bounds from explicit region inputs ---------------------
    frame_svg = display_region_to_svg_bounds(
        region_origin_x, region_origin_y, region_w, region_h
    )
    logger.info(
        "Region mapped to SVG bounds: x0=%.3f y0=%.3f x1=%.3f y1=%.3f",
        *frame_svg,
    )

    # Convert SVG user-unit frame bounds to Inkscape physical-pixel coordinates.
    # Inkscape's --export-area parameter expects CSS px (96 dpi) referenced to
    # the document's viewBox origin, NOT raw SVG user units.
    try:
        sx, sy, ox, oy = read_doc_scale(SVG_PATH)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    frame = svg_frame_to_inkscape(frame_svg, sx, sy, ox, oy)
    logger.info(
        "Region in Inkscape px: x0=%.2f y0=%.2f x1=%.2f y1=%.2f  (%.1f x %.1f px)",
        *frame, frame[2] - frame[0], frame[3] - frame[1],
    )

    if not math.isclose(region_w, region_h):
        logger.info(
            "Non-square region detected (width %.3f != height %.3f). "
            "The export/splitting pipeline supports this.",
            region_w, region_h,
        )

    # -- Print viewer config -----------------------------------------------
    print(viewer_config_string(args.min_zoom, args.max_zoom, args.tileset, region_extent))

    # -- Sanity check: z=1 full extent should equal the frame exactly ------
    region_x_min, region_y_min, region_x_max, region_y_max = region_extent
    region_span_x = region_x_max - region_x_min
    region_span_y = region_y_max - region_y_min
    tiles_x_z1 = 2
    tiles_y_z1 = max(1, int(round(tiles_x_z1 * (region_span_y / region_span_x))))
    area_z1 = block_export_area(
        frame,
        tiles_x_z1,
        tiles_y_z1,
        0,
        0,
        tiles_x_z1,
        tiles_y_z1,
    )
    logger.info(
        "Sanity check z=1 full-extent area (Inkscape px): x0=%.2f y0=%.2f x1=%.2f y1=%.2f",
        *area_z1,
    )
    logger.info("  Region bounds (Inkscape px):                   x0=%.2f y0=%.2f x1=%.2f y1=%.2f", *frame)

    # -- Create temporary SVG ----------------------------------------------
    logger.info("Preparing temporary SVG (layer visibility + frame hidden)...")
    logger.info("  Visible layers: %s", sorted(VISIBLE_LAYERS))
    tmp_svg = create_temp_svg(VISIBLE_LAYERS)
    logger.info("  Temporary SVG: %s", tmp_svg)

    # -- Generate tiles ----------------------------------------------------
    grand_total = 0
    try:
        for z in range(args.min_zoom, args.max_zoom + 1):
            logger.info("==========================================")
            logger.info("Zoom level %d / %d", z, args.max_zoom)
            n = generate_zoom_level(
                z           = z,
                frame       = frame,
                region_extent = region_extent,
                inkscape    = inkscape,
                svg_path    = tmp_svg,
                out_dir     = out_dir,
                block_tiles = args.block_tiles,
                background  = args.background,
                dry_run     = args.dry_run,
            )
            grand_total += n
            if not args.dry_run:
                logger.info("Zoom %d complete: %d tiles written.", z, n)
    finally:
        try:
            tmp_svg.unlink()
            logger.debug("Temporary SVG removed.")
        except Exception:
            logger.warning("Could not remove temporary SVG: %s", tmp_svg)

    logger.info("==========================================")
    if args.dry_run:
        logger.info("Dry run complete.  No files written.")
    else:
        logger.info("Tile generation complete.  Grand total: %d tiles.", grand_total)
        logger.info("Output: %s", out_dir)


if __name__ == "__main__":
    main()
