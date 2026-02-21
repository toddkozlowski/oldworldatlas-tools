"""generate_tiles.py — Generate OpenLayers-compatible TMS PNG tiles from the Old World Atlas SVG.

Frame-based approach (gdal2tiles-compatible):
    The SVG contains a rectangle object named "Frame (Old World)" whose corners
    define the exact region to be tiled.  Its SVG bounding box is read at
    startup and ALL tile export areas are derived by linearly subdividing that
    frame — guaranteeing perfect alignment across every zoom level.

Geographic extent (defined by the Frame object):
    West:  -19 degrees longitude
    East:   18 degrees longitude
    South:  32 degrees latitude
    North:  69 degrees latitude
    Span:   37 x 37 degrees (square grid)

Tile convention (TMS, matching OpenLayers TileGrid with y=0 at south):
    On disk: <output_root>/<tileset_name>/<z>/<x>/<y>.png

    z = 1 ->   2 x   2 tiles
    z = 2 ->   4 x   4 tiles
    z = 8 -> 256 x 256 tiles

    Each tile is always 256 x 256 pixels.

Layer visibility:
    Only the layers listed in VISIBLE_LAYERS are rendered.  The "Frame (Old
    World)" rect is also hidden in the temporary SVG so it does not appear in
    the rendered tiles.  The original SVG is never modified.

Usage:
    python generate_tiles.py [--tileset NAME] [--min-zoom Z] [--max-zoom Z]
                             [--block-tiles N] [--inkscape PATH]
                             [--background COLOR] [--dry-run] [--verbose]

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
    Path(__file__).parent.parent.parent / "oldworldatlas-maps" / "OLD_WORLD_ATLAS.svg"
)
OUTPUT_ROOT = Path(
    r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-repository"
)

# ---------------------------------------------------------------------------
# Geographic extent -- MUST match the "Frame (Old World)" object's geo meaning
# ---------------------------------------------------------------------------

GEO_X_MIN: float = -19.0   # westernmost longitude (degrees)
GEO_X_MAX: float =  18.0   # easternmost longitude
GEO_Y_MIN: float =  32.0   # southernmost latitude
GEO_Y_MAX: float =  69.0   # northernmost latitude
GEO_X_SPAN: float = GEO_X_MAX - GEO_X_MIN   # 37.0 degrees
GEO_Y_SPAN: float = GEO_Y_MAX - GEO_Y_MIN   # 37.0 degrees

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
# Frame configuration
# ---------------------------------------------------------------------------

# The inkscape:label of the rectangle that defines the tile region.
FRAME_LABEL: str = "Frame (Old World)"

# ---------------------------------------------------------------------------
# Layer visibility
# ---------------------------------------------------------------------------
#
# These top-level layer labels (inkscape:label) are made visible in the
# temporary SVG.  Every other top-level layer is hidden with display:none.
# ---------------------------------------------------------------------------

VISIBLE_LAYERS: frozenset = frozenset({
    "topography",         # elevation colour scale
    "Forests",
    "Marshes",
    "Urban Areas",
    "Bretonnian Rivers",
    "Rivers",
    "Bretonnia Lakes",
    "Lakes",
    "Roads",
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
# Frame detection
# ---------------------------------------------------------------------------


def find_frame_bounds(svg_path: Path) -> Tuple[float, float, float, float]:
    """
    Locate the 'Frame (Old World)' rectangle in the SVG and return its
    axis-aligned bounding box in SVG user coordinates: (x0, y0, x1, y1).

    x0/y0 = north-west corner (smaller SVG x and y values).
    x1/y1 = south-east corner (larger SVG x and y values).

    SVG y increases downward, so y0 is the NORTH edge and y1 is the SOUTH edge.

    The search walks the entire document tree accumulating transforms; handles
    translate and matrix forms used by Inkscape.

    Raises RuntimeError if the frame cannot be found or is not a <rect>.
    """
    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    identity: Tuple[float, float, float, float, float, float] = (
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    )

    def _walk(elem: ET.Element, parent_tf: Tuple) -> Optional[Tuple]:
        """DFS; returns (element, cumulative_transform) when frame is found."""
        tf = _compose(parent_tf, _parse_svg_transform(elem.get("transform", "")))
        label = elem.get(INKSCAPE_LABEL, "")
        eid   = elem.get("id", "")
        if label == FRAME_LABEL or eid == FRAME_LABEL:
            return (elem, tf)
        for child in elem:
            result = _walk(child, tf)
            if result is not None:
                return result
        return None

    found = _walk(root, identity)
    if found is None:
        raise RuntimeError(
            f"'{FRAME_LABEL}' not found in {svg_path}.\n"
            "  Open the SVG in Inkscape, select the bounding rectangle, and\n"
            f"  set Object Properties -> Label to exactly: {FRAME_LABEL}"
        )

    elem, tf = found
    if elem.tag not in (SVG_RECT, f"rect"):
        raise RuntimeError(
            f"'{FRAME_LABEL}' has element tag '{elem.tag}' -- expected a <rect>.\n"
            "  Draw the frame with Inkscape's Rectangle tool."
        )

    rx = float(elem.get("x", 0.0))
    ry = float(elem.get("y", 0.0))
    rw = float(elem.get("width", 0.0))
    rh = float(elem.get("height", 0.0))

    corners = [
        _apply(tf, rx,      ry),
        _apply(tf, rx + rw, ry),
        _apply(tf, rx,      ry + rh),
        _apply(tf, rx + rw, ry + rh),
    ]

    x0 = min(c[0] for c in corners)
    y0 = min(c[1] for c in corners)   # north edge (smallest SVG y)
    x1 = max(c[0] for c in corners)
    y1 = max(c[1] for c in corners)   # south edge (largest SVG y)

    logger.info(
        "Frame '%s' SVG user-unit bounds: x0=%.3f y0=%.3f x1=%.3f y1=%.3f  (%.1f x %.1f uu)",
        FRAME_LABEL, x0, y0, x1, y1, x1 - x0, y1 - y0,
    )
    logger.info(
        "  Geographically: lon [%.1f, %.1f], lat [%.1f, %.1f]",
        GEO_X_MIN, GEO_X_MAX, GEO_Y_MIN, GEO_Y_MAX,
    )
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
    z: int,
    col_start: int,
    tms_row_bottom: int,
    block_tiles: int,
) -> Tuple[float, float, float, float]:
    """
    Return the Inkscape --export-area (x0:y0:x1:y1) for a rectangular block
    of tiles by linearly subdividing the frame bounding box.

    This is the gdal2tiles approach: the frame is divided into 2^z x 2^z
    equal-sized cells.  Because every zoom level subdivides the SAME frame
    boundaries, tiles at adjacent zoom levels align perfectly.

    TMS convention: tms_row_bottom=0 is the SOUTH/bottom row.
    Y increases downward, so the south edge has the LARGER y value.

    Parameters
    ----------
    frame          : (x0, y0, x1, y1) in Inkscape CSS-pixel coordinates,
                     as returned by svg_frame_to_inkscape().
                     y0 < y1:  y0 is the NORTH/top edge, y1 is the SOUTH/bottom.
    z              : zoom level
    col_start      : westernmost tile column (0 = left/west edge of frame)
    tms_row_bottom : southernmost TMS row in the block (0 = south edge of frame)
    block_tiles    : number of tiles per block edge

    Returns
    -------
    (x0, y0, x1, y1) in Inkscape CSS-pixel coordinates, ready for
    --export-area.  y0 < y1 (north edge first).
    """
    fx0, fy0, fx1, fy1 = frame
    n = 2 ** z

    tile_w = (fx1 - fx0) / n
    tile_h = (fy1 - fy0) / n

    x0 = fx0 + col_start * tile_w
    x1 = fx0 + (col_start + block_tiles) * tile_w

    # TMS row 0 = south = bottom of the frame (fy1).
    # TMS row n-1 = north = top of the frame (fy0).
    # North edge of this block (top of rendered image, smaller y):
    y0 = fy0 + (n - (tms_row_bottom + block_tiles)) * tile_h
    # South edge of this block (bottom of rendered image, larger y):
    y1 = fy0 + (n - tms_row_bottom) * tile_h

    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Viewer config output
# ---------------------------------------------------------------------------


def viewer_config_string(min_zoom: int, max_zoom: int, tileset: str) -> str:
    """
    Return an OpenLayers TileGrid configuration snippet for map-manager.js
    that exactly matches the tile grid produced by this script.
    """
    resolutions = [
        GEO_X_SPAN / (TILE_SIZE * (2 ** z))
        for z in range(min_zoom, max_zoom + 1)
    ]
    res_strs = ",\n            ".join(
        f"{r:.10f}  // z={z}"
        for z, r in zip(range(min_zoom, max_zoom + 1), resolutions)
    )
    return (
        "\n=== VIEWER CONFIG (paste into map-manager.js createTileLayer) ===\n"
        "\n"
        f"// Tile extent matches Frame (Old World) geographic bounds\n"
        f"const tileExtent = [{GEO_X_MIN}, {GEO_Y_MIN}, {GEO_X_MAX}, {GEO_Y_MAX}];\n"
        "\n"
        "new ol.layer.Tile({\n"
        f"    title: 'Map Tiles ({tileset})',\n"
        "    source: new ol.source.TileImage({\n"
        "        tileGrid: new ol.tilegrid.TileGrid({\n"
        "            extent:      tileExtent,\n"
        f"            origin:      [{GEO_X_MIN}, {GEO_Y_MIN}],  // SW corner\n"
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
        f"// z={min_zoom} = lowest res ({2**min_zoom}x{2**min_zoom} tiles), "
        f"z={max_zoom} = highest res ({2**max_zoom}x{2**max_zoom} tiles).\n"
        f"// Tile extent: lon [{GEO_X_MIN}, {GEO_X_MAX}], lat [{GEO_Y_MIN}, {GEO_Y_MAX}]\n"
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
    Hide the 'Frame (Old World)' rectangle so it does not appear in tiles.
    Searches the entire element tree and sets display:none on the element.
    """
    def _walk(elem: ET.Element) -> bool:
        label = elem.get(INKSCAPE_LABEL, "")
        eid   = elem.get("id", "")
        if label == FRAME_LABEL or eid == FRAME_LABEL:
            elem.set("style", _set_css_display(elem.get("style", ""), "none"))
            logger.debug("Frame element hidden (id=%s, label=%s)", eid, label)
            return True
        for child in elem:
            if _walk(child):
                return True
        return False

    if not _walk(root):
        logger.warning(
            "Could not find '%s' to hide in the temporary SVG; "
            "it may appear in rendered tiles.",
            FRAME_LABEL,
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

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".svg", prefix="owa_tiles_")
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
    frame : SVG bounding box of the 'Frame (Old World)' rectangle,
            as returned by find_frame_bounds().

    Returns the total number of tiles written (0 in dry-run mode).
    """
    tiles_per_side  = 2 ** z
    actual_block    = min(block_tiles, tiles_per_side)
    blocks_per_side = tiles_per_side // actual_block
    px_block        = actual_block * TILE_SIZE

    total_tiles  = 0
    total_blocks = blocks_per_side * blocks_per_side
    block_num    = 0

    logger.info(
        "Zoom %d: %dx%d tiles  |  block=%d tiles (%dpx)  |  %dx%d blocks = %d Inkscape calls",
        z, tiles_per_side, tiles_per_side,
        actual_block, px_block,
        blocks_per_side, blocks_per_side,
        total_blocks,
    )

    tile_deg_x = GEO_X_SPAN / tiles_per_side
    tile_deg_y = GEO_Y_SPAN / tiles_per_side

    with tempfile.TemporaryDirectory(prefix="owa_blocks_") as tmpdir:
        for block_row in range(blocks_per_side):
            # block_row=0 is the northernmost block row
            for block_col in range(blocks_per_side):
                block_num += 1

                col_start = block_col * actual_block

                # block_row=0 -> northernmost -> highest TMS rows
                tms_row_top    = tiles_per_side - 1 - block_row * actual_block
                tms_row_bottom = tms_row_top - actual_block + 1

                area = block_export_area(frame, z, col_start, tms_row_bottom, actual_block)

                # Human-readable geographic bounds for logging
                lon_lo = GEO_X_MIN + col_start * tile_deg_x
                lon_hi = GEO_X_MIN + (col_start + actual_block) * tile_deg_x
                lat_lo = GEO_Y_MIN + tms_row_bottom * tile_deg_y
                lat_hi = GEO_Y_MIN + (tms_row_bottom + actual_block) * tile_deg_y

                logger.info(
                    "  [%d/%d] z=%d block(row=%d,col=%d) tms_y=%d..%d "
                    "geo=(%.2f..%.2f E, %.2f..%.2f N) "
                    "svg=(%.1f:%.1f:%.1f:%.1f) %dpx",
                    block_num, total_blocks,
                    z, block_row, block_col,
                    tms_row_bottom, tms_row_top,
                    lon_lo, lon_hi, lat_lo, lat_hi,
                    area[0], area[1], area[2], area[3],
                    px_block,
                )

                if dry_run:
                    logger.info(
                        "  [DRY RUN] tiles %d/%d/%d..%d  through  %d/%d/%d..%d",
                        z, col_start, tms_row_bottom, tms_row_top,
                        z, col_start + actual_block - 1, tms_row_bottom, tms_row_top,
                    )
                    continue

                block_png = Path(tmpdir) / f"z{z}_br{block_row}_bc{block_col}.png"

                if not render_block(
                    inkscape, svg_path, area,
                    px_block, px_block,
                    block_png, background,
                ):
                    logger.error("  Block failed -- skipping.")
                    continue

                n_written = split_block(
                    block_png, actual_block, actual_block,
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
            "Generate OpenLayers TMS PNG tiles from the Old World Atlas SVG.\n"
            "Output: <output_root>/<tileset>/<z>/<x>/<y>.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tileset", default="test_tiles",
        help="Tileset directory name under the repository root (default: test_tiles)",
    )
    p.add_argument(
        "--min-zoom", type=int, default=MIN_ZOOM,
        help=f"First zoom level to generate (default: {MIN_ZOOM})",
    )
    p.add_argument(
        "--max-zoom", type=int, default=MAX_ZOOM,
        help=f"Last zoom level to generate (default: {MAX_ZOOM})",
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

    if args.min_zoom < 1 or args.max_zoom > 8 or args.min_zoom > args.max_zoom:
        logger.error("Zoom levels must satisfy 1 <= min-zoom <= max-zoom <= 8.")
        sys.exit(1)

    inkscape = find_inkscape(args.inkscape)
    logger.info("Inkscape: %s", inkscape)

    out_dir = OUTPUT_ROOT / args.tileset
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # -- Read frame bounds from the SVG ------------------------------------
    logger.info("Reading '%s' bounds from SVG...", FRAME_LABEL)
    try:
        frame_svg = find_frame_bounds(SVG_PATH)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

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
        "Frame in Inkscape px: x0=%.2f y0=%.2f x1=%.2f y1=%.2f  (%.1f x %.1f px)",
        *frame, frame[2] - frame[0], frame[3] - frame[1],
    )

    # -- Print viewer config -----------------------------------------------
    print(viewer_config_string(args.min_zoom, args.max_zoom, args.tileset))

    # -- Sanity check: z=1 full extent should equal the frame exactly ------
    area_z1 = block_export_area(frame, 1, 0, 0, 2)
    logger.info(
        "Sanity check z=1 full-extent area (Inkscape px): x0=%.2f y0=%.2f x1=%.2f y1=%.2f",
        *area_z1,
    )
    logger.info(
        "  Frame bounds (Inkscape px):                    x0=%.2f y0=%.2f x1=%.2f y1=%.2f",
        *frame,
    )

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
