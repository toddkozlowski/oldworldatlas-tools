"""
Comprehensive SVG map processing tool for the Old World Atlas.
Extracts settlements, points of interest, and labels from the FULL_MAP_CLEANED.svg file.
"""

import json
import csv
import logging
import math
from platform import processor
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from xml.etree import ElementTree as ET
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from scipy.interpolate import CubicSpline
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
SVG_PATH = Path(__file__).parent.parent.parent / "oldworldatlas-maps" / "OLD_WORLD_ATLAS.svg"
INPUT_DIR = Path(__file__).parent.parent / "input" / "gazetteers"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
LOGS_DIR = Path(__file__).parent.parent / "logs"

# ---------------------------------------------------------------------------
# Coordinate conversion: Inkscape display formula
# ---------------------------------------------------------------------------
# The Inkscape document has been configured so that its *display* coordinates
# (the values shown in the Inkscape status bar) equal geospatial coordinates
# (degrees longitude / latitude) for an equirectangular projection.
# Settings used:
#   - Display unit : inches, scale = 0.017504 (so 1 SVG user-unit = SCALE degrees)
#   - Viewbox origin : x = 428.530, y = -1470.000 (SVG user units)
#   - Y-axis inverted : positive display-y points north (up)
# Conversion from *absolute* SVG coordinates (after all layer transforms have
# been flattened and applied) to geographic coordinates:
#
#   geo_lon = (svg_x_abs - INKSCAPE_VB_X) * INKSCAPE_SCALE
#   geo_lat = (INKSCAPE_C_Y - svg_y_abs) * INKSCAPE_SCALE
#
# INKSCAPE_C_Y is derived empirically by averaging the four calibration points
# below (≈ 3347.85).  It encodes both the viewbox y-origin and the document
# height, and does not need to be changed unless the SVG canvas is resized.
# ---------------------------------------------------------------------------
INKSCAPE_VB_X   = 428.530   # SVG absolute x corresponding to geo longitude 0.0°
INKSCAPE_SCALE  = 0.017504  # Degrees per SVG user unit (both axes)
INKSCAPE_C_Y    = 3347.85   # y-axis constant; geo_lat = (C_Y - svg_y_abs) * SCALE

# Reference calibration points (for validation only).
# SVG coords are the *absolute* positions after flattening all layer transforms.
# Geo coords are the Inkscape display coordinates (= geospatial coordinates).
CALIBRATION_POINTS = [
    {"svg": (428.519, 407.885), "geo": (0.000, 51.460),
     "settlement": "Altdorf",           "province": "Reikland"},
    {"svg": (495.376, 187.843), "geo": (1.172, 55.312),
     "settlement": "Middenheim",        "province": "Middenland"},
    {"svg": (738.284, 778.673), "geo": (5.422, 44.970),
     "settlement": "Wachdorf",          "province": "Averland"},
    {"svg": (891.496, 479.299), "geo": (8.104, 50.210),
     "settlement": "Waldenhof (Sylvania)", "province": "Stirland"},
]

NS = {
    'svg': 'http://www.w3.org/2000/svg',
    'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
}


@dataclass
class Settlement:
    """Represents a settlement."""
    name: str
    province: str
    svg_x: float
    svg_y: float
    geo_lon: float = 0.0
    geo_lat: float = 0.0
    population: int = 0
    size_category: int = 1
    tags: List[str] = None
    notes: List[str] = None
    wiki: Dict = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.notes is None:
            self.notes = []
        if self.wiki is None:
            self.wiki = {
                "title": None,
                "url": None,
                "description": None,
                "image": None
            }


@dataclass
class PointOfInterest:
    """Represents a point of interest."""
    name: str
    poi_type: str
    svg_x: float
    svg_y: float
    geo_lon: float = 0.0
    geo_lat: float = 0.0


@dataclass
class Road:
    """Represents a road."""
    road_id: str
    road_type: str
    svg_path: str
    geo_coordinates: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.geo_coordinates is None:
            self.geo_coordinates = []


@dataclass
class ProvinceLabel:
    """Represents a political/province label."""
    name: str
    province_type: str
    svg_x: float
    svg_y: float
    geo_lon: float = 0.0
    geo_lat: float = 0.0
    formal_title: str = ""
    part_of: str = ""
    population: int = 0
    info: Dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {
                "wiki_url": None,
                "image": None,
                "description": None
            }


@dataclass
class WaterLabel:
    """Represents a water body label."""
    name: str
    waterbody_type: str
    svg_x: float
    svg_y: float
    geo_lon: float = 0.0
    geo_lat: float = 0.0


@dataclass
class DwarfHold:
    """Represents a Dwarf Hold (Karaz Ankor settlement)."""
    name: str
    svg_x: float
    svg_y: float
    geo_lon: float = 0.0
    geo_lat: float = 0.0
    hold_type: str = ""
    tags: List[str] = None
    notes: List[str] = None
    wiki: Dict = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.notes is None:
            self.notes = []
        if self.wiki is None:
            self.wiki = {
                "title": None,
                "url": None,
                "description": None,
                "image": None
            }


# SVG affine transform matrix type alias: (a, b, c, d, e, f)
# Applies as:  x' = a*x + c*y + e
#              y' = b*x + d*y + f
# Identity  :  (1, 0, 0, 1, 0, 0)
TransformMatrix = Tuple[float, float, float, float, float, float]
IDENTITY_MATRIX: TransformMatrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


class CoordinateConverter:
    """Converts absolute SVG coordinates to geographic coordinates.

    Uses the direct Inkscape display formula:
        geo_lon = (svg_x_abs - INKSCAPE_VB_X) * INKSCAPE_SCALE
        geo_lat = (INKSCAPE_C_Y - svg_y_abs) * INKSCAPE_SCALE

    This works because Inkscape has been configured so that the display
    coordinates (shown in the status bar) ARE the geospatial coordinates.
    The svg_x_abs / svg_y_abs values must be *absolute* SVG coordinates,
    i.e., the element position after all ancestor layer transforms have
    been composed and applied.
    """

    def __init__(self, calibration_points: List[Dict]):
        """Initialize with calibration points (used for validation only)."""
        self.calibration_points = calibration_points

    def svg_to_geo(self, svg_x: float, svg_y: float) -> Tuple[float, float]:
        """Convert absolute SVG coordinates to geographic (lon, lat)."""
        lon = (svg_x - INKSCAPE_VB_X) * INKSCAPE_SCALE
        lat = (INKSCAPE_C_Y - svg_y) * INKSCAPE_SCALE
        return (lon, lat)

    def validate_calibration(self):
        """Validate the formula against all calibration points and log results."""
        logger.info("Validating coordinate conversion formula:")
        for point in self.calibration_points:
            calc_geo = self.svg_to_geo(point["svg"][0], point["svg"][1])
            expected_geo = point["geo"]
            lon_err = calc_geo[0] - expected_geo[0]
            lat_err = calc_geo[1] - expected_geo[1]
            dist_err = math.sqrt(lon_err**2 + lat_err**2)
            logger.info(
                f"  {point['settlement']:25s}: "
                f"calc=({calc_geo[0]:.4f}, {calc_geo[1]:.4f})  "
                f"expected=({expected_geo[0]:.4f}, {expected_geo[1]:.4f})  "
                f"error={dist_err:.5f}°"
            )


class SVGMapProcessor:
    """Processes the SVG map file."""

    def __init__(self):
        """Initialize processor."""
        self.tree = ET.parse(str(SVG_PATH))
        self.root = self.tree.getroot()
        self.converter = CoordinateConverter(CALIBRATION_POINTS)
        self.converter.validate_calibration()

        self.settlements_empire = []
        self.settlements_westerland = []
        self.settlements_bretonnia = []
        self.settlements_karaz_ankor = []
        self.points_of_interest = []
        self.roads = []
        self.province_labels = []
        self.water_labels = []

        self.invalid_settlements = []
        self.duplicate_settlements = defaultdict(list)
        self.missing_population_data = defaultdict(list)
        
        # Track CSV data
        self.csv_data_empire = {}  # {province: {name: row_data}}
        self.csv_data_westerland = {}  # {name: row_data}
        self.csv_data_bretonnia = {}  # {name: row_data}
        self.csv_data_karaz_ankor = {}  # {name: row_data}
        self.csv_data_provinces = {}  # {name: row_data}
        
        # Track validation issues
        self.csv_settlements_not_in_svg = defaultdict(list)  # {province: [names]}
        self.svg_settlements_not_in_csv = defaultdict(list)  # {province: [names]}
        self.province_mismatches = []  # List of {settlement, province_svg, province_csv}
        self.invalid_tags = []  # List of {settlement, tags, issues}
        
        # Track province label validation issues
        self.csv_provinces_not_in_svg = []  # List of province names
        self.svg_provinces_not_in_csv = []  # List of province names

        # Track all groups/layers whose transforms were detected and absorbed.
        # Each entry: {"layer_path": str, "transform": str, "context": str}
        self.layers_with_transforms: List[Dict] = []

    def _get_text_element_label(self, elem) -> Optional[str]:
        """Extract text from text/tspan elements."""
        text_content = []
        
        # Check if it's a text element directly
        if elem.tag == f"{{{NS['svg']}}}text":
            # Look for tspan children
            for tspan in elem.findall(f"{{{NS['svg']}}}tspan"):
                if tspan.text:
                    text_content.append(tspan.text.strip())
            # If no tspan, check text directly
            if not text_content and elem.text:
                text_content.append(elem.text.strip())
        
        return " ".join(text_content) if text_content else None

    def _parse_transform_to_matrix(self, transform_str: str) -> TransformMatrix:
        """Parse an SVG transform string into an affine matrix (a, b, c, d, e, f).

        Handles: translate, scale, matrix, rotate and sequences of transforms.
        Returns the identity matrix for empty or unrecognised input.

        SVG matrix convention (column-vector):
            x' = a*x + c*y + e
            y' = b*x + d*y + f
        """
        if not transform_str or not transform_str.strip():
            return IDENTITY_MATRIX

        result: TransformMatrix = IDENTITY_MATRIX

        for match in re.finditer(r'([a-zA-Z]+)\s*\(([^)]+)\)', transform_str):
            func = match.group(1)
            args_str = match.group(2).strip()
            try:
                args = [float(v) for v in re.split(r'[,\s]+', args_str) if v]
            except ValueError:
                logger.warning(f"Unreadable transform args in: {match.group(0)!r}")
                continue

            if func == 'translate':
                tx = args[0] if len(args) >= 1 else 0.0
                ty = args[1] if len(args) >= 2 else 0.0
                t: TransformMatrix = (1.0, 0.0, 0.0, 1.0, tx, ty)
            elif func == 'scale':
                sx = args[0] if len(args) >= 1 else 1.0
                sy = args[1] if len(args) >= 2 else sx
                t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
            elif func == 'matrix':
                if len(args) >= 6:
                    t = (args[0], args[1], args[2], args[3], args[4], args[5])
                else:
                    logger.warning(f"Incomplete matrix transform (need 6 values): {match.group(0)!r}")
                    continue
            elif func == 'rotate':
                angle = math.radians(args[0]) if args else 0.0
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                if len(args) >= 3:
                    cx, cy = args[1], args[2]
                    t = (cos_a, sin_a, -sin_a, cos_a,
                         cx - cx * cos_a + cy * sin_a,
                         cy - cx * sin_a - cy * cos_a)
                else:
                    t = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            else:
                logger.warning(f"Unrecognised transform function: {func!r} in {transform_str!r}")
                continue

            # Compose: accumulate left-to-right per SVG spec (result applied first, t applied second)
            result = self._compose_transforms(outer=result, inner=t)

        return result

    def _compose_transforms(self, outer: TransformMatrix, inner: TransformMatrix) -> TransformMatrix:
        """Compose two affine transforms: apply *inner* first, then *outer*.

        Equivalent to the matrix product outer * inner.
        Uses standard SVG/matrix convention (column vectors):
            (outer \u2218 inner)(x) = outer(inner(x))
        """
        a1, b1, c1, d1, e1, f1 = outer
        a2, b2, c2, d2, e2, f2 = inner
        return (
            a1 * a2 + c1 * b2,   # a
            b1 * a2 + d1 * b2,   # b
            a1 * c2 + c1 * d2,   # c
            b1 * c2 + d1 * d2,   # d
            a1 * e2 + c1 * f2 + e1,  # e
            b1 * e2 + d1 * f2 + f1,  # f
        )

    def _apply_transform_matrix(self, x: float, y: float, matrix: TransformMatrix) -> Tuple[float, float]:
        """Apply affine transform matrix (a,b,c,d,e,f) to point (x, y)."""
        a, b, c, d, e, f = matrix
        return (a * x + c * y + e, b * x + d * y + f)

    def _get_group_transform(
        self,
        group_elem,
        layer_path: str,
        parent_matrix: TransformMatrix = IDENTITY_MATRIX,
    ) -> TransformMatrix:
        """Return the composed transform for *group_elem* within its parent context.

        If the element has a 'transform' attribute, that transform is:
          1. Parsed into an affine matrix.
          2. Logged to self.layers_with_transforms.
          3. Composed with parent_matrix (parent is applied first, then element).

        The composed matrix represents the total coordinate mapping from the
        group's local coordinate space to absolute SVG space.
        """
        transform_str = group_elem.get("transform", "")
        if not transform_str:
            return parent_matrix

        label = (
            group_elem.get(f"{{{NS['inkscape']}}}label")
            or group_elem.get("id", "<unlabeled group>")
        )
        full_path = f"{layer_path}/{label}" if layer_path else label

        self.layers_with_transforms.append({
            "layer_path": full_path,
            "transform": transform_str,
            "context": f"Group '{label}' in layer path '{layer_path}'",
        })
        logger.debug(f"Layer transform absorbed: {full_path!r} -> {transform_str}")

        elem_matrix = self._parse_transform_to_matrix(transform_str)
        # child_absolute = parent_matrix * elem_matrix
        return self._compose_transforms(outer=parent_matrix, inner=elem_matrix)

    # ------------------------------------------------------------------
    # Legacy helper kept for backward compatibility with any callers that
    # still pass a transform string directly.  New code should use
    # _get_group_transform / _apply_transform_matrix instead.
    # ------------------------------------------------------------------
    def _apply_svg_transform(self, x: float, y: float, transform: str) -> Tuple[float, float]:
        """Apply an SVG transform string to (x, y) and return the new coordinates."""
        if not transform:
            return (x, y)
        matrix = self._parse_transform_to_matrix(transform)
        return self._apply_transform_matrix(x, y, matrix)

    def _validate_settlement_element(
        self, elem, province: str,
        parent_matrix: TransformMatrix = IDENTITY_MATRIX,
        layer_path: str = "Settlements",
    ) -> Optional[Tuple[str, float, float]]:
        """Validate a text element and return (name, abs_svg_x, abs_svg_y).

        All ancestor layer transforms must already be encoded in *parent_matrix*.
        Any transform on the text element itself is also detected, logged, and
        absorbed so that the returned coordinates are absolute SVG coordinates
        (suitable for direct conversion with CoordinateConverter.svg_to_geo).
        """
        # Must be a text element
        if elem.tag != f"{{{NS['svg']}}}text":
            self.invalid_settlements.append({
                "province": province,
                "element": elem.tag,
                "reason": "Not a text element",
                "attributes": dict(elem.attrib)
            })
            return None

        # Get text content
        name = self._get_text_element_label(elem)
        if not name:
            self.invalid_settlements.append({
                "province": province,
                "element": "text",
                "reason": "No text content",
                "attributes": dict(elem.attrib)
            })
            return None

        # Extract base coordinates from the text element
        try:
            svg_x = float(elem.get("x", 0))
            svg_y = float(elem.get("y", 0))

            # Detect and absorb any transform on the text element itself
            elem_transform_str = elem.get("transform", "")
            if elem_transform_str:
                elem_path = f"{layer_path}/text:'{name}'"
                self.layers_with_transforms.append({
                    "layer_path": elem_path,
                    "transform": elem_transform_str,
                    "context": f"Text element '{name}' in province '{province}'",
                })
                logger.debug(f"Text element transform absorbed: {elem_path!r}")
                elem_matrix = self._parse_transform_to_matrix(elem_transform_str)
                svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, elem_matrix)

            # Apply the fully accumulated ancestor transform to reach absolute SVG space
            svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, parent_matrix)

            return (name, svg_x, svg_y)
        except ValueError:
            self.invalid_settlements.append({
                "province": province,
                "element": "text",
                "name": name,
                "reason": "Invalid coordinates",
                "attributes": dict(elem.attrib)
            })
            return None

    def _process_settlement_elements(
        self, parent_elem, province_name: str,
        settlements_dict: dict, settlements_list: list,
        parent_matrix: TransformMatrix = IDENTITY_MATRIX,
        layer_path: str = "Settlements",
    ):
        """Recursively process settlement text elements, flattening all group transforms.

        Every group encountered is checked for a 'transform' attribute; if found,
        that transform is logged and composed into the running matrix before
        recursing into the group's children.
        """
        for elem in parent_elem:
            if elem.tag == f"{{{NS['svg']}}}g":
                # Compose this group's transform with the accumulated parent matrix
                label = (
                    elem.get(f"{{{NS['inkscape']}}}label")
                    or elem.get("id", "")
                )
                child_path = f"{layer_path}/{label}" if label else layer_path
                child_matrix = self._get_group_transform(elem, layer_path, parent_matrix)
                self._process_settlement_elements(
                    elem, province_name, settlements_dict, settlements_list,
                    child_matrix, child_path
                )
            else:
                result = self._validate_settlement_element(
                    elem, province_name, parent_matrix, layer_path
                )
                if result:
                    name, svg_x, svg_y = result

                    # Track duplicates
                    if name in settlements_dict:
                        self.duplicate_settlements[province_name].append({
                            "name": name,
                            "occurrences": 2,
                            "coordinates": [settlements_dict[name], (svg_x, svg_y)]
                        })
                    else:
                        settlements_dict[name] = (svg_x, svg_y)

                        geo_lon, geo_lat = self.converter.svg_to_geo(svg_x, svg_y)
                        settlement = Settlement(
                            name=name,
                            province=province_name,
                            svg_x=svg_x,
                            svg_y=svg_y,
                            geo_lon=geo_lon,
                            geo_lat=geo_lat
                        )
                        settlements_list.append(settlement)

    def process_settlements_empire(self):
        """Process all Empire settlements."""
        logger.info("Processing Empire settlements...")

        # Find Settlements layer
        settlements_layer = None
        for g in self.root.findall(f".//{{{NS['svg']}}}g"):
            if g.get(f"{{{NS['inkscape']}}}label") == "Settlements":
                settlements_layer = g
                break

        if settlements_layer is None:
            logger.error("Settlements layer not found!")
            return

        # Find Empire faction
        empire_faction = None
        for child in settlements_layer:
            if child.get(f"{{{NS['inkscape']}}}label") == "Empire":
                empire_faction = child
                break

        if empire_faction is None:
            logger.error("Empire faction not found!")
            return

        # Build the transform chain: Settlements layer -> Empire faction
        settlements_matrix = self._get_group_transform(
            settlements_layer, "Settlements", IDENTITY_MATRIX
        )
        empire_matrix = self._get_group_transform(
            empire_faction, "Settlements/Empire", settlements_matrix
        )

        # Process each province
        provinces_seen = set()
        for province_group in empire_faction:
            province_name = province_group.get(f"{{{NS['inkscape']}}}label")
            if not province_name:
                continue

            provinces_seen.add(province_name)
            logger.info(f"  Processing province: {province_name}")

            settlements_in_province = {}
            prov_path = f"Settlements/Empire/{province_name}"

            # Absorb province-level transform and recurse into its children
            province_matrix = self._get_group_transform(
                province_group, "Settlements/Empire", empire_matrix
            )
            self._process_settlement_elements(
                province_group, province_name,
                settlements_in_province, self.settlements_empire,
                province_matrix, prov_path
            )

        logger.info(
            f"  Found {len(self.settlements_empire)} valid settlements "
            f"across {len(provinces_seen)} provinces"
        )

    def process_settlements_westerland(self):
        """Process all Westerland settlements."""
        logger.info("Processing Westerland settlements...")

        # Find Settlements layer
        settlements_layer = None
        for g in self.root.findall(f".//{{{NS['svg']}}}g"):
            if g.get(f"{{{NS['inkscape']}}}label") == "Settlements":
                settlements_layer = g
                break

        if settlements_layer is None:
            logger.error("Settlements layer not found!")
            return

        # Find Westerland faction
        westerland_faction = None
        for child in settlements_layer:
            if child.get(f"{{{NS['inkscape']}}}label") == "Westerland":
                westerland_faction = child
                break

        if westerland_faction is None:
            logger.error("Westerland faction not found!")
            return

        settlements_in_faction = {}

        # Build transform chain: Settlements -> Westerland faction
        settlements_matrix = self._get_group_transform(
            settlements_layer, "Settlements", IDENTITY_MATRIX
        )
        westerland_matrix = self._get_group_transform(
            westerland_faction, "Settlements/Westerland", settlements_matrix
        )
        self._process_settlement_elements(
            westerland_faction, "Westerland",
            settlements_in_faction, self.settlements_westerland,
            westerland_matrix, "Settlements/Westerland"
        )

        logger.info(f"  Found {len(self.settlements_westerland)} valid Westerland settlements")

    def process_settlements_bretonnia(self):
        """Process all Bretonnia settlements."""
        logger.info("Processing Bretonnia settlements...")

        # Find Settlements layer
        settlements_layer = None
        for g in self.root.findall(f".//{{{NS['svg']}}}g"):
            if g.get(f"{{{NS['inkscape']}}}label") == "Settlements":
                settlements_layer = g
                break

        if settlements_layer is None:
            logger.error("Settlements layer not found!")
            return

        # Find Bretonnia faction
        bretonnia_faction = None
        for child in settlements_layer:
            if child.get(f"{{{NS['inkscape']}}}label") == "Bretonnia":
                bretonnia_faction = child
                break

        if bretonnia_faction is None:
            logger.error("Bretonnia faction not found!")
            return

        settlements_in_faction = {}

        # Build transform chain: Settlements -> Bretonnia faction
        settlements_matrix = self._get_group_transform(
            settlements_layer, "Settlements", IDENTITY_MATRIX
        )
        bretonnia_matrix = self._get_group_transform(
            bretonnia_faction, "Settlements/Bretonnia", settlements_matrix
        )
        self._process_settlement_elements(
            bretonnia_faction, "Bretonnia",
            settlements_in_faction, self.settlements_bretonnia,
            bretonnia_matrix, "Settlements/Bretonnia"
        )

        logger.info(f"  Found {len(self.settlements_bretonnia)} valid Bretonnia settlements")

    def process_settlements_karaz_ankor(self):
        """Process all Karaz Ankor (Dwarf Holds) settlements."""
        logger.info("Processing Karaz Ankor settlements...")

        # Find Settlements layer
        settlements_layer = None
        for g in self.root.findall(f".//{{{NS['svg']}}}g"):
            if g.get(f"{{{NS['inkscape']}}}label") == "Settlements":
                settlements_layer = g
                break

        if settlements_layer is None:
            logger.error("Settlements layer not found!")
            return

        # Find Dwarf Holds faction
        dwarf_holds_faction = None
        for child in settlements_layer:
            if child.get(f"{{{NS['inkscape']}}}label") == "Dwarf Holds":
                dwarf_holds_faction = child
                break

        if dwarf_holds_faction is None:
            logger.error("Dwarf Holds faction layer not found!")
            return

        settlements_in_faction = {}

        # Build transform chain: Settlements -> Dwarf Holds faction
        settlements_matrix = self._get_group_transform(
            settlements_layer, "Settlements", IDENTITY_MATRIX
        )
        faction_matrix = self._get_group_transform(
            dwarf_holds_faction, "Settlements/Dwarf Holds", settlements_matrix
        )

        # Recurse into Dwarf Holds with fully composed initial matrix
        self._process_dwarf_hold_elements(
            dwarf_holds_faction, settlements_in_faction,
            faction_matrix, "Settlements/Dwarf Holds"
        )

        logger.info(f"  Found {len(self.settlements_karaz_ankor)} valid Karaz Ankor settlements")

    def _process_dwarf_hold_elements(
        self, parent_elem, settlements_dict: dict,
        parent_matrix: TransformMatrix = IDENTITY_MATRIX,
        layer_path: str = "Settlements/Dwarf Holds",
    ):
        """Recursively process dwarf hold elements, flattening all group transforms."""
        for elem in parent_elem:
            if elem.tag == f"{{{NS['svg']}}}g":
                label = (
                    elem.get(f"{{{NS['inkscape']}}}label")
                    or elem.get("id", "")
                )
                child_path = f"{layer_path}/{label}" if label else layer_path
                child_matrix = self._get_group_transform(elem, layer_path, parent_matrix)
                self._process_dwarf_hold_elements(
                    elem, settlements_dict, child_matrix, child_path
                )
            elif elem.tag == f"{{{NS['svg']}}}text":
                result = self._validate_dwarf_hold_element(elem, parent_matrix, layer_path)
                if result:
                    name, svg_x, svg_y = result

                    if name in settlements_dict:
                        self.duplicate_settlements["Karaz Ankor"].append({
                            "name": name,
                            "occurrences": 2,
                            "coordinates": [settlements_dict[name], (svg_x, svg_y)]
                        })
                    else:
                        settlements_dict[name] = (svg_x, svg_y)
                        geo_lon, geo_lat = self.converter.svg_to_geo(svg_x, svg_y)
                        hold = DwarfHold(
                            name=name,
                            svg_x=svg_x,
                            svg_y=svg_y,
                            geo_lon=geo_lon,
                            geo_lat=geo_lat
                        )
                        self.settlements_karaz_ankor.append(hold)

    def _validate_dwarf_hold_element(
        self, elem,
        parent_matrix: TransformMatrix = IDENTITY_MATRIX,
        layer_path: str = "Settlements/Dwarf Holds",
    ) -> Optional[Tuple[str, float, float]]:
        """Validate a dwarf-hold text element; return (name, abs_svg_x, abs_svg_y).

        Mirrors _validate_settlement_element but for Dwarf Holds (no province field).
        All ancestor transforms must be encoded in parent_matrix.
        """
        if elem.tag != f"{{{NS['svg']}}}text":
            return None

        name = self._get_text_element_label(elem)
        if not name:
            return None

        try:
            svg_x = float(elem.get("x", 0))
            svg_y = float(elem.get("y", 0))

            # Detect and absorb any transform on the text element itself
            elem_transform_str = elem.get("transform", "")
            if elem_transform_str:
                elem_path = f"{layer_path}/text:'{name}'"
                self.layers_with_transforms.append({
                    "layer_path": elem_path,
                    "transform": elem_transform_str,
                    "context": f"Dwarf hold text element '{name}'",
                })
                logger.debug(f"Text element transform absorbed: {elem_path!r}")
                elem_matrix = self._parse_transform_to_matrix(elem_transform_str)
                svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, elem_matrix)

            # Apply accumulated ancestor transform
            svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, parent_matrix)
            return (name, svg_x, svg_y)
        except ValueError:
            return None

    def load_population_data(self, faction: str, province: Optional[str] = None) -> Dict[str, int]:
        """Load population data from CSV files."""
        populations = {}

        if faction == "Empire" and province:
            csv_file = INPUT_DIR / "The-Empire" / f"{province.lower()}.csv"
            if csv_file.exists():
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            settlement_name = row[0].strip()
                            try:
                                population = int(row[1].strip())
                                populations[settlement_name] = population
                            except ValueError:
                                pass

        elif faction == "Westerland":
            csv_file = INPUT_DIR / "westerland.csv"
            if csv_file.exists():
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            settlement_name = row[0].strip()
                            try:
                                population = int(row[1].strip())
                                populations[settlement_name] = population
                            except ValueError:
                                pass

        return populations

    def load_csv_data(self, faction: str, province: Optional[str] = None) -> Dict[str, List]:
        """Load full CSV data for a faction/province."""
        csv_data = {}
        csv_file = None

        if faction == "Empire":
            # Use the empire_combined.csv file which contains all provinces
            csv_file = INPUT_DIR / "empire.csv"
        elif faction == "Westerland":
            csv_file = INPUT_DIR / "westerland.csv"
        elif faction == "Bretonnia":
            csv_file = INPUT_DIR / "bretonnia.csv"

        if csv_file and csv_file.exists():
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        settlement_name = row['Settlement'].strip()
                        # If province specified, only include matching settlements
                        if province and row.get('Province_2515', '').strip() == province:
                            csv_data[settlement_name] = row
                        elif not province:
                            csv_data[settlement_name] = row
            except Exception as e:
                logger.warning(f"Error loading CSV from {csv_file}: {e}")

        return csv_data

    def parse_tags(self, tags_str: str, trade_str: str) -> List[str]:
        """Parse tags from CSV, including trade goods."""
        tags = []
        
        # Parse tags column
        if tags_str and tags_str.strip():
            # Remove quotes and split on semicolon
            tags_str = tags_str.strip().strip('"')
            if tags_str:
                for tag in tags_str.split(';'):
                    tag = tag.strip()
                    if tag:
                        tags.append(tag)
        
        # Parse trade column and add as tags with prefix
        if trade_str and trade_str.strip():
            trade_str = trade_str.strip().strip('"')
            if trade_str:
                for trade in trade_str.split(';'):
                    trade = trade.strip()
                    if trade:
                        tags.append(f"trade:{trade}")
        
        return tags

    def validate_tags(self, tags: List[str], settlement_name: str) -> List[str]:
        """Validate tags and log any issues."""
        valid_sources = {"AndyLaw", "2eSH", "4eAotE1", "4eEiS", "4ePBtTC", "4eSCoSaS",
                        "4eCRB", "4eDotRC", "NCC", "WFB8e", "AmbChron", "G&FT", "TOW", "1eMSDtR",
                        "4eLoSaS","4eTHRC","4eMCotWW","1eDSaS","TWW3","2eKAAotDC", "MadAlfred", "MA",
                        "4eStarter", "4eUA1", "4eAotE3", "4eUA2", "4eAotE1"}
        issues = []
        
        for tag in tags:
            if ':' not in tag:
                issues.append(f"Tag '{tag}' missing format 'type:value'")
            else:
                tag_type, tag_value = tag.split(':', 1)
                
                # Validate source tags
                if tag_type == "source":
                    if tag_value not in valid_sources:
                        issues.append(f"Invalid source '{tag_value}' not in {valid_sources}")
        
        if issues:
            self.invalid_tags.append({
                "settlement": settlement_name,
                "tags": tags,
                "issues": issues
            })
        
        return tags

    def parse_notes(self, notes_str: str) -> List[str]:
        """Parse notes from CSV."""
        notes = []
        
        if notes_str and notes_str.strip():
            notes_str = notes_str.strip().strip('"')
            if notes_str:
                for note in notes_str.split(';'):
                    note = note.strip()
                    if note:
                        notes.append(note)
        
        return notes

    def calculate_size_category(self, population: int) -> int:
        """Calculate size category based on population."""
        if population <= 300:
            return 1  # Village
        elif population <= 900:
            return 2  # Small Town
        elif population <= 3000:
            return 3  # Town
        elif population <= 15000:
            return 4  # Large Town
        elif population <= 49999:
            return 5  # City
        else:
            return 6  # Metropolis

    def populate_settlement_data(self):
        """Load population and additional data from CSVs and assign to settlements."""
        logger.info("Loading and processing CSV data...")

        # Process Empire settlements
        for settlement in self.settlements_empire:
            csv_data = self.load_csv_data("Empire", settlement.province)
            
            if settlement.name in csv_data:
                row = csv_data[settlement.name]
                
                # Population
                try:
                    settlement.population = int(row['Population'].strip())
                except (ValueError, KeyError):
                    settlement.population = self._assign_random_population()
                    self.missing_population_data[settlement.province].append(settlement.name)
                
                # Province validation
                if row.get('Province_2515'):
                    csv_province = row['Province_2515'].strip()
                    if csv_province and csv_province != settlement.province:
                        self.province_mismatches.append({
                            "settlement": settlement.name,
                            "province_svg": settlement.province,
                            "province_csv": csv_province
                        })
                        # Log warning but continue with SVG province
                        logger.warning(f"Province mismatch for {settlement.name}: SVG={settlement.province}, CSV={csv_province}")
                
                # Tags
                tags_str = row.get('Tags', '')
                trade_str = row.get('Trade', '')
                settlement.tags = self.parse_tags(tags_str, trade_str)
                settlement.tags = self.validate_tags(settlement.tags, settlement.name)
                
                # Notes
                settlement.notes = self.parse_notes(row.get('Notes', ''))
                
                # Wiki data
                settlement.wiki = {
                    "title": row.get('wiki_title') or None,
                    "url": row.get('wiki_url') or None,
                    "description": row.get('wiki_description') or None,
                    "image": row.get('wiki_image') or None
                }
            else:
                # Settlement in SVG but not in CSV - assign random population
                settlement.population = self._assign_random_population()
                self.missing_population_data[settlement.province].append(settlement.name)
                settlement.tags = []
                settlement.notes = []
            
            settlement.size_category = self.calculate_size_category(settlement.population)

        # Process Westerland settlements
        csv_data = self.load_csv_data("Westerland")
        svg_settlement_names = {s.name for s in self.settlements_westerland}
        
        for settlement in self.settlements_westerland:
            if settlement.name in csv_data:
                row = csv_data[settlement.name]
                
                # Population
                try:
                    settlement.population = int(row['Population'].strip())
                except (ValueError, KeyError):
                    settlement.population = self._assign_random_population()
                    self.missing_population_data["Westerland"].append(settlement.name)
                
                # Tags
                tags_str = row.get('Tags', '')
                trade_str = row.get('Trade', '')
                settlement.tags = self.parse_tags(tags_str, trade_str)
                settlement.tags = self.validate_tags(settlement.tags, settlement.name)
                
                # Notes
                settlement.notes = self.parse_notes(row.get('Notes', ''))
                
                # Wiki data
                settlement.wiki = {
                    "title": row.get('wiki_title') or None,
                    "url": row.get('wiki_url') or None,
                    "description": row.get('wiki_description') or None,
                    "image": row.get('wiki_image') or None
                }
            else:
                settlement.population = self._assign_random_population()
                self.missing_population_data["Westerland"].append(settlement.name)
                settlement.tags = []
                settlement.notes = []
            
            settlement.size_category = self.calculate_size_category(settlement.population)

        # Process Bretonnia settlements
        csv_data = self.load_csv_data("Bretonnia")
        for settlement in self.settlements_bretonnia:
            if settlement.name in csv_data:
                row = csv_data[settlement.name]

                # Population
                try:
                    settlement.population = int(row['Population'].strip())
                except (ValueError, KeyError):
                    settlement.population = self._assign_random_population()
                    self.missing_population_data["Bretonnia"].append(settlement.name)

                # Tags
                tags_str = row.get('Tags', '')
                trade_str = row.get('Trade', '')
                settlement.tags = self.parse_tags(tags_str, trade_str)
                settlement.tags = self.validate_tags(settlement.tags, settlement.name)

                # Notes
                settlement.notes = self.parse_notes(row.get('Notes', ''))

                # Wiki data
                settlement.wiki = {
                    "title": row.get('wiki_title') or None,
                    "url": row.get('wiki_url') or None,
                    "description": row.get('wiki_description') or None,
                    "image": row.get('wiki_image') or None
                }
            else:
                settlement.population = self._assign_random_population()
                self.missing_population_data["Bretonnia"].append(settlement.name)
                settlement.tags = []
                settlement.notes = []

            settlement.size_category = self.calculate_size_category(settlement.population)

        # Track CSV settlements not in SVG
        # Load all Empire CSV data and check against SVG by province
        empire_csv_file = INPUT_DIR / "empire.csv"
        if empire_csv_file.exists():
            try:
                with open(empire_csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        csv_name = row['Settlement'].strip()
                        csv_province = row.get('Province_2515', '').strip()
                        if csv_province:
                            svg_names = {s.name for s in self.settlements_empire if s.province == csv_province}
                            if csv_name not in svg_names:
                                self.csv_settlements_not_in_svg[csv_province].append(csv_name)
            except Exception as e:
                logger.warning(f"Error tracking CSV settlements: {e}")
        
        westerland_csv = self.load_csv_data("Westerland")
        westerland_svg_names = {s.name for s in self.settlements_westerland}
        for csv_name in westerland_csv.keys():
            if csv_name not in westerland_svg_names:
                self.csv_settlements_not_in_svg["Westerland"].append(csv_name)

        bretonnia_csv = self.load_csv_data("Bretonnia")
        bretonnia_svg_names = {s.name for s in self.settlements_bretonnia}
        for csv_name in bretonnia_csv.keys():
            if csv_name not in bretonnia_svg_names:
                self.csv_settlements_not_in_svg["Bretonnia"].append(csv_name)

        # Log summary
        if self.missing_population_data:
            logger.warning("Settlements with randomly assigned populations:")
            for province, settlements in self.missing_population_data.items():
                logger.warning(f"  {province}: {len(settlements)} settlements")

    def populate_karaz_ankor_data(self):
        """Load data from Karaz Ankor CSV and assign to dwarf holds."""
        logger.info("Loading and processing Karaz Ankor CSV data...")

        # Load CSV data
        csv_file = INPUT_DIR / "karaz_ankor.csv"
        csv_data = {}
        
        if csv_file.exists():
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        settlement_name = row['Settlement'].strip()
                        csv_data[settlement_name] = row
            except Exception as e:
                logger.warning(f"Error loading CSV from {csv_file}: {e}")
                return

        # Get set of SVG settlement names
        svg_settlement_names = {hold.name for hold in self.settlements_karaz_ankor}
        
        # Process each dwarf hold
        for hold in self.settlements_karaz_ankor:
            if hold.name in csv_data:
                row = csv_data[hold.name]
                
                # Hold Type (instead of population)
                hold.hold_type = row.get('Type', '').strip()
                
                # Tags
                tags_str = row.get('Tags', '')
                trade_str = row.get('Trade', '')
                hold.tags = self.parse_tags(tags_str, trade_str)
                hold.tags = self.validate_tags(hold.tags, hold.name)
                
                # Notes
                hold.notes = self.parse_notes(row.get('Notes', ''))
                
                # Wiki data
                hold.wiki = {
                    "title": row.get('wiki_title') or None,
                    "url": row.get('wiki_url') or None,
                    "description": row.get('wiki_description') or None,
                    "image": row.get('wiki_image') or None
                }
            else:
                # Settlement in SVG but not in CSV
                self.svg_settlements_not_in_csv["Karaz Ankor"].append(hold.name)
                hold.hold_type = ""
                hold.tags = []
                hold.notes = []

        # Track CSV settlements not in SVG
        for csv_name in csv_data.keys():
            if csv_name not in svg_settlement_names:
                self.csv_settlements_not_in_svg["Karaz Ankor"].append(csv_name)
        
        # Log summaries
        if self.svg_settlements_not_in_csv["Karaz Ankor"]:
            logger.warning(f"Karaz Ankor settlements in SVG but not in CSV: {len(self.svg_settlements_not_in_csv['Karaz Ankor'])}")
        if self.csv_settlements_not_in_svg["Karaz Ankor"]:
            logger.warning(f"Karaz Ankor settlements in CSV but not in SVG: {len(self.csv_settlements_not_in_svg['Karaz Ankor'])}")

    def populate_province_data(self):
        """Load data from provinces.csv and assign to province labels."""
        logger.info("Loading and processing Province CSV data...")

        # Load CSV data
        csv_file = INPUT_DIR / "provinces.csv"
        
        if csv_file.exists():
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        province_name = row['name'].strip()
                        self.csv_data_provinces[province_name] = row
            except Exception as e:
                logger.warning(f"Error loading CSV from {csv_file}: {e}")
                return

        # Get set of SVG province label names
        svg_province_names = {label.name for label in self.province_labels}
        
        # Process each province label
        for label in self.province_labels:
            if label.name in self.csv_data_provinces:
                row = self.csv_data_provinces[label.name]
                
                # Formal title
                label.formal_title = row.get('formal_title', '').strip()
                
                # Part of (parent region)
                label.part_of = row.get('part_of', '').strip()
                
                # Population
                try:
                    pop_str = row.get('population', '').strip()
                    if pop_str:
                        label.population = int(pop_str)
                    else:
                        label.population = 0
                except (ValueError, TypeError):
                    label.population = 0
                
                # Province type from CSV (this may differ from SVG layer classification)
                csv_province_type = row.get('province_type', '').strip()
                if csv_province_type:
                    # Keep the SVG-derived province_type but could add validation here if needed
                    pass
                
                # Info dictionary
                label.info = {
                    "wiki_url": row.get('info_wiki_url', '').strip() or None,
                    "image": row.get('info_image', '').strip() or None,
                    "description": row.get('info_description', '').strip() or None
                }
            else:
                # Province label in SVG but not in CSV
                self.svg_provinces_not_in_csv.append(label.name)
                label.formal_title = ""
                label.part_of = ""
                label.population = 0
                label.info = {
                    "wiki_url": None,
                    "image": None,
                    "description": None
                }

        # Track CSV provinces not in SVG
        for csv_name in self.csv_data_provinces.keys():
            if csv_name not in svg_province_names:
                self.csv_provinces_not_in_svg.append(csv_name)
        
        # Log summaries
        if self.svg_provinces_not_in_csv:
            logger.warning(f"Province labels in SVG but not in CSV: {len(self.svg_provinces_not_in_csv)}")
        if self.csv_provinces_not_in_svg:
            logger.warning(f"Province labels in CSV but not in SVG: {len(self.csv_provinces_not_in_svg)}")


    def _assign_random_population(self) -> int:
        """Assign random population using log-normal distribution between 100 and 800."""
        # Use log-normal distribution for realistic settlement populations
        # Shape and scale chosen to give reasonable distribution in 100-800 range
        _random_population = int(np.random.lognormal(mean=5.0, sigma=0.8))
        if _random_population > 800:
            return 782
        return _random_population

    def _process_poi_elements(
        self, parent_elem, poi_type: str, poi_list: list,
        parent_matrix: TransformMatrix = IDENTITY_MATRIX,
        layer_path: str = "Points of Interest",
    ):
        """Recursively process POI text elements, flattening all group transforms."""
        for elem in parent_elem:
            if elem.tag == f"{{{NS['svg']}}}g":
                label = (
                    elem.get(f"{{{NS['inkscape']}}}label")
                    or elem.get("id", "")
                )
                child_path = f"{layer_path}/{label}" if label else layer_path
                child_matrix = self._get_group_transform(elem, layer_path, parent_matrix)
                self._process_poi_elements(elem, poi_type, poi_list, child_matrix, child_path)
            elif elem.tag == f"{{{NS['svg']}}}text":
                name = self._get_text_element_label(elem)
                if name:
                    try:
                        svg_x = float(elem.get("x", 0))
                        svg_y = float(elem.get("y", 0))

                        # Detect and absorb any element-level transform
                        elem_transform_str = elem.get("transform", "")
                        if elem_transform_str:
                            elem_path = f"{layer_path}/text:'{name}'"
                            self.layers_with_transforms.append({
                                "layer_path": elem_path,
                                "transform": elem_transform_str,
                                "context": f"POI text element '{name}' (type={poi_type})",
                            })
                            elem_matrix = self._parse_transform_to_matrix(elem_transform_str)
                            svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, elem_matrix)

                        # Apply accumulated ancestor transform
                        svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, parent_matrix)

                        geo_lon, geo_lat = self.converter.svg_to_geo(svg_x, svg_y)
                        poi = PointOfInterest(
                            name=name,
                            poi_type=poi_type,
                            svg_x=svg_x,
                            svg_y=svg_y,
                            geo_lon=geo_lon,
                            geo_lat=geo_lat
                        )
                        poi_list.append(poi)
                    except (ValueError, TypeError):
                        pass

    def process_points_of_interest(self):
        """Process all points of interest."""
        logger.info("Processing Points of Interest...")

        # Find Points of Interest layer
        poi_layer = None
        for g in self.root.findall(f".//{{{NS['svg']}}}g"):
            if g.get(f"{{{NS['inkscape']}}}label") == "Points of Interest":
                poi_layer = g
                break

        if poi_layer is None:
            logger.error("Points of Interest layer not found!")
            return

        # Build the initial transform from the POI layer itself
        poi_layer_matrix = self._get_group_transform(
            poi_layer, "Points of Interest", IDENTITY_MATRIX
        )

        poi_types = {
            "Other": "Other",
            "City Districts": "City Districts",
            "Forts and Castles": "Forts and Castles",
            "Monastaries and Temples": "Monasteries and Temples",
            "Taverns and Inns": "Taverns and Inns"
        }

        for poi_group in poi_layer:
            poi_type = poi_group.get(f"{{{NS['inkscape']}}}label")
            if not poi_type:
                continue

            poi_type = poi_types.get(poi_type, poi_type)
            logger.info(f"  Processing POI type: {poi_type}")

            # Compose sub-group transform with the POI layer matrix
            group_path = f"Points of Interest/{poi_type}"
            group_matrix = self._get_group_transform(poi_group, "Points of Interest", poi_layer_matrix)

            initial_count = len(self.points_of_interest)
            self._process_poi_elements(
                poi_group, poi_type, self.points_of_interest,
                group_matrix, group_path
            )
            count = len(self.points_of_interest) - initial_count
            logger.info(f"    Found {count} POI")

    def parse_svg_path(self, path_d: str) -> List[Tuple[float, float]]:
        """Parse SVG path data and extract coordinates, handling both absolute and relative commands."""
        points = []
        import re

        # Remove extra spaces and split on commands and commas
        path_d = re.sub(r'([MmLlHhVvCcSsQqTtAaZz])', r' \1 ', path_d)
        path_d = path_d.replace(',', ' ')  # Split on commas too
        tokens = [t for t in path_d.split() if t.strip()]

        x, y = 0, 0  # Current position
        start_x, start_y = 0, 0  # Start position for closepath
        command = None
        last_cp2 = (0, 0)  # Last control point for smooth curves

        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token in 'MmLlHhVvCcSsQqTtAaZz':
                command = token
                i += 1
            else:
                try:
                    if command in 'Mm':
                        # Move command
                        num_x = float(token)
                        i += 1
                        if i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                            num_y = float(tokens[i])
                            i += 1
                        else:
                            continue
                        
                        if command == 'M':
                            # Absolute move
                            x, y = num_x, num_y
                        else:
                            # Relative move
                            x += num_x
                            y += num_y
                        
                        start_x, start_y = x, y
                        points.append((x, y))
                        
                    elif command in 'Ll':
                        # Line command
                        num_x = float(token)
                        i += 1
                        if i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                            num_y = float(tokens[i])
                            i += 1
                        else:
                            continue
                        
                        if command == 'L':
                            x, y = num_x, num_y
                        else:
                            x += num_x
                            y += num_y
                        
                        points.append((x, y))
                        
                    elif command == 'H':
                        # Horizontal line (absolute)
                        x = float(token)
                        i += 1
                        points.append((x, y))
                        
                    elif command == 'h':
                        # Horizontal line (relative)
                        x += float(token)
                        i += 1
                        points.append((x, y))
                        
                    elif command == 'V':
                        # Vertical line (absolute)
                        y = float(token)
                        i += 1
                        points.append((x, y))
                        
                    elif command == 'v':
                        # Vertical line (relative)
                        y += float(token)
                        i += 1
                        points.append((x, y))
                        
                    elif command in 'Cc':
                        # Bezier curve - collect all 6 numbers
                        curve_points = [float(token)]
                        i += 1
                        for _ in range(5):
                            if i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                                curve_points.append(float(tokens[i]))
                                i += 1
                            else:
                                break
                        
                        if len(curve_points) == 6:
                            cp1_x, cp1_y, cp2_x, cp2_y, end_x, end_y = curve_points
                            
                            if command == 'c':
                                # Relative Bezier
                                cp1_x += x
                                cp1_y += y
                                cp2_x += x
                                cp2_y += y
                                end_x += x
                                end_y += y
                            
                            # Sample along the curve
                            sampled = self._sample_bezier_curve(
                                (x, y),
                                (cp1_x, cp1_y),
                                (cp2_x, cp2_y),
                                (end_x, end_y)
                            )
                            points.extend(sampled)
                            
                            x, y = end_x, end_y
                            last_cp2 = (cp2_x, cp2_y)
                    else:
                        i += 1
                except ValueError:
                    i += 1

        return points

    def _sample_bezier_curve(self, start: Tuple[float, float], cp1: Tuple[float, float],
                             cp2: Tuple[float, float], end: Tuple[float, float],
                             samples: int = 20) -> List[Tuple[float, float]]:
        """Sample points along a cubic Bezier curve."""
        points = []
        for t in np.linspace(0, 1, samples):
            mt = 1 - t
            x = (mt**3 * start[0] + 3*mt**2*t * cp1[0] + 3*mt*t**2 * cp2[0] + t**3 * end[0])
            y = (mt**3 * start[1] + 3*mt**2*t * cp1[1] + 3*mt*t**2 * cp2[1] + t**3 * end[1])
            points.append((x, y))
        return points

    def _process_road_elements(self, parent_elem, road_type: str, road_list: list, road_id_ref: list):
        """Recursively process road elements, handling nested layers."""
        for elem in parent_elem:
            # Check if this is a layer/group
            if elem.tag == f"{{{NS['svg']}}}g":
                # Recursively process children of this layer
                self._process_road_elements(elem, road_type, road_list, road_id_ref)
            elif elem.tag == f"{{{NS['svg']}}}path":
                path_d = elem.get("d", "")
                if path_d:
                    try:
                        svg_points = self.parse_svg_path(path_d)
                        
                        if svg_points:
                            # Convert to geographic coordinates
                            geo_points = [
                                self.converter.svg_to_geo(pt[0], pt[1])
                                for pt in svg_points
                            ]

                            road = Road(
                                road_id=f"road_{road_id_ref[0]:03d}",
                                road_type=road_type,
                                svg_path=path_d,
                                geo_coordinates=geo_points
                            )
                            road_list.append(road)
                            road_id_ref[0] += 1
                    except Exception as e:
                        logger.warning(f"    Error processing road: {e}")

    # Road processing functionality maintained for potential future implementation
    # def process_roads(self):
    #     """Process all roads."""
    #     logger.info("Processing Roads...")
    #
    #     # Find ALL Roads layers
    #     roads_layers = []
    #     for g in self.root.findall(f".//{{{NS['svg']}}}g"):
    #         if g.get(f"{{{NS['inkscape']}}}label") == "Roads":
    #             roads_layers.append(g)
    #
    #     if not roads_layers:
    #         logger.error("Roads layers not found!")
    #         return
    #
    #     road_type_map = {
    #         "Imperial Highways": "Imperial Highways",
    #         "Roads": "Roads",
    #         "Paths": "Paths"
    #     }
    #
    #     road_id_ref = [0]  # Use list to allow modification in nested function
    #
    #     # Process each Roads layer
    #     for roads_idx, roads_layer in enumerate(roads_layers):
    #         logger.info(f"  Processing Roads layer {roads_idx + 1}...")
    #         
    #         # Check if this layer has direct path elements (unlabeled roads)
    #         direct_paths = [child for child in roads_layer if child.tag == f"{{{NS['svg']}}}path"]
    #         
    #         if direct_paths:
    #             # This layer contains direct paths - process them as unlabeled roads
    #             logger.info(f"    Found {len(direct_paths)} unlabeled road paths")
    #             successful = 0
    #             for path_idx, path_elem in enumerate(direct_paths):
    #                 path_d = path_elem.get("d", "")
    #                 if path_d:
    #                     try:
    #                         svg_points = self.parse_svg_path(path_d)
    #                         
    #                         if svg_points:
    #                             # Convert to geographic coordinates
    #                             geo_points = [
    #                                 self.converter.svg_to_geo(pt[0], pt[1])
    #                                 for pt in svg_points
    #                             ]
    #
    #                             road = Road(
    #                                 road_id=f"road_{road_id_ref[0]:03d}",
    #                                 road_type="Road",
    #                                 svg_path=path_d,
    #                                 geo_coordinates=geo_points
    #                             )
    #                             self.roads.append(road)
    #                             road_id_ref[0] += 1
    #                             successful += 1
    #                     except Exception as e:
    #                         if path_idx < 3:  # Log first 3 errors only
    #                             logger.debug(f"    Error processing road {path_idx}: {str(e)[:100]}")
    #         else:
    #             # This layer contains labeled road type groups
    #             for road_group in roads_layer:
    #                 road_type = road_group.get(f"{{{NS['inkscape']}}}label}")
    #                 if not road_type or road_type not in road_type_map:
    #                     continue
    #
    #                 road_type = road_type_map.get(road_type, road_type)
    #                 logger.info(f"    Processing road type: {road_type}")
    #
    #                 # Extract roads from this group (may be nested in sub-layers)
    #                 initial_count = len(self.roads)
    #                 self._process_road_elements(road_group, road_type, self.roads, road_id_ref)
    #                 count = len(self.roads) - initial_count
    #
    #                 logger.info(f"      Found {count} roads")

    def process_province_labels(self):
        """Process all political/province labels."""
        logger.info("Processing Province Labels...")

        # Find the Region-Labels-post2512 layer
        regions_layer = None
        for g in self.root.findall(f".//{{{NS['svg']}}}g"):
            if g.get(f"{{{NS['inkscape']}}}label") == "Region-Labels-post2512":
                regions_layer = g
                break

        if regions_layer is None:
            logger.error("Region-Labels-post2512 layer not found!")
            return

        # Build initial transform from the Region-Labels layer itself
        regions_layer_matrix = self._get_group_transform(
            regions_layer, "Region-Labels-post2512", IDENTITY_MATRIX
        )

        # Map layer names to province types
        province_type_map = {
            "Nation-States": "Nation-State",
            "Grand-Provinces": "Grand-Province",
            "Provinces": "Province"
        }

        for region_group in regions_layer:
            layer_name = region_group.get(f"{{{NS['inkscape']}}}label")
            if not layer_name or layer_name not in province_type_map:
                continue

            province_type = province_type_map[layer_name]
            logger.info(f"  Processing province type: {province_type}")

            group_path = f"Region-Labels-post2512/{layer_name}"
            group_matrix = self._get_group_transform(
                region_group, "Region-Labels-post2512", regions_layer_matrix
            )

            initial_count = len(self.province_labels)
            self._process_province_label_elements(
                region_group, province_type, self.province_labels,
                group_matrix, group_path
            )
            count = len(self.province_labels) - initial_count
            logger.info(f"    Found {count} labels")

    def _process_province_label_elements(
        self, parent_elem, province_type: str, label_list: list,
        parent_matrix: TransformMatrix = IDENTITY_MATRIX,
        layer_path: str = "Region-Labels-post2512",
    ):
        """Recursively process province label elements, flattening all group transforms."""
        for elem in parent_elem:
            if elem.tag == f"{{{NS['svg']}}}g":
                label = (
                    elem.get(f"{{{NS['inkscape']}}}label")
                    or elem.get("id", "")
                )
                child_path = f"{layer_path}/{label}" if label else layer_path
                child_matrix = self._get_group_transform(elem, layer_path, parent_matrix)
                self._process_province_label_elements(
                    elem, province_type, label_list, child_matrix, child_path
                )
            elif elem.tag == f"{{{NS['svg']}}}text":
                name = self._get_text_element_label(elem)
                if name:
                    try:
                        svg_x = float(elem.get("x", 0))
                        svg_y = float(elem.get("y", 0))

                        # Detect and absorb any element-level transform
                        elem_transform_str = elem.get("transform", "")
                        if elem_transform_str:
                            elem_path = f"{layer_path}/text:'{name}'"
                            self.layers_with_transforms.append({
                                "layer_path": elem_path,
                                "transform": elem_transform_str,
                                "context": f"Province label text element '{name}'",
                            })
                            elem_matrix = self._parse_transform_to_matrix(elem_transform_str)
                            svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, elem_matrix)

                        # Apply accumulated ancestor transform
                        svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, parent_matrix)

                        geo_lon, geo_lat = self.converter.svg_to_geo(svg_x, svg_y)
                        label = ProvinceLabel(
                            name=name,
                            province_type=province_type,
                            svg_x=svg_x,
                            svg_y=svg_y,
                            geo_lon=geo_lon,
                            geo_lat=geo_lat,
                            formal_title="",
                            part_of=""
                        )
                        label_list.append(label)
                    except (ValueError, TypeError):
                        pass

    def process_water_labels(self):
        """Process all water body labels."""
        logger.info("Processing Water Labels...")

        # Find the Water Labels layer
        water_layer = None
        for g in self.root.findall(f".//{{{NS['svg']}}}g"):
            if g.get(f"{{{NS['inkscape']}}}label") == "Water Labels":
                water_layer = g
                break

        if water_layer is None:
            logger.error("Water Labels layer not found!")
            return

        # Build the initial transform from the Water Labels layer itself.
        # This is critical: the Water Labels layer has its own translate that must
        # be composed with every sub-group's transform before extracting coordinates.
        water_layer_matrix = self._get_group_transform(
            water_layer, "Water Labels", IDENTITY_MATRIX
        )

        # Map layer names to waterbody types
        waterbody_type_map = {
            "ocean": "Ocean",
            "major-sea": "Major Sea",
            "large-sea": "Large Sea",
            "medium-sea": "Medium Sea",
            "small-sea": "Small Sea",
            "small-marsh": "Small Marsh",
            "large-marsh": "Large Marsh",
            "lakes": "Lake"
        }

        for water_group in water_layer:
            layer_name = water_group.get(f"{{{NS['inkscape']}}}label")
            if not layer_name:
                continue

            # Handle marshes which has sub-layers
            if layer_name == "marshes":
                logger.info(f"  Processing marshes...")
                # The marshes group itself may have a transform
                marshes_matrix = self._get_group_transform(
                    water_group, "Water Labels", water_layer_matrix
                )
                for marsh_group in water_group:
                    marsh_layer_name = marsh_group.get(f"{{{NS['inkscape']}}}label")
                    if marsh_layer_name and marsh_layer_name in waterbody_type_map:
                        waterbody_type = waterbody_type_map[marsh_layer_name]
                        marsh_path = f"Water Labels/marshes/{marsh_layer_name}"
                        marsh_matrix = self._get_group_transform(
                            marsh_group, "Water Labels/marshes", marshes_matrix
                        )
                        initial_count = len(self.water_labels)
                        self._process_water_label_elements(
                            marsh_group, waterbody_type, self.water_labels,
                            marsh_matrix, marsh_path
                        )
                        count = len(self.water_labels) - initial_count
                        logger.info(f"    Found {count} {waterbody_type} labels")

            elif layer_name in waterbody_type_map:
                waterbody_type = waterbody_type_map[layer_name]
                logger.info(f"  Processing {waterbody_type}...")

                group_path = f"Water Labels/{layer_name}"
                group_matrix = self._get_group_transform(
                    water_group, "Water Labels", water_layer_matrix
                )
                initial_count = len(self.water_labels)
                self._process_water_label_elements(
                    water_group, waterbody_type, self.water_labels,
                    group_matrix, group_path
                )
                count = len(self.water_labels) - initial_count
                logger.info(f"    Found {count} labels")

    def _process_water_label_elements(
        self, parent_elem, waterbody_type: str, label_list: list,
        parent_matrix: TransformMatrix = IDENTITY_MATRIX,
        layer_path: str = "Water Labels",
    ):
        """Recursively process water label elements, flattening all group transforms."""
        for elem in parent_elem:
            if elem.tag == f"{{{NS['svg']}}}g":
                label = (
                    elem.get(f"{{{NS['inkscape']}}}label")
                    or elem.get("id", "")
                )
                child_path = f"{layer_path}/{label}" if label else layer_path
                child_matrix = self._get_group_transform(elem, layer_path, parent_matrix)
                self._process_water_label_elements(
                    elem, waterbody_type, label_list, child_matrix, child_path
                )
            elif elem.tag == f"{{{NS['svg']}}}text":
                name = self._get_text_element_label(elem)
                if name:
                    try:
                        svg_x = float(elem.get("x", 0))
                        svg_y = float(elem.get("y", 0))

                        # Detect and absorb any element-level transform
                        elem_transform_str = elem.get("transform", "")
                        if elem_transform_str:
                            elem_path = f"{layer_path}/text:'{name}'"
                            self.layers_with_transforms.append({
                                "layer_path": elem_path,
                                "transform": elem_transform_str,
                                "context": f"Water label text element '{name}' (type={waterbody_type})",
                            })
                            elem_matrix = self._parse_transform_to_matrix(elem_transform_str)
                            svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, elem_matrix)

                        # Apply accumulated ancestor transform
                        svg_x, svg_y = self._apply_transform_matrix(svg_x, svg_y, parent_matrix)

                        geo_lon, geo_lat = self.converter.svg_to_geo(svg_x, svg_y)
                        label = WaterLabel(
                            name=name,
                            waterbody_type=waterbody_type,
                            svg_x=svg_x,
                            svg_y=svg_y,
                            geo_lon=geo_lon,
                            geo_lat=geo_lat
                        )
                        label_list.append(label)
                    except (ValueError, TypeError):
                        pass

    def generate_empire_geojson(self):
        """Generate GeoJSON for Empire settlements."""
        features = []
        for settlement in self.settlements_empire:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [settlement.geo_lon, settlement.geo_lat]
                },
                "properties": {
                    "name": settlement.name,
                    "province": settlement.province,
                    "population": settlement.population,
                    "tags": settlement.tags,
                    "notes": settlement.notes,
                    "size_category": settlement.size_category,
                    "inkscape_coordinates": [settlement.svg_x, settlement.svg_y],
                    "wiki": settlement.wiki
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_file = OUTPUT_DIR / "empire_settlements.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {output_file}: {len(features)} settlements")

    def generate_westerland_geojson(self):
        """Generate GeoJSON for Westerland settlements."""
        features = []
        for settlement in self.settlements_westerland:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [settlement.geo_lon, settlement.geo_lat]
                },
                "properties": {
                    "name": settlement.name,
                    "province": "Westerland",
                    "population": settlement.population,
                    "tags": settlement.tags,
                    "notes": settlement.notes,
                    "size_category": settlement.size_category,
                    "inkscape_coordinates": [settlement.svg_x, settlement.svg_y],
                    "wiki": settlement.wiki
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_file = OUTPUT_DIR / "westerland_settlements.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {output_file}: {len(features)} settlements")

    def generate_bretonnia_geojson(self):
        """Generate GeoJSON for Bretonnia settlements."""
        features = []
        for settlement in self.settlements_bretonnia:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [settlement.geo_lon, settlement.geo_lat]
                },
                "properties": {
                    "name": settlement.name,
                    "province": "Bretonnia",
                    "population": settlement.population,
                    "tags": settlement.tags,
                    "notes": settlement.notes,
                    "size_category": settlement.size_category,
                    "inkscape_coordinates": [settlement.svg_x, settlement.svg_y],
                    "wiki": settlement.wiki
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_file = OUTPUT_DIR / "bretonnia_settlements.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {output_file}: {len(features)} settlements")

    def generate_karaz_ankor_geojson(self):
        """Generate GeoJSON for Karaz Ankor (Dwarf Holds) settlements."""
        features = []
        for hold in self.settlements_karaz_ankor:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [hold.geo_lon, hold.geo_lat]
                },
                "properties": {
                    "name": hold.name,
                    "hold_type": hold.hold_type,
                    "tags": hold.tags,
                    "notes": hold.notes,
                    "inkscape_coordinates": [hold.svg_x, hold.svg_y],
                    "wiki": hold.wiki
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_file = OUTPUT_DIR / "karaz_ankor.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {output_file}: {len(features)} holds")


    def generate_poi_geojson(self):
        """Generate GeoJSON for points of interest."""
        features = []
        for poi in self.points_of_interest:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [poi.geo_lon, poi.geo_lat]
                },
                "properties": {
                    "name": poi.name,
                    "type": poi.poi_type,
                    "inkscape_coordinates": [poi.svg_x, poi.svg_y]
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_file = OUTPUT_DIR / "points_of_interest.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {output_file}: {len(features)} POI")

    def generate_roads_geojson(self):
        """Generate GeoJSON for roads with horizontally-formatted coordinates."""
        features = []
        for road in self.roads:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": road.geo_coordinates
                },
                "properties": {
                    "road_type": road.road_type,
                    "road_id": road.road_id,
                    "inkscape_coordinates": road.svg_path
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_file = OUTPUT_DIR / "empire_roads.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Custom JSON formatting for readability - keep coordinate arrays on single lines
            f.write('{\n  "type": "FeatureCollection",\n  "features": [\n')
            for i, feature in enumerate(features):
                f.write('    {\n')
                f.write('      "type": "Feature",\n')
                f.write('      "geometry": {\n')
                f.write('        "type": "LineString",\n')
                f.write('        "coordinates": [')
                # Format coordinates horizontally
                coords_str = ', '.join(f'[{lon}, {lat}]' for lon, lat in feature['geometry']['coordinates'])
                f.write(coords_str)
                f.write(']\n')
                f.write('      },\n')
                f.write('      "properties": {\n')
                f.write(f'        "road_type": "{feature["properties"]["road_type"]}",\n')
                f.write(f'        "road_id": "{feature["properties"]["road_id"]}",\n')
                f.write(f'        "inkscape_coordinates": "{feature["properties"]["inkscape_coordinates"]}"\n')
                f.write('      }\n')
                f.write('    }')
                if i < len(features) - 1:
                    f.write(',')
                f.write('\n')
            f.write('  ]\n}\n')

        logger.info(f"Generated {output_file}: {len(features)} roads")

    def generate_province_labels_geojson(self):
        """Generate GeoJSON for province labels."""
        features = []
        for label in self.province_labels:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [label.geo_lon, label.geo_lat]
                },
                "properties": {
                    "name": label.name,
                    "province_type": label.province_type,
                    "formal_title": label.formal_title,
                    "part_of": label.part_of,
                    "inkscape_coordinates": [label.svg_x, label.svg_y],
                    "info": label.info,
                    "population": label.population
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_file = OUTPUT_DIR / "province_labels.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {output_file}: {len(features)} province labels")

    def generate_water_labels_geojson(self):
        """Generate GeoJSON for water labels."""
        features = []
        for label in self.water_labels:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [label.geo_lon, label.geo_lat]
                },
                "properties": {
                    "name": label.name,
                    "waterbody_type": label.waterbody_type,
                    "inkscape_coordinates": [label.svg_x, label.svg_y]
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        output_file = OUTPUT_DIR / "water_labels.geojson"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {output_file}: {len(features)} water labels")

    def write_invalid_settlements_log(self):
        """Write log of invalid settlement elements."""
        if not self.invalid_settlements:
            return

        output_file = LOGS_DIR / "invalid_settlement_elements.log"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Invalid Settlement Elements Log\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total invalid elements: {len(self.invalid_settlements)}\n\n")

            for item in self.invalid_settlements:
                f.write(f"Province: {item.get('province', 'N/A')}\n")
                f.write(f"Reason: {item.get('reason', 'N/A')}\n")
                if 'name' in item:
                    f.write(f"Name: {item['name']}\n")
                f.write(f"Attributes: {item.get('attributes', {})}\n")
                f.write("-" * 80 + "\n")

        logger.info(f"Generated {output_file}")

    def write_duplicate_settlements_log(self):
        """Write log of duplicate settlements."""
        if not self.duplicate_settlements:
            return

        output_file = LOGS_DIR / "duplicate_settlements.log"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Duplicate Settlements Log\n")
            f.write("=" * 80 + "\n\n")
            total_duplicates = sum(len(v) for v in self.duplicate_settlements.values())
            f.write(f"Total provinces with duplicates: {len(self.duplicate_settlements)}\n")
            f.write(f"Total duplicate entries: {total_duplicates}\n\n")

            for province, duplicates in self.duplicate_settlements.items():
                f.write(f"Province: {province}\n")
                for dup in duplicates:
                    f.write(f"  Name: {dup['name']}\n")
                    f.write(f"  Coordinates: {dup.get('coordinates', [])}\n")
                f.write("-" * 80 + "\n")

        logger.info(f"Generated {output_file}")

    def generate_report(self):
        """Generate the processing report."""
        output_file = LOGS_DIR / "processing_report.txt"

        # Calculate statistics
        empire_pop_by_province = defaultdict(int)
        empire_count_by_province = defaultdict(int)

        for settlement in self.settlements_empire:
            empire_pop_by_province[settlement.province] += settlement.population
            empire_count_by_province[settlement.province] += 1

        westerland_total_pop = sum(s.population for s in self.settlements_westerland)
        westerland_total_count = len(self.settlements_westerland)

        bretonnia_total_pop = sum(s.population for s in self.settlements_bretonnia)
        bretonnia_total_count = len(self.settlements_bretonnia)

        karaz_ankor_total_count = len(self.settlements_karaz_ankor)
        karaz_ankor_by_type = defaultdict(int)
        for hold in self.settlements_karaz_ankor:
            if hold.hold_type:
                karaz_ankor_by_type[hold.hold_type] += 1

        total_road_points = sum(len(road.geo_coordinates) for road in self.roads)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Old World Atlas Map Processing Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("SETTLEMENTS SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Empire Total: {len(self.settlements_empire)} settlements\n")
            f.write(f"Westerland Total: {westerland_total_count} settlements\n")
            f.write(f"Bretonnia Total: {bretonnia_total_count} settlements\n")
            f.write(f"Karaz Ankor Total: {karaz_ankor_total_count} holds\n")
            f.write(f"Grand Total: {len(self.settlements_empire) + westerland_total_count + bretonnia_total_count + karaz_ankor_total_count} settlements/holds\n\n")

            f.write("EMPIRE SETTLEMENTS BY PROVINCE\n")
            f.write("-" * 80 + "\n")
            for province in sorted(empire_count_by_province.keys()):
                count = empire_count_by_province[province]
                pop = empire_pop_by_province[province]
                f.write(f"{province:20s} - {count:3d} settlements, {pop:10,d} total population\n")

            f.write(f"\nEmpire Total Population: {sum(empire_pop_by_province.values()):,d}\n")
            f.write(f"Westerland Total Population: {westerland_total_pop:,d}\n")
            f.write(f"Bretonnia Total Population: {bretonnia_total_pop:,d}\n\n")

            f.write("BRETONNIA SETTLEMENTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total: {bretonnia_total_count} settlements, {bretonnia_total_pop:,d} total population\n\n")

            f.write("KARAZ ANKOR (DWARF HOLDS) BY TYPE\n")
            f.write("-" * 80 + "\n")
            for hold_type in sorted(karaz_ankor_by_type.keys()):
                count = karaz_ankor_by_type[hold_type]
                f.write(f"{hold_type:40s} - {count:3d} holds\n")
            # Count holds without type
            holds_without_type = karaz_ankor_total_count - sum(karaz_ankor_by_type.values())
            if holds_without_type > 0:
                f.write(f"{'(No type assigned)':40s} - {holds_without_type:3d} holds\n")
            f.write(f"\nTotal Karaz Ankor Holds: {karaz_ankor_total_count}\n\n")


            f.write("POINTS OF INTEREST\n")
            f.write("-" * 80 + "\n")
            poi_by_type = defaultdict(int)
            for poi in self.points_of_interest:
                poi_by_type[poi.poi_type] += 1
            for poi_type in sorted(poi_by_type.keys()):
                count = poi_by_type[poi_type]
                f.write(f"{poi_type:30s} - {count:3d} POI\n")
            f.write(f"\nTotal Points of Interest: {len(self.points_of_interest)}\n\n")

            f.write("ROADS\n")
            f.write("-" * 80 + "\n")
            roads_by_type = defaultdict(int)
            for road in self.roads:
                roads_by_type[road.road_type] += 1
            for road_type in sorted(roads_by_type.keys()):
                count = roads_by_type[road_type]
                f.write(f"{road_type:30s} - {count:3d} roads\n")
            f.write(f"\nTotal Roads: {len(self.roads)}\n")
            f.write(f"Total Coordinate Points (all roads): {total_road_points:,d}\n\n")

            f.write("PROVINCE LABELS\n")
            f.write("-" * 80 + "\n")
            labels_by_type = defaultdict(int)
            for label in self.province_labels:
                labels_by_type[label.province_type] += 1
            for label_type in sorted(labels_by_type.keys()):
                count = labels_by_type[label_type]
                f.write(f"{label_type:30s} - {count:3d} labels\n")
            f.write(f"\nTotal Province Labels: {len(self.province_labels)}\n\n")

            f.write("WATER LABELS\n")
            f.write("-" * 80 + "\n")
            water_by_type = defaultdict(int)
            for label in self.water_labels:
                water_by_type[label.waterbody_type] += 1
            for water_type in sorted(water_by_type.keys()):
                count = water_by_type[water_type]
                f.write(f"{water_type:30s} - {count:3d} labels\n")
            f.write(f"\nTotal Water Labels: {len(self.water_labels)}\n\n")

            # ----------------------------------------------------------------
            # Layer transform report
            # ----------------------------------------------------------------
            f.write("LAYER TRANSFORMS DETECTED AND ABSORBED\n")
            f.write("-" * 80 + "\n")
            f.write(
                "All transforms listed below were detected on SVG groups/layers.\n"
                "Each transform was absorbed (applied to element coordinates) so that\n"
                "extracted coordinates are in absolute SVG space, consistent across\n"
                "all layers regardless of how Inkscape organises its layer hierarchy.\n\n"
            )
            if self.layers_with_transforms:
                f.write(f"Total groups/layers with transforms absorbed: {len(self.layers_with_transforms)}\n\n")
                # Group by top-level layer for readability
                by_layer: Dict[str, List[Dict]] = defaultdict(list)
                for entry in self.layers_with_transforms:
                    top = entry["layer_path"].split("/")[0]
                    by_layer[top].append(entry)
                for top_layer in sorted(by_layer.keys()):
                    f.write(f"  [{top_layer}] ({len(by_layer[top_layer])} transform(s))\n")
                    for entry in by_layer[top_layer]:
                        f.write(f"    Path      : {entry['layer_path']}\n")
                        f.write(f"    Transform : {entry['transform']}\n")
                        f.write(f"    Context   : {entry['context']}\n")
                        f.write("\n")
            else:
                f.write("  (No layer transforms detected.)\n\n")

            f.write("COORDINATE CONVERSION\n")
            f.write("-" * 80 + "\n")
            f.write(
                "Calibration: Inkscape display formula (direct, no least-squares fitting).\n"
                f"  INKSCAPE_VB_X  = {INKSCAPE_VB_X}  (SVG x at longitude 0.0\u00b0)\n"
                f"  INKSCAPE_C_Y   = {INKSCAPE_C_Y}  (SVG y constant for latitude)\n"
                f"  INKSCAPE_SCALE = {INKSCAPE_SCALE}  (degrees per SVG user unit)\n"
                "  geo_lon = (svg_x_abs - VB_X)  * SCALE\n"
                "  geo_lat = (C_Y       - svg_y_abs) * SCALE\n\n"
            )
            f.write("Reference calibration points (validation only):\n")
            for pt in CALIBRATION_POINTS:
                calc = self.converter.svg_to_geo(pt["svg"][0], pt["svg"][1])
                lon_err = calc[0] - pt["geo"][0]
                lat_err = calc[1] - pt["geo"][1]
                f.write(
                    f"  {pt['settlement']:25s}: "
                    f"svg=({pt['svg'][0]:.3f},{pt['svg'][1]:.3f})  "
                    f"expected=({pt['geo'][0]:.4f},{pt['geo'][1]:.4f})  "
                    f"calc=({calc[0]:.4f},{calc[1]:.4f})  "
                    f"err=({lon_err:+.4f},{lat_err:+.4f})\u00b0\n"
                )
            f.write("\n")

            f.write("DATA QUALITY ISSUES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Groups/Layers with Transforms Absorbed: {len(self.layers_with_transforms)}\n")
            f.write(f"Invalid Settlement Elements: {len(self.invalid_settlements)}\n")
            f.write(f"Provinces with Duplicate Names: {len(self.duplicate_settlements)}\n")
            f.write(f"Settlements with Assigned Population Data: {sum(len(v) for v in self.missing_population_data.values())}\n")
            f.write(f"CSV Settlements Not in SVG: {sum(len(v) for v in self.csv_settlements_not_in_svg.values())}\n")
            f.write(f"SVG Settlements Not in CSV: {sum(len(v) for v in self.svg_settlements_not_in_csv.values())}\n")
            f.write(f"Province Mismatches: {len(self.province_mismatches)}\n")
            f.write(f"CSV Province Labels Not in SVG: {len(self.csv_provinces_not_in_svg)}\n")
            f.write(f"SVG Province Labels Not in CSV: {len(self.svg_provinces_not_in_csv)}\n")
            f.write(f"Invalid Tags: {len(self.invalid_tags)}\n\n")

            if self.missing_population_data:
                f.write("Settlements with Randomly Assigned Populations:\n")
                for province, settlements in sorted(self.missing_population_data.items()):
                    f.write(f"  {province}: {len(settlements)} settlements\n")
                f.write("\n")

            if self.csv_settlements_not_in_svg:
                f.write("CSV Settlements Not Found in SVG (should be added to map):\n")
                for province, settlements in sorted(self.csv_settlements_not_in_svg.items()):
                    f.write(f"  {province}:\n")
                    for settlement in sorted(settlements):
                        f.write(f"    - {settlement}\n")
                f.write("\n")

            if self.svg_settlements_not_in_csv:
                f.write("SVG Settlements Not Found in CSV (should be added to gazetteer):\n")
                for province, settlements in sorted(self.svg_settlements_not_in_csv.items()):
                    f.write(f"  {province}:\n")
                    for settlement in sorted(settlements):
                        f.write(f"    - {settlement}\n")
                f.write("\n")

            if self.csv_provinces_not_in_svg:
                f.write("CSV Province Labels Not Found in SVG (should be added to map):\n")
                for province_name in sorted(self.csv_provinces_not_in_svg):
                    f.write(f"  - {province_name}\n")
                f.write("\n")

            if self.svg_provinces_not_in_csv:
                f.write("SVG Province Labels Not Found in CSV (should be added to gazetteer):\n")
                for province_name in sorted(self.svg_provinces_not_in_csv):
                    f.write(f"  - {province_name}\n")
                f.write("\n")


            if self.province_mismatches:
                f.write("Province Name Mismatches (SVG vs CSV):\n")
                for item in self.province_mismatches:
                    f.write(f"  {item['settlement']}: SVG='{item['province_svg']}', CSV='{item['province_csv']}'\n")
                f.write("\n")

            if self.invalid_tags:
                f.write("Settlements with Invalid Tag Format:\n")
                for item in self.invalid_tags:
                    f.write(f"  {item['settlement']}:\n")
                    f.write(f"    Tags: {item['tags']}\n")
                    for issue in item['issues']:
                        f.write(f"    - {issue}\n")
                f.write("\n")

        logger.info(f"Generated {output_file}")


def main():
    """Main entry point."""
    logger.info("Starting SVG map processing...")

    processor = SVGMapProcessor()

    # Process all data - ONLY Empire settlements enabled
    processor.process_settlements_empire()
    processor.process_settlements_westerland()
    processor.process_settlements_bretonnia()
    processor.process_settlements_karaz_ankor()
    processor.populate_settlement_data()
    processor.populate_karaz_ankor_data()
    processor.process_points_of_interest()
    # processor.process_roads()  # Disabled: road extraction not needed currently
    processor.process_province_labels()
    processor.populate_province_data()
    processor.process_water_labels()

    # Generate output files - ONLY Empire GeoJSON enabled
    processor.generate_empire_geojson()
    processor.generate_westerland_geojson()
    processor.generate_bretonnia_geojson()
    processor.generate_karaz_ankor_geojson()
    processor.generate_poi_geojson()
    # processor.generate_roads_geojson()  # Disabled: road extraction not needed currently
    processor.generate_province_labels_geojson()
    processor.generate_water_labels_geojson()

    # Write logs
    processor.write_invalid_settlements_log()
    processor.write_duplicate_settlements_log()

    # Generate report
    processor.generate_report()

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
