"""
Comprehensive SVG map processing tool for the Old World Atlas.
Extracts settlements, points of interest, and labels from the FULL_MAP_CLEANED.svg file.
"""

import json
import csv
import logging
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
SVG_PATH = Path(__file__).parent.parent.parent / "oldworldatlas-maps" / "FULL_MAP_CLEANED.svg"
INPUT_DIR = Path(__file__).parent.parent / "input" / "gazetteers"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
LOGS_DIR = Path(__file__).parent.parent / "logs"

# Calibration points for coordinate conversion
CALIBRATION_POINTS = [
    # SVG coords -> Geographic coords (longitude, latitude)
    {"svg": (429.058, 408.152), "geo": (0.007, 51.465), "settlement": "Altdorf", "province": "Reikland"},
    {"svg": (495.263, 187.911), "geo": (1.167, 55.318), "settlement": "Middenheim", "province": "Middenland"},
    {"svg": (738.171, 778.741), "geo": (5.421, 44.977), "settlement": "Wachdorf", "province": "Averland"},
    {"svg": (891.383, 479.367), "geo": (8.100, 50.219), "settlement": "Waldenhof (Stirland)", "province": "Stirland"},
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


class CoordinateConverter:
    """Converts between SVG and geographic coordinate systems."""

    def __init__(self, calibration_points: List[Dict]):
        """Initialize with calibration points."""
        self.calibration_points = calibration_points
        self._calculate_transformation()

    def _calculate_transformation(self):
        """Calculate affine transformation parameters from calibration points."""
        svg_coords = np.array([p["svg"] for p in self.calibration_points])
        geo_coords = np.array([p["geo"] for p in self.calibration_points])

        # Use least squares to fit affine transformation
        # geo = A * svg + B where A is 2x2 and B is 2x1
        A_matrix = np.hstack([svg_coords, np.ones((svg_coords.shape[0], 1))])
        
        # Solve for longitude
        self.lon_coeffs = np.linalg.lstsq(A_matrix, geo_coords[:, 0], rcond=None)[0]
        # Solve for latitude
        self.lat_coeffs = np.linalg.lstsq(A_matrix, geo_coords[:, 1], rcond=None)[0]

    def svg_to_geo(self, svg_x: float, svg_y: float) -> Tuple[float, float]:
        """Convert SVG coordinates to geographic coordinates."""
        lon = self.lon_coeffs[0] * svg_x + self.lon_coeffs[1] * svg_y + self.lon_coeffs[2]
        lat = self.lat_coeffs[0] * svg_x + self.lat_coeffs[1] * svg_y + self.lat_coeffs[2]
        return (lon, lat)

    def validate_calibration(self):
        """Validate the transformation against calibration points."""
        logger.info("Validating coordinate transformation:")
        for point in self.calibration_points:
            calc_geo = self.svg_to_geo(point["svg"][0], point["svg"][1])
            expected_geo = point["geo"]
            error = np.sqrt((calc_geo[0] - expected_geo[0])**2 + (calc_geo[1] - expected_geo[1])**2)
            logger.info(f"  {point['settlement']}: Calculated {calc_geo}, Expected {expected_geo}, Error: {error:.6f}")


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

    def _apply_svg_transform(self, x: float, y: float, transform: str) -> Tuple[float, float]:
        """Apply SVG transform attribute to coordinates. Currently supports translate only."""
        if not transform:
            return (x, y)
        
        # Parse translate(x, y) or translate(x y)
        translate_match = re.search(r'translate\s*\(\s*([-\d.]+)\s*[,\s]\s*([-\d.]+)\s*\)', transform)
        if translate_match:
            tx = float(translate_match.group(1))
            ty = float(translate_match.group(2))
            return (x + tx, y + ty)
        
        return (x, y)

    def _validate_settlement_element(self, elem, province: str, parent_transform: str = "") -> Optional[Tuple[str, float, float]]:
        """Validate that element is a textbox with valid coordinates and return (name, x, y).
        
        Args:
            elem: The XML element to validate
            province: The province name for logging
            parent_transform: Accumulated transform from parent groups
            
        Returns:
            Tuple of (name, x, y) or None if invalid
        """
        # Check if it's a text element
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

        # Get coordinates and apply fudge factor (+3 to X, +4 to Y)
        try:
            svg_x = float(elem.get("x", 0)) + 3
            svg_y = float(elem.get("y", 0)) + 4
            
            # Apply element-level transform if present
            elem_transform = elem.get("transform", "")
            if elem_transform:
                svg_x, svg_y = self._apply_svg_transform(svg_x, svg_y, elem_transform)
            
            # Apply accumulated parent transforms
            if parent_transform:
                svg_x, svg_y = self._apply_svg_transform(svg_x, svg_y, parent_transform)
            
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

    def _process_settlement_elements(self, parent_elem, province_name: str, settlements_dict: dict, settlements_list: list, parent_transform: str = ""):
        """Recursively process settlement elements, handling nested layers (like Reikland estates) and transforms."""
        for elem in parent_elem:
            # Check if this is a layer/group
            if elem.tag == f"{{{NS['svg']}}}g":
                # Get group transform and accumulate with parent
                group_transform = elem.get("transform", "")
                # Combine transforms: if both exist, concatenate them
                combined_transform = (parent_transform + " " + group_transform).strip() if group_transform else parent_transform
                # Recursively process children of this layer with accumulated transform
                self._process_settlement_elements(elem, province_name, settlements_dict, settlements_list, combined_transform)
            else:
                # Process as settlement element
                result = self._validate_settlement_element(elem, province_name, parent_transform)
                if result:
                    name, svg_x, svg_y = result
                    
                    # Check for duplicates
                    if name in settlements_dict:
                        self.duplicate_settlements[province_name].append({
                            "name": name,
                            "occurrences": 2,
                            "coordinates": [settlements_dict[name], (svg_x, svg_y)]
                        })
                    else:
                        settlements_dict[name] = (svg_x, svg_y)

                        # Convert coordinates
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

        # Process each province
        provinces_seen = set()
        for province_group in empire_faction:
            province_name = province_group.get(f"{{{NS['inkscape']}}}label")
            if not province_name:
                continue

            provinces_seen.add(province_name)
            logger.info(f"  Processing province: {province_name}")

            settlements_in_province = {}

            # Extract settlements from this province (may be nested in sub-layers like estates)
            self._process_settlement_elements(province_group, province_name, settlements_in_province, self.settlements_empire)

        logger.info(f"  Found {len(self.settlements_empire)} valid settlements across {len(provinces_seen)} provinces")

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

        # Extract settlements from Westerland (may be nested in sub-layers)
        self._process_settlement_elements(westerland_faction, "Westerland", settlements_in_faction, self.settlements_westerland)

        logger.info(f"  Found {len(self.settlements_westerland)} valid Westerland settlements")

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

        # Extract settlements from Dwarf Holds, accumulating any transforms
        faction_transform = dwarf_holds_faction.get("transform", "")
        
        for elem in dwarf_holds_faction:
            # Check if this is a layer/group
            if elem.tag == f"{{{NS['svg']}}}g":
                # Get group transform and combine with faction transform
                group_transform = elem.get("transform", "")
                combined_transform = (faction_transform + " " + group_transform).strip() if group_transform else faction_transform
                # Process nested elements
                self._process_dwarf_hold_elements(elem, settlements_in_faction, combined_transform)
            elif elem.tag == f"{{{NS['svg']}}}text":
                # Direct text element
                result = self._validate_dwarf_hold_element(elem, faction_transform)
                if result:
                    name, svg_x, svg_y = result
                    if name not in settlements_in_faction:
                        settlements_in_faction[name] = (svg_x, svg_y)
                        
                        # Convert coordinates
                        geo_lon, geo_lat = self.converter.svg_to_geo(svg_x, svg_y)
                        
                        hold = DwarfHold(
                            name=name,
                            svg_x=svg_x,
                            svg_y=svg_y,
                            geo_lon=geo_lon,
                            geo_lat=geo_lat
                        )
                        self.settlements_karaz_ankor.append(hold)

        logger.info(f"  Found {len(self.settlements_karaz_ankor)} valid Karaz Ankor settlements")

    def _process_dwarf_hold_elements(self, parent_elem, settlements_dict: dict, parent_transform: str = ""):
        """Recursively process dwarf hold elements, handling nested layers and transforms."""
        for elem in parent_elem:
            # Check if this is a layer/group
            if elem.tag == f"{{{NS['svg']}}}g":
                # Get group transform and accumulate with parent
                group_transform = elem.get("transform", "")
                combined_transform = (parent_transform + " " + group_transform).strip() if group_transform else parent_transform
                # Recursively process children with accumulated transform
                self._process_dwarf_hold_elements(elem, settlements_dict, combined_transform)
            elif elem.tag == f"{{{NS['svg']}}}text":
                # Process text element
                result = self._validate_dwarf_hold_element(elem, parent_transform)
                if result:
                    name, svg_x, svg_y = result
                    
                    # Check for duplicates
                    if name in settlements_dict:
                        self.duplicate_settlements["Karaz Ankor"].append({
                            "name": name,
                            "occurrences": 2,
                            "coordinates": [settlements_dict[name], (svg_x, svg_y)]
                        })
                    else:
                        settlements_dict[name] = (svg_x, svg_y)

                        # Convert coordinates
                        geo_lon, geo_lat = self.converter.svg_to_geo(svg_x, svg_y)

                        hold = DwarfHold(
                            name=name,
                            svg_x=svg_x,
                            svg_y=svg_y,
                            geo_lon=geo_lon,
                            geo_lat=geo_lat
                        )
                        self.settlements_karaz_ankor.append(hold)

    def _validate_dwarf_hold_element(self, elem, parent_transform: str = "") -> Optional[Tuple[str, float, float]]:
        """Validate that element is a textbox with valid coordinates and return (name, x, y).
        
        Similar to _validate_settlement_element but for Dwarf Holds (no province parameter).
        """
        # Check if it's a text element
        if elem.tag != f"{{{NS['svg']}}}text":
            return None

        # Get text content
        name = self._get_text_element_label(elem)
        if not name:
            return None

        # Get coordinates and apply fudge factor (+3 to X, +4 to Y)
        try:
            svg_x = float(elem.get("x", 0)) + 3
            svg_y = float(elem.get("y", 0)) + 4
            
            # Apply element-level transform if present
            elem_transform = elem.get("transform", "")
            if elem_transform:
                svg_x, svg_y = self._apply_svg_transform(svg_x, svg_y, elem_transform)
            
            # Apply accumulated parent transforms
            if parent_transform:
                svg_x, svg_y = self._apply_svg_transform(svg_x, svg_y, parent_transform)
            
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
            # Use the main empire.csv file which contains all provinces
            csv_file = INPUT_DIR / "empire.csv"
        elif faction == "Westerland":
            csv_file = INPUT_DIR / "westerland.csv"

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
                        "4eLoSaS","4eTHRC","4eMCotWW","1eDSaS","TWW3","2eKAAotDC"}
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

    def _process_poi_elements(self, parent_elem, poi_type: str, poi_list: list, parent_transform: str = ""):
        """Recursively process POI elements, handling nested layers and transforms."""
        for elem in parent_elem:
            # Check if this is a layer/group
            if elem.tag == f"{{{NS['svg']}}}g":
                # Get group transform and accumulate with parent
                group_transform = elem.get("transform", "")
                combined_transform = (parent_transform + " " + group_transform).strip() if group_transform else parent_transform
                # Recursively process children of this layer with accumulated transform
                self._process_poi_elements(elem, poi_type, poi_list, combined_transform)
            elif elem.tag == f"{{{NS['svg']}}}text":
                name = self._get_text_element_label(elem)
                if name:
                    try:
                        # Get base coordinates (no fudge factor needed for POI)
                        svg_x = float(elem.get("x", 0))
                        svg_y = float(elem.get("y", 0))
                        
                        # Apply element-level transform if present
                        elem_transform = elem.get("transform", "")
                        if elem_transform:
                            svg_x, svg_y = self._apply_svg_transform(svg_x, svg_y, elem_transform)
                        
                        # Apply accumulated parent transforms
                        if parent_transform:
                            svg_x, svg_y = self._apply_svg_transform(svg_x, svg_y, parent_transform)
                        
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

            # Extract POI from this group (may be nested in sub-layers)
            initial_count = len(self.points_of_interest)
            group_transform = poi_group.get("transform", "")
            self._process_poi_elements(poi_group, poi_type, self.points_of_interest, group_transform)
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

            # Extract labels from this group
            initial_count = len(self.province_labels)
            self._process_province_label_elements(region_group, province_type, self.province_labels)
            count = len(self.province_labels) - initial_count

            logger.info(f"    Found {count} labels")

    def _process_province_label_elements(self, parent_elem, province_type: str, label_list: list):
        """Recursively process province label elements, handling nested layers."""
        for elem in parent_elem:
            # Check if this is a layer/group
            if elem.tag == f"{{{NS['svg']}}}g":
                # Recursively process children of this layer
                self._process_province_label_elements(elem, province_type, label_list)
            elif elem.tag == f"{{{NS['svg']}}}text":
                name = self._get_text_element_label(elem)
                if name:
                    try:
                        svg_x = float(elem.get("x", 0))
                        svg_y = float(elem.get("y", 0))
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
                for marsh_group in water_group:
                    marsh_layer_name = marsh_group.get(f"{{{NS['inkscape']}}}label")
                    if marsh_layer_name and marsh_layer_name in waterbody_type_map:
                        waterbody_type = waterbody_type_map[marsh_layer_name]
                        initial_count = len(self.water_labels)
                        # Get transform from marsh_group if it exists
                        marsh_transform = marsh_group.get("transform", "")
                        self._process_water_label_elements(marsh_group, waterbody_type, self.water_labels, marsh_transform)
                        count = len(self.water_labels) - initial_count
                        logger.info(f"    Found {count} {waterbody_type} labels")
            # Handle lakes which is a new tier
            elif layer_name == "lakes":
                logger.info(f"  Processing lakes...")
                waterbody_type = waterbody_type_map[layer_name]
                initial_count = len(self.water_labels)
                lakes_transform = water_group.get("transform", "")
                self._process_water_label_elements(water_group, waterbody_type, self.water_labels, lakes_transform)
                count = len(self.water_labels) - initial_count
                logger.info(f"    Found {count} {waterbody_type} labels")
            elif layer_name in waterbody_type_map:
                waterbody_type = waterbody_type_map[layer_name]
                logger.info(f"  Processing {waterbody_type}...")
                
                # Extract labels from this group
                initial_count = len(self.water_labels)
                group_transform = water_group.get("transform", "")
                self._process_water_label_elements(water_group, waterbody_type, self.water_labels, group_transform)
                count = len(self.water_labels) - initial_count

                logger.info(f"    Found {count} labels")

    def _process_water_label_elements(self, parent_elem, waterbody_type: str, label_list: list, parent_transform: str = ""):
        """Recursively process water label elements, handling nested layers and transforms."""
        for elem in parent_elem:
            # Check if this is a layer/group
            if elem.tag == f"{{{NS['svg']}}}g":
                # Get group transform and accumulate with parent
                group_transform = elem.get("transform", "")
                # Combine transforms: if both exist, concatenate them
                combined_transform = (parent_transform + " " + group_transform).strip() if group_transform else parent_transform
                # Recursively process children with accumulated transform
                self._process_water_label_elements(elem, waterbody_type, label_list, combined_transform)
            elif elem.tag == f"{{{NS['svg']}}}text":
                name = self._get_text_element_label(elem)
                if name:
                    try:
                        svg_x = float(elem.get("x", 0))
                        svg_y = float(elem.get("y", 0))
                        
                        # Apply element-level transform if present
                        transform = elem.get("transform", "")
                        svg_x, svg_y = self._apply_svg_transform(svg_x, svg_y, transform)
                        
                        # Apply parent/group transform
                        svg_x, svg_y = self._apply_svg_transform(svg_x, svg_y, parent_transform)
                        
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
            f.write(f"Karaz Ankor Total: {karaz_ankor_total_count} holds\n")
            f.write(f"Grand Total: {len(self.settlements_empire) + westerland_total_count + karaz_ankor_total_count} settlements/holds\n\n")

            f.write("EMPIRE SETTLEMENTS BY PROVINCE\n")
            f.write("-" * 80 + "\n")
            for province in sorted(empire_count_by_province.keys()):
                count = empire_count_by_province[province]
                pop = empire_pop_by_province[province]
                f.write(f"{province:20s} - {count:3d} settlements, {pop:10,d} total population\n")

            f.write(f"\nEmpire Total Population: {sum(empire_pop_by_province.values()):,d}\n")
            f.write(f"Westerland Total Population: {westerland_total_pop:,d}\n\n")

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

            f.write("DATA QUALITY ISSUES\n")
            f.write("-" * 80 + "\n")
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

    # Process all data
    # processor.process_settlements_empire()
    # processor.process_settlements_westerland()
    # processor.process_settlements_karaz_ankor()
    # processor.populate_settlement_data()
    # processor.populate_karaz_ankor_data()
    processor.process_points_of_interest()
    # processor.process_roads()  # Disabled: road extraction not needed currently
    # processor.process_province_labels()
    # processor.populate_province_data()
    # processor.process_water_labels()

    # Generate output files
    # processor.generate_empire_geojson()
    # processor.generate_westerland_geojson()
    # processor.generate_karaz_ankor_geojson()
    processor.generate_poi_geojson()
    # processor.generate_roads_geojson()  # Disabled: road extraction not needed currently
    # processor.generate_province_labels_geojson()
    # processor.generate_water_labels_geojson()

    # Write logs
    # processor.write_invalid_settlements_log()
    # processor.write_duplicate_settlements_log()

    # Generate report
    # processor.generate_report()

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
