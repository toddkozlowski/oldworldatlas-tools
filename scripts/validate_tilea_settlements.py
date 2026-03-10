"""Validate and relabel settlements in JUST_TILEA.svg using tilea.csv.

This script performs three checks for the layer "Tilea Settlements":
1. Renames each settlement text element's inkscape:label to match text content.
2. Detects duplicate settlement names in the SVG layer.
3. Verifies every settlement from input/gazetteers/tilea.csv exists on the SVG.
"""

import csv
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("svg", "http://www.w3.org/2000/svg")
ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_PATH = PROJECT_ROOT / "input" / "gazetteers" / "tilea.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "tilea_settlement_validation.log"
TARGET_LAYER_LABEL = "Tilea Settlements"
SVG_CANDIDATES = [
    PROJECT_ROOT / "JUST_TILEA.svg",
    PROJECT_ROOT / "input" / "JUST_TILEA.svg",
    PROJECT_ROOT / "input" / "svg" / "JUST_TILEA.svg",
    PROJECT_ROOT.parent / "oldworldatlas-maps" / "JUST_TILEA.svg",
]

NS = {
    "svg": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
}

INKSCAPE_LABEL = f"{{{NS['inkscape']}}}label"
SVG_TEXT = f"{{{NS['svg']}}}text"
SVG_TSPAN = f"{{{NS['svg']}}}tspan"
SVG_G = f"{{{NS['svg']}}}g"


@dataclass
class SettlementOccurrence:
    """Represents one settlement text object found in the SVG."""

    name: str
    element_id: str
    layer_path: str
    x: Optional[float]
    y: Optional[float]


def find_layer_by_label(root: ET.Element, label: str) -> Optional[ET.Element]:
    """Find the first group with the given inkscape layer label."""
    for group in root.iter(SVG_G):
        if group.get(INKSCAPE_LABEL) == label:
            return group
    return None


def get_text_content(text_elem: ET.Element) -> Optional[str]:
    """Extract text content from a text element and its tspans."""
    parts: List[str] = []
    for tspan in text_elem.findall(SVG_TSPAN):
        if tspan.text and tspan.text.strip():
            parts.append(tspan.text.strip())
    if not parts and text_elem.text and text_elem.text.strip():
        parts.append(text_elem.text.strip())
    return " ".join(parts) if parts else None


def parse_float_attribute(elem: ET.Element, attr_name: str) -> Optional[float]:
    """Parse an element attribute as float, returning None if unavailable."""
    value = elem.get(attr_name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def iter_text_elements(parent: ET.Element, current_path: str) -> Iterable[Tuple[ET.Element, str]]:
    """Yield all descendant text elements and their layer paths."""
    for elem in parent:
        if elem.tag == SVG_G:
            child_label = elem.get(INKSCAPE_LABEL) or elem.get("id") or "<group>"
            child_path = f"{current_path}/{child_label}" if current_path else child_label
            yield from iter_text_elements(elem, child_path)
        elif elem.tag == SVG_TEXT:
            yield elem, current_path


def load_csv_settlements(csv_path: Path) -> Tuple[List[str], List[str]]:
    """Load settlement names from tilea.csv and return names and CSV duplicates."""
    settlements: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Settlement" not in (reader.fieldnames or []):
            raise ValueError("CSV does not contain a 'Settlement' column.")

        for row in reader:
            name = (row.get("Settlement") or "").strip()
            if name:
                settlements.append(name)

    duplicate_names = sorted(
        [name for name, count in Counter(settlements).items() if count > 1],
        key=str.casefold,
    )
    return settlements, duplicate_names


def resolve_svg_path() -> Optional[Path]:
    """Resolve JUST_TILEA.svg from common project locations."""
    for candidate in SVG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def write_report(
    svg_path: Path,
    csv_path: Path,
    occurrences_by_name: Dict[str, List[SettlementOccurrence]],
    csv_duplicates: List[str],
    csv_not_in_svg: List[str],
    svg_not_in_csv: List[str],
    no_text_items: List[str],
    updated_count: int,
) -> None:
    """Write a detailed validation report to logs/tilea_settlement_validation.log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    duplicate_svg_names = {
        name: items for name, items in occurrences_by_name.items() if len(items) > 1
    }

    lines: List[str] = []
    lines.append("Tilea Settlement Validation Report")
    lines.append("=" * 80)
    lines.append(f"SVG: {svg_path}")
    lines.append(f"CSV: {csv_path}")
    lines.append(f"Layer: {TARGET_LAYER_LABEL}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 80)
    lines.append(f"SVG unique settlement names: {len(occurrences_by_name)}")
    lines.append(f"SVG duplicate names: {len(duplicate_svg_names)}")
    lines.append(f"CSV duplicate names: {len(csv_duplicates)}")
    lines.append(f"CSV settlements not in SVG: {len(csv_not_in_svg)}")
    lines.append(f"SVG settlements not in CSV: {len(svg_not_in_csv)}")
    lines.append(f"Text elements with no text content: {len(no_text_items)}")
    lines.append(f"Labels updated in SVG: {updated_count}")
    lines.append("")

    if duplicate_svg_names:
        lines.append("Duplicate settlement names in SVG")
        lines.append("-" * 80)
        for name in sorted(duplicate_svg_names.keys(), key=str.casefold):
            lines.append(f"{name} ({len(duplicate_svg_names[name])} occurrences)")
            for occurrence in duplicate_svg_names[name]:
                lines.append(
                    f"  - id={occurrence.element_id} path={occurrence.layer_path} "
                    f"x={occurrence.x} y={occurrence.y}"
                )
        lines.append("")

    if csv_duplicates:
        lines.append("Duplicate settlement names in CSV")
        lines.append("-" * 80)
        for name in csv_duplicates:
            lines.append(name)
        lines.append("")

    if csv_not_in_svg:
        lines.append("Settlements present in CSV but missing from SVG")
        lines.append("-" * 80)
        for name in csv_not_in_svg:
            lines.append(name)
        lines.append("")

    if svg_not_in_csv:
        lines.append("Settlements present in SVG but missing from CSV")
        lines.append("-" * 80)
        for name in svg_not_in_csv:
            lines.append(name)
        lines.append("")

    if no_text_items:
        lines.append("Text elements with missing text content")
        lines.append("-" * 80)
        for item in no_text_items:
            lines.append(item)
        lines.append("")

    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run Tilea settlement relabeling and validation checks."""
    svg_path = resolve_svg_path()
    if svg_path is None:
        logger.error("JUST_TILEA.svg not found. Checked:")
        for candidate in SVG_CANDIDATES:
            logger.error(f"  - {candidate}")
        return 1

    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        return 1

    raw_svg = svg_path.read_text(encoding="utf-8")
    xml_declaration = ""
    if raw_svg.startswith("<?xml"):
        end_index = raw_svg.find("?>")
        if end_index != -1:
            xml_declaration = raw_svg[: end_index + 2]

    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    settlements_layer = find_layer_by_label(root, TARGET_LAYER_LABEL)
    if settlements_layer is None:
        logger.error(f"Layer not found: {TARGET_LAYER_LABEL}")
        return 1

    occurrences_by_name: Dict[str, List[SettlementOccurrence]] = defaultdict(list)
    no_text_items: List[str] = []
    updated_count = 0

    for text_elem, layer_path in iter_text_elements(settlements_layer, TARGET_LAYER_LABEL):
        text_content = get_text_content(text_elem)
        element_id = text_elem.get("id", "<no-id>")

        if not text_content:
            no_text_items.append(f"id={element_id} path={layer_path}")
            continue

        current_label = text_elem.get(INKSCAPE_LABEL)
        if current_label != text_content:
            text_elem.set(INKSCAPE_LABEL, text_content)
            updated_count += 1

        occurrences_by_name[text_content].append(
            SettlementOccurrence(
                name=text_content,
                element_id=element_id,
                layer_path=layer_path,
                x=parse_float_attribute(text_elem, "x"),
                y=parse_float_attribute(text_elem, "y"),
            )
        )

    if updated_count > 0:
        svg_body = ET.tostring(root, encoding="unicode", xml_declaration=False)
        output_svg = (
            f"{xml_declaration}\n{svg_body}"
            if xml_declaration
            else svg_body
        )
        svg_path.write_text(output_svg, encoding="utf-8")
        logger.info(f"Updated {updated_count} settlement labels in SVG.")
    else:
        logger.info("All settlement labels already matched text content.")

    csv_settlements, csv_duplicates = load_csv_settlements(CSV_PATH)
    csv_names = set(csv_settlements)
    svg_names = set(occurrences_by_name.keys())

    csv_not_in_svg = sorted(csv_names - svg_names, key=str.casefold)
    svg_not_in_csv = sorted(svg_names - csv_names, key=str.casefold)

    duplicate_svg_names = [
        name for name, items in occurrences_by_name.items() if len(items) > 1
    ]

    logger.info(f"SVG unique settlement names: {len(svg_names)}")
    logger.info(f"CSV settlement names: {len(csv_names)}")
    logger.info(f"SVG duplicate names: {len(duplicate_svg_names)}")
    logger.info(f"CSV duplicate names: {len(csv_duplicates)}")
    logger.info(f"CSV settlements missing from SVG: {len(csv_not_in_svg)}")
    logger.info(f"SVG settlements missing from CSV: {len(svg_not_in_csv)}")
    logger.info(f"Text elements with no text content: {len(no_text_items)}")

    write_report(
        svg_path=svg_path,
        csv_path=CSV_PATH,
        occurrences_by_name=occurrences_by_name,
        csv_duplicates=csv_duplicates,
        csv_not_in_svg=csv_not_in_svg,
        svg_not_in_csv=svg_not_in_csv,
        no_text_items=no_text_items,
        updated_count=updated_count,
    )
    logger.info(f"Wrote report: {LOG_PATH}")

    has_issues = any(
        [
            duplicate_svg_names,
            csv_duplicates,
            csv_not_in_svg,
            no_text_items,
        ]
    )
    return 1 if has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())