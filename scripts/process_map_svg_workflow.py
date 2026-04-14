"""Refactored, standalone SVG map processing workflow with interactive CLI.

This module orchestrates map processing in optional stages:
1. Download/update gazetteer CSV files (with validation and versioned backups).
2. Optionally relabel SVG text elements for Inkscape labels.
3. Optionally fetch missing wiki metadata for CSV rows.
4. Process the SVG and merge with gazetteer CSV data.
5. Generate standardized GeoJSON outputs.
6. Optionally deploy generated GeoJSON files to another directory.

Design goals:
- Preserve extraction behavior by reusing the existing process_map_svg processor.
- Standardize workflow and reduce one-off script usage.
- Keep the workflow extensible via small step handlers.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import re
import shutil
import sys
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from xml.etree import ElementTree as ET

import process_map_svg as legacy


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_DIR = REPO_ROOT / "input" / "gazetteers"
OUTPUT_DIR = REPO_ROOT / "output"
LOGS_DIR = REPO_ROOT / "logs"
CSV_BACKUP_DIR = INPUT_DIR / "_backups"
DEFAULT_SVG_PATH = Path(__file__).parent.parent.parent / "oldworldatlas-maps" / "OLD_WORLD_ATLAS.svg"


NS = {
    "svg": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
}


STANDARD_SETTLEMENT_COLUMNS = [
    "Settlement",
    "Population",
    "Estate",
    "Trade",
    "Tags",
    "Notes",
    "Coordinates",
    "Province_2515",
    "Province_2512",
    "Province_2276",
    "Ruler_2515",
    "Ruler_2512",
    "Ruler_2276",
    "wiki_url",
    "wiki_title",
    "wiki_description",
    "wiki_image",
]

SKAVENDOM_COLUMNS = [
    "Settlement",
    "Type",
    "Population",
    "Major Clan(s)",
    "Minor Clan(s)",
    "Trade",
    "Tags",
    "Notes",
    "Coordinates",
    "wiki_url",
    "wiki_title",
    "wiki_description",
    "wiki_image",
]

PROVINCES_COLUMNS = [
    "name",
    "formal_title",
    "part_of",
    "population",
    "province_type",
    "local_category",
    "wiki_url",
    "wiki_title",
    "wiki_description",
    "wiki_image",
]

GEO_FEATURE_COLUMNS = [
    "name",
    "type",
    "wiki_url",
    "wiki_title",
    "wiki_description",
    "wiki_image",
]

POI_COLUMNS = [
    "name",
    "type",
    "tags",
    "notes",
    "wiki_url",
    "wiki_title",
    "wiki_description",
    "wiki_image",
]


@dataclass(frozen=True)
class GazetteerSchema:
    """Schema definition for one CSV gazetteer file."""

    filename: str
    required_columns: list[str]
    required_non_empty: list[str]
    allowed_values: dict[str, set[str]] = field(default_factory=dict)
    gid: Optional[str] = None


@dataclass
class WorkflowConfig:
    """Runtime options for interactive/non-interactive workflow execution."""

    interactive: bool = True
    auto_yes: bool = False
    download_csv: Optional[bool] = None
    relabel_svg: Optional[bool] = None
    fetch_wiki: Optional[bool] = None
    update_csv_with_wiki: Optional[bool] = None
    push_geojson: Optional[bool] = None
    deploy_target: Optional[Path] = None
    svg_path: Path = DEFAULT_SVG_PATH


@dataclass
class WorkflowState:
    """Shared mutable state for all step handlers."""

    config: WorkflowConfig
    report_lines: list[str] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)
    csv_not_in_svg: dict[str, list[str]] = field(default_factory=dict)
    duplicate_names_by_province: dict[str, dict[str, int]] = field(default_factory=dict)
    generated_files: list[Path] = field(default_factory=list)
    processor: Optional[legacy.SVGMapProcessor] = None


@dataclass(frozen=True)
class WorkflowStep:
    """One workflow stage."""

    key: str
    description: str
    handler: Callable[[WorkflowState], None]
    optional: bool = True


GAZETTEER_SCHEMAS: dict[str, GazetteerSchema] = {
    "empire.csv": GazetteerSchema(
        filename="empire.csv",
        required_columns=STANDARD_SETTLEMENT_COLUMNS,
        required_non_empty=["Settlement"],
        gid="1095068342",
    ),
    "westerland.csv": GazetteerSchema(
        filename="westerland.csv",
        required_columns=STANDARD_SETTLEMENT_COLUMNS,
        required_non_empty=["Settlement"],
        gid="1817375421",
    ),
    "bretonnia.csv": GazetteerSchema(
        filename="bretonnia.csv",
        required_columns=STANDARD_SETTLEMENT_COLUMNS,
        required_non_empty=["Settlement"],
        gid="890230032",
    ),
    "tilea.csv": GazetteerSchema(
        filename="tilea.csv",
        required_columns=STANDARD_SETTLEMENT_COLUMNS,
        required_non_empty=["Settlement"],
        gid="1839982734",
    ),
    "estalia.csv": GazetteerSchema(
        filename="estalia.csv",
        required_columns=STANDARD_SETTLEMENT_COLUMNS,
        required_non_empty=["Settlement"],
        gid="124941632",
    ),
    "norsca.csv": GazetteerSchema(
        filename="norsca.csv",
        required_columns=STANDARD_SETTLEMENT_COLUMNS,
        required_non_empty=["Settlement"],
        gid="2036460226",
    ),
    "border_princes.csv": GazetteerSchema(
        filename="border_princes.csv",
        required_columns=STANDARD_SETTLEMENT_COLUMNS,
        required_non_empty=["Settlement"],
        gid="1123781756",
    ),
    "kislev.csv": GazetteerSchema(
        filename="kislev.csv",
        required_columns=STANDARD_SETTLEMENT_COLUMNS,
        required_non_empty=["Settlement"],
        gid="249472016",
    ),
    "karaz_ankor.csv": GazetteerSchema(
        filename="karaz_ankor.csv",
        required_columns=["Settlement", "Type", *STANDARD_SETTLEMENT_COLUMNS[1:]],
        required_non_empty=["Settlement"],
        gid="2134669291",
    ),
    "wood_elves.csv": GazetteerSchema(
        filename="wood_elves.csv",
        required_columns=["Settlement", "Type", *STANDARD_SETTLEMENT_COLUMNS[1:]],
        required_non_empty=["Settlement"],
        gid="904575690",
    ),
    "skavendom.csv": GazetteerSchema(
        filename="skavendom.csv",
        required_columns=SKAVENDOM_COLUMNS,
        required_non_empty=["Settlement", "Type"],
        gid="1825010677",
    ),
    "provinces.csv": GazetteerSchema(
        filename="provinces.csv",
        required_columns=PROVINCES_COLUMNS,
        required_non_empty=["name", "province_type"],
        allowed_values={
            "province_type": {"Nation", "Major Division", "Minor Division"},
        },
        gid="2020870558",
    ),
    "geographic_feature_labels.csv": GazetteerSchema(
        filename="geographic_feature_labels.csv",
        required_columns=GEO_FEATURE_COLUMNS,
        required_non_empty=["name", "type"],
        allowed_values={
            "type": {
                "Ocean",
                "Major Sea",
                "Large Sea",
                "Medium Sea",
                "Small Sea",
                "Lake",
                "Large Marsh",
                "Small Marsh",
                "Large River",
                "Medium River",
                "Small River",
                "Large Forest",
                "Small Forest",
                "Other",
            }
        },
        gid="1527667508",
    ),
    "points_of_interest.csv": GazetteerSchema(
        filename="points_of_interest.csv",
        required_columns=POI_COLUMNS,
        required_non_empty=["name", "type"],
        allowed_values={
            "type": {
                "City Districts",
                "Taverns and Inns",
                "Forts and Castles",
                "Monasteries and Temples",
                "Chaos Shrines",
                "Other",
                "Geographic Landmarks",
                "High Elf Colonies",
                "Gnome Burrows",
                "Greenskin Landmarks",
            }
        },
        gid="695954623"
    ),
}


# (Human settlements (processed with generic handler)
# Metadata: (csv_filename, default_province_name, type_property)
HUMAN_SETTLEMENTS = {
    "Empire": ("empire.csv", None, None),
    "Bretonnia": ("bretonnia.csv", "Bretonnia", None),
    "Westerland": ("westerland.csv", "Westerland", None),
    "Kislev": ("kislev.csv", "Kislev", None),
    "Tilea": ("tilea.csv", "Tilea", None),
    "Norsca": ("norsca.csv", "Norsca", None),
    "Border Princes": ("border_princes.csv", "Border Princes", None),
    "Estalia": ("estalia.csv", "Estalia", None),
}


def now_stamp() -> str:
    """Return a filesystem-safe ISO-like timestamp."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def prompt_bool(question: str, default: bool = True, auto_yes: bool = False) -> bool:
    """Prompt a yes/no question in interactive mode."""
    if auto_yes:
        return True
    default_label = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{question} [{default_label}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def normalize_name(name: str) -> str:
    """Normalize a name for stable key matching."""
    value = unicodedata.normalize("NFKD", (name or "").strip())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", value).lower()


def parse_tags(value: str) -> list[str]:
    """Parse semi-colon tags into a list."""
    if not value:
        return []
    cleaned = value.strip().strip('"')
    if not cleaned:
        return []
    return [part.strip() for part in cleaned.split(";") if part.strip()]


def parse_notes(value: str) -> list[str]:
    """Parse notes into a list while preserving delimiter intent."""
    if not value:
        return []
    cleaned = value.strip().strip('"')
    if not cleaned:
        return []
    if ";" in cleaned:
        return [part.strip() for part in cleaned.split(";") if part.strip()]
    return [cleaned]


def safe_int(value: Any, default: int = 0) -> int:
    """Convert to int safely."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def calculate_size_category(population: int) -> int:
    """Map population to size category using existing thresholds."""
    if population <= 300:
        return 1
    if population <= 900:
        return 2
    if population <= 3000:
        return 3
    if population <= 15000:
        return 4
    if population <= 49999:
        return 5
    return 6


def random_population() -> int:
    """Replicate legacy random population strategy."""
    generated = int(random.lognormvariate(5.0, 0.8))
    return 782 if generated > 800 else generated


def ensure_dirs() -> None:
    """Ensure required runtime directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def csv_path(filename: str) -> Path:
    """Resolve a gazetteer path."""
    return INPUT_DIR / filename


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV to header+rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return (reader.fieldnames or [], [dict(row) for row in reader])


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write full CSV rows back to disk."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_with_retention(source_file: Path, backup_dir: Path, max_versions: int = 3) -> Optional[Path]:
    """Backup file to timestamped copy and prune old backups."""
    if not source_file.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()
    backup_file = backup_dir / f"{source_file.stem}_{stamp}{source_file.suffix}"
    shutil.copy2(source_file, backup_file)

    pattern = f"{source_file.stem}_*{source_file.suffix}"
    backups = sorted(backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in backups[max_versions:]:
        stale.unlink(missing_ok=True)

    return backup_file


def download_google_sheet_csv(gid: str) -> str:
    """Fetch CSV text from Google Sheets export endpoint."""
    spreadsheet_id = "1NjA45QfX9vfy97ZA8a7HcN-Zc-TKdr-nWGhKvaMRpVQ"
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def validate_coordinates_format(value: str) -> bool:
    """Validate expected coordinate format 'longitude latitude'."""
    if not value or not value.strip():
        return True
    return re.match(r"^\s*-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\s*$", value) is not None


def validate_csv_against_schema(schema: GazetteerSchema, rows: list[dict[str, str]], header: list[str]) -> list[str]:
    """Validate one CSV file according to schema rules."""
    issues: list[str] = []
    missing_columns = [col for col in schema.required_columns if col not in header]
    if missing_columns:
        issues.append(f"{schema.filename}: Missing columns: {missing_columns}")
        return issues

    for idx, row in enumerate(rows, start=2):
        for col in schema.required_non_empty:
            value = (row.get(col) or "").strip()
            if not value:
                issues.append(f"{schema.filename}:{idx}: Required column '{col}' is empty")

        if "Coordinates" in row and not validate_coordinates_format(row.get("Coordinates", "")):
            issues.append(
                f"{schema.filename}:{idx}: Invalid Coordinates format: {row.get('Coordinates', '')}"
            )

        for col, allowed in schema.allowed_values.items():
            value = (row.get(col) or "").strip()
            if value and value not in allowed:
                issues.append(
                    f"{schema.filename}:{idx}: Invalid '{col}' value '{value}'. Allowed: {sorted(allowed)}"
                )

    return issues


def step_sync_and_validate_csv(state: WorkflowState) -> None:
    """Download/update gazetteers and validate all available CSVs."""
    cfg = state.config

    should_download = cfg.download_csv
    if should_download is None and cfg.interactive:
        should_download = prompt_bool("Download/update gazetteers from Google Sheets?", default=False, auto_yes=cfg.auto_yes)
    should_download = bool(should_download)

    if should_download:
        logger.info("Downloading configured gazetteers...")
        for filename, schema in GAZETTEER_SCHEMAS.items():
            if not schema.gid:
                continue
            target = csv_path(filename)
            try:
                csv_text = download_google_sheet_csv(schema.gid)
                header = next(csv.reader(io.StringIO(csv_text)), [])
                rows = list(csv.DictReader(io.StringIO(csv_text)))
                issues = validate_csv_against_schema(schema, rows, header)
                if issues:
                    state.validation_issues.extend(issues)
                    logger.warning("Validation failed for downloaded %s; existing file retained", filename)
                    continue

                backup_file = backup_with_retention(target, CSV_BACKUP_DIR, max_versions=3)
                target.write_text(csv_text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")
                if backup_file:
                    logger.info("Updated %s (backup: %s)", target.name, backup_file.name)
                else:
                    logger.info("Created %s", target.name)
            except Exception as exc:  # noqa: BLE001
                message = f"{filename}: Download failed: {exc}"
                state.validation_issues.append(message)
                logger.warning(message)

    logger.info("Validating local gazetteer CSV files...")
    for filename, schema in GAZETTEER_SCHEMAS.items():
        path = csv_path(filename)
        if not path.exists():
            state.validation_issues.append(f"{filename}: File not found")
            continue
        try:
            header, rows = read_csv_rows(path)
            issues = validate_csv_against_schema(schema, rows, header)
            state.validation_issues.extend(issues)
        except Exception as exc:  # noqa: BLE001
            state.validation_issues.append(f"{filename}: Failed to read/validate CSV: {exc}")

    if state.validation_issues:
        logger.warning("CSV validation found %d issue(s)", len(state.validation_issues))
    else:
        logger.info("CSV validation passed without issues")


def find_layer_by_label(root: ET.Element, layer_label: str) -> Optional[ET.Element]:
    """Find a layer/group by inkscape label."""
    label_attr = f"{{{NS['inkscape']}}}label"
    g_tag = f"{{{NS['svg']}}}g"
    for elem in root.iter(g_tag):
        if elem.get(label_attr) == layer_label:
            return elem
    return None


def text_content(text_elem: ET.Element) -> Optional[str]:
    """Extract plain text from a text element."""
    tspan_tag = f"{{{NS['svg']}}}tspan"
    parts: list[str] = []
    for tspan in text_elem.findall(tspan_tag):
        if tspan.text and tspan.text.strip():
            parts.append(tspan.text.strip())
    if not parts and text_elem.text and text_elem.text.strip():
        parts.append(text_elem.text.strip())
    return " ".join(parts) if parts else None


def relabel_text_nodes(parent: ET.Element, updated: list[int]) -> None:
    """Recursively relabel text nodes with their displayed text."""
    g_tag = f"{{{NS['svg']}}}g"
    text_tag = f"{{{NS['svg']}}}text"
    label_attr = f"{{{NS['inkscape']}}}label"

    for child in parent:
        if child.tag == g_tag:
            relabel_text_nodes(child, updated)
        elif child.tag == text_tag:
            label = text_content(child)
            if not label:
                continue
            if child.get(label_attr) != label:
                child.set(label_attr, label)
                updated[0] += 1


def step_relabel_svg(state: WorkflowState) -> None:
    """Optional label sync for Settlements and Points of Interest text nodes."""
    cfg = state.config
    should_run = cfg.relabel_svg
    if should_run is None and cfg.interactive:
        should_run = prompt_bool(
            "Relabel SVG text elements to match displayed names? (modifies SVG file)",
            default=False,
            auto_yes=cfg.auto_yes,
        )
    if not should_run:
        return

    svg_path = cfg.svg_path
    if not svg_path.exists():
        state.validation_issues.append(f"SVG not found for relabel step: {svg_path}")
        return

    ET.register_namespace("", NS["svg"])
    ET.register_namespace("svg", NS["svg"])
    ET.register_namespace("inkscape", NS["inkscape"])
    ET.register_namespace("sodipodi", NS["sodipodi"])

    raw_text = svg_path.read_text(encoding="utf-8")
    xml_decl = ""
    if raw_text.startswith("<?xml"):
        end_idx = raw_text.index("?>") + 2
        xml_decl = raw_text[:end_idx]

    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    targets = ["Settlements", "Points of Interest"]
    updated_count = [0]

    for target in targets:
        layer = find_layer_by_label(root, target)
        if layer is None:
            state.validation_issues.append(f"Relabel step: missing SVG layer '{target}'")
            continue
        relabel_text_nodes(layer, updated_count)

    if updated_count[0] == 0:
        logger.info("Relabel step made no SVG changes")
        return

    backup_with_retention(svg_path, svg_path.parent / "_backups", max_versions=3)
    xml_body = ET.tostring(root, encoding="unicode", xml_declaration=False)
    output = (xml_decl + "\n" + xml_body) if xml_decl else xml_body
    svg_path.write_text(output, encoding="utf-8")
    logger.info("Relabel step updated %d SVG text labels", updated_count[0])


def extract_name_variants(name: str) -> list[str]:
    """Generate lookup variants for a name with optional parenthetical alias."""
    variants = [name]
    match = re.search(r"(.+?)\s*\((.+?)\)", name)
    if match:
        without_parens = match.group(1).strip()
        alias = match.group(2).strip()
        if without_parens and without_parens not in variants:
            variants.append(without_parens)
        if alias and alias not in variants:
            variants.append(alias)

    expanded: list[str] = []
    for variant in variants:
        expanded.append(variant)
        normalized = unicodedata.normalize("NFD", variant)
        latin = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        if latin != variant:
            expanded.append(latin)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        key = normalize_name(item)
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped


def first_n_sentences(text: str, n: int = 3) -> str:
    """Return first n sentence-like chunks from plain text."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return " ".join(sentences[:n]).strip()


def fetch_wiki_metadata(name: str) -> Optional[dict[str, str]]:
    """Fetch wiki metadata for a settlement/label candidate name."""
    base_url = "https://warhammerfantasy.fandom.com/api.php"
    variants = extract_name_variants(name)

    for candidate in variants:
        params = {
            "action": "query",
            "format": "json",
            "titles": candidate,
            "prop": "info|pageimages|extracts",
            "inprop": "url",
            "piprop": "original",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            continue

        pages = payload.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                continue
            title = page_data.get("title", "")
            full_url = page_data.get("fullurl", "")
            extract = first_n_sentences(page_data.get("extract", ""), n=3)
            image = ""
            if "original" in page_data:
                image = page_data.get("original", {}).get("source", "")
            return {
                "wiki_title": title,
                "wiki_url": full_url,
                "wiki_description": extract,
                "wiki_image": image,
            }

    return None


def csv_name_field(filename: str) -> str:
    """Return the canonical key column for a gazetteer."""
    if filename in {"provinces.csv", "points_of_interest.csv", "geographic_feature_labels.csv"}:
        return "name"
    return "Settlement"


def gazetteer_display_name(filename: str) -> str:
    """Return a human-readable faction/collection label from filename."""
    stem = Path(filename).stem
    return stem.replace("_", " ").title()


def enrich_csv_with_wiki(path: Path) -> tuple[int, int, int]:
    """Fill missing wiki fields and emit live progress for this CSV.

    Returns:
        (checked_missing_count, existing_metadata_count, newly_added_count)
    """
    fieldnames, rows = read_csv_rows(path)
    if not rows:
        return (0, 0, 0)

    key_field = csv_name_field(path.name)
    if key_field not in fieldnames:
        return (0, 0, 0)

    faction_label = gazetteer_display_name(path.name)
    candidate_rows = [row for row in rows if (row.get(key_field) or "").strip()]
    total_candidates = len(candidate_rows)

    logger.info("Checking for %s wiki articles...", faction_label)
    if total_candidates == 0:
        logger.info("Checking [0 / 0], [0] with existing wiki metadata, [0] newly added.")
        return (0, 0, 0)

    updated = 0
    checked = 0
    existing = 0

    for idx, row in enumerate(candidate_rows, 1):
        wiki_url = (row.get("wiki_url") or "").strip()
        wiki_title = (row.get("wiki_title") or "").strip()
        if wiki_url or wiki_title:
            existing += 1
            continue

        name = (row.get(key_field) or "").strip()
        if not name:
            continue

        checked += 1
        metadata = fetch_wiki_metadata(name)
        if not metadata:
            continue

        row["wiki_url"] = metadata.get("wiki_url", "")
        row["wiki_title"] = metadata.get("wiki_title", "")
        row["wiki_description"] = metadata.get("wiki_description", "")
        row["wiki_image"] = metadata.get("wiki_image", "")
        updated += 1
        logger.info(
            "Checking [%d / %d], [%d] with existing wiki metadata, [%d] newly added.",
            idx,
            total_candidates,
            existing,
            updated,
        )

    logger.info(
        "Checking [%d / %d], [%d] with existing wiki metadata, [%d] newly added.",
        total_candidates,
        total_candidates,
        existing,
        updated,
    )

    if updated > 0:
        write_csv_rows(path, fieldnames, rows)

    return (checked, existing, updated)


def step_fetch_wiki_metadata(state: WorkflowState) -> None:
    """Optional wiki enrichment stage for gazetteer CSV files."""
    cfg = state.config
    should_run = cfg.fetch_wiki
    if should_run is None and cfg.interactive:
        should_run = prompt_bool("Fetch missing wiki metadata from Warhammer Wiki?", default=False, auto_yes=cfg.auto_yes)
    if not should_run:
        return

    should_update_csv = cfg.update_csv_with_wiki
    if should_update_csv is None and cfg.interactive:
        should_update_csv = prompt_bool(
            "Update gazetteer CSV files with fetched wiki metadata?",
            default=True,
            auto_yes=cfg.auto_yes,
        )
    should_update_csv = bool(should_update_csv)

    if not should_update_csv:
        logger.info("Wiki metadata lookup enabled, but CSV update skipped by user choice")
        return

    total_checked = 0
    total_existing = 0
    total_updated = 0

    for filename in GAZETTEER_SCHEMAS:
        path = csv_path(filename)
        if not path.exists():
            continue
        if not {"wiki_url", "wiki_title", "wiki_description", "wiki_image"}.issubset(set(read_csv_rows(path)[0])):
            continue

        backup_with_retention(path, CSV_BACKUP_DIR, max_versions=3)
        checked, existing, updated = enrich_csv_with_wiki(path)
        total_checked += checked
        total_existing += existing
        total_updated += updated
        logger.info(
            "Wiki metadata: %s checked_missing=%d existing=%d updated=%d",
            filename,
            checked,
            existing,
            updated,
        )

    logger.info(
        "Wiki metadata summary: checked_missing=%d existing=%d updated=%d",
        total_checked,
        total_existing,
        total_updated,
    )


def csv_settlement_rows(filename: str) -> list[dict[str, str]]:
    """Read settlement gazetteer rows if available."""
    path = csv_path(filename)
    if not path.exists():
        return []
    _, rows = read_csv_rows(path)
    return rows


def wiki_payload_from_row(row: dict[str, str]) -> dict[str, Optional[str]]:
    """Create standardized wiki payload."""
    def null_or_value(value: str) -> Optional[str]:
        stripped = (value or "").strip()
        return stripped if stripped else None

    return {
        "title": null_or_value(row.get("wiki_title", "")),
        "url": null_or_value(row.get("wiki_url", "")),
        "description": null_or_value(row.get("wiki_description", "")),
        "image": null_or_value(row.get("wiki_image", "")),
    }


def build_settlement_features(
    filename: str,
    extracted: list[Any],
    state: WorkflowState,
    default_province: str = "",
    type_property: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Merge extracted settlement coordinates with CSV properties."""
    rows = csv_settlement_rows(filename)

    # Index extracted by (name, province)
    extracted_index: dict[tuple[str, str], list[Any]] = {}
    for item in extracted:
        province = getattr(item, "province", "") or default_province
        key = (normalize_name(getattr(item, "name", "")), normalize_name(province))
        extracted_index.setdefault(key, []).append(item)

    features: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str]] = set()
    missing_from_svg: list[str] = []

    province_duplicates: dict[str, dict[str, int]] = {}
    for row in rows:
        name = (row.get("Settlement") or "").strip()
        if not name:
            continue
        province = (
            row.get("Province_2515")
            or row.get("Province_2512")
            or row.get("Province_2276")
            or default_province
            or ""
        ).strip()

        province_key = province or default_province or ""
        province_duplicates.setdefault(province_key, {})
        province_duplicates[province_key][name] = province_duplicates[province_key].get(name, 0) + 1

        lookup_key = (normalize_name(name), normalize_name(province))
        candidates = extracted_index.get(lookup_key, [])

        # Fallback for non-province nations
        if not candidates and default_province:
            lookup_key = (normalize_name(name), normalize_name(default_province))
            candidates = extracted_index.get(lookup_key, [])

        if not candidates:
            # Last fallback by name only
            name_matches = [
                entry
                for (entry_name, _prov), entries in extracted_index.items()
                if entry_name == normalize_name(name)
                for entry in entries
            ]
            candidates = name_matches

        if not candidates:
            missing_from_svg.append(name)
            continue

        match = candidates[0]
        used_keys.add((normalize_name(getattr(match, "name", "")), normalize_name(getattr(match, "province", "") or default_province)))

        population = safe_int(row.get("Population"), default=0)
        if population <= 0:
            population = random_population()

        tags = parse_tags(row.get("Tags", ""))
        trade = (row.get("Trade") or "").strip()
        if trade:
            tags.append(f"trade:{trade}")

        properties: dict[str, Any] = {
            "name": name,
            "province": province or default_province,
            "estate": (row.get("Estate") or "").strip(),
            "population": population,
            "tags": tags,
            "notes": parse_notes(row.get("Notes", "")),
            "size_category": calculate_size_category(population),
            "wiki": wiki_payload_from_row(row),
        }

        if type_property:
            properties[type_property] = (row.get("Type") or "").strip()

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [getattr(match, "geo_lon", 0.0), getattr(match, "geo_lat", 0.0)],
                },
                "properties": properties,
            }
        )

    # Keep SVG-only entries so we do not lose extracted coordinates.
    for item in extracted:
        item_name = getattr(item, "name", "")
        item_province = getattr(item, "province", "") or default_province
        item_key = (normalize_name(item_name), normalize_name(item_province))
        if item_key in used_keys:
            continue

        pop = random_population()
        properties: dict[str, Any] = {
            "name": item_name,
            "province": item_province,
            "estate": "",
            "population": pop,
            "tags": list(getattr(item, "tags", []) or []),
            "notes": list(getattr(item, "notes", []) or []),
            "size_category": calculate_size_category(pop),
            "wiki": getattr(item, "wiki", {"title": None, "url": None, "description": None, "image": None}),
        }
        if type_property:
            properties[type_property] = getattr(item, type_property, "")

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [getattr(item, "geo_lon", 0.0), getattr(item, "geo_lat", 0.0)],
                },
                "properties": properties,
            }
        )

    if missing_from_svg:
        state.csv_not_in_svg[filename] = sorted(set(missing_from_svg))

    duplicate_summary = {
        province: {name: count for name, count in names.items() if count > 1}
        for province, names in province_duplicates.items()
    }
    duplicate_summary = {k: v for k, v in duplicate_summary.items() if v}
    if duplicate_summary:
        state.duplicate_names_by_province[filename] = {
            f"{province}:{name}": count
            for province, names in duplicate_summary.items()
            for name, count in names.items()
        }

    return features


def build_index_by_name(extracted: list[Any]) -> dict[str, Any]:
    """Build name-normalized index for one-to-one label merge."""
    index: dict[str, Any] = {}
    for item in extracted:
        key = normalize_name(getattr(item, "name", ""))
        if key and key not in index:
            index[key] = item
    return index


def build_points_of_interest_features(state: WorkflowState) -> list[dict[str, Any]]:
    """Merge POI CSV records with extracted POI coordinates."""
    assert state.processor is not None
    extracted = state.processor.points_of_interest
    index = build_index_by_name(extracted)

    path = csv_path("points_of_interest.csv")
    rows: list[dict[str, str]] = []
    if path.exists():
        _, rows = read_csv_rows(path)

    features: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    missing: list[str] = []

    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        key = normalize_name(name)
        candidate = index.get(key)
        if not candidate:
            missing.append(name)
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [candidate.geo_lon, candidate.geo_lat],
                },
                "properties": {
                    "name": name,
                    "type": (row.get("type") or getattr(candidate, "poi_type", "")).strip(),
                    "description": (row.get("description") or "").strip(),
                    "tags": parse_tags(row.get("tags", "")),
                    "wiki": wiki_payload_from_row(row),
                },
            }
        )
        seen_names.add(key)

    for candidate in extracted:
        key = normalize_name(candidate.name)
        if key in seen_names:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [candidate.geo_lon, candidate.geo_lat]},
                "properties": {
                    "name": candidate.name,
                    "type": candidate.poi_type,
                    "description": "",
                    "tags": [],
                    "wiki": {"title": None, "url": None, "description": None, "image": None},
                },
            }
        )

    if missing:
        state.csv_not_in_svg["points_of_interest.csv"] = sorted(set(missing))
    return features


def build_province_features(state: WorkflowState) -> list[dict[str, Any]]:
    """Merge province CSV records with extracted province label coordinates."""
    assert state.processor is not None
    extracted = state.processor.province_labels
    index = build_index_by_name(extracted)

    path = csv_path("provinces.csv")
    rows: list[dict[str, str]] = []
    if path.exists():
        _, rows = read_csv_rows(path)

    features: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing: list[str] = []

    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        key = normalize_name(name)
        item = index.get(key)
        if not item:
            missing.append(name)
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item.geo_lon, item.geo_lat]},
                "properties": {
                    "name": name,
                    "province_type": (row.get("province_type") or item.province_type).strip(),
                    "formal_title": (row.get("formal_title") or "").strip(),
                    "part_of": (row.get("part_of") or "").strip(),
                    "population": safe_int(row.get("population"), default=0),
                    "wiki": wiki_payload_from_row(row),
                },
            }
        )
        seen.add(key)

    for item in extracted:
        key = normalize_name(item.name)
        if key in seen:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item.geo_lon, item.geo_lat]},
                "properties": {
                    "name": item.name,
                    "province_type": item.province_type,
                    "formal_title": "",
                    "part_of": "",
                    "population": 0,
                    "wiki": {"title": None, "url": None, "description": None, "image": None},
                },
            }
        )

    if missing:
        state.csv_not_in_svg["provinces.csv"] = sorted(set(missing))
    return features


def build_geographic_feature_features(state: WorkflowState) -> list[dict[str, Any]]:
    """Merge geographic feature CSV records with extracted water label coordinates."""
    assert state.processor is not None
    extracted = state.processor.water_labels
    index = build_index_by_name(extracted)

    path = csv_path("geographic_feature_labels.csv")
    rows: list[dict[str, str]] = []
    if path.exists():
        _, rows = read_csv_rows(path)

    features: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing: list[str] = []

    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        key = normalize_name(name)
        item = index.get(key)
        if not item:
            missing.append(name)
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item.geo_lon, item.geo_lat]},
                "properties": {
                    "name": name,
                    "type": (row.get("type") or item.waterbody_type).strip(),
                    "description": None,
                    "wiki": wiki_payload_from_row(row),
                },
            }
        )
        seen.add(key)

    for item in extracted:
        key = normalize_name(item.name)
        if key in seen:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item.geo_lon, item.geo_lat]},
                "properties": {
                    "name": item.name,
                    "type": item.waterbody_type,
                    "description": None,
                    "wiki": {"title": None, "url": None, "description": None, "image": None},
                },
            }
        )

    if missing:
        state.csv_not_in_svg["geographic_feature_labels.csv"] = sorted(set(missing))
    return features


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    """Write a FeatureCollection GeoJSON file."""
    payload = {"type": "FeatureCollection", "features": features}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def step_process_svg_and_generate_outputs(state: WorkflowState) -> None:
    """Run extraction and write standardized output GeoJSON files."""
    if not state.config.svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {state.config.svg_path}")

    processor = legacy.SVGMapProcessor()
    state.processor = processor

    # Process human settlements generically
    for region_name in HUMAN_SETTLEMENTS.keys():
        processor.process_human_settlements(region_name)

    # Process special cases (non-human settlements)
    processor.process_settlements_karaz_ankor()
    processor.process_settlements_wood_elves()
    processor.populate_settlement_data()
    processor.populate_karaz_ankor_data()
    processor.populate_wood_elves_data()
    processor.process_points_of_interest()
    processor.process_province_labels()
    processor.populate_province_data()
    processor.process_water_labels()

    outputs: dict[str, list[dict[str, Any]]] = {}

    # Generate outputs for all human settlements generically
    for region_name, (csv_filename, default_province, type_property) in HUMAN_SETTLEMENTS.items():
        # Get the settlements list from processor
        attr_key = region_name.lower().replace(" ", "_")
        settlements_attr = f"settlements_{attr_key}"
        settlements_list = getattr(processor, settlements_attr, [])
        
        # Generate output filename
        output_filename = f"settlements_{csv_filename.replace('.csv', '')}.geojson"
        
        # Build features with appropriate parameters
        kwargs: dict[str, Any] = {"state": state}
        if default_province is not None:
            kwargs["default_province"] = default_province
        if type_property is not None:
            kwargs["type_property"] = type_property
        
        outputs[output_filename] = build_settlement_features(csv_filename, settlements_list, **kwargs)

    # Process special cases (karaz_ankor, wood_elves)
    outputs["settlements_karaz_ankor.geojson"] = build_settlement_features(
        "karaz_ankor.csv",
        processor.settlements_karaz_ankor,
        state,
        default_province="Karaz Ankor",
        type_property="hold_type",
    )
    outputs["settlements_wood_elves.geojson"] = build_settlement_features(
        "wood_elves.csv",
        processor.settlements_wood_elves,
        state,
        default_province="Wood Elves",
        type_property="settlement_type",
    )

    # Ensure placeholder entries exist for unsupported settlements
    for missing_name in ["skavendom"]:
        path = csv_path(f"{missing_name}.csv")
        if path.exists() and f"settlements_{missing_name}.geojson" not in outputs:
            outputs[f"settlements_{missing_name}.geojson"] = []

    outputs["points_of_interest.geojson"] = build_points_of_interest_features(state)
    outputs["province_labels.geojson"] = build_province_features(state)
    outputs["geographic_feature_labels.geojson"] = build_geographic_feature_features(state)

    for filename, features in outputs.items():
        out_path = OUTPUT_DIR / filename
        write_geojson(out_path, features)
        state.generated_files.append(out_path)
        logger.info("Generated %s (%d features)", out_path.name, len(features))


def backup_destination_file(dest_file: Path, backup_dir: Path, max_versions: int = 3) -> Optional[Path]:
    """Move existing destination file to timestamped backup with retention."""
    if not dest_file.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()
    backup_file = backup_dir / f"{dest_file.stem}_backup_{stamp}{dest_file.suffix}"
    shutil.move(str(dest_file), backup_file)

    backups = sorted(
        backup_dir.glob(f"{dest_file.stem}_backup_*{dest_file.suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in backups[max_versions:]:
        stale.unlink(missing_ok=True)

    return backup_file


def step_deploy_outputs(state: WorkflowState) -> None:
    """Optional deployment of generated GeoJSON to target directory."""
    cfg = state.config

    should_push = cfg.push_geojson
    if should_push is None and cfg.interactive:
        should_push = prompt_bool("Push generated GeoJSON files to another directory?", default=False, auto_yes=cfg.auto_yes)
    if not should_push:
        return

    target_dir = cfg.deploy_target
    if target_dir is None and cfg.interactive:
        raw = input("Destination directory path: ").strip()
        target_dir = Path(raw) if raw else None

    if target_dir is None:
        raise ValueError("Deployment requested but no destination directory was provided")

    target_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = target_dir / "backup"

    changes: list[str] = []
    for src in state.generated_files:
        if not src.exists():
            continue
        dest = target_dir / src.name
        backup = backup_destination_file(dest, backup_dir, max_versions=3)
        if backup:
            changes.append(f"backup: {backup.name}")
        action = "overwritten" if dest.exists() else "new"
        shutil.copy2(src, dest)
        changes.append(f"{action}: {dest.name}")

    logger.info("Deployment complete: %d change(s)", len(changes))
    for change in changes:
        logger.info("  %s", change)


def step_generate_report(state: WorkflowState) -> None:
    """Write consolidated processing report log."""
    report_path = LOGS_DIR / "processing_report_refactored.txt"

    lines: list[str] = []
    lines.append("Old World Atlas Refactored Workflow Report")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("CSV VALIDATION ISSUES")
    lines.append("-" * 80)
    if state.validation_issues:
        lines.extend(state.validation_issues)
    else:
        lines.append("No validation issues detected.")
    lines.append("")

    lines.append("CSV ENTRIES WITHOUT SVG MATCH")
    lines.append("-" * 80)
    if state.csv_not_in_svg:
        for source_name, names in sorted(state.csv_not_in_svg.items()):
            lines.append(f"{source_name}: {len(names)} unmatched")
            for entry in names:
                lines.append(f"  - {entry}")
    else:
        lines.append("No CSV-to-SVG mismatches detected.")
    lines.append("")

    lines.append("DUPLICATE SETTLEMENT NAMES WITHIN SAME PROVINCE")
    lines.append("-" * 80)
    if state.duplicate_names_by_province:
        for source_name, dup_map in sorted(state.duplicate_names_by_province.items()):
            lines.append(source_name)
            for key, count in sorted(dup_map.items()):
                lines.append(f"  - {key} => {count}")
    else:
        lines.append("No duplicates found.")
    lines.append("")

    lines.append("OUTPUT SUMMARY")
    lines.append("-" * 80)
    total_features = 0
    for generated in state.generated_files:
        try:
            payload = json.loads(generated.read_text(encoding="utf-8"))
            count = len(payload.get("features", []))
        except Exception:  # noqa: BLE001
            count = -1
        total_features += max(count, 0)
        lines.append(f"{generated.name}: {count} features")
    lines.append(f"Total features generated: {total_features}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Generated report: %s", report_path)


def should_run_step(step: WorkflowStep, cfg: WorkflowConfig) -> bool:
    """Resolve whether a step should run based on config and prompts."""
    if not cfg.interactive:
        return True
    if not step.optional:
        return True
    return prompt_bool(f"Run step '{step.description}'?", default=True, auto_yes=cfg.auto_yes)


def build_steps() -> list[WorkflowStep]:
    """Build ordered workflow steps."""
    return [
        WorkflowStep(
            key="sync_validate",
            description="Download/update and validate gazetteer CSV files",
            handler=step_sync_and_validate_csv,
            optional=True,
        ),
        WorkflowStep(
            key="relabel_svg",
            description="Relabel settlement/POI SVG text elements",
            handler=step_relabel_svg,
            optional=True,
        ),
        WorkflowStep(
            key="wiki_enrich",
            description="Fetch missing wiki metadata and optionally update CSV files",
            handler=step_fetch_wiki_metadata,
            optional=True,
        ),
        WorkflowStep(
            key="process_generate",
            description="Process SVG and generate standardized GeoJSON output",
            handler=step_process_svg_and_generate_outputs,
            optional=True,
        ),
        WorkflowStep(
            key="deploy",
            description="Deploy generated GeoJSON files to target directory",
            handler=step_deploy_outputs,
            optional=True,
        ),
        WorkflowStep(
            key="report",
            description="Write processing report",
            handler=step_generate_report,
            optional=False,
        ),
    ]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Refactored SVG processing workflow")
    parser.add_argument("--non-interactive", action="store_true", help="Run all steps without prompts")
    parser.add_argument("--yes", action="store_true", help="Answer yes to interactive prompts")
    parser.add_argument("--download-csv", action="store_true", help="Force enable CSV download step")
    parser.add_argument("--skip-download-csv", action="store_true", help="Force disable CSV download step")
    parser.add_argument("--relabel-svg", action="store_true", help="Force enable SVG relabel step")
    parser.add_argument("--skip-relabel-svg", action="store_true", help="Force disable SVG relabel step")
    parser.add_argument("--fetch-wiki", action="store_true", help="Force enable wiki enrichment")
    parser.add_argument("--skip-fetch-wiki", action="store_true", help="Force disable wiki enrichment")
    parser.add_argument("--update-csv-with-wiki", action="store_true", help="Write wiki metadata into CSV files")
    parser.add_argument("--skip-update-csv-with-wiki", action="store_true", help="Do not write wiki metadata to CSV files")
    parser.add_argument("--push-geojson", action="store_true", help="Force enable deployment step")
    parser.add_argument("--skip-push-geojson", action="store_true", help="Force disable deployment step")
    parser.add_argument("--deploy-target", type=Path, default=None, help="Target directory for deployment")
    parser.add_argument("--svg-path", type=Path, default=DEFAULT_SVG_PATH, help="Path to source SVG")
    return parser.parse_args(argv)


def bool_override(enable: bool, disable: bool) -> Optional[bool]:
    """Resolve paired enable/disable flags to tri-state bool."""
    if enable and disable:
        raise ValueError("Conflicting enable/disable flags")
    if enable:
        return True
    if disable:
        return False
    return None


def run_workflow(config: WorkflowConfig) -> int:
    """Execute all workflow steps."""
    ensure_dirs()

    state = WorkflowState(config=config)
    steps = build_steps()

    logger.info("Starting refactored SVG workflow")
    logger.info("SVG source: %s", config.svg_path)

    for step in steps:
        if should_run_step(step, config):
            logger.info("STEP: %s", step.description)
            try:
                step.handler(state)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Step failed: %s", step.key)
                state.validation_issues.append(f"Step {step.key} failed: {exc}")
                if config.interactive:
                    if not prompt_bool("Continue after this step failure?", default=False, auto_yes=config.auto_yes):
                        return 1
                else:
                    return 1
        else:
            logger.info("Skipping step: %s", step.description)

    logger.info("Workflow complete")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    config = WorkflowConfig(
        interactive=not args.non_interactive,
        auto_yes=args.yes,
        download_csv=bool_override(args.download_csv, args.skip_download_csv),
        relabel_svg=bool_override(args.relabel_svg, args.skip_relabel_svg),
        fetch_wiki=bool_override(args.fetch_wiki, args.skip_fetch_wiki),
        update_csv_with_wiki=bool_override(args.update_csv_with_wiki, args.skip_update_csv_with_wiki),
        push_geojson=bool_override(args.push_geojson, args.skip_push_geojson),
        deploy_target=args.deploy_target,
        svg_path=args.svg_path,
    )

    return run_workflow(config)


if __name__ == "__main__":
    sys.exit(main())
