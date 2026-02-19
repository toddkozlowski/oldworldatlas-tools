"""
Renames every settlement text element in the Settlements layer of the SVG by
setting (or updating) its inkscape:label attribute to match its text content.

Only inkscape:label attributes on settlement text elements are modified.
All other content is preserved exactly as-is.

The updated SVG is saved back to the original file path.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Namespaces – register before parsing to preserve prefixes on write
from xml.etree import ElementTree as ET

ET.register_namespace('', 'http://www.w3.org/2000/svg')
ET.register_namespace('svg', 'http://www.w3.org/2000/svg')
ET.register_namespace('inkscape', 'http://www.inkscape.org/namespaces/inkscape')
ET.register_namespace('sodipodi', 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')

SVG_PATH = (
    Path(__file__).parent.parent.parent / "oldworldatlas-maps" / "OLD_WORLD_ATLAS.svg"
)

NS = {
    'svg': 'http://www.w3.org/2000/svg',
    'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
    'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd',
}

INKSCAPE_LABEL = f"{{{NS['inkscape']}}}label"
SVG_TEXT = f"{{{NS['svg']}}}text"
SVG_TSPAN = f"{{{NS['svg']}}}tspan"
SVG_G = f"{{{NS['svg']}}}g"
INKSCAPE_GROUP_MODE = f"{{{NS['inkscape']}}}groupmode"


def get_text_content(text_elem: ET.Element) -> Optional[str]:
    """Extract the text content of a <text> element from its <tspan> children."""
    parts = []
    for tspan in text_elem.findall(SVG_TSPAN):
        if tspan.text:
            parts.append(tspan.text.strip())
    # Fallback: direct text on the element
    if not parts and text_elem.text:
        parts.append(text_elem.text.strip())
    return " ".join(parts) if parts else None


def relabel_text_elements(parent: ET.Element, count: dict, current_layer: str = '') -> None:
    """Recursively walk settlement group children and relabel every text element."""
    for elem in parent:
        if elem.tag == SVG_G:
            # Use this group's label as the layer name if it has one, else inherit
            layer_label = elem.get(INKSCAPE_LABEL) or current_layer
            relabel_text_elements(elem, count, layer_label)
        elif elem.tag == SVG_TEXT:
            label = get_text_content(elem)
            if label:
                existing = elem.get(INKSCAPE_LABEL)
                if existing != label:
                    elem.set(INKSCAPE_LABEL, label)
                    count['updated'] += 1
                else:
                    count['already_correct'] += 1
            else:
                count['no_text'] += 1
                count['no_text_by_layer'][current_layer or '<root>'] += 1


def find_layer_by_label(root: ET.Element, label: str) -> Optional[ET.Element]:
    """Find the first <g inkscape:label="label"> layer element in the tree."""
    for g in root.iter(SVG_G):
        if g.get(INKSCAPE_LABEL) == label:
            return g
    return None


def main() -> None:
    """Entry point: parse SVG, relabel settlement text elements, save."""
    if not SVG_PATH.exists():
        logger.error(f"SVG not found at: {SVG_PATH}")
        return

    logger.info(f"Parsing SVG: {SVG_PATH}")

    # Preserve the original XML declaration and DOCTYPE (ET strips them)
    raw_text = SVG_PATH.read_text(encoding='utf-8')
    xml_declaration = ''
    if raw_text.startswith('<?xml'):
        end = raw_text.index('?>') + 2
        xml_declaration = raw_text[:end]

    tree = ET.parse(str(SVG_PATH))
    root = tree.getroot()

    # Locate the top-level Settlements layer
    settlements_layer = find_layer_by_label(root, 'Settlements')
    if settlements_layer is None:
        logger.error("Could not find Settlements layer – aborting.")
        return

    factions = [
        child for child in settlements_layer
        if child.tag == SVG_G
    ]
    logger.info(
        f"Found Settlements layer with {len(factions)} faction sub-layer(s): "
        + ", ".join(f.get(INKSCAPE_LABEL, '<unnamed>') for f in factions)
    )

    count = {'updated': 0, 'already_correct': 0, 'no_text': 0, 'no_text_by_layer': defaultdict(int)}
    relabel_text_elements(settlements_layer, count)

    logger.info(
        f"Labels updated: {count['updated']:,}  |  "
        f"Already correct: {count['already_correct']:,}  |  "
        f"No text content: {count['no_text']}"
    )
    if count['no_text_by_layer']:
        logger.warning("Text elements with no content, by layer:")
        for layer, n in sorted(count['no_text_by_layer'].items()):
            logger.warning(f"  {layer}: {n}")

    if count['updated'] == 0:
        logger.info("No changes required – file not written.")
        return

    # Write back, prepending the original XML declaration if present
    logger.info(f"Saving updated SVG to: {SVG_PATH}")
    xml_bytes = ET.tostring(root, encoding='unicode', xml_declaration=False)
    output = (xml_declaration + '\n' + xml_bytes) if xml_declaration else xml_bytes
    SVG_PATH.write_text(output, encoding='utf-8')
    logger.info("Done.")


if __name__ == '__main__':
    main()
