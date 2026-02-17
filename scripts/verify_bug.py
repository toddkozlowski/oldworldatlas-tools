"""
Verify the bug: Mootland province group transform is not being extracted.
"""

from xml.etree import ElementTree as ET

tree = ET.parse('../oldworldatlas-maps/FULL_MAP_CLEANED.svg')
root = tree.getroot()

NS = {
    'svg': 'http://www.w3.org/2000/svg',
    'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
}

# Find Settlements layer
settlements_layer = None
for g in root.findall(f".//{{{NS['svg']}}}g"):
    if g.get(f"{{{NS['inkscape']}}}label") == "Settlements":
        settlements_layer = g
        break

if settlements_layer:
    # Find Empire faction
    for faction in settlements_layer:
        faction_label = faction.get(f"{{{NS['inkscape']}}}label")
        if faction_label == "Empire":
            print("Empire provinces and their transforms:\n")
            # Check each province
            for province_group in faction:
                province_label = province_group.get(f"{{{NS['inkscape']}}}label")
                if province_label:
                    transform = province_group.get('transform')
                    print(f"Province group: {province_label}")
                    print(f"  Transform on group element: {transform}")
                    
                    # Check if children would get this transform
                    has_child_transforms = False
                    for child in province_group:
                        if child.get('transform'):
                            has_child_transforms = True
                            break
                    
                    if transform or has_child_transforms:
                        print(f"  ^ This transform needs to be passed to _process_settlement_elements!")
                    print()
            break
