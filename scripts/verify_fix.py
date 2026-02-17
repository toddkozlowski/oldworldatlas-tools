"""
Check coordinates for settlements from multiple provinces to verify the fix doesn't affect others.
"""

import json

with open('output/empire_settlements.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check a few settlements from different provinces
test_settlements = ['Altdorf', 'Middenheim', 'Eicheschatten', 'Kurbisdorf', 'Wachdorf']

for feature in data['features']:
    if feature['properties']['name'] in test_settlements:
        print(f"{feature['properties']['name']} ({feature['properties']['province']}):")
        print(f"  Inkscape coords: {feature['properties']['inkscape_coordinates']}")
        print(f"  Map coords: {feature['geometry']['coordinates']}")
        print()
