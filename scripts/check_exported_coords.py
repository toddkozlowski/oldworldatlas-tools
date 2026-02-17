"""
Check the exported coordinates for Eicheschatten and Kurbisdorf.
"""

import json

with open('output/empire_settlements.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

for feature in data['features']:
    if feature['properties']['name'] in ['Eicheschatten', 'Kurbisdorf']:
        print(f"{feature['properties']['name']}:")
        print(f"  Inkscape coords: {feature['properties']['inkscape_coordinates']}")
        print(f"  Map coords: {feature['geometry']['coordinates']}")
        print()
