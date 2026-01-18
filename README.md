# oldworldatlas-tools
Scripting tools for the Old World Atlas

## Project Structure

```
oldworldatlas-tools/
├── input/              # Input data
│   └── gazetteers/     # Population data and settlement information
│       ├── The-Empire/ # Empire province CSV files
│       └── Westerland/ # Westerland CSV files
├── output/             # Generated GeoJSON files
│   ├── empire_settlements.geojson
│   ├── westerland_settlements.geojson
│   ├── points_of_interest.geojson
│   ├── empire_roads.geojson
│   ├── province_labels.geojson
│   └── water_labels.geojson
├── logs/               # Processing reports and logs
│   ├── processing_report.txt
│   ├── invalid_settlement_elements.log
│   └── duplicate_settlements.log
└── scripts/            # Processing scripts
    └── process_map_svg.py
```

## Usage

Run the main processing script to extract data from the SVG map:

```bash
python scripts/process_map_svg.py
```

This will:
- Extract settlements from Empire and Westerland regions
- Extract points of interest (forts, temples, taverns, etc.)
- Extract road networks
- Extract political/province labels
- Extract water body labels
- Generate GeoJSON files in the `output/` directory
- Create processing reports and logs in the `logs/` directory
