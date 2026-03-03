# SVG Route Finding System

A Python-based pathfinding system that parses Inkscape SVG files containing hand-drawn paths and location markers, builds a node-branch network, and computes optimal routes between locations using Dijkstra's algorithm.

## Features

- **SVG Parsing**: Extracts paths and location markers from Inkscape SVG files
- **Smart Location Mapping**: Automatically maps location markers to the nearest point on any path
- **Intersection Detection**: Identifies path crossings and self-intersections to create a network graph
- **Weighted Pathfinding**: Supports path weighting based on stroke width (thicker lines = preferred routes)
- **Dijkstra's Algorithm**: Finds optimal routes considering actual path curves and distances
- **Multiple Route Options**: Can find the k-shortest paths between locations
- **ASCII Network Map**: Generates a visual representation of the node network
- **Interactive CLI**: User-friendly command-line interface for route queries

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package and environment management.

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone or navigate to the project directory:
```bash
cd route-finding
```

3. Sync dependencies:
```bash
uv sync
```

## Usage

Run the pathfinding system:

```bash
uv run python main.py
```

The system will:
1. Parse the `example.svg` file
2. Build a node-branch network from the paths and locations
3. Display the network map
4. Enter an interactive mode where you can query routes

### Interactive Mode

```
Enter start location (or 'q' to quit): A
Enter end location: E

======================================================================
ROUTE ANALYSIS: A to E
======================================================================

1. SHORTEST PATH (ignoring weighting):
   Distance: 338.47mm
   Route: A -> INT1 -> INT2 -> C -> INT4 -> E

2. SHORTEST PATH (considering weighting):
   Actual Distance: 338.47mm
   Weighted Distance: 338.47mm
   Route: A -> INT1 -> INT2 -> C -> INT4 -> E
   
3. SECOND-BEST PATH: None (only one route exists)

======================================================================
```

## SVG File Requirements

The system expects an Inkscape SVG file with two layers:

### 1. Paths Layer
- Layer name: "paths" (case-insensitive)
- Contains bezier paths representing routes
- Paths can have different stroke widths for weighting
- All paths must have a `stroke-width` style property

### 2. Locations Layer
- Layer name: "Locations" (case-insensitive)
- Contains text elements marking locations
- Each text element should have a `<tspan>` with the location name (e.g., A, B, C)
- Location names are automatically converted to uppercase

## Configuration

Key parameters can be adjusted in the code:

### Location Threshold (default: 10mm)
Maximum distance from a location marker to the nearest path node:
```python
parser = SVGParser(str(svg_file), location_threshold=10.0)
```

### Intersection Tolerance (default: 2mm)
Distance tolerance for detecting path intersections:
```python
graph = GraphBuilder(parser, intersection_tolerance=2.0)
```

### Path Weighting
- Base weight: 1.0
- Base thickness: 2.05mm
- Thicker paths have lower weighted distances (preferred routes)
- Weight is calculated as: `weight = length * (base_thickness / stroke_width) * base_weight`

## Project Structure

```
route-finding/
├── main.py              # Main entry point and CLI interface
├── svg_parser.py        # SVG file parser
├── graph_builder.py     # Node-branch network builder
├── pathfinder.py        # Dijkstra's pathfinding algorithm
├── pyproject.toml       # Project dependencies
├── network_map.txt      # Generated network visualization (after running)
└── example.svg          # Sample SVG file (in parent directory)
```

## How It Works

1. **Parsing**: The `SVGParser` extracts paths and location markers from the SVG file
   - Paths are parsed using `svgpathtools` library
   - Location coordinates are extracted from text elements
   - Each location is mapped to the nearest point on any path

2. **Graph Building**: The `GraphBuilder` creates a node network
   - Detects intersections between paths (including self-intersections)
   - Creates nodes for: locations, intersections, and path endpoints
   - Calculates segment lengths along curved paths
   - Applies weighting based on stroke width

3. **Pathfinding**: The `Pathfinder` uses Dijkstra's algorithm
   - Finds shortest paths with and without weighting
   - Can find k-shortest paths for alternative routes
   - Validates locations and provides detailed route reports

## Dependencies

- **svgpathtools** (>=1.6.1): SVG path parsing and manipulation
- **numpy** (>=1.24.0): Numerical computations
- **lxml** (>=4.9.0): XML parsing for SVG structure

## Limitations

- Locations not within the threshold distance of a path are excluded from pathfinding
- Path intersection detection uses sampling (may miss very close intersections)
- Alternative path finding uses a simplified algorithm (not full Yen's algorithm)
- All coordinates use SVG/Inkscape units (typically mm)

## Example Output

The system generates a `network_map.txt` file showing all nodes and connections:

```
============================================================
NODE-BRANCH NETWORK MAP
============================================================

NODES:
  A            (location    ) at (  89.2,   33.0)
  C            (location    ) at ( 125.7,  149.3)
  ...

EDGES (showing distance and weighted distance):
  A: INT1 [35.60mm, weighted:35.60mm]
  C: INT2 [198.68mm, weighted:198.68mm], INT4 [2.04mm, weighted:2.04mm]
  ...
============================================================
```

## License

This project is provided as-is for educational and practical use.
