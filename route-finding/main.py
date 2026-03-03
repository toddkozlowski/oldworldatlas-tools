#!/usr/bin/env python3
"""
SVG Route Finding System
A tool for parsing SVG paths and finding optimal routes between locations.
"""

import sys
from pathlib import Path
from svg_parser import SVGParser
from graph_builder import GraphBuilder
from pathfinder import Pathfinder
from svg_visualizer import SVGVisualizer


def main():
    """Main entry point for the route finding system."""
    
    print("=" * 70)
    print("SVG ROUTE FINDING SYSTEM")
    print("=" * 70)
    print()
    
    # Find the SVG file
    svg_file = Path(__file__).parent.parent / "example.svg"
    
    if not svg_file.exists():
        print(f"ERROR: SVG file not found at {svg_file}")
        print("Please ensure 'example.svg' exists in the project root directory.")
        sys.exit(1)
    
    try:
        # Step 1: Parse the SVG file
        print(f"Parsing SVG file: {svg_file.name}")
        parser = SVGParser(str(svg_file), location_threshold=10.0)
        paths, locations, location_nodes = parser.parse()
        
        print(f"Found {len(paths)} paths")
        print(f"Found {len(locations)} locations: {', '.join(sorted(locations.keys()))}")
        print(f"Mapped {len(location_nodes)} locations to paths")
        print()
        
        # Step 2: Build the node-branch network
        graph = GraphBuilder(parser, intersection_tolerance=2.0)
        graph.build()
        print()
        
        # Step 3: Create annotated SVG with visualization
        print("Creating annotated SVG with pathfinding nodes...")
        visualizer = SVGVisualizer(str(svg_file), graph)
        annotated_file = visualizer.add_visualization()
        print(f"Annotated SVG saved to: {Path(annotated_file).name}")
        print()
        
        # Step 4: Initialize pathfinder
        pathfinder = Pathfinder(graph)
        
        # Step 5: Display the network map
        print("\nGenerating network map...")
        network_map = graph.generate_ascii_map()
        print(network_map)
        
        # Save the network map to a file
        map_file = Path(__file__).parent / "network_map.txt"
        with open(map_file, 'w') as f:
            f.write(network_map)
        print(f"\nNetwork map saved to: {map_file.name}")
        print()
        
        # Step 5: Interactive route finding
        available_locations = pathfinder.get_available_locations()
        
        if not available_locations:
            print("ERROR: No valid locations found for pathfinding.")
            sys.exit(1)
        
        print("Available locations:", ", ".join(sorted(available_locations)))
        print()
        
        # Interactive loop
        while True:
            print("-" * 70)
            
            # Get start location
            start = input("Enter start location (or 'q' to quit): ").strip().upper()
            if start.lower() == 'q':
                print("Exiting route finder. Goodbye!")
                break
            
            if not pathfinder.validate_location(start):
                print(f"ERROR: '{start}' is not a valid location.")
                print(f"Available locations: {', '.join(sorted(available_locations))}")
                continue
            
            # Get end location
            end = input("Enter end location: ").strip().upper()
            if end.lower() == 'q':
                print("Exiting route finder. Goodbye!")
                break
            
            if not pathfinder.validate_location(end):
                print(f"ERROR: '{end}' is not a valid location.")
                print(f"Available locations: {', '.join(sorted(available_locations))}")
                continue
            
            if start == end:
                print("ERROR: Start and end locations must be different.")
                continue
            
            # Generate and display route report
            print()
            report = pathfinder.format_route_report(start, end)
            print(report)
            print()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
