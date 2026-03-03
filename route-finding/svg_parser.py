"""
SVG Parser Module
Parses Inkscape SVG files to extract paths and locations for pathfinding.
"""

from lxml import etree
import svgpathtools as svg
import numpy as np
from typing import Dict, List, Tuple, Optional


class SVGParser:
    """Parser for Inkscape SVG files containing paths and location markers."""
    
    # SVG namespaces
    NS = {
        'svg': 'http://www.w3.org/2000/svg',
        'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
        'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
    }
    
    def __init__(self, svg_file: str, location_threshold: float = 10.0):
        """
        Initialize the SVG parser.
        
        Args:
            svg_file: Path to the SVG file
            location_threshold: Maximum distance (in mm) from a location to a path node
        """
        self.svg_file = svg_file
        self.location_threshold = location_threshold
        self.tree = etree.parse(svg_file)
        self.root = self.tree.getroot()
        
        self.paths = {}  # Dict of path_id -> (path_object, stroke_width)
        self.locations = {}  # Dict of location_name -> (x, y)
        self.location_nodes = {}  # Dict of location_name -> (path_id, t_param)
        
    def parse(self) -> Tuple[Dict, Dict, Dict]:
        """
        Parse the SVG file and extract paths and locations.
        
        Returns:
            Tuple of (paths, locations, location_nodes)
        """
        self._extract_paths()
        self._extract_locations()
        self._map_locations_to_paths()
        
        return self.paths, self.locations, self.location_nodes
    
    def _extract_paths(self):
        """Extract all paths from the 'paths' layer with their stroke widths."""
        # Find the paths layer
        for layer in self.root.findall('.//svg:g[@inkscape:groupmode="layer"]', self.NS):
            layer_name = layer.get('{http://www.inkscape.org/namespaces/inkscape}label', '')
            
            if layer_name.lower() == 'paths':
                # Extract all paths in this layer
                for path_elem in layer.findall('.//svg:path', self.NS):
                    path_id = path_elem.get('id')
                    d_attr = path_elem.get('d')
                    style = path_elem.get('style', '')
                    
                    # Parse stroke width from style attribute
                    stroke_width = self._parse_stroke_width(style)
                    
                    # Parse the path using svgpathtools
                    try:
                        path_obj = svg.parse_path(d_attr)
                        self.paths[path_id] = (path_obj, stroke_width)
                    except Exception as e:
                        print(f"Warning: Could not parse path {path_id}: {e}")
    
    def _parse_stroke_width(self, style: str) -> float:
        """Extract stroke width from style string."""
        for prop in style.split(';'):
            if 'stroke-width:' in prop:
                width_str = prop.split(':')[1].strip()
                # Remove units if present
                width_str = width_str.replace('px', '').replace('mm', '').replace('pt', '')
                try:
                    return float(width_str)
                except ValueError:
                    pass
        return 2.05  # Default width from the example
    
    def _extract_locations(self):
        """Extract location markers from the 'Locations' layer."""
        for layer in self.root.findall('.//svg:g[@inkscape:groupmode="layer"]', self.NS):
            layer_name = layer.get('{http://www.inkscape.org/namespaces/inkscape}label', '')
            
            if layer_name.lower() == 'locations':
                # Extract all text elements
                for text_elem in layer.findall('.//svg:text', self.NS):
                    x = float(text_elem.get('x', 0))
                    y = float(text_elem.get('y', 0))
                    
                    # Get the text content from tspan
                    tspan = text_elem.find('.//svg:tspan', self.NS)
                    if tspan is not None and tspan.text:
                        location_name = tspan.text.strip()
                        self.locations[location_name] = (x, y)
    
    def _map_locations_to_paths(self):
        """Map each location to the nearest point on any path."""
        for loc_name, (loc_x, loc_y) in self.locations.items():
            best_distance = float('inf')
            best_path_id = None
            best_t = None
            
            # Check all paths
            for path_id, (path_obj, _) in self.paths.items():
                # Sample the path at many points to find the closest
                t_values = np.linspace(0, 1, 1000)
                
                for t in t_values:
                    point = path_obj.point(t)
                    px, py = point.real, point.imag
                    
                    distance = np.sqrt((px - loc_x)**2 + (py - loc_y)**2)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_path_id = path_id
                        best_t = t
            
            # Check if within threshold
            if best_distance <= self.location_threshold:
                self.location_nodes[loc_name] = (best_path_id, best_t)
                # Update location position to the actual path point
                path_obj, _ = self.paths[best_path_id]
                point = path_obj.point(best_t)
                self.locations[loc_name] = (point.real, point.imag)
            else:
                print(f"Warning: Location '{loc_name}' is {best_distance:.2f}mm from nearest path (threshold: {self.location_threshold}mm)")
                print(f"  Location will not be available for pathfinding.")
    
    def get_path_length(self, path_id: str, t_start: float = 0.0, t_end: float = 1.0) -> float:
        """
        Calculate the length of a path segment.
        
        Args:
            path_id: ID of the path
            t_start: Start parameter (0 to 1)
            t_end: End parameter (0 to 1)
            
        Returns:
            Length of the path segment in mm
        """
        if path_id not in self.paths:
            return 0.0
        
        path_obj, _ = self.paths[path_id]
        
        # Get the cropped path
        if t_start == 0.0 and t_end == 1.0:
            return path_obj.length()
        
        # Sample points to calculate length
        num_samples = 100
        t_values = np.linspace(t_start, t_end, num_samples)
        total_length = 0.0
        
        for i in range(len(t_values) - 1):
            p1 = path_obj.point(t_values[i])
            p2 = path_obj.point(t_values[i + 1])
            total_length += abs(p2 - p1)
        
        return total_length
    
    def get_point_at_t(self, path_id: str, t: float) -> Tuple[float, float]:
        """Get the (x, y) coordinates at parameter t on a path."""
        if path_id not in self.paths:
            return (0, 0)
        
        path_obj, _ = self.paths[path_id]
        point = path_obj.point(t)
        return (point.real, point.imag)
