"""
Graph Builder Module
Creates a node-branch network from SVG paths, finding intersections and building a graph.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import svgpathtools as svg


@dataclass
class Node:
    """Represents a node in the pathfinding graph."""
    id: str
    x: float
    y: float
    node_type: str  # 'location', 'intersection', or 'endpoint'
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id


@dataclass
class Edge:
    """Represents an edge (path segment) between two nodes."""
    node1_id: str
    node2_id: str
    path_id: str
    t_start: float
    t_end: float
    length: float
    weight: float  # Weighted length considering path preferences
    
    def get_other_node(self, node_id: str) -> str:
        """Get the ID of the node on the other end of this edge."""
        return self.node2_id if node_id == self.node1_id else self.node1_id


class GraphBuilder:
    """Builds a node-branch network from parsed SVG data."""
    
    def __init__(self, svg_parser, intersection_tolerance: float = 2.0, 
                 base_weight: float = 1.0, base_thickness: float = 2.05):
        """
        Initialize the graph builder.
        
        Args:
            svg_parser: SVGParser instance with parsed data
            intersection_tolerance: Distance tolerance for detecting intersections (mm)
            base_weight: Weight factor for base thickness paths
            base_thickness: The reference stroke width for weight calculations
        """
        self.parser = svg_parser
        self.intersection_tolerance = intersection_tolerance
        self.base_weight = base_weight
        self.base_thickness = base_thickness
        
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[Edge]] = {}  # node_id -> list of edges
        
    def build(self):
        """Build the complete graph with nodes and edges."""
        print("\nBuilding node-branch network...")
        
        # Step 1: Find all intersections
        intersections = self._find_intersections()
        print(f"Found {len(intersections)} intersection points")
        
        # Step 2: Create nodes for intersections
        self._create_intersection_nodes(intersections)
        
        # Step 3: Create nodes for locations
        self._create_location_nodes()
        
        # Step 4: Create nodes for path endpoints
        self._create_endpoint_nodes()
        
        # Step 5: Create edges between nodes
        self._create_edges()
        
        print(f"Created {len(self.nodes)} nodes and {len(self.edges)} edges")
        
        # Build adjacency list
        self._build_adjacency()
        
    def _find_intersections(self) -> List[Tuple[str, float, str, float, float, float]]:
        """
        Find all TRUE intersections between paths (actual crossings).
        Multiple intersections between the same pair of paths are detected.
        
        Returns:
            List of (path1_id, t1, path2_id, t2, x, y) tuples
        """
        intersections = []
        path_ids = list(self.parser.paths.keys())
        
        # Intersections must be at least this far apart to be considered separate
        min_separation = 10.0  # mm
        
        for i, path1_id in enumerate(path_ids):
            path1_obj, _ = self.parser.paths[path1_id]
            
            for j, path2_id in enumerate(path_ids):
                if i == j:
                    # Skip self-path - we'll handle self-intersections separately
                    continue
                if j < i:
                    # Skip pairs already checked
                    continue
                    
                path2_obj, _ = self.parser.paths[path2_id]
                
                # Find all candidate intersection points
                candidates = []
                
                # Sample both paths densely
                num_samples = 300
                t1_values = np.linspace(0, 1, num_samples)
                t2_values = np.linspace(0, 1, num_samples)
                
                # Check for close approaches between the paths
                for t1 in t1_values:
                    p1 = path1_obj.point(t1)
                    
                    for t2 in t2_values:
                        p2 = path2_obj.point(t2)
                        dist = abs(p2 - p1)
                        
                        # If close enough, it's a candidate intersection
                        if dist < self.intersection_tolerance:
                            x, y = p1.real, p1.imag
                            candidates.append((path1_id, t1, path2_id, t2, x, y, dist))
                
                # Cluster candidates: group those within min_separation distance
                # and keep the best (closest) one from each cluster
                while candidates:
                    # Take the best candidate (smallest distance)
                    candidates.sort(key=lambda c: c[6])
                    best = candidates.pop(0)
                    
                    # Add to intersections if not already near an existing intersection
                    if not self._is_duplicate_intersection(intersections, best[4], best[5], min_separation):
                        intersections.append(best[:6])  # Exclude distance from result
                    
                    # Remove all candidates within min_separation of this one
                    candidates = [
                        c for c in candidates 
                        if np.sqrt((c[4] - best[4])**2 + (c[5] - best[5])**2) >= min_separation
                    ]
        
        return intersections
    
    def _is_duplicate_intersection(self, intersections: List[Tuple], 
                                   x: float, y: float, tolerance: float = None) -> bool:
        """Check if an intersection point is already recorded."""
        if tolerance is None:
            tolerance = self.intersection_tolerance
        for _, _, _, _, ix, iy in intersections:
            if np.sqrt((x - ix)**2 + (y - iy)**2) < tolerance:
                return True
        return False
    
    def _create_intersection_nodes(self, intersections: List[Tuple]):
        """Create nodes for intersection points."""
        for idx, (path1_id, t1, path2_id, t2, x, y) in enumerate(intersections):
            node_id = f"INT{idx+1}"
            self.nodes[node_id] = Node(node_id, x, y, 'intersection')
    
    def _create_location_nodes(self):
        """Create nodes for valid location markers."""
        for loc_name, (path_id, t) in self.parser.location_nodes.items():
            x, y = self.parser.locations[loc_name]
            # Check if already have a node very close to this
            existing = self._find_node_at(x, y)
            if existing:
                # Rename existing node to location name
                self.nodes[loc_name] = self.nodes.pop(existing.id)
                self.nodes[loc_name].id = loc_name
                self.nodes[loc_name].node_type = 'location'
            else:
                self.nodes[loc_name] = Node(loc_name, x, y, 'location')
    
    def _create_endpoint_nodes(self):
        """Create nodes for path endpoints that don't overlap with existing nodes."""
        for path_id, (path_obj, _) in self.parser.paths.items():
            # Start point
            start_point = path_obj.point(0)
            sx, sy = start_point.real, start_point.imag
            if not self._find_node_at(sx, sy):
                node_id = f"EP_{path_id}_0"
                self.nodes[node_id] = Node(node_id, sx, sy, 'endpoint')
            
            # End point
            end_point = path_obj.point(1)
            ex, ey = end_point.real, end_point.imag
            if not self._find_node_at(ex, ey):
                node_id = f"EP_{path_id}_1"
                self.nodes[node_id] = Node(node_id, ex, ey, 'endpoint')
    
    def _find_node_at(self, x: float, y: float) -> Optional[Node]:
        """Find a node at or very near the given coordinates."""
        for node in self.nodes.values():
            dist = np.sqrt((node.x - x)**2 + (node.y - y)**2)
            if dist < self.intersection_tolerance:
                return node
        return None
    
    def _create_edges(self):
        """Create edges between nodes along each path."""
        for path_id, (path_obj, stroke_width) in self.parser.paths.items():
            # Find all nodes on this path
            nodes_on_path = []
            
            # Check each node
            for node in self.nodes.values():
                # Sample the path to find if this node is on it
                t_values = np.linspace(0, 1, 1000)
                for t in t_values:
                    point = path_obj.point(t)
                    px, py = point.real, point.imag
                    dist = np.sqrt((node.x - px)**2 + (node.y - py)**2)
                    
                    if dist < self.intersection_tolerance:
                        nodes_on_path.append((node, t))
                        break
            
            # Sort nodes by their t parameter
            nodes_on_path.sort(key=lambda x: x[1])
            
            # Create edges between consecutive nodes
            for i in range(len(nodes_on_path) - 1):
                node1, t1 = nodes_on_path[i]
                node2, t2 = nodes_on_path[i + 1]
                
                # Calculate edge length and weight
                length = self.parser.get_path_length(path_id, t1, t2)
                weight = self._calculate_weight(length, stroke_width)
                
                edge = Edge(node1.id, node2.id, path_id, t1, t2, length, weight)
                self.edges.append(edge)
    
    def _calculate_weight(self, length: float, stroke_width: float) -> float:
        """
        Calculate weighted length based on stroke width.
        Thicker lines have higher preferential weighting (shorter weighted distance).
        """
        # Weight inversely proportional to thickness (thicker = preferred = lower weight)
        thickness_factor = self.base_thickness / stroke_width
        return length * thickness_factor * self.base_weight
    
    def _build_adjacency(self):
        """Build adjacency list for efficient pathfinding."""
        self.adjacency = {node_id: [] for node_id in self.nodes}
        
        for edge in self.edges:
            self.adjacency[edge.node1_id].append(edge)
            self.adjacency[edge.node2_id].append(edge)
    
    def get_neighbors(self, node_id: str) -> List[Tuple[str, float, float]]:
        """
        Get neighboring nodes with distances.
        
        Returns:
            List of (neighbor_id, unweighted_distance, weighted_distance) tuples
        """
        neighbors = []
        
        if node_id not in self.adjacency:
            return neighbors
        
        for edge in self.adjacency[node_id]:
            neighbor_id = edge.get_other_node(node_id)
            neighbors.append((neighbor_id, edge.length, edge.weight))
        
        return neighbors
    
    def generate_ascii_map(self) -> str:
        """Generate an ASCII representation of the node network."""
        lines = []
        lines.append("=" * 60)
        lines.append("NODE-BRANCH NETWORK MAP")
        lines.append("=" * 60)
        lines.append("")
        
        # List all nodes
        lines.append("NODES:")
        for node_id, node in sorted(self.nodes.items()):
            lines.append(f"  {node_id:12} ({node.node_type:12}) at ({node.x:6.1f}, {node.y:6.1f})")
        
        lines.append("")
        lines.append("EDGES (showing distance and weighted distance):")
        
        # Group edges by node
        displayed_edges = set()
        for node_id in sorted(self.nodes.keys()):
            connections = []
            
            for edge in self.adjacency.get(node_id, []):
                edge_key = tuple(sorted([edge.node1_id, edge.node2_id]))
                if edge_key not in displayed_edges:
                    displayed_edges.add(edge_key)
                    other_node = edge.get_other_node(node_id)
                    connections.append(f"{other_node} [{edge.length:.2f}mm, weighted:{edge.weight:.2f}mm]")
            
            if connections:
                lines.append(f"  {node_id}: {', '.join(connections)}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
