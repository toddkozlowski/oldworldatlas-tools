"""
Pathfinding Module
Implements Dijkstra's algorithm for finding shortest paths in the node network.
"""

import heapq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PathResult:
    """Result of a pathfinding operation."""
    start: str
    end: str
    path: List[str]  # List of node IDs
    distance: float
    weighted_distance: float
    exists: bool
    
    def __str__(self):
        if not self.exists:
            return f"No path found from {self.start} to {self.end}"
        
        path_str = " -> ".join(self.path)
        return (f"Path from {self.start} to {self.end}:\n"
                f"  Route: {path_str}\n"
                f"  Distance: {self.distance:.2f}mm\n"
                f"  Weighted Distance: {self.weighted_distance:.2f}mm")


class Pathfinder:
    """Implements Dijkstra's algorithm for pathfinding."""
    
    def __init__(self, graph_builder):
        """
        Initialize the pathfinder.
        
        Args:
            graph_builder: GraphBuilder instance with the node network
        """
        self.graph = graph_builder
    
    def find_path(self, start: str, end: str, use_weights: bool = False) -> PathResult:
        """
        Find the shortest path between two nodes using Dijkstra's algorithm.
        
        Args:
            start: Starting node ID
            end: Ending node ID
            use_weights: If True, use weighted distances; otherwise use actual distances
            
        Returns:
            PathResult object containing the path and distances
        """
        # Validate nodes
        if start not in self.graph.nodes:
            return PathResult(start, end, [], 0, 0, False)
        if end not in self.graph.nodes:
            return PathResult(start, end, [], 0, 0, False)
        if start == end:
            return PathResult(start, end, [start], 0, 0, True)
        
        # Dijkstra's algorithm
        distances = {node_id: float('inf') for node_id in self.graph.nodes}
        weighted_distances = {node_id: float('inf') for node_id in self.graph.nodes}
        previous = {node_id: None for node_id in self.graph.nodes}
        
        distances[start] = 0
        weighted_distances[start] = 0
        
        # Priority queue: (priority, node_id)
        # Priority is weighted_distance if use_weights, otherwise actual distance
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_priority, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == end:
                break
            
            # Check all neighbors
            for neighbor_id, dist, weight in self.graph.get_neighbors(current):
                if neighbor_id in visited:
                    continue
                
                # Calculate new distances
                new_distance = distances[current] + dist
                new_weighted_distance = weighted_distances[current] + weight
                
                # Update if we found a better path
                priority_distance = new_weighted_distance if use_weights else new_distance
                old_priority = weighted_distances[neighbor_id] if use_weights else distances[neighbor_id]
                
                if priority_distance < old_priority:
                    distances[neighbor_id] = new_distance
                    weighted_distances[neighbor_id] = new_weighted_distance
                    previous[neighbor_id] = current
                    heapq.heappush(pq, (priority_distance, neighbor_id))
        
        # Reconstruct path
        if distances[end] == float('inf'):
            return PathResult(start, end, [], 0, 0, False)
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return PathResult(start, end, path, distances[end], weighted_distances[end], True)
    
    def find_k_shortest_paths(self, start: str, end: str, k: int = 2, 
                             use_weights: bool = True) -> List[PathResult]:
        """
        Find the k shortest paths using Yen's algorithm (simplified version).
        
        Args:
            start: Starting node ID
            end: Ending node ID
            k: Number of shortest paths to find
            use_weights: If True, use weighted distances
            
        Returns:
            List of PathResult objects, sorted by distance
        """
        # Find the first shortest path
        first_path = self.find_path(start, end, use_weights)
        
        if not first_path.exists:
            return [first_path]
        
        results = [first_path]
        
        if k == 1:
            return results
        
        # For simplicity, we'll use a basic approach: try removing edges and finding alternatives
        # This is a simplified version - a full Yen's algorithm implementation would be more complex
        candidate_paths = []
        
        # Try removing each edge in the shortest path and finding alternatives
        for i in range(len(first_path.path) - 1):
            # Temporarily "remove" an edge by modifying the graph
            node1 = first_path.path[i]
            node2 = first_path.path[i + 1]
            
            # Find and temporarily remove the edge
            removed_edges = []
            for edge in self.graph.adjacency.get(node1, []):
                if edge.get_other_node(node1) == node2:
                    removed_edges.append(edge)
            
            # Remove edges from adjacency lists
            for edge in removed_edges:
                self.graph.adjacency[edge.node1_id].remove(edge)
                self.graph.adjacency[edge.node2_id].remove(edge)
            
            # Find alternative path
            alt_path = self.find_path(start, end, use_weights)
            
            if alt_path.exists and alt_path.path != first_path.path:
                candidate_paths.append(alt_path)
            
            # Restore edges
            for edge in removed_edges:
                self.graph.adjacency[edge.node1_id].append(edge)
                self.graph.adjacency[edge.node2_id].append(edge)
        
        # Sort candidates by distance and add unique paths
        candidate_paths.sort(key=lambda p: p.weighted_distance if use_weights else p.distance)
        
        seen_paths = {tuple(first_path.path)}
        for path in candidate_paths:
            if len(results) >= k:
                break
            
            path_tuple = tuple(path.path)
            if path_tuple not in seen_paths:
                results.append(path)
                seen_paths.add(path_tuple)
        
        return results
    
    def validate_location(self, location: str) -> bool:
        """
        Check if a location is valid for pathfinding.
        
        Args:
            location: Location name
            
        Returns:
            True if the location exists in the node network
        """
        return location in self.graph.nodes and self.graph.nodes[location].node_type == 'location'
    
    def get_available_locations(self) -> List[str]:
        """Get a list of all valid location names."""
        return [node_id for node_id, node in self.graph.nodes.items() 
                if node.node_type == 'location']
    
    def format_route_report(self, start: str, end: str) -> str:
        """
        Generate a comprehensive route report.
        
        Args:
            start: Starting location
            end: Ending location
            
        Returns:
            Formatted string with route information
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"ROUTE ANALYSIS: {start} to {end}")
        lines.append("=" * 70)
        lines.append("")
        
        # Find path without weights
        path_unweighted = self.find_path(start, end, use_weights=False)
        
        if not path_unweighted.exists:
            lines.append(f"ERROR: No path exists between {start} and {end}")
            lines.append("=" * 70)
            return "\n".join(lines)
        
        # Find path with weights
        path_weighted = self.find_path(start, end, use_weights=True)
        
        # Find second-best path with weights
        all_paths = self.find_k_shortest_paths(start, end, k=2, use_weights=True)
        
        lines.append("1. SHORTEST PATH (ignoring weighting):")
        lines.append(f"   Distance: {path_unweighted.distance:.2f}mm")
        lines.append(f"   Route: {' -> '.join(path_unweighted.path)}")
        lines.append("")
        
        lines.append("2. SHORTEST PATH (considering weighting):")
        lines.append(f"   Actual Distance: {path_weighted.distance:.2f}mm")
        lines.append(f"   Weighted Distance: {path_weighted.weighted_distance:.2f}mm")
        lines.append(f"   Route: {' -> '.join(path_weighted.path)}")
        lines.append("")
        
        if len(all_paths) > 1:
            second_path = all_paths[1]
            lines.append("3. SECOND-BEST PATH (considering weighting):")
            lines.append(f"   Actual Distance: {second_path.distance:.2f}mm")
            lines.append(f"   Weighted Distance: {second_path.weighted_distance:.2f}mm")
            lines.append(f"   Route: {' -> '.join(second_path.path)}")
        else:
            lines.append("3. SECOND-BEST PATH: None (only one route exists)")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
