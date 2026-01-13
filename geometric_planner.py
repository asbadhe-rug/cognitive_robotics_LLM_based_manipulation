import numpy as np
from typing import List, Tuple, Dict

class GeometricPlanner:
    """Converts shape descriptions to actual coordinates"""
    
    def __init__(self, workspace_bounds=None):
        self.bounds = workspace_bounds or {
            'x': (-0.35, 0.35),
            'y': (-0.85, -0.15),
            'z': 1.0
        }
        self.center = (0.0, -0.5, 1.0)
    
    def _line(self, n: int, axis: str = 'x', spacing: float = None, 
            orientation: str = None, length: float = None,
            start: Tuple = None, end: Tuple = None) -> List[Tuple]:
        """
        Flexible line with multiple parameterization options.
        
        Args:
            orientation: 'horizontal' or 'vertical' (overrides axis)
            length: specific length (overrides automatic calculation)
            start/end: explicit start and end points
        """
        
        # Handle orientation parameter
        if orientation == 'horizontal':
            axis = 'x'
        elif orientation == 'vertical':
            axis = 'y'
        
        # Handle explicit start/end
        if start is not None and end is not None:
            coords = []
            for i in range(n):
                t = i / (n - 1) if n > 1 else 0.5
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                coords.append((x, y, self.bounds['z']))
            return coords
        
        # Handle custom length
        if length is not None:
            if spacing is None:
                spacing = length / (n - 1) if n > 1 else 0
            start_pos = -length / 2
        else:
            if spacing is None:
                range_size = self.bounds[axis][1] - self.bounds[axis][0]
                spacing = range_size / (n - 1) if n > 1 else 0
            start_pos = self.bounds[axis][0] if axis == 'x' else self.bounds[axis][1]
        
        coords = []
        for i in range(n):
            if axis == 'x':
                x = start_pos + i * spacing
                y = self.center[1]
            else:
                x = self.center[0]
                y = start_pos - i * spacing
            
            coords.append((x, y, self.bounds['z']))
        return coords

    def _diagonal(self, n: int, direction: str = "topleft-bottomright",
                start: Tuple = None, end: Tuple = None) -> List[Tuple]:
        """
        Diagonal line with configurable direction or explicit endpoints.
        
        Args:
            direction: 'topleft-bottomright', 'topright-bottomleft', etc.
            start/end: explicit (x, y) coordinates
        """
        
        if start is not None and end is not None:
            # Explicit endpoints
            coords = []
            for i in range(n):
                t = i / (n - 1) if n > 1 else 0.5
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                coords.append((x, y, self.bounds['z']))
            return coords
        
        # Direction-based
        if direction == "topleft-bottomright":
            start_x, start_y = self.bounds['x'][0], self.bounds['y'][0]
            end_x, end_y = self.bounds['x'][1], self.bounds['y'][1]
        elif direction == "topright-bottomleft":
            start_x, start_y = self.bounds['x'][1], self.bounds['y'][0]
            end_x, end_y = self.bounds['x'][0], self.bounds['y'][1]
        else:
            # Default
            start_x, start_y = self.bounds['x'][0], self.bounds['y'][0]
            end_x, end_y = self.bounds['x'][1], self.bounds['y'][1]
        
        coords = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            coords.append((x, y, self.bounds['z']))
        
        return coords
    
    def plan_shape(self, shape: str, n_objects: int, **kwargs) -> List[Tuple[float, float, float]]:
        """
        Generate coordinates for a given shape.
        """
        shape = shape.lower()
        
        if shape == "circle":
            return self._circle(n_objects, kwargs.get('radius', 0.35))
        elif shape == "hexagon":
            return self._regular_polygon(n_objects, 6, kwargs.get('radius', 0.35))
        elif shape == "pentagon":
            return self._regular_polygon(n_objects, 5, kwargs.get('radius', 0.35))
        elif shape == "square":
            return self._regular_polygon(n_objects, 4, kwargs.get('radius', 0.35))
        elif shape == "triangle":
            return self._regular_polygon(n_objects, 3, kwargs.get('radius', 0.35))
        elif shape == "line" or shape == "horizontal_line":
            return self._line(
                n_objects,
                axis=kwargs.get('axis', 'x'),
                spacing=kwargs.get('spacing'),
                orientation=kwargs.get('orientation', 'horizontal'),
                length=kwargs.get('length'),
                start=kwargs.get('start'),
                end=kwargs.get('end')
            )
        elif shape == "vertical_line":
            return self._line(
                n_objects,
                axis='y',
                spacing=kwargs.get('spacing'),
                orientation='vertical',
                length=kwargs.get('length'),
                start=kwargs.get('start'),
                end=kwargs.get('end')
            )
        elif shape == "diagonal":
            # ✅ Only pass parameters that _diagonal accepts
            return self._diagonal(
                n_objects,
                direction=kwargs.get('direction', 'topleft-bottomright'),
                start=kwargs.get('start'),
                end=kwargs.get('end')
            )
        elif shape == "grid":
            return self._grid(n_objects, kwargs.get('rows', None), kwargs.get('cols', None))
        elif shape == "star":
            return self._star(n_objects, kwargs.get('radius', 0.35))
        elif shape == "spiral":
            return self._spiral(n_objects, kwargs.get('turns', 2))
        elif shape == "arc":
            return self._arc(n_objects, kwargs.get('radius', 0.35), 
                        kwargs.get('start_angle', 0), kwargs.get('end_angle', 180))
        else:
            raise ValueError(f"Unknown shape: {shape}")
    
    def _circle(self, n: int, radius: float) -> List[Tuple]:
        """Evenly distribute n objects in a circle"""
        coords = []
        for i in range(n):
            angle = 2 * np.pi * i / n - np.pi/2  # Start at top
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            coords.append((x, y, self.bounds['z']))
        return coords
    
    def _regular_polygon(self, n: int, sides: int, radius: float) -> List[Tuple]:
        """
        Distribute objects along a regular polygon's vertices.
        If n > sides, distribute remaining objects along edges.
        """
        coords = []
        
        # Place objects at vertices
        for i in range(min(n, sides)):
            angle = 2 * np.pi * i / sides - np.pi/2
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            coords.append((x, y, self.bounds['z']))
        
        # If more objects than vertices, distribute along edges
        if n > sides:
            remaining = n - sides
            objects_per_edge = remaining // sides
            for edge_idx in range(sides):
                start_idx = edge_idx
                end_idx = (edge_idx + 1) % sides
                for i in range(1, objects_per_edge + 1):
                    t = i / (objects_per_edge + 1)
                    x = coords[start_idx][0] * (1-t) + coords[end_idx][0] * t
                    y = coords[start_idx][1] * (1-t) + coords[end_idx][1] * t
                    coords.append((x, y, self.bounds['z']))
        
        return coords[:n]
    
    def _line(self, n: int, axis: str = 'x', spacing: float = None,
            orientation: str = None, length: float = None,
            start: Tuple = None, end: Tuple = None) -> List[Tuple]:
        """
        Flexible line with multiple parameterization options.
        
        Args:
            axis: 'x' for horizontal, 'y' for vertical
            spacing: distance between objects
            orientation: 'horizontal' or 'vertical' (overrides axis)
            length: specific line length
            start: explicit (x, y) start point
            end: explicit (x, y) end point
        """
        
        # Handle orientation parameter
        if orientation == 'horizontal':
            axis = 'x'
        elif orientation == 'vertical':
            axis = 'y'
        
        # Handle explicit start/end points
        if start is not None and end is not None:
            coords = []
            for i in range(n):
                t = i / (n - 1) if n > 1 else 0.5
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                coords.append((x, y, self.bounds['z']))
            return coords
        
        # Calculate spacing and start position
        if length is not None:
            if spacing is None:
                spacing = length / (n - 1) if n > 1 else 0
            start_pos = self.center[0 if axis == 'x' else 1] - length / 2
        else:
            if spacing is None:
                range_size = self.bounds[axis][1] - self.bounds[axis][0]
                spacing = range_size / (n - 1) if n > 1 else 0
            start_pos = self.bounds[axis][0] if axis == 'x' else self.bounds[axis][1]
        
        coords = []
        for i in range(n):
            if axis == 'x':
                x = start_pos + i * spacing
                y = self.center[1]
            else:
                x = self.center[0]
                y = start_pos - i * spacing  # y is negative
            
            coords.append((x, y, self.bounds['z']))
        return coords
    
    def _grid(self, n: int, rows: int = None, cols: int = None) -> List[Tuple]:
        """Arrange in grid pattern"""
        if rows is None and cols is None:
            # Auto-determine best grid
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        elif rows is None:
            rows = int(np.ceil(n / cols))
        elif cols is None:
            cols = int(np.ceil(n / rows))
        
        x_spacing = (self.bounds['x'][1] - self.bounds['x'][0]) / (cols + 1)
        y_spacing = (self.bounds['y'][1] - self.bounds['y'][0]) / (rows + 1)
        
        coords = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x = self.bounds['x'][0] + (col + 1) * x_spacing
            y = self.bounds['y'][0] + (row + 1) * y_spacing
            coords.append((x, y, self.bounds['z']))
        
        return coords
    
    def _diagonal(self, n: int, direction: str = "topleft-bottomright",
              start: Tuple = None, end: Tuple = None) -> List[Tuple]:
        """
        Diagonal line with configurable direction or explicit endpoints.
        
        Args:
            direction: 'topleft-bottomright', 'topright-bottomleft'
            start: explicit (x, y) start point
            end: explicit (x, y) end point
        """
        
        if start is not None and end is not None:
            # Explicit endpoints provided
            coords = []
            for i in range(n):
                t = i / (n - 1) if n > 1 else 0.5
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                coords.append((x, y, self.bounds['z']))
            return coords
        
        # Direction-based (if no explicit start/end)
        if direction == "topleft-bottomright":
            start_x, start_y = self.bounds['x'][0], self.bounds['y'][0]
            end_x, end_y = self.bounds['x'][1], self.bounds['y'][1]
        elif direction == "topright-bottomleft":
            start_x, start_y = self.bounds['x'][1], self.bounds['y'][0]
            end_x, end_y = self.bounds['x'][0], self.bounds['y'][1]
        else:
            # Default to topleft-bottomright
            start_x, start_y = self.bounds['x'][0], self.bounds['y'][0]
            end_x, end_y = self.bounds['x'][1], self.bounds['y'][1]
        
        coords = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            coords.append((x, y, self.bounds['z']))
        
        return coords
    
    def _star(self, n: int, radius: float) -> List[Tuple]:
        """5-pointed star pattern"""
        if n < 5:
            return self._regular_polygon(n, 5, radius)
        
        coords = []
        # Outer points
        for i in range(5):
            angle = 2 * np.pi * i / 5 - np.pi/2
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            coords.append((x, y, self.bounds['z']))
        
        # Inner points
        inner_radius = radius * 0.4
        for i in range(min(n - 5, 5)):
            angle = 2 * np.pi * i / 5 - np.pi/2 + np.pi/5
            x = self.center[0] + inner_radius * np.cos(angle)
            y = self.center[1] + inner_radius * np.sin(angle)
            coords.append((x, y, self.bounds['z']))
        
        # Center if needed
        if n > 10:
            coords.append(self.center)
        
        return coords[:n]
    
    def _spiral(self, n: int, turns: float = 2) -> List[Tuple]:
        """Archimedean spiral"""
        coords = []
        max_radius = 0.35
        
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            angle = t * turns * 2 * np.pi
            radius = max_radius * t
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            coords.append((x, y, self.bounds['z']))
        
        return coords
    
    def _arc(self, n: int, radius: float, start_angle: float, end_angle: float) -> List[Tuple]:
        """Arc from start_angle to end_angle (in degrees)"""
        start_rad = np.radians(start_angle) - np.pi/2
        end_rad = np.radians(end_angle) - np.pi/2
        
        coords = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            angle = start_rad + t * (end_rad - start_rad)
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            coords.append((x, y, self.bounds['z']))
        
        return coords