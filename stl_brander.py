"""
STL Text Carving - Pure Python Implementation
Carves text into STL files (engraved/sunken effect)

Installation:
    uv pip install trimesh numpy pillow scipy scikit-image
"""

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Literal, Optional

class STLBrander:
    """Carve text branding into STL files"""
    
    def __init__(self, font_path: Optional[str] = None):
        """
        Initialize brander
        
        Args:
            font_path: Path to TTF font file (optional)
        """
        self.font_path = font_path or self._get_default_font()
    
    def _get_default_font(self) -> str:
        """Get default system font path"""
        possible_fonts = [
            "/Users/enrique/Library/Fonts/xkcd.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
        
        for font in possible_fonts:
            if Path(font).exists():
                return font
        
        return possible_fonts[0]
    
    def create_text_mesh(
        self,
        text: str,
        width_mm: float,
        height_mm: float,
        depth_mm: float = 2.0,
        font_height_mm: Optional[float] = None
    ) -> trimesh.Trimesh:
        """
        Create 3D text mesh for carving
        
        Args:
            text: Text to create
            width_mm: Width in mm
            height_mm: Height in mm  
            depth_mm: Extrusion depth in mm
            font_height_mm: Desired font height in mm (optional, will auto-fit if not provided)
        
        Returns:
            Text mesh
        """
        # Create high-res image with text
        img_width = 1024
        img_height = 512
        img = Image.new('L', (img_width, img_height), color=0)
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font = ImageFont.truetype(self.font_path, 200)  # Start with large font
        except:
            try:
                font = ImageFont.truetype("/Users/enrique/Library/Fonts/xkcd.ttf", 200)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 200)
                except:
                    font = ImageFont.load_default()
                    print("Warning: Using default font")
        
        # Get text size and center it
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # If font_height_mm is specified, adjust font size to match requested height
        if font_height_mm is not None:
            # Calculate required font size to achieve desired height in mm
            pixels_per_mm = img_height / height_mm
            required_height_px = font_height_mm * pixels_per_mm
            
            if text_height > 0:
                font_scale = required_height_px / text_height
                new_font_size = max(10, int(200 * font_scale))  # Ensure minimum font size
                
                # Reload font with correct size
                try:
                    font = ImageFont.truetype(self.font_path, new_font_size)
                except:
                    try:
                        font = ImageFont.truetype("/Users/enrique/Library/Fonts/xkcd.ttf", new_font_size)
                    except:
                        font = ImageFont.load_default()
                
                print(f"Debug: Using font size {new_font_size} for height {font_height_mm}mm")
                
                # Recalculate text dimensions with new font
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
        
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill=255, font=font)
        
        # Convert to numpy array
        img_array = np.array(img).astype(float) / 255.0
        
        # Create 3D mesh from text image
        mesh = self._image_to_3d_mesh(
            img_array,
            width_mm,
            height_mm,
            depth_mm
        )
        
        return mesh
    
    def _image_to_3d_mesh(
        self,
        img_array: np.ndarray,
        width: float,
        height: float,
        depth: float
    ) -> trimesh.Trimesh:
        """
        Convert 2D image to 3D mesh
        
        Args:
            img_array: 2D array with values 0-1
            width: Width in mm
            height: Height in mm
            depth: Depth in mm
        
        Returns:
            3D mesh
        """
        h, w = img_array.shape
        
        # Threshold to binary
        binary = (img_array > 0.5).astype(np.uint8)
        
        # Create voxel grid
        voxel_depth = max(3, int(depth * 2))
        voxels = np.zeros((h, w, voxel_depth), dtype=bool)
        
        # Fill voxels where text exists
        for i in range(h):
            for j in range(w):
                if binary[i, j]:
                    voxels[i, j, :] = True
        
        # Convert voxels to mesh
        try:
            voxel_grid = trimesh.voxel.VoxelGrid(voxels)
            mesh = voxel_grid.marching_cubes
            
            # Scale to desired dimensions
            current_size = mesh.bounds[1] - mesh.bounds[0]
            scale_x = width / current_size[0] if current_size[0] > 0 else 1
            scale_y = height / current_size[1] if current_size[1] > 0 else 1
            scale_z = depth / current_size[2] if current_size[2] > 0 else 1
            
            mesh.apply_scale([scale_x, scale_y, scale_z])
            
            return mesh
        except Exception as e:
            print(f"Warning: Voxel conversion failed: {e}")
            # Fallback: create simple extruded shape
            return self._create_fallback_mesh(binary, width, height, depth)
    
    def _create_fallback_mesh(
        self,
        binary: np.ndarray,
        width: float,
        height: float,
        depth: float
    ) -> trimesh.Trimesh:
        """
        Fallback method to create text mesh
        """
        from scipy import ndimage
        
        # Get coordinates of text pixels
        coords = np.argwhere(binary > 0)
        
        if len(coords) == 0:
            return trimesh.Trimesh()
        
        h, w = binary.shape
        meshes = []
        
        # Sample coordinates to reduce complexity
        step = max(1, len(coords) // 2000)
        sampled = coords[::step]
        
        pixel_width = width / w
        pixel_height = height / h
        
        for y, x in sampled:
            # Create small box for each pixel
            box = trimesh.creation.box(
                extents=[pixel_width * 1.5, pixel_height * 1.5, depth]
            )
            box.apply_translation([
                x * pixel_width - width / 2,
                y * pixel_height - height / 2,
                depth / 2
            ])
            meshes.append(box)
        
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.merge_vertices()
            return combined
        
        return trimesh.Trimesh()
    
    def carve_text(
        self,
        input_stl: str,
        output_stl: str,
        brand_text: str,
        position: Literal["bottom", "top", "front", "back", "left", "right"] = "bottom",
        text_scale: float = 0.7,
        carve_depth: float = 1.0,
        font_height_mm: Optional[float] = None
    ) -> bool:
        """
        Carve text into STL file
        
        Args:
            input_stl: Input STL path
            output_stl: Output STL path
            brand_text: Text to carve
            position: Where to place text
            text_scale: Scale relative to model size (0.0-1.0)
            carve_depth: How deep to carve (mm)
            font_height_mm: Desired font height in mm (optional, will auto-fit if not provided)
        
        Returns:
            True if successful
        """
        try:
            print(f"Loading {input_stl}...")
            model = trimesh.load(input_stl)
            
            if not isinstance(model, trimesh.Trimesh):
                print("Error: Not a valid mesh")
                return False
            
            # Get model dimensions
            bounds = model.bounds
            size = bounds[1] - bounds[0]
            
            # Calculate text dimensions based on position
            if font_height_mm is not None:
                # Use absolute font height - set text_height to font_height_mm
                text_height = font_height_mm
                
                # Calculate text_width based on actual text dimensions and character count
                # Create a temporary image to measure actual text width
                temp_img = Image.new('L', (1024, 512), color=0)
                temp_draw = ImageDraw.Draw(temp_img)
                
                try:
                    temp_font = ImageFont.truetype(self.font_path, 200)
                except:
                    try:
                        temp_font = ImageFont.truetype("/Users/enrique/Library/Fonts/xkcd.ttf", 200)
                    except:
                        temp_font = ImageFont.load_default()
                
                temp_bbox = temp_draw.textbbox((0, 0), brand_text, font=temp_font)
                temp_text_width = temp_bbox[2] - temp_bbox[0]
                temp_text_height = temp_bbox[3] - temp_bbox[1]
                
                # Calculate aspect ratio of actual text
                text_aspect_ratio = temp_text_width / temp_text_height if temp_text_height > 0 else 3
                
                # Calculate available space
                if position in ["bottom", "top"]:
                    available_width = size[0] * text_scale
                    available_height = size[1] * text_scale
                elif position in ["front", "back"]:
                    available_width = size[0] * text_scale
                    available_height = size[2] * text_scale
                else:  # left, right
                    available_width = size[1] * text_scale
                    available_height = size[2] * text_scale
                
                # Use actual aspect ratio to calculate text_width
                text_width = min(available_width, text_height * text_aspect_ratio)
                
                # Check if text_height fits available height
                if text_height > available_height:
                    scale_factor = available_height / text_height
                    text_height *= scale_factor
                    text_width *= scale_factor
                    print(f"Warning: Requested font height {font_height_mm}mm is too tall for available space. Scaled down to fit.")
            else:
                # Original logic: calculate based on text_scale
                if position in ["bottom", "top"]:
                    text_width = size[0] * text_scale
                    text_height = size[1] * text_scale * 0.3
                elif position in ["front", "back"]:
                    text_width = size[0] * text_scale
                    text_height = size[2] * text_scale * 0.3
                else:  # left, right
                    text_width = size[1] * text_scale
                    text_height = size[2] * text_scale * 0.3
            
            print(f"Creating text: '{brand_text}'")
            
            # Pass the actual font height to create_text_mesh
            actual_font_height = font_height_mm if font_height_mm is not None else None
            
            text_mesh = self.create_text_mesh(
                brand_text,
                text_width,
                text_height,
                carve_depth,
                actual_font_height
            )
            
            if len(text_mesh.vertices) == 0:
                print("Error: Failed to create text mesh")
                return False
            
            print(f"Positioning text on {position}...")
            text_mesh = self._position_on_model(
                text_mesh,
                model,
                position,
                carve_depth
            )
            
            print("Carving text into model (this may take a moment)...")
            try:
                # Try boolean difference
                result = model.difference(text_mesh)
                
                if result is None or len(result.vertices) == 0:
                    print("Error: Boolean operation failed")
                    return False
                    
            except Exception as e:
                print(f"Error during boolean operation: {e}")
                return False
            
            print(f"Exporting to {output_stl}...")
            result.export(output_stl)
            
            print("✓ Success!")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _position_on_model(
        self,
        text_mesh: trimesh.Trimesh,
        model: trimesh.Trimesh,
        position: str,
        depth: float
    ) -> trimesh.Trimesh:
        """Position text mesh on model for carving"""
        
        bounds = model.bounds
        center = model.centroid
        text_bounds = text_mesh.bounds
        text_center = (text_bounds[0] + text_bounds[1]) / 2
        
        # Center text at origin
        text_mesh.apply_translation(-text_center)
        
        # Position and orient based on face
        if position == "bottom":
            # Mirror text horizontally for bottom viewing
            text_mesh.apply_scale([-1, 1, 1])
            text_mesh.apply_translation([
                center[0],
                center[1],
                bounds[0][2]
            ])
        
        elif position == "top":
            # Flip upside down
            text_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            )
            text_mesh.apply_translation([
                center[0],
                center[1],
                bounds[1][2]
            ])
        
        elif position == "front":
            # Rotate to vertical
            text_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
            )
            text_mesh.apply_translation([
                center[0],
                bounds[0][1],
                center[2]
            ])
        
        elif position == "back":
            # Rotate to vertical, facing back
            text_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
            )
            text_mesh.apply_translation([
                center[0],
                bounds[1][1],
                center[2]
            ])
        
        elif position == "left":
            # Rotate to vertical, facing left
            text_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
            )
            text_mesh.apply_translation([
                bounds[0][0],
                center[1],
                center[2]
            ])
        
        elif position == "right":
            # Rotate to vertical, facing right
            text_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0])
            )
            text_mesh.apply_translation([
                bounds[1][0],
                center[1],
                center[2]
            ])
        
        return text_mesh
    
    def batch_carve(
        self,
        input_files: list[str],
        output_folder: str,
        brand_text: str,
        **kwargs
    ) -> dict:
        """
        Carve text into multiple STL files
        
        Args:
            input_files: List of input STL file paths
            output_folder: Folder to save carved files
            brand_text: Text to carve into the models
            **kwargs: Additional arguments for carve_text
        
        Returns:
            Dictionary with success/failure counts
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        results = {
            'success': [],
            'failed': [],
            'total': len(input_files)
        }
        
        for input_file in input_files:
            filename = Path(input_file).name
            output_file = output_path / filename
            
            print(f"\n{'='*50}")
            print(f"Processing: {filename}")
            print('='*50)
            
            if self.carve_text(
                input_file,
                str(output_file),
                brand_text,
                **kwargs
            ):
                results['success'].append(str(output_file))
            else:
                results['failed'].append(input_file)
        
        return results


# Command-line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python stl_brander.py input.stl output.stl 'BRAND TEXT' [position] [carve_depth] [font_height_mm]")
        print("  position: bottom, top, front, back, left, right (default: bottom)")
        print("  carve_depth: depth in mm (default: 1.0)")
        print("  font_height_mm: font height in mm (optional, will auto-fit if not provided)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    brand_text = sys.argv[3]
    position = sys.argv[4] if len(sys.argv) > 4 else "bottom"
    carve_depth = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    font_height_mm = float(sys.argv[6]) if len(sys.argv) > 6 else None
    
    brander = STLBrander()
    
    success = brander.carve_text(
        input_file,
        output_file,
        brand_text,
        position=position,
        text_scale=0.7,
        carve_depth=carve_depth,
        font_height_mm=font_height_mm
    )
    
    if success:
        print(f"\n✓ Successfully created {output_file}")
    else:
        print(f"\n✗ Failed to process {input_file}")
        sys.exit(1)