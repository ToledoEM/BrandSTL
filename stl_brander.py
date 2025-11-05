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
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
        
        for font in possible_fonts:
            if Path(font).exists():
                return font
        
        return possible_fonts[0]
    
    def _load_font(self, font_size: int) -> ImageFont.ImageFont:
        """Load font with proper fallback chain"""
        font_paths = [
            self.font_path if self.font_path else None,
            "xkcd.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/System/Library/Fonts/SFNS.ttf"
        ]
        
        for font_path in font_paths:
            if font_path is None:
                continue
            try:
                if Path(font_path).exists():
                    return ImageFont.truetype(font_path, font_size)
            except (OSError, IOError):
                continue
        
        # Final fallback
        return ImageFont.load_default()
    
    def create_text_mesh(
        self,
        text: str,
        depth_mm: float = 2.0,
        font_size: Optional[int] = None
    ) -> trimesh.Trimesh:
        """
        Create 3D text mesh for carving
        
        Args:
            text: Text to create
            depth_mm: Extrusion depth in mm
            font_size: Font size in points (optional, will auto-fit if not provided)
        
        Returns:
            Text mesh
        """
        # Create high-res image with text
        img_width = 1024
        img_height = 512
        img = Image.new('L', (img_width, img_height), color=0)
        draw = ImageDraw.Draw(img)
        
        # Load font
        font_size_to_use = font_size if font_size is not None else 200
        font = self._load_font(font_size_to_use)
        
        # Get text size and center it
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
            depth_mm,
            text,
            font_size_to_use
        )
        
        return mesh
    
    def _image_to_3d_mesh(
        self,
        img_array: np.ndarray,
        depth: float,
        text: str = "",
        font_size: int = 200
    ) -> trimesh.Trimesh:
        """
        Convert 2D image to 3D mesh
        
        Args:
            img_array: 2D array with values 0-1
            depth: Depth in mm
            text: Text string for high-res rendering
            font_size: Font size for high-res rendering
        
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
        
        # Convert voxels to mesh using improved voxel method
        try:
            # Use moderate resolution for balance of quality and file size
            scale_factor = 2
            large_w, large_h = w * scale_factor, h * scale_factor
            
            # Create high-resolution image
            large_img = Image.new('L', (large_w, large_h), 0)
            large_draw = ImageDraw.Draw(large_img)
            
            # Scale font size accordingly
            large_font_size = int(font_size * scale_factor)
            large_font = self._load_font(large_font_size)
            
            # Get text dimensions and center it
            bbox = large_draw.textbbox((0, 0), text, font=large_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (large_w - text_width) // 2
            y = (large_h - text_height) // 2
            
            # Draw text on high-res image
            large_draw.text((x, y), text, fill=255, font=large_font)
            
            # Convert to numpy array and create binary mask
            large_array = np.array(large_img)
            large_binary = large_array > 128
            
            # Create high-resolution voxel grid with moderate depth
            voxel_depth = max(4, int(depth * 8))
            large_voxels = np.zeros((large_w, large_h, voxel_depth), dtype=bool)
            for z in range(large_voxels.shape[2]):
                large_voxels[:, :, z] = large_binary.T
            
            # Generate mesh from high-res voxels
            voxel_grid = trimesh.voxel.VoxelGrid(large_voxels)
            mesh = voxel_grid.marching_cubes
            
            # Scale back down and normalize
            mesh.apply_scale([1/scale_factor, 1/scale_factor, depth/large_voxels.shape[2]])
            
            # Center the mesh
            mesh.apply_translation([-0.5, -0.5, 0])
            
            # Skip subdivision to reduce file size
            # Apply basic smoothing by subdividing
            # try:
            #     mesh = mesh.subdivide()
            # except:
            #     pass
            
            # Ensure mesh is watertight
            if not mesh.is_watertight:
                mesh.fill_holes()
                if not mesh.is_watertight:
                    mesh.remove_degenerate_faces()
                    mesh.remove_duplicate_faces()
                    mesh.merge_vertices()
            
            return mesh
            
        except Exception as e:
            print(f"Warning: High-res method failed ({e}), using basic voxel method")
            # Fallback to original voxel method
            voxel_grid = trimesh.voxel.VoxelGrid(voxels)
            mesh = voxel_grid.marching_cubes
            return mesh
        except Exception as e:
            print(f"Warning: Voxel conversion failed: {e}")
            # Fallback: create simple extruded shape
            return self._create_fallback_mesh(binary, depth)
    
    def _create_fallback_mesh(
        self,
        binary: np.ndarray,
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
        
        pixel_width = 1.0 / w
        pixel_height = 1.0 / h
        
        for y, x in sampled:
            # Create small box for each pixel
            box = trimesh.creation.box(
                extents=[pixel_width * 1.5, pixel_height * 1.5, depth]
            )
            box.apply_translation([
                x * pixel_width - 0.5,
                y * pixel_height - 0.5,
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
        text_scale: float = 0.7,
        carve_depth: float = 1.0,
        font_size: Optional[int] = None
    ) -> bool:
        """
        Carve text into STL file (bottom position only)
        
        Args:
            input_stl: Input STL path
            output_stl: Output STL path
            brand_text: Text to carve
            text_scale: Scale relative to model size (0.0-1.0)
            carve_depth: How deep to carve (mm)
            font_size: Font size in points (optional, will auto-fit if not provided)
        
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
            if font_size is not None:
                # Use absolute font size - set text_height to font_size
                text_height = font_size
                
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
                
                # Calculate available space (bottom position only)
                available_width = size[0] * text_scale
                available_height = size[1] * text_scale
                
                # Use actual aspect ratio to calculate text_width
                text_width = min(available_width, text_height * text_aspect_ratio)
                
                # Check if text_height fits available height
                if text_height > available_height:
                    scale_factor = available_height / text_height
                    text_height *= scale_factor
                    text_width *= scale_factor
                    print(f"Warning: Requested font size {font_size}pt is too tall for available space. Scaled down to fit.")
            else:
                # Original logic: calculate based on text_scale (bottom position only)
                text_width = size[0] * text_scale
                text_height = size[1] * text_scale * 0.3
            
            print(f"Creating text: '{brand_text}'")
            
            # Pass the actual font size to create_text_mesh
            actual_font_size = font_size if font_size is not None else None
            
            text_mesh = self.create_text_mesh(
                brand_text,
                carve_depth,
                actual_font_size
            )
            
            if len(text_mesh.vertices) == 0:
                print("Error: Failed to create text mesh")
                return False
            
            print("Positioning text on bottom...")
            text_mesh = self._position_on_model(
                text_mesh,
                model,
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
            if output_stl.endswith('_ascii.stl'):
                result.export(output_stl, file_type='stl_ascii')
            else:
                # Force binary format by using default STL export
                with open(output_stl, 'wb') as f:
                    result.export(f, file_type='stl')
            
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
        depth: float
    ) -> trimesh.Trimesh:
        """Position text mesh on model for carving (bottom position only)"""
        
        bounds = model.bounds
        center = model.centroid
        text_bounds = text_mesh.bounds
        text_center = (text_bounds[0] + text_bounds[1]) / 2
        
        # Center text at origin
        text_mesh.apply_translation(-text_center)
        
        # Scale text to reasonable size relative to model
        model_size = np.max(bounds[1] - bounds[0])
        text_size = np.max(text_bounds[1] - text_bounds[0])
        scale_factor = (model_size * 0.3) / text_size  # Text should be 30% of model size
        text_mesh.apply_scale(scale_factor)
        
        # Position for bottom (mirrored text)
        text_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
        )
        text_mesh.apply_translation([
            center[0],
            center[1],
            bounds[0][2] + depth/2  # Extend into model from bottom
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
        print("Usage: python stl_brander.py input.stl output.stl 'BRAND TEXT' [carve_depth] [font_size]")
        print("  carve_depth: depth in mm (default: 1.0)")
        print("  font_size: font size in points (optional, will auto-fit if not provided)")
        print("  Note: Text is always positioned on bottom with mirroring")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    brand_text = sys.argv[3]
    carve_depth = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    font_size = int(sys.argv[5]) if len(sys.argv) > 5 else None
    
    brander = STLBrander()
    
    success = brander.carve_text(
        input_file,
        output_file,
        brand_text,
        text_scale=0.7,
        carve_depth=carve_depth,
        font_size=font_size
    )
    
    if success:
        print(f"\n✓ Successfully created {output_file}")
    else:
        print(f"\n✗ Failed to process {input_file}")
        sys.exit(1)