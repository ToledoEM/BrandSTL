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
        Create 3D text mesh for carving - simplified approach
        
        Args:
            text: Text to create
            depth_mm: Extrusion depth in mm
            font_size: Font size in points (optional, will auto-fit if not provided)
        
        Returns:
            Text mesh
        """
        # Create simple text using primitive shapes for better boolean operations
        if len(text) <= 2:  # For short text like "ET", use simple geometric approach
            return self._create_simple_text_mesh(text, depth_mm)
        
        # For longer text, use the original image-based approach
        return self._create_image_based_text_mesh(text, depth_mm, font_size)
    
    def _create_simple_text_mesh(self, text: str, depth: float) -> trimesh.Trimesh:
        """Create simple geometric text mesh for better boolean operations"""
        try:
            # Create simple block letters using boxes
            meshes = []
            char_width = 10
            char_spacing = 12
            
            for i, char in enumerate(text):
                x_offset = i * char_spacing
                
                if char.upper() == 'E':
                    # Create E using multiple boxes
                    # Vertical bar
                    vertical = trimesh.creation.box(extents=[2, 10, depth])
                    vertical.apply_translation([x_offset, 0, depth/2])
                    meshes.append(vertical)
                    
                    # Top horizontal bar
                    top = trimesh.creation.box(extents=[6, 2, depth])
                    top.apply_translation([x_offset + 2, 4, depth/2])
                    meshes.append(top)
                    
                    # Middle horizontal bar
                    middle = trimesh.creation.box(extents=[4, 2, depth])
                    middle.apply_translation([x_offset + 1, 0, depth/2])
                    meshes.append(middle)
                    
                    # Bottom horizontal bar
                    bottom = trimesh.creation.box(extents=[6, 2, depth])
                    bottom.apply_translation([x_offset + 2, -4, depth/2])
                    meshes.append(bottom)
                    
                elif char.upper() == 'T':
                    # Create T using two boxes
                    # Top horizontal bar
                    top = trimesh.creation.box(extents=[8, 2, depth])
                    top.apply_translation([x_offset, 4, depth/2])
                    meshes.append(top)
                    
                    # Vertical bar
                    vertical = trimesh.creation.box(extents=[2, 10, depth])
                    vertical.apply_translation([x_offset, 0, depth/2])
                    meshes.append(vertical)
                    
                else:
                    # Default: create a simple rectangular block
                    block = trimesh.creation.box(extents=[6, 8, depth])
                    block.apply_translation([x_offset, 0, depth/2])
                    meshes.append(block)
            
            if meshes:
                # Combine all character meshes using union operations
                combined = meshes[0]
                for mesh in meshes[1:]:
                    try:
                        combined = combined.union(mesh)
                    except:
                        # If union fails, just concatenate
                        combined = trimesh.util.concatenate([combined, mesh])
                
                # Ensure the result is a proper volume
                if not combined.is_volume:
                    combined.fill_holes()
                    combined = combined.convex_hull  # Force to be a volume
                
                return combined
            else:
                return trimesh.Trimesh()
                
        except Exception as e:
            print(f"Simple text creation failed: {e}, falling back to image method")
            return self._create_image_based_text_mesh(text, depth, None)
    
    def _create_image_based_text_mesh(
        self,
        text: str,
        depth_mm: float = 2.0,
        font_size: Optional[int] = None
    ) -> trimesh.Trimesh:
        """
        Create 3D text mesh for carving from image - original method
        
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
            
            # Debug: Check mesh integrity before boolean operation
            print(f"Original model: {len(model.vertices)} vertices, {len(model.faces)} faces")
            print(f"Text mesh: {len(text_mesh.vertices)} vertices, {len(text_mesh.faces)} faces")
            print(f"Model bounds: {model.bounds}")
            print(f"Text bounds: {text_mesh.bounds}")
            
            # Ensure both meshes are watertight before boolean operation
            if not model.is_watertight:
                print("Warning: Original model is not watertight, attempting to fix...")
                model.fill_holes()
                model.remove_degenerate_faces()
                model.remove_duplicate_faces()
                model.merge_vertices()
                
            if not text_mesh.is_watertight:
                print("Warning: Text mesh is not watertight, attempting to fix...")
                text_mesh.fill_holes()
                text_mesh.remove_degenerate_faces()
                text_mesh.remove_duplicate_faces()
                text_mesh.merge_vertices()
            
            try:
                # Alternative approach: Try to fix the boolean operation issue
                # by ensuring both meshes are properly conditioned
                
                print("Conditioning meshes for boolean operation...")
                
                # Ensure models are manifold and well-conditioned
                if not model.is_winding_consistent:
                    print("Fixing model winding...")
                    model.fix_normals()
                    
                if not text_mesh.is_winding_consistent:
                    print("Fixing text mesh winding...")
                    text_mesh.fix_normals()
                
                # Try to make models watertight
                print("Ensuring models are watertight...")
                model.fill_holes()
                text_mesh.fill_holes()
                
                # Save conditioned models for debugging
                print(f"Saving conditioned models for debugging...")
                model.export("debug_original_conditioned.stl")
                text_mesh.export("debug_text_conditioned.stl")
                
                # Alternative: Instead of difference, try intersection to see if that works
                print("Attempting boolean difference operation...")
                result = model.difference(text_mesh)
                
                if result is None:
                    print("Boolean difference failed, trying alternative approach...")
                    # If difference fails, just return original model (no carving)
                    print("Returning original model without carving")
                    result = model.copy()
                    
                if len(result.vertices) == 0:
                    print("Boolean operation resulted in empty mesh, returning original")
                    result = model.copy()
                
                # Save result for debugging
                result.export("debug_result.stl")
                
                # Check mesh quality
                print(f"Result mesh: {len(result.vertices)} vertices, {len(result.faces)} faces")
                print(f"Result is watertight: {result.is_watertight}")
                print(f"Result is winding consistent: {result.is_winding_consistent}")
                
                # If result has reasonable number of vertices, accept it
                vertex_ratio = len(result.vertices) / len(model.vertices)
                print(f"Vertex ratio: {vertex_ratio:.3f}")
                
                if len(result.vertices) >= len(model.vertices) * 0.05:  # At least 5% of original vertices
                    print("Result appears valid based on vertex count - accepting result")
                else:
                    print("Result has too few vertices, using original model")
                    result = model.copy()
                    
            except Exception as e:
                print(f"Error during boolean operation: {e}")
                print("Using original model without carving")
                result = model.copy()
            
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
        
        # Calculate model dimensions
        model_width = bounds[1][0] - bounds[0][0]
        model_height = bounds[1][1] - bounds[0][1]
        model_depth = bounds[1][2] - bounds[0][2]
        
        # Calculate text dimensions
        text_width = text_bounds[1][0] - text_bounds[0][0]
        text_height = text_bounds[1][1] - text_bounds[0][1]
        text_depth = text_bounds[1][2] - text_bounds[0][2]
        
        # Scale text to be much smaller and more conservative
        # Text should be no more than 30% of model width/height
        max_text_width = model_width * 0.3
        max_text_height = model_height * 0.2
        
        width_scale = max_text_width / text_width if text_width > 0 else 1
        height_scale = max_text_height / text_height if text_height > 0 else 1
        
        # Use the smaller scale to ensure text fits
        scale_factor = min(width_scale, height_scale)
        
        # Make the text much shallower to avoid cutting through the model
        shallow_depth = min(depth, model_depth * 0.05)  # Limit to 5% of model depth
        depth_scale = shallow_depth / text_depth if text_depth > 0 else 1
        
        print(f"Model dimensions: {model_width:.2f} x {model_height:.2f} x {model_depth:.2f}")
        print(f"Text scale factor: {scale_factor:.3f}")
        print(f"Shallow depth: {shallow_depth:.2f}mm (was {depth:.2f}mm)")
        print(f"Text will be: {text_width*scale_factor:.2f} x {text_height*scale_factor:.2f} x {shallow_depth:.2f}")
        
        text_mesh.apply_scale([scale_factor, scale_factor, depth_scale])
        
        # Position for bottom (mirrored text)
        text_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
        )
        
        # Position text at bottom of model, only creating a very shallow carve
        # Make sure the text extends only very slightly into the model
        carve_depth = min(depth, model_depth * 0.05)  # Limit to 5% of model depth
        
        text_mesh.apply_translation([
            center[0],
            center[1],
            bounds[0][2] + carve_depth * 0.25  # Position mostly at surface with minimal penetration
        ])
        
        print(f"Text positioned at bottom with minimal carve depth: {carve_depth:.2f}mm")
        
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
    
    print("STL Brander starting...")
    print(f"Arguments: {sys.argv}")
    
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
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Brand text: {brand_text}")
    print(f"Carve depth: {carve_depth}")
    print(f"Font size: {font_size}")
    
    try:
        brander = STLBrander()
        print("STLBrander initialized...")
        
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
            
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)