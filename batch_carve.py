#!/usr/bin/env python3
"""
Batch carve text branding into multiple STL files
Usage: uv run python batch_carve.py
"""

from pathlib import Path
import sys

# Import the STLBrander class
try:
    from stl_brander import STLBrander
except ImportError:
    print("Error: stl_brander.py not found in current directory")
    sys.exit(1)

# ============= CONFIGURATION =============
BRAND_TEXT = "YOUR BRAND"
INPUT_FOLDER = "input_stls"
OUTPUT_FOLDER = "branded_stls"
POSITION = "bottom"  # bottom, top, front, back, left, right
TEXT_SCALE = 0.7     # 0.0 to 1.0 (relative to model size)
CARVE_DEPTH = 1.0    # Depth in mm
FONT_HEIGHT_MM = None  # Font height in mm (None to prompt for each file)
# =========================================

def main():
    """Main batch processing function"""
    
    # Initialize brander
    brander = STLBrander()
    
    # Setup paths
    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    
    # Create output folder
    output_path.mkdir(exist_ok=True)
    
    # Find all STL files
    stl_files = list(input_path.glob("*.stl"))
    
    if not stl_files:
        print(f"❌ No STL files found in {INPUT_FOLDER}/")
        print(f"Please add STL files to the {INPUT_FOLDER}/ directory")
        return
    
    # Prompt for font height if not configured
    font_height = FONT_HEIGHT_MM
    if font_height is None:
        try:
            height_input = input("How tall should the font be in mm? (press Enter for auto-fit): ").strip()
            font_height = float(height_input) if height_input else None
        except (ValueError, EOFError):
            print("Invalid input, using auto-fit")
            font_height = None
    
    print(f"{'='*60}")
    print(f"STL Batch Branding Tool")
    print(f"{'='*60}")
    print(f"Brand Text: {BRAND_TEXT}")
    print(f"Position: {POSITION}")
    print(f"Carve Depth: {CARVE_DEPTH}mm")
    print(f"Text Scale: {TEXT_SCALE * 100}%")
    print(f"Font Height: {font_height}mm" if font_height else "Font Height: Auto-fit")
    print(f"Files to process: {len(stl_files)}")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_count = 0
    success_files = []
    failed_files = []
    
    # Process each file
    for i, stl_file in enumerate(stl_files, 1):
        output_file = output_path / stl_file.name
        
        print(f"\n[{i}/{len(stl_files)}] Processing: {stl_file.name}")
        print("-" * 60)
        
        if brander.carve_text(
            str(stl_file),
            str(output_file),
            BRAND_TEXT,
            position=POSITION,
            text_scale=TEXT_SCALE,
            carve_depth=CARVE_DEPTH,
            font_height_mm=font_height
        ):
            success_count += 1
            success_files.append(stl_file.name)
        else:
            failed_count += 1
            failed_files.append(stl_file.name)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files: {len(stl_files)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failed_count}")
    print(f"Output folder: {OUTPUT_FOLDER}/")
    print(f"{'='*60}")
    
    if success_files:
        print(f"\n✓ Successfully processed:")
        for filename in success_files:
            print(f"  - {filename}")
    
    if failed_files:
        print(f"\n✗ Failed to process:")
        for filename in failed_files:
            print(f"  - {filename}")
    
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)