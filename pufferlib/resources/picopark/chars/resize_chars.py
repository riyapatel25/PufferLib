#!/usr/bin/env python3
"""
Resize all character sprites to fit within 96x96 pixels (2 blocks height).
- Maintains aspect ratio (no distortion)
- Centers on transparent canvas
- Uses highest quality resampling
"""

from PIL import Image
import os

TARGET_SIZE = 96  # 2 blocks × 48 pixels

def resize_sprite_preserve_aspect(filepath):
    """
    Resize sprite to fit within TARGET_SIZE while preserving aspect ratio.
    Centers the result on a transparent canvas.
    """
    img = Image.open(filepath)
    original_size = img.size
    
    # Convert to RGBA if not already (preserve transparency)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Calculate scale factor to fit within target size
    # Scale so the LARGEST dimension becomes TARGET_SIZE
    width, height = img.size
    scale = TARGET_SIZE / max(width, height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize with high-quality resampling
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create transparent canvas of target size
    canvas = Image.new('RGBA', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0))
    
    # Center the resized image on the canvas
    paste_x = (TARGET_SIZE - new_width) // 2
    paste_y = (TARGET_SIZE - new_height) // 2
    
    # Paste using alpha channel as mask (preserves transparency)
    canvas.paste(img_resized, (paste_x, paste_y), img_resized)
    
    # Save back
    canvas.save(filepath, 'PNG')
    print(f"Resized: {filepath} {original_size} -> ({TARGET_SIZE}, {TARGET_SIZE}) [scaled to {new_width}x{new_height}, centered]")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    count = 0
    for filename in sorted(os.listdir('.')):
        if filename.endswith('.png') and 'resize' not in filename:
            resize_sprite_preserve_aspect(filename)
            count += 1
    
    print(f"\nDone! Processed {count} images to {TARGET_SIZE}x{TARGET_SIZE} pixels (aspect ratio preserved).")

if __name__ == '__main__':
    main()
