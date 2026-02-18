#!/usr/bin/env python3
"""Process PICO PARK sprites: remove white backgrounds and generate block sprites."""

from PIL import Image
import os

# Cat sprite files
CAT_FILES = [
    'blue_cat.png',
    'gray_cat.png', 
    'green_cat.png',
    'orange_cat.png',
    'pink_cat.png',
    'purple.png',
    'red_cat.png',
    'yellow_cat.png',
]

# Colors extracted from the cat sprites (approximate RGB values)
COLORS = {
    'blue': (70, 130, 230),
    'gray': (128, 128, 128),
    'green': (70, 200, 70),
    'orange': (255, 140, 50),
    'pink': (255, 150, 200),
    'purple': (150, 80, 200),
    'red': (230, 70, 70),
    'yellow': (255, 220, 70),
}

def remove_white_background(img_path):
    """Remove white background from an image, making it transparent."""
    img = Image.open(img_path).convert('RGBA')
    data = img.getdata()
    
    new_data = []
    for item in data:
        # If pixel is white or near-white, make it transparent
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    
    img.putdata(new_data)
    img.save(img_path)
    print(f'Removed background from {img_path}')

def create_block(color_name, color_rgb, size=64):
    """Create a colored block sprite."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    # Draw main block
    margin = 2
    for y in range(margin, size - margin):
        for x in range(margin, size - margin):
            # Add slight 3D effect
            if x < margin + 3 or y < margin + 3:
                # Highlight
                r = min(color_rgb[0] + 40, 255)
                g = min(color_rgb[1] + 40, 255)
                b = min(color_rgb[2] + 40, 255)
                img.putpixel((x, y), (r, g, b, 255))
            elif x > size - margin - 4 or y > size - margin - 4:
                # Shadow
                r = max(color_rgb[0] - 40, 0)
                g = max(color_rgb[1] - 40, 0)
                b = max(color_rgb[2] - 40, 0)
                img.putpixel((x, y), (r, g, b, 255))
            else:
                img.putpixel((x, y), (*color_rgb, 255))
    
    return img

def create_goal(size=64):
    """Create a goal/key sprite."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    gold = (255, 215, 0)
    gold_dark = (200, 160, 0)
    
    center = size // 2
    
    # Simple star/goal shape
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = (dx*dx + dy*dy) ** 0.5
            
            if dist < size // 3:
                # Inner bright
                img.putpixel((x, y), (*gold, 255))
            elif dist < size // 2.5:
                # Outer ring
                img.putpixel((x, y), (*gold_dark, 200))
    
    return img

def create_bullet(size=32):
    """Create a bullet sprite."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    center = size // 2
    
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = (dx*dx + dy*dy) ** 0.5
            
            if dist < size // 4:
                img.putpixel((x, y), (255, 255, 255, 255))
            elif dist < size // 3:
                img.putpixel((x, y), (255, 255, 200, 200))
            elif dist < size // 2:
                img.putpixel((x, y), (255, 255, 150, 100))
    
    return img

def main():
    # Remove white backgrounds from cat sprites
    print("Removing white backgrounds from cat sprites...")
    for cat_file in CAT_FILES:
        if os.path.exists(cat_file):
            remove_white_background(cat_file)
        else:
            print(f'Warning: {cat_file} not found')
    
    # Generate block sprites
    print("\nGenerating block sprites...")
    for color_name, color_rgb in COLORS.items():
        block = create_block(color_name, color_rgb, 64)
        filename = f'block_{color_name}.png'
        block.save(filename)
        print(f'Created {filename}')
    
    # Generate goal sprite
    goal = create_goal(64)
    goal.save('goal.png')
    print('Created goal.png')
    
    # Generate bullet sprite
    bullet = create_bullet(32)
    bullet.save('bullet.png')
    print('Created bullet.png')
    
    print('\nAll sprites processed successfully!')

if __name__ == '__main__':
    main()

