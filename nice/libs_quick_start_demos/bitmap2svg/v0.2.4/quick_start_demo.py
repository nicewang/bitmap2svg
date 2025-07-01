from PIL import Image
from bitmap2svg import bitmap_to_svg

try:
    img = Image.open("0013.png")

    # Convert to SVG
    svg_output = bitmap_to_svg(img, num_colors=None, use_processed_mask=True)

    # Save the SVG to a file
    with open("processed_mask/output.svg", "w") as f:
        f.write(svg_output)

    print("SVG generated successfully!")

except FileNotFoundError:
    print("Error: Image file not found.")
except Exception as e:
    print(f"An error occurred: {e}")