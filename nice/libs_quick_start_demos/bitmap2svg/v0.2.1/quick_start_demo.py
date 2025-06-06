from PIL import Image
from bitmap2svg import bitmap_to_svg

try:
    img = Image.open("WechatIMG31.jpg")

    # Convert to SVG
    svg_output = bitmap_to_svg(img, num_colors=None)

    # Save the SVG to a file
    with open("output_ocean.svg", "w") as f:
        f.write(svg_output)

    print("SVG generated successfully!")

except FileNotFoundError:
    print("Error: Image file not found.")
except Exception as e:
    print(f"An error occurred: {e}")
