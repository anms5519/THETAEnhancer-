import sys
from enhancer import enhance_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    enhanced_image = enhance_image(image_path)
    enhanced_image.save("enhanced_" + image_path)

if __name__ == "__main__":
    main()
