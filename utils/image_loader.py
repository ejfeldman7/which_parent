from PIL import Image

def convert_to_png(file_path):
    try:
        image = Image.open(file_path).convert("RGB")
        if image.format != 'PNG':
            image = image.convert('RGBA')
            image.save('output.png', format='PNG')
        return image
    except IOError:
        print("Unable to open or convert the image.")