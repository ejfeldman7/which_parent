from PIL import Image

def convert_to_png(file_path: str) -> Image.Image:
    '''
    Convert file path to PNG format
    '''
    try:
        image = Image.open(file_path).convert("RGB")
        if image.format not in ['JPG', 'JPEG', 'PNG']:
            image = image.convert('RGBA')
            image.save('output.png', format='PNG')
        return image
    except IOError:
        print("Unable to open or convert the image.")