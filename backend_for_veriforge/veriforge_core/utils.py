# image_processing/utils.py

from PIL import Image, ImageChops, ImageEnhance
import io

def ela_image(image, quality=90):
    """Perform Error Level Analysis (ELA) on the image using JPEG compression artifacts."""
    # Save the original image to a temporary in-memory file with reduced quality
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Load the compressed image from the in-memory buffer
    compressed_image = Image.open(buffer)

    # Calculate the difference between the original and compressed image
    ela_image = ImageChops.difference(image.convert('RGB'), compressed_image)

    # Enhance brightness based on the maximum difference
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image
