import logging
from PIL import Image
import os

def generate_collage(photos, output_path, options):
    """
    Generates a collage image from the given photos and saves it to output_path.
    """
    collage_width_px = options['collage_width_px']
    collage_height_px = options['collage_height_px']
    photo_width_px = options['photo_width_px']
    photo_height_px = options['photo_height_px']
    rows = options['rows']
    cols = options['cols']
    wrap_method = options['wrap_method']

    # Create a blank image for the collage
    collage_image = Image.new('RGB', (collage_width_px, collage_height_px), color='white')

    # Resize photos to the specified photo dimensions
    resized_photos = []
    for photo in photos:
        img = Image.open(photo.processed_image_path)
        img_resized = img.resize((photo_width_px, photo_height_px))
        resized_photos.append(img_resized)

    # Arrange photos on the grid according to the wrap method
    index = 0
    total_photos = len(resized_photos)
    for row in range(rows):
        for col in range(cols):
            if index >= total_photos:
                break
            photo_img = resized_photos[index]

            # Calculate position based on wrap method
            if wrap_method == 'lr_down':
                x = col * photo_width_px
                y = row * photo_height_px
            elif wrap_method == 'lr_down_snake':
                if row % 2 == 0:
                    x = col * photo_width_px
                else:
                    x = (cols - 1 - col) * photo_width_px
                y = row * photo_height_px
            elif wrap_method == 'tb_right':
                x = row * photo_width_px
                y = col * photo_height_px
            elif wrap_method == 'tb_right_snake':
                if col % 2 == 0:
                    x = row * photo_width_px
                else:
                    x = (rows - 1 - row) * photo_width_px
                y = col * photo_height_px
            else:
                x = col * photo_width_px
                y = row * photo_height_px

            collage_image.paste(photo_img, (int(x), int(y)))
            index += 1

    # Save the collage image
    collage_image.save(output_path, format='JPEG', quality=95)
    logging.info(f"Collage saved to {output_path}")
