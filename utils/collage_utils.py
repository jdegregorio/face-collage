import logging
from PIL import Image
import os

def generate_collage(faces, output_path, options):
    """
    Generates a collage image from the given faces and saves it to output_path.
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

    # Resize faces to the specified photo dimensions
    resized_faces = []
    for face in faces:
        img = Image.open(face.image_path)
        img_resized = img.resize((photo_width_px, photo_height_px))
        resized_faces.append(img_resized)

    # Arrange faces on the grid according to the wrap method
    index = 0
    total_faces = len(resized_faces)
    for row in range(rows):
        for col in range(cols):
            if index >= total_faces:
                break
            face_img = resized_faces[index]

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

            collage_image.paste(face_img, (int(x), int(y)))
            index += 1

    # Save the collage image
    collage_image.save(output_path, format='JPEG', quality=95)
    logging.info(f"Collage saved to {output_path}")
