import numpy as np
import os
from PIL import Image

# Combines images to form one large image from multiple channels

def combine_images_vertically(image_paths, row_nbr):
    # Open the images and convert them to numpy arrays
    images = [np.asarray(Image.open(path)) for path in image_paths]
    
    # Combine the images vertically
    combined_image_array = np.vstack(images)
    
    # Convert the combined array back to an image
    combined_image = Image.fromarray(combined_image_array)

    combined_image.save(f"D:\\LUMO\\IMGC\\img_row_{row_nbr}.png")
    
    return combined_image

if __name__ == "__main__":
    nbr_instances = 3157
    for row in range(nbr_instances):
        x = [f"D:\\LUMO\\IMG\\img_ch_{i}_row_{row}.png" for i in range(6)]
        combine_images_vertically(x, row)
