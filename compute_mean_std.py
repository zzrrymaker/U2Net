import os

import numpy as np
from PIL import Image


def main():
    image_dir = "./kidney_data/train/slice"
    image_names = sorted(name for name in os.listdir(image_dir) if name.endswith(".png"))
    assert image_names, f"No PNG images found in: {image_dir}"

    pixel_sum = None
    squared_sum = None
    pixel_count = 0

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        image = np.asarray(Image.open(image_path), dtype=np.float64) / 255.0
        assert image.ndim == 3, f"Expected a multi-channel image: {image_path}"

        if pixel_sum is None:
            pixel_sum = np.zeros(image.shape[-1])
            squared_sum = np.zeros(image.shape[-1])

        pixel_sum += image.sum(axis=(0, 1))
        squared_sum += np.square(image).sum(axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]

    mean = pixel_sum / pixel_count
    std = np.sqrt(squared_sum / pixel_count - mean ** 2)
    print("mean:", np.array2string(mean, precision=3))
    print("std:", np.array2string(std, precision=3))


if __name__ == "__main__":
    main()
