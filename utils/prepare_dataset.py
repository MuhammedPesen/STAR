import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class PrepareDataset:
    def __init__(self, image_paths, mask_paths, output_dir, train_ratio = 0.75, patch_size = 256, label_threshold = 0, overlap_ratio = 0):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.label_threshold = label_threshold
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.ensure_directories_exist()

    def ensure_directories_exist(self):
        """
        Ensure that necessary directories exist for storing processed images and masks, both for training and validation sets.
        """
        directories = [
            os.path.join(self.output_dir, 'train/images'),
            os.path.join(self.output_dir, 'train/masks'),
            os.path.join(self.output_dir, 'val/images'),
            os.path.join(self.output_dir, 'val/masks'),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def rescale_to_uint8(self, image_array):
        """
        Rescale any image array to the 0-255 range and convert to uint8 for image processing compatibility.
        """
        image_rescaled = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
        return image_rescaled.astype(np.uint8)

    def save_patches(self, patches, images_dir, masks_dir, start_index):
        """
        Save image and mask patches to the specified directories and return the last index used.
        """
        for i, (img_patch, mask_patch) in enumerate(patches, start=start_index):
            img = Image.fromarray(img_patch)
            mask = Image.fromarray(mask_patch)
            img.save(os.path.join(images_dir, f"{i}.png"))
            mask.save(os.path.join(masks_dir, f"{i}.png"))
        return i

    def process_image_mask_pair(self, image_path, mask_path, start_index):
        """
        Process each image-mask pair by extracting patches, filtering by label threshold, and splitting into train and validation sets.
        """
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.rescale_to_uint8(np.array(image))
        mask = self.rescale_to_uint8(np.array(mask))

        stride = int(self.patch_size * (1 - self.overlap_ratio))
        num_patches_x, num_patches_y = (image.shape[0] - self.patch_size) // stride + 1, (image.shape[1] - self.patch_size) // stride + 1

        patches = []
        for x in range(num_patches_x):
            for y in range(num_patches_y):
                img_patch = image[x*stride:(x*stride)+self.patch_size, y*stride:(y*stride)+self.patch_size]
                mask_patch = mask[x*stride:(x*stride)+self.patch_size, y*stride:(y*stride)+self.patch_size]
                if np.sum(mask_patch > 0) / (self.patch_size**2) >= self.label_threshold:
                    patches.append((img_patch, mask_patch))

        train_patches, val_patches = train_test_split(patches, test_size=1-self.train_ratio, random_state=42)
        index = self.save_patches(train_patches, os.path.join(self.output_dir, 'train/images'), os.path.join(self.output_dir, 'train/masks'), start_index)
        index = self.save_patches(val_patches, os.path.join(self.output_dir, 'val/images'), os.path.join(self.output_dir, 'val/masks'), index + 1)
        return index


    def divide_into_patches(self):
        """
        Divide all images and masks into patches and process them for training and validation datasets.
        """
        assert len(self.image_paths) == len(self.mask_paths), "Image paths and mask paths lists must have the same length"
        start_index = 0
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            start_index = self.process_image_mask_pair(image_path, mask_path, start_index) + 1