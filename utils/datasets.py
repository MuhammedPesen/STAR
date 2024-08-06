from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision

class AstrocyteDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path):
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)

        # Creating dictionaries that map image IDs to their respective file paths
        self.image = {p.stem: p for p in image_dir.iterdir()}
        self.mask = {p.stem: p for p in mask_dir.iterdir()}

        # List of image IDs (to ensure we can index them)
        self.ids = list(self.image.keys())

        # Define transformations to apply to images (conversion to tensor)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # Fetch the ID of the image and mask using the provided index
        id_ = self.ids[index]

        # Load the image file, apply transformations, and ensure dtype consistency
        image = Image.open(self.image[id_])
        image = self.transforms(image)

        # Load the corresponding mask file, apply transformations
        mask = Image.open(self.mask[id_])
        mask = self.transforms(mask)

        # Normalize mask to range 0-1 if necessary
        if mask.max() > 1:
            mask = mask / mask.max()
        
        # Normalize image to range 0-1 if necessary
        if image.max() > 1:
            image = image / image.max()

        return image, mask

    def __len__(self):
        # Returns the total number of images/masks
        return len(self.ids)