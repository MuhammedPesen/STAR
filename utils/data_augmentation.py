from pathlib import Path
from tqdm import tqdm
import cv2

def save_dataset(image_dir, mask_dir, num_rep, transforms=None):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    # Get the names of the images
    train_dict = {p.stem: p for p in image_dir.iterdir()}
    mask_dict = {p.stem: p for p in mask_dir.iterdir()}

    ids = list(train_dict.keys())

    for train_id in tqdm(ids):
        # Get the original img, mask path
        img_path = image_dir / train_dict[train_id]
        mask_path = mask_dir / mask_dict[train_id]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Add original image and mask to dataset
        train_list = [img]
        train_mask_list = [mask]

        if transforms != None:
            for i in range(num_rep): # How many copies of the image
                transformed = transforms(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']

                train_list.append(transformed_image)
                train_mask_list.append(transformed_mask)

        idx = 1
        for img, mask in zip(train_list, train_mask_list):
            
            name = f"{int(train_id)}_{idx}.png"

            save_img_path = image_dir / name
            save_mask_path = mask_dir / name

            # If you want to save original data too, remove the statement
            if idx != 1:
                cv2.imwrite(str(save_img_path), img)
                cv2.imwrite(str(save_mask_path), mask)

            idx += 1