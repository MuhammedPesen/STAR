import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from patchify import patchify
import imageio
from tqdm import tqdm
import tifffile as tiff
from PIL import Image, ImageSequence
from skimage.measure import label, regionprops
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


class SingleTileProcessor:
    def __init__(self, size1, size2, patch_size, input_path, output_path, model, device, loss_fn, model_name):
        self.size1 = size1
        self.size2 = size2
        self.patch_size = patch_size
        self.input_path = input_path
        self.output_path = output_path
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.model_name = model_name

    def load_image(self):
        im = Image.open(f"{self.input_path}")
        if im.mode == 'I;16B':
            print("16-bit image detected")
            im = im.convert('I')  # Convert to 32-bit integer pixels
            im = im.point(lambda i: i * (1 / 256))  # Scale down to 8-bit
            im = im.convert('L')  # Convert to 8-bit grayscale
        #im = im.resize((self.size1, self.size2), Image.LANCZOS)
        return im

    def process_image(self, image, overlap_size, threshold):
        patches = patchify(np.array(image), (self.patch_size, self.patch_size), step=self.patch_size - overlap_size)
        predicted_patches = []
        for i in tqdm(range(patches.shape[0])):
            for j in range(patches.shape[1]):
                self.model.eval()
                single_patch = torch.tensor(patches[i, j, :, :], dtype=torch.float32)
                single_patch_input = single_patch.unsqueeze(0).unsqueeze(0).to(self.device)
                single_patch_input /= 255.0

                with torch.no_grad():
                    single_patch_prediction = self.model(single_patch_input).squeeze().cpu()

                pred_mask = F.sigmoid(single_patch_prediction)
                pred_mask = (pred_mask > threshold).float()

                predicted_patches.append(pred_mask.numpy())
        return self.reconstruct_from_patches(predicted_patches, overlap_size)

    def reconstruct_from_patches(self, patches, overlap_size):
        full_img_shape = (self.size1, self.size2)
        step = self.patch_size - overlap_size

        full_img = np.zeros(full_img_shape, dtype=np.float32)
        count_mat = np.zeros(full_img_shape, dtype=np.float32)

        num_patches_x = (full_img_shape[1] - self.patch_size) // step + 1
        num_patches_y = (full_img_shape[0] - self.patch_size) // step + 1

        patch_idx = 0
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                start_x = j * step
                end_x = start_x + self.patch_size
                start_y = i * step
                end_y = start_y + self.patch_size

                full_img[start_y:end_y, start_x:end_x] += patches[patch_idx]
                count_mat[start_y:end_y, start_x:end_x] += 1
                patch_idx += 1

        overlap_areas = count_mat > 1
        # full_img[overlap_areas] /= count_mat[overlap_areas]

        return np.clip(full_img * 255, 0, 255).astype(np.uint8)


    def execute(self, overlap_size=128, threshold=0.5):
        image = self.load_image()
        result = self.process_image(image, overlap_size, threshold)

        # Save the original and segmented image
        tile_name = self.input_path.split('/')[-1].split('.')[0]
        imageio.imwrite(f"{self.output_path}/{tile_name}_ov{overlap_size}_{self.loss_fn}_{self.model_name}.tif", result, format='TIFF')

class ImageAnalysis:
    def __init__(self, gt_path, seg_path, plot_conf_matrix=False):
        self.gt_path = gt_path
        self.seg_path = seg_path
        self.plot_conf_matrix = plot_conf_matrix

    def analyze(self):
        gt = self._read_gt(self.gt_path)
        seg = self._read_seg(self.seg_path)

        proc_seg = self.__eliminateComp(seg, THR=500)

        gt_flat = gt.ravel()
        seg_proc_flat = proc_seg.ravel()

        # Calculating metrics
        f1 = f1_score(gt_flat, seg_proc_flat)
        recall = recall_score(gt_flat, seg_proc_flat)
        precision = precision_score(gt_flat, seg_proc_flat)

        cm = confusion_matrix(gt_flat, seg_proc_flat)


        # Calculate accuracy
        accuracy = np.trace(cm) / float(np.sum(cm))
    
        # Extract TP, TN, FP, FN from the confusion matrix
        tn, fp, fn, tp = cm.ravel()
    
        # Create a dictionary with formatted metrics
        metrics = {
            "TP": f"{(tp/np.sum(cm))*100:.2f}%",
            "TN": f"{(tn/np.sum(cm))*100:.2f}%",
            "FP": f"{(fp/np.sum(cm))*100:.2f}%",
            "FN": f"{(fn/np.sum(cm))*100:.2f}%",
            "Accuracy": f"{accuracy*100:.2f}%",
            "F1 Score": f"{f1*100:.2f}%",
            "Recall": f"{recall*100:.2f}%",
            "Precision": f"{precision*100:.2f}%"
        }
    
        return metrics

    def _read_gt(self, file_path):
        img = tiff.imread(file_path)
        img = img.astype(np.float64) / img.max()
        img[img > 0] = 1
        return img

    def _read_seg(self, file_path):
        seg_file = Image.open(file_path)
        seg = None  # Initialize seg to None
    
        for i, page_image in enumerate(ImageSequence.Iterator(seg_file)):
            seg = np.array(page_image)
            break  # Exit the loop once the desired page is found
    
        
        seg_n = seg.astype(np.float64) / seg.max()
        return seg_n
        
    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap=plt.cm.cividis)
        fig.colorbar(cax)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f'{value*100 / float(np.sum(cm))}%', ha='center', va='center')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.show()

    def __eliminateComp(self, image, THR):
        labeled_img = label(image)
        for region in regionprops(labeled_img):
            if region.area < THR:
                labeled_img[labeled_img == region.label] = 0
        return labeled_img > 0
