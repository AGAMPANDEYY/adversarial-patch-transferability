# adversarial-patch-transferability/dataset/cityscapes.py

import os
import cv2
import numpy as np
from PIL import Image
import torch
from dataset.base_dataset import BaseDataset

class Cityscapes(BaseDataset):
    """
    A “hard-coded” Cityscapes loader for Kaggle’s layout.

    Expects:
      root = "/kaggle/input/cityscapes-for-segmentation/Cityscapes"
      Inside that folder:
        ├─ train/
        │   ├─ images/
        │   │   ├─ <city>/…_leftImg8bit.png
        │   │   └─ …
        │   └─ gtFine/
        │       ├─ <city>/…_gtFine_labelIds.png
        │       └─ …
        ├─ val/
        │   ├─ images/…
        │   └─ gtFine/…
        └─ test/
            └─ images/…
    This loader will:
      • collect all train (or val, or test) images under root/[split]/images/…
      • for train/val, also load corresponding labels from root/[split]/gtFine/…
      • for test split, only return (image, size, name).
    """

    def __init__(
        self,
        root,
        split="train",          # one of "train", "val", or "test"
        num_classes=19,
        multi_scale=True,
        flip=True,
        ignore_label=255,
        base_size=2048,
        crop_size=(512, 1024),
        scale_factor=16,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        bd_dilate_size=4
    ):
        super(Cityscapes, self).__init__(
            ignore_label, base_size, crop_size, scale_factor, mean, std
        )

        self.root = root.rstrip("/")  # e.g. "/kaggle/input/cityscapes-for-segmentation/Cityscapes"
        self.split = split           # "train", "val", or "test"
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.bd_dilate_size = bd_dilate_size

        # Build a list of (image_path, label_path, name) dictionaries
        # For "train" and "val", each <city> folder has two subfolders:
        #   images/  (contains *_leftImg8bit.png)
        #   gtFine/  (contains *_gtFine_labelIds.png)
        # For "test", only images/ exists; we won't load labels.

        self.files = []
        img_dir = os.path.join(self.root, split, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Expected folder not found: {img_dir}")

        # Walk through each city subfolder under images/
        for city in sorted(os.listdir(img_dir)):
            city_img_folder = os.path.join(img_dir, city)
            if not os.path.isdir(city_img_folder):
                continue

            for fname in sorted(os.listdir(city_img_folder)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                # Full relative paths from <root>:
                rel_img = os.path.join(split, "images", city, fname)
                name = os.path.splitext(fname)[0]  # e.g. "aachen_000000_000019_leftImg8bit"
                if split in ["train", "val"]:
                    # Construct corresponding label filename:
                    # e.g. if fname = "aachen_000000_000019_leftImg8bit.png",
                    # label_fn = "aachen_000000_000019_gtFine_labelIds.png"
                    label_fn = name.replace("_leftImg8bit", "_gtFine_labelIds") + ".png"
                    rel_label = os.path.join(split, "gtFine", city, label_fn)
                    self.files.append({
                        "img": rel_img,
                        "label": rel_label,
                        "name": name
                    })
                else:  # split == "test"
                    self.files.append({
                        "img": rel_img,
                        "name": name
                    })

        # Create the label‐conversion mapping here (same as before)
        self.label_mapping = {
            -1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
             3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
             7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3,
            13: 4, 14: ignore_label, 15: ignore_label, 16: ignore_label,
            17: 5, 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
            24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: ignore_label,
            30: ignore_label, 31: 16, 32: 17, 33: 18
        }
        self.class_weights = torch.FloatTensor([
            0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
            0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
            1.0865, 1.1529, 1.0507
        ])


    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label


    def __getitem__(self, index):
        """
        Returns:
          If split=="train" or "val":
            ( image_tensor, label_tensor, edge_tensor, np.array(size), name )
          If split=="test":
            ( image_tensor, np.array(size), name )
        """
        item = self.files[index]
        name = item["name"]

        # ─── LOAD IMAGE ─────────────────────────────────────────────────────────────
        rel_img = item["img"]
        image_path = os.path.join(self.root, rel_img)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")
        size = image.shape  # (H, W, 3)

        # ─── IF TEST SPLIT ──────────────────────────────────────────────────────────
        if self.split == "test":
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

        # ─── LOAD LABEL (TRAIN/VAL) ─────────────────────────────────────────────────
        rel_lbl = item["label"]
        label_path = os.path.join(self.root, rel_lbl)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Cannot read label at {label_path}")
        label = self.convert_label(label)

        # ─── GENERATE PATCHED SAMPLE ───────────────────────────────────────────────
        image_t, label_t, edge = self.gen_sample(
            image,
            label,
            self.multi_scale,
            self.flip,
            edge_size=self.bd_dilate_size
        )

        return image_t.copy(), label_t.copy(), edge.copy(), np.array(size), name


    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
