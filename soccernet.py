from multiprocessing import process
from matplotlib.pylab import annotations
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os
from transformers import AutoImageProcessor
import pdb


class SoccerNetDataset(Dataset):
    """
    A dataset class for loading and preprocessing images from the SoccerNet dataset for object detection tasks.

    Attributes:
        root (str): The root directory of the dataset (e.g., 'data/tracking/train').
        processor (callable, optional): A processor for preprocessing the images.
        data (list): A list to store the images and their corresponding annotations.
        labelsToId (dict): A dictionary mapping class labels to their respective IDs.
    """
    def __init__(self, root, processor=None):
        """
        Initializes the SoccerNetDataset with the specified root directory and optional processor.

        Args:
            root (str): The root directory of the dataset.
            processor (callable, optional): A processor for preprocessing the images.
        """
        self.root = root
        self.processor = processor
        self.data = []
        self.labelsToId = {"player_team_left": 0, "player_team_right": 1, "ball": 2, "referee": 3, "goalkeeper_team_left": 4, "goalkeeper_team_right": 5, "other":6}
        self.id_to_label = {v: k for k, v in self.labelsToId.items()}
        for folder in os.listdir(root):
            if os.path.isdir(os.path.join(root, folder)):
                idToLabelLocal = self._parse_labels(os.path.join(root, folder, "gameinfo.ini"))
                img_folder = os.path.join(root, folder, "img1")
                gt = pd.read_csv(os.path.join(root, folder, "gt", "gt.txt"), header=None)
                gt.columns = ["frame", "class", "x", "y", "w", "h"] + [f"extra_{i}" for i in range(4)]
                annotations = {}
                for _, row in gt.iterrows():
                    imgName = f"{str(row['frame']).zfill(6)}.jpg"
                    # img = Image.open(os.path.join(img_folder, imgName))
                    label = idToLabelLocal[str(row["class"])]
                    # if annotations key is not present in annotations, add it
                    if imgName not in annotations:
                        annotations[imgName] = []
                    # do i need image_id in the annotations?
                    annotations[imgName].append({
                        "bbox": row[["x", "y", "w", "h"]].tolist(),
                        "bbox_mode": 0,
                        "category_id": self.labelsToId[label],
                        "iscrowd": 0,
                        "area" : row["w"] * row["h"]
                    })
            
                for imgName in os.listdir(img_folder):
                    image_id = int(folder.split('-')[1] + imgName.split('.')[0])
                    img_data = {"id": image_id,
                                "img": Image.open(os.path.join(img_folder, imgName))}
                    self.data.append((img_data, annotations[imgName])) 
            break


    def _parse_labels(self, filepath):
        """
        Parses the gameinfo.ini file to map class IDs to labels.

        Args:
            filepath (str): The path to the gameinfo.ini file.

        Returns:
            dict: A dictionary mapping class IDs to labels.
        """
        labels = {}
        with open(filepath, "r") as file:
            for line in file:
                if line.startswith("trackletID"):
                    parts = line.split("=")
                    class_id = parts[0].split("_")[1]
                    label = parts[1].split(";")[0]
                    labels[class_id] = label.strip().replace(" ", "_")
                    # bug in the labels, fix it
                    if labels[class_id] == "goalkeepers_team_left": labels[class_id] = "goalkeeper_team_left"
                    elif labels[class_id] == "goalkeepers_team_right": labels[class_id] = "goalkeeper_team_right"
        print(labels)
        return labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the image and corresponding annotations for the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its annotations. If a processor is provided, the image is preprocessed before being returned.
            image is a tensor of shape (channels, height, width) 
            annotations is a list of dictionaries containing the bounding box coordinates, category ID, and iscrowd flag for each object in the image
        """
        img_data, annotations = self.data[idx]
        
        # category_id is the index of the label in the list of labels
        target = {
            "image_id": img_data["id"],
            "annotations": annotations
        }
        if self.processor is None:
            return img_data["img"], target
        # pdb.set_trace()
        inputs = self.processor(images=img_data["img"], annotations=target, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) # remove batch dimension
        labels = inputs['labels'][0] # remove batch dimension
        return pixel_values, labels
    
def collate_fn(pixel_values, labels):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors='pt')
    labels = [item[1] for item in batch]
    batch = {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }

processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
train_dataset = SoccerNetDataset("data/tracking/train", processor=processor)
test_dataset = SoccerNetDataset("data/tracking/test", processor=processor)

# split the dataset into training and validation sets stratified by class
# train_size = int(0.8 * len(train_dataset))
# val_size = len(train_dataset) - train_size
# train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# data loader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)


# visualize one image from the dataset with bounding boxes and labels
# also for each line of code, explain what it does

import matplotlib.pyplot as plt
import matplotlib.patches as patches

img, labels = train_dataset[100]
# why? because matplotlib expects channels last format but pytorch uses channels first format
# meaning the image tensor has shape (channels, height, width) but matplotlib expects (height, width, channels)
# so permute the dimensions to match the expected format
plt.imshow(img.permute(1, 2, 0))
ax = plt.gca() # why? to get the current axes of the plot to add patches to it later on for bounding boxes and labels in the image 
# axes are the subplots meaning the region of the image where the data is plotted
# so to add bounding boxes and labels to the image, we need to get the current axes of the plot
# so that we can add patches to it
# plot the bounding boxes and labels
print(labels)
for bbox, label in zip(labels["boxes"], labels["class_labels"]):
    # bbox is a tensor of shape (4,) containing the bounding box coordinates in (x, y, w, h) format and normalized to [0, 1] based on the image size
    # label is a tensor containing the class ID of the object
    # convert the bounding box coordinates to absolute values
    # convert bbox based on the image size
    bbox = [bbox[0]*img.shape[2], bbox[1]*img.shape[1], bbox[2]*img.shape[2], bbox[3]*img.shape[1]]
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),bbox[2],bbox[3], linewidth=1, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(bbox[0], bbox[1], f"{train_dataset.id_to_label[label.item()]}", color="red")
    break
plt.show()
# for annotation in target["annotations"]:
#     bbox = annotation["bbox"]
#     category_id = annotation["category_id"]
#     # convert category_id to label
#     label = list(train_dataset.labelsToId.keys())[category_id]
#     rect = patches.Rectangle(
#         (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor="r", facecolor="none"
#     )
#     ax.add_patch(rect)
#     ax.text(bbox[0], bbox[1], label, color="red")
# plt.show()