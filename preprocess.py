import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split

import wandb

# Log in to wandb with API key
wandb.login(key='eb4c4a1fa7eec1ffbabc36420ba1166f797d4ac5')

train_img_path = "/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/data/train/images"
train_lbl_path = "/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/data/train/labels"
valid_img_path = "/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/data/valid/images"
valid_lbl_path = "/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/data/valid/labels"
test_img_path = "/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/data/test/images"
test_lbl_path = "/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/data/test/labels"
model_path = "/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/best.pt"
data_yaml_path = "/Users/lecourtoistristan/Documents/projet_IA/Mars_Crater_detection/data.yaml"

# Data preprocessing 

def load_labels(label_path):
    label_files = os.listdir(label_path)
    data = []
    classes = set()
    for file in label_files:
        with open(os.path.join(label_path, file), 'r') as f:
            lines= f.readlines()
            for line in lines:
                parts= list(map(float, line.strip().split()))
                data.append([file, *parts])
                classes.add(int(parts[0]))
    df = pd.DataFrame(data,columns=['file', 'class', 'x_center', 'y_center', 'width', 'height'])
    return df, sorted(classes)

train_labels, train_classes = load_labels(train_lbl_path)
valid_labels, valid_classes = load_labels(valid_lbl_path)
test_labels, test_classes = load_labels(test_lbl_path)

# Get all unique classes
all_classes = sorted(set(train_classes + valid_classes + test_classes))
class_names = [f'class_{i}' for i in all_classes]

# Display first few rows of the labels
print("Train Labels")
print(train_labels.head())
print("\nValidation Labels")
print(valid_labels.head())
print("\nTest Labels")
print(test_labels.head())

# Create data.yaml
data_yaml_content = f"""
train: {train_img_path}
val: {valid_img_path}
test: {test_img_path}

nc: {len(all_classes)}  # number of classes
names: {class_names}  # class names
"""

with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

def visualize_sample_images(image_path, label_df, n_samples=5):
    image_files = os.listdir(image_path)[:n_samples]
    for img_file in image_files:
        img_path = os.path.join(image_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        
        labels = label_df[label_df['file'] == img_file]
        for _, label in labels.iterrows():
            x_center = int(label['x_center'] * img.shape[1])
            y_center = int(label['y_center'] * img.shape[0])
            width = int(label['width'] * img.shape[1])
            height = int(label['height'] * img.shape[0])
            x_min = x_center - width // 2
            y_min = y_center - height // 2
            
            rect = plt.Rectangle((x_min, y_min), width, height, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
        
        plt.title(f'Sample Image: {img_file}')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
  
    visualize_sample_images(train_img_path, train_labels)
    visualize_sample_images(valid_img_path, valid_labels)
    visualize_sample_images(test_img_path, test_labels)
