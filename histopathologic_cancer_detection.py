import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set data directory
DATA_DIR = "D:\\Backup\\Documents\\School"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_PATH = os.path.join(DATA_DIR, "train_labels.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# Parameters
IMG_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 3

# Train dataset
class HistopathologicCancerDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, f"{img_name}.tif")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Test dataset
class TestDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.tif")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training data
    df = pd.read_csv(LABELS_PATH)
    print(f"Total labeled images: {len(df)}")

    df_sampled = df.groupby("label").apply(lambda x: x.sample(1000, random_state=42)).reset_index(drop=True)
    train_df, val_df = train_test_split(df_sampled, test_size=0.2, stratify=df_sampled['label'], random_state=42)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = HistopathologicCancerDataset(train_df, TRAIN_DIR, transform=transform)
    val_dataset = HistopathologicCancerDataset(val_df, TRAIN_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Prediction block
    test_df = pd.read_csv(SUBMISSION_PATH)

    test_dataset = TestDataset(test_df, TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("Sanity check: iterating over test_loader for 1 batch")
    for inputs, image_ids in test_loader:
        print(f"Loaded batch of size: {inputs.size(0)}")
        print(f"First image ID: {image_ids[0]}")
        break

    print("Checking a few test image paths...")
    for i in range(5):
        img_name = test_df.iloc[i, 0]
        img_path = os.path.join(TEST_DIR, f"{img_name}.tif")
        print(f"Looking for: {img_path}")
        if not os.path.exists(img_path):
            print(f"❌ MISSING: {img_path}")
        else:
            print(f"✅ FOUND: {img_path}")

    model.eval()
    predictions = []

    with torch.no_grad():
        processed = 0
        total = len(test_loader.dataset)

        for batch_idx, (inputs, image_ids) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            for img_id, pred in zip(image_ids, preds):
                predictions.append([img_id, pred])
                processed += 1

            # Progress update every 100 images
            if processed % 100 < BATCH_SIZE:
                print(f"[{processed}/{total}] images processed...")

    submission_df = pd.DataFrame(predictions, columns=["id", "label"])
    submission_df.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
