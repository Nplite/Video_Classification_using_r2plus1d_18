import os
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

class VideoActionRecognition:
    def __init__(self, dataset_path, batch_size=4, epochs=10, lr=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        # Initialize model and loaders
        self.train_loader, self.val_loader, self.label_map = self._get_loaders()
        self.model = self._create_model(num_classes=len(self.label_map))

    class VideoDataset(Dataset):
        def __init__(self, root_dir, label_map, frames_per_clip=32, transform=None):
            self.root_dir = root_dir
            self.label_map = label_map
            self.frames_per_clip = frames_per_clip
            self.transform = transform
            self.videos = self._get_video_paths()

        def _get_video_paths(self):
            videos = []
            for class_name in os.listdir(self.root_dir):
                class_dir = os.path.join(self.root_dir, class_name)
                if os.path.isdir(class_dir):
                    for video in os.listdir(class_dir):
                        videos.append((os.path.join(class_dir, video), self.label_map[class_name]))
            return videos

        def __len__(self):
            return len(self.videos)

        def __getitem__(self, idx):
            video_path, label = self.videos[idx]
            frames = self._extract_frames(video_path)
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            frames = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
            return frames, label

        def _extract_frames(self, video_path):
            cap = cv2.VideoCapture(video_path)
            frames = []
            while len(frames) < self.frames_per_clip:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (112, 112))  # Resize to model input size
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
                frames.append(frame)
            cap.release()
            return frames

    def _get_loaders(self):
        label_map = {class_name: i for i, class_name in enumerate(os.listdir(self.dataset_path))}
        dataset = self.VideoDataset(self.dataset_path, label_map, 
                                    transform=transforms.Normalize((0.5,), (0.5,)))

        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, label_map

    def _create_model(self, num_classes):
        model = models.video.r2plus1d_18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {train_loss / len(self.train_loader)}")
            self._validate()

        # Save the trained model
        torch.save(self.model.state_dict(), 'action_recognition_model.pth')

    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss / len(self.val_loader)}, Accuracy: {accuracy}%")


trainer = VideoActionRecognition(
        dataset_path='/home/alluvium/Desktop/Video_classification/dataset',
        batch_size=4,
        epochs=10,
        lr=1e-4)

trainer.train()
