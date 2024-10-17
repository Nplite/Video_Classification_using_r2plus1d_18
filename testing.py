import torch
import cv2
import torch.nn.functional as F
from torchvision import models

class VideoActionTester:
    def __init__(self, model_path, device, label_map, frames_per_clip=32):
        self.device = device
        self.label_map = label_map
        self.frames_per_clip = frames_per_clip

        # Initialize the model and load the state dictionary
        self.model = self._initialize_model(num_classes=len(label_map))
        self._load_model(model_path)

    def _initialize_model(self, num_classes):
        """Initialize the model structure."""
        model = models.video.r2plus1d_18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the output layer
        return model

    def _load_model(self, model_path):
        """Load the model weights from the specified path."""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)  # Load state dict into the model
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def _load_video_frames(self, video_path):
        """Load frames from a video file and prepare them for the model."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (112, 112))  # Resize to match model input
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
            frames.append(frame)

        cap.release()

        # If not enough frames, pad with the last frame
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1].clone())

        # Stack frames along the time dimension and add batch dimension
        video_tensor = torch.stack(frames, dim=1)  # (C, T, H, W)
        return video_tensor.unsqueeze(0)  # (1, C, T, H, W)

    def predict(self, video_path, threshold=0.5):
        """Predict the action in the video and display the results."""
        inputs = self._load_video_frames(video_path).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            class_name = list(self.label_map.keys())[list(self.label_map.values()).index(predicted.item())]

            if confidence.item() < threshold:
                print(f"Predicted Action: Unknown, Confidence: {confidence.item():.2f}")
            else:
                print(f"Predicted Action: {class_name}, Confidence: {confidence.item():.2f}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
video_path = 'test_data/16:12:51.mp4'
label_map = { 'CricketBowling':0, 'JavelineThrow': 1,  'ThrowDiscus':2}  # Replace with your label map

tester = VideoActionTester(
    model_path='action_recognition_model.pth',
    device=device,
    label_map=label_map,
    frames_per_clip=32
)
tester.predict(video_path, threshold=0.5)
