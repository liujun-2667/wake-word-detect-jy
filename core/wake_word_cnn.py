import numpy as np
import os
import config

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = object
    DataLoader = None


if TORCH_AVAILABLE:
    class WakeWordCNN(nn.Module):
        def __init__(self, num_classes=1, input_dim=39):
            super(WakeWordCNN, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes),
                nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = x.unsqueeze(1) if x.dim() == 3 else x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


    class WakeWordDataset(Dataset):
        def __init__(self, features_list, labels, max_frames=100):
            self.features = []
            self.labels = labels
            self.max_frames = max_frames

            for feat in features_list:
                padded = self._pad_or_truncate(feat)
                self.features.append(padded)

        def _pad_or_truncate(self, features):
            if features.shape[0] >= self.max_frames:
                return features[:self.max_frames, :]
            else:
                pad_width = self.max_frames - features.shape[0]
                padding = np.zeros((pad_width, features.shape[1]))
                return np.vstack([features, padding])

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            feat = self.features[idx]
            label = self.labels[idx]
            return torch.FloatTensor(feat), torch.FloatTensor([label])


    class WakeWordCNNClassifier:
        def __init__(self, num_classes=1, input_dim=39, max_frames=100, device=None):
            self.num_classes = num_classes
            self.input_dim = input_dim
            self.max_frames = max_frames

            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device

            self.model = WakeWordCNN(num_classes=num_classes, input_dim=input_dim).to(self.device)
            self.is_trained = False

        def train(self, positive_features, negative_features, epochs=50, batch_size=16, learning_rate=0.001):
            features = positive_features + negative_features
            labels = [1] * len(positive_features) + [0] * len(negative_features)

            dataset = WakeWordDataset(features, labels, max_frames=self.max_frames)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0

                for batch_features, batch_labels in dataloader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == batch_labels).sum().item()
                    total += batch_labels.size(0)

                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    accuracy = correct / total
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            self.is_trained = True

        def predict(self, features):
            self.model.eval()
            with torch.no_grad():
                padded = self._pad_or_truncate(features)
                tensor = torch.FloatTensor(padded).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                confidence = output.cpu().numpy()[0][0]
            return confidence

        def _pad_or_truncate(self, features):
            if features.shape[0] >= self.max_frames:
                return features[:self.max_frames, :]
            else:
                pad_width = self.max_frames - features.shape[0]
                padding = np.zeros((pad_width, features.shape[1]))
                return np.vstack([features, padding])

        def save(self, filepath):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'input_dim': self.input_dim,
                'max_frames': self.max_frames,
                'is_trained': self.is_trained
            }, filepath)

        def load(self, filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.num_classes = checkpoint['num_classes']
            self.input_dim = checkpoint['input_dim']
            self.max_frames = checkpoint['max_frames']
            self.is_trained = checkpoint['is_trained']
            self.model = WakeWordCNN(num_classes=self.num_classes, input_dim=self.input_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()


    class MultiClassWakeWordCNN:
        def __init__(self, num_wake_words, input_dim=39, max_frames=100, device=None):
            self.num_wake_words = num_wake_words
            self.input_dim = input_dim
            self.max_frames = max_frames

            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device

            self.model = WakeWordCNN(num_classes=num_wake_words + 1, input_dim=input_dim).to(self.device)
            self.is_trained = False
            self.wake_word_names = []

        def train(self, wake_word_data, negative_features, epochs=50, batch_size=16, learning_rate=0.001):
            all_features = []
            all_labels = []

            self.wake_word_names = list(wake_word_data.keys())

            for idx, (name, features_list) in enumerate(wake_word_data.items()):
                for feat in features_list:
                    all_features.append(feat)
                    all_labels.append(idx + 1)

            for feat in negative_features:
                all_features.append(feat)
                all_labels.append(0)

            features_array = []
            labels_array = []
            for feat, label in zip(all_features, all_labels):
                padded = self._pad_or_truncate(feat)
                features_array.append(padded)
                labels_array.append(label)

            features_tensor = torch.FloatTensor(np.array(features_array))
            labels_tensor = torch.LongTensor(labels_array)

            dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0

                for batch_features, batch_labels in dataloader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_labels).sum().item()
                    total += batch_labels.size(0)

                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    accuracy = correct / total
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            self.is_trained = True

        def predict(self, features):
            self.model.eval()
            with torch.no_grad():
                padded = self._pad_or_truncate(features)
                tensor = torch.FloatTensor(padded).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

            results = {}
            for i, name in enumerate(self.wake_word_names):
                results[name] = float(probabilities[i + 1])

            results['background'] = float(probabilities[0])

            best_wake_word = max(self.wake_word_names, key=lambda x: results[x])
            best_confidence = results[best_wake_word]

            return best_wake_word, best_confidence, results

        def _pad_or_truncate(self, features):
            if features.shape[0] >= self.max_frames:
                return features[:self.max_frames, :]
            else:
                pad_width = self.max_frames - features.shape[0]
                padding = np.zeros((pad_width, features.shape[1]))
                return np.vstack([features, padding])

        def save(self, filepath):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_wake_words': self.num_wake_words,
                'input_dim': self.input_dim,
                'max_frames': self.max_frames,
                'is_trained': self.is_trained,
                'wake_word_names': self.wake_word_names
            }, filepath)

        def load(self, filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.num_wake_words = checkpoint['num_wake_words']
            self.input_dim = checkpoint['input_dim']
            self.max_frames = checkpoint['max_frames']
            self.is_trained = checkpoint['is_trained']
            self.wake_word_names = checkpoint['wake_word_names']
            self.model = WakeWordCNN(num_classes=self.num_wake_words + 1, input_dim=self.input_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

else:
    class WakeWordCNNClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. Please install torch to use CNN-based wake word detection.")

    class MultiClassWakeWordCNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. Please install torch to use CNN-based wake word detection.")
