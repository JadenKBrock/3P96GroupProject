import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as f
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights
import os
from torch.optim import lr_scheduler
import torchvision.models as models
import kagglehub
from PIL import Image

# Disable some TensorFlow optimization options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
dataset_path = "./archive/PetImages"#path + "/PetImages"

def check_and_remove_corrupted_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the image file integrity
        return False  # Image is not corrupted
    except (IOError, SyntaxError) as e:
        print(f"Removing corrupted image: {file_path} - {e}")
        os.remove(file_path)  # Remove corrupted image file
        return True  # Image was corrupted and removed

def scan_and_clean_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            check_and_remove_corrupted_image(file_path)

def main():
    # Store experiment data
    experiment_data = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "version": "1.0",
        "optimizer": "Adam",
        "model": "resnet18"
    }

    # Test with different learning rates
    learning_rates = [0.01]

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")

        # Hyperparameter settings
        hyperparams = {
            "batch_size": 128,
            "learning_rate": lr,  # Use the current loop's learning rate
            "momentum": 0.9,
            "epochs": 91
        }

        params = f"Experiment_LR_{hyperparams['learning_rate']}_BS_{hyperparams['batch_size']}_MOM_{hyperparams['momentum']}_EP_{hyperparams['epochs']}_OPTIMIZER_{experiment_data['optimizer']}_MOD_{experiment_data['model']}"
        # TensorBoard log directory
        log_dir = f"runs/Cats-vs-Dogs_experiments/{experiment_data['version']}/{params}"

        # Initialize TensorBoard's SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Data preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(20),  
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        batch_size = hyperparams['batch_size']

        # Load training set
        dataset = ImageFolder(root=dataset_path, transform=transform)
        print(len(dataset))

        train_size = int(0.8 * len(dataset))
        print(f"Train size: {train_size}")
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Load test set
        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
        valloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
        print(f"Trainloader len: {len(trainloader)}")
        # Cat and Dog classes
        classes = ["Cat", "Dog"]

        # Function to display images
        def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        class net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16*53*53, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 2)

            def forward(self, x):
                x = self.pool(f.relu(self.conv1(x)))
                x = self.pool(f.relu(self.conv2(x)))
                x = torch.flatten(x, 1)
                x = f.relu(self.fc1(x))
                x = f.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = net().to(device)
        model = model.to(device)

        
        #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion = nn.CrossEntropyLoss()

        # SGD optimizer
        #optimizer = optim.SGD(model.parameters(), lr=hyperparams['learning_rate'], momentum=hyperparams['momentum'])
        
        # Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

        # Add learning rate scheduler
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)  # Multiply learning rate by 0.8 every 10 epochs

        early_stop = 10
        best_loss = float('inf')
        stopping_counter = 0

        # Training loop
        for epoch in range(hyperparams["epochs"]):
            running_loss = 0.0
            model.train()  # Set to training mode
            print(f"Epoch: {epoch}")


            for i, data in enumerate(trainloader, 0):
                print(f"I: {i}")
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()  # Zero the gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update parameters

                running_loss += loss.item()
                if i % 2000 == 1999:  # Print loss every 2000 batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

            # Record training loss
            experiment_data["train_loss"].append(running_loss / len(trainloader))
            writer.add_scalar("Loss/Train", running_loss / len(trainloader), epoch)

            # Validation loop
            model.eval()  # Set to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            val_loss /= len(valloader)
            # Record validation loss and accuracy
            experiment_data["val_loss"].append(val_loss)
            experiment_data["val_accuracy"].append(val_accuracy)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

            print(f"Epoch {epoch + 1}/{hyperparams['epochs']}: Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.2f}%")

            if val_loss < best_loss:
                best_loss = val_loss
                stopping_counter = 0
            else:
                stopping_counter += 1

            if stopping_counter >= early_stop:
                print("Early stopping triggered!")
                break

            # Update learning rate
            scheduler.step(val_loss)

        # Save model
        #torch.save(model.state_dict(), f"model_VER_{experiment_data['version']}_{params}.pth")
        writer.flush()
        writer.close()

    # Print experiment information
    print(f"Experiment info: {experiment_data}")

if __name__ == '__main__':
    directory = dataset_path + "/Cat/"
    scan_and_clean_directory(directory)
    directory = dataset_path + "/Dog/"
    scan_and_clean_directory(directory)
    main()