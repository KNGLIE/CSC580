from multiprocessing import freeze_support
from sklearn import model_selection
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data():
    data = unpickle('cifar-10-batches-py/data_batch_1')
    data2 = unpickle('cifar-10-batches-py/data_batch_2')
    data3 = unpickle('cifar-10-batches-py/data_batch_3')
    data4 = unpickle('cifar-10-batches-py/data_batch_4')
    data5 = unpickle('cifar-10-batches-py/data_batch_5')
    test = unpickle('cifar-10-batches-py/test_batch')
    return data, data2, data3, data4, data5, test

def transform_data():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return train_transform, test_transform

def image_data(train_transform, test_transform):
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, test_loader

def display_image(image):
    image = image.numpy().transpose((1, 2, 0))
    plt.imshow(image)
    plt.show()

def mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt((channels_squared_sum / num_batches) - (mean ** 2))
    return mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

def normalize_data(data, mean, std):
    return (data - mean) / std

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 10)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define Loss Function
loss_function = torch.nn.CrossEntropyLoss()

# Define Training Loop
def train(model, loss_function, optimizer, scheduler, num_epochs, mean, std, train_loader, test_loader):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        train_loss = 0

        for images, labels in train_loader:
            images = normalize_data(images, mean, std)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(correct / total)

        test_loss, test_acc = test(model, loss_function, test_loader, mean, std)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1] * 100:.2f}%, "
              f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1] * 100:.2f}%")

    return train_losses, test_losses, train_accs, test_accs

# Define Testing Loop
def test(model, loss_function, test_loader, mean, std):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = normalize_data(images, mean, std)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / total

    return test_loss, test_acc

# Define Main Function
def main():
    train_transform, test_transform = transform_data()
    train_loader, test_loader = image_data(train_transform, test_transform)

    model = CNN().to(device)
    mean, std = mean_std(train_loader)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_losses, test_losses, train_accs, test_accs = train(model, loss_function, optimizer, scheduler, num_epochs=100,
                                                            mean=mean, std=std, train_loader=train_loader,
                                                            test_loader=test_loader)

    # Plotting training and validation curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'model.ckpt')

if __name__ == '__main__':
    freeze_support()
    main()

 
# Code to load the model
#model = CNN()
#model.load_state_dict(torch.load('model.ckpt'))
#model.eval()

