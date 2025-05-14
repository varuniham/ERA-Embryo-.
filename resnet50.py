from torch.utils.data import random_split, DataLoader

from torchvision import datasets, transforms # type: ignore

data_path = "C:/Users/VARUNIHA/OneDrive/Desktop/eraembryo/data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_path, transform=transform)

data_path = "C:/Users/VARUNIHA/OneDrive/Desktop/data"
dataset = datasets.ImageFolder(data_path, transform=transform)



print(dataset.classes)  # ['post_receptive', 'pre_receptive', 'receptive']


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.models as models # type: ignore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes

model = model.to(device)


import torch.optim as optim # type: ignore

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), "resnet50.py")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:.2f}%')
