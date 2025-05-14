import torch
import torch.nn as nn
import torchvision.models as models

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Replace final layer for 3-class classification
model.fc = nn.Linear(model.fc.in_features, 3)

# Move model to device
model = model.to(device)

# Save the model
torch.save(model.state_dict(), "resnet50_embryo.pth")

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm

# Set title
st.title(" ERA Embryo Classification: Post-receptive, Pre-receptive, Receptive")
st.write("Upload an embryo image to classify it as **Post-Receptive**, **Pre-Receptive**, or **Receptive**.")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
    model.load_state_dict(torch.load("resnet50_embryo.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels
class_names = ['post_receptive', 'pre_receptive', 'receptive']

# Image upload
uploaded_file = st.file_uploader("Upload embryo image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Classify"):
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]
        st.success(f"**Prediction: {prediction.upper()}**")

