import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load('cat_dog_model.pth', map_location='cpu'))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit app
st.title(" Cat vs Dog Classifier")
st.markdown("Upload an image and let the model decide!")

file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output).item()

    label = "Cat" if prediction == 0 else "Dog"
    st.success(f"Prediction: **{label}**")
