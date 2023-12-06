from torchvision.models import efficientnet_v2_s
import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import v2
from torch import nn
import gdown

url1 = 'https://drive.google.com/uc?id=10MH20a4ohol_ESXmajt6Vmn_uxHIebLE'
output1 = 'Weights for Fruit Classification.pth'
gdown.download(url1, output1, quiet=False)

url2 = 'https://drive.google.com/uc?id=1-rHdFq2GXLh21DF1fWWtuUj98nzk6BL3'
output2 = 'Weights for Quality Classification.pth'
gdown.download(url2, output2, quiet=False)

# Define the preprocessing steps
preprocess = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model architectures
model_fruit_type = efficientnet_v2_s(weights='IMAGENET1K_V1',progress=True)
model_fruit_quality = efficientnet_v2_s(weights='IMAGENET1K_V1',progress=True)

# Load the state dicts into your models
# Number of classes in your dataset
num_classes = 6

# Modify the last layer
num_features = model_fruit_type.classifier[1].in_features

# Replace the final linear layer with a new one
model_fruit_type.classifier[1] = nn.Linear(num_features, num_classes)

model_fruit_type.load_state_dict(torch.load(output1, map_location=device))

# Number of classes in your dataset
num_classes = 3

# Modify the last layer
num_features = model_fruit_quality.classifier[1].in_features

# Replace the final linear layer with a new one
model_fruit_quality.classifier[1] = nn.Linear(num_features, num_classes)

model_fruit_quality.load_state_dict(torch.load(output2, map_location=device))

# Load your models to the device
model_fruit_type = model_fruit_type.to(device)
model_fruit_quality = model_fruit_quality.to(device)

# Mapping from index to label
idx_to_fruit = {0: 'Apple', 1: 'Banana', 2: 'Guava', 3: 'Lime', 4: 'Orange', 5: 'Pomegranate'}
idx_to_quality = {0: 'Bad Quality', 1: 'Good Quality', 2: 'Mixed Quality'}

# Mapping from label to index
fruit_to_idx = {v: k for k, v in idx_to_fruit.items()}
quality_to_idx = {v: k for k, v in idx_to_quality.items()}

# Put models in evaluation mode
model_fruit_type.eval()
model_fruit_quality.eval()

def predict(image):
    # Prepare your image for input to your model
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    # Pass your image tensor through your model
    with torch.no_grad():
        outputs_fruit_type = model_fruit_type(image_tensor)
        outputs_fruit_quality = model_fruit_quality(image_tensor)
    # Get the top prediction
    _, predicted_fruit_type = torch.max(outputs_fruit_type.data, 1)
    _, predicted_fruit_quality = torch.max(outputs_fruit_quality.data, 1)
    # Convert your model's output to the corresponding label
    predicted_fruit_label = idx_to_fruit[predicted_fruit_type.item()]
    predicted_quality_label = idx_to_quality[predicted_fruit_quality.item()]
    # Get the probabilities by applying the softmax function
    prob_fruit_type = torch.nn.functional.softmax(outputs_fruit_type, dim=1)[0] * 100
    prob_fruit_quality = torch.nn.functional.softmax(outputs_fruit_quality, dim=1)[0] * 100
    return predicted_fruit_label, predicted_quality_label, prob_fruit_type[predicted_fruit_type.item()].item(), prob_fruit_quality[predicted_fruit_quality.item()].item()

header_html = "<img src=https://i.imgur.com/p2H3AbF.png' style='height:250px;display: block;margin-left: auto;margin-right: auto;width: 100%;'/>"
st.markdown(header_html, unsafe_allow_html=True)
st.title("Multi-Fruit Classification and Grading")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    fruit_label, quality_label, fruit_prob, quality_prob = predict(image)
    st.write('Predicted fruit type: %s (probability: %.2f%%)' % (fruit_label, fruit_prob))
    st.write('Predicted fruit quality: %s (probability: %.2f%%)' % (quality_label, quality_prob))
