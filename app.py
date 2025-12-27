import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from ultralytics import YOLO

# ==============================================================
# MODEL PATHS
# ==============================================================
CLASSIFICATION_MODEL_PATH = "clinicscan_classifier_15class.pth"
DETECTION_MODEL_PATH = "runs/detect/train38/weights/best.pt"

# ==============================================================
# LOAD RESNET50 CLASSIFIER (15 CLASSES)
# ==============================================================
num_classes = 15

def load_classifier():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

classifier = load_classifier()

# ==============================================================
# LOAD YOLO
# ==============================================================
det_model = YOLO(DETECTION_MODEL_PATH)

# ==============================================================
# PREPROCESSING
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==============================================================
# LABELS & DESCRIPTIONS
# ==============================================================
CLASS_LABELS = {
    0:'Aortic enlargement',1:'Atelectasis',2:'Calcification',3:'Cardiomegaly',
    4:'Consolidation',5:'ILD',6:'Infiltration',7:'Lung Opacity',8:'Nodule/Mass',
    9:'Other lesion',10:'Pleural effusion',11:'Pleural thickening',
    12:'Pneumothorax',13:'Pulmonary fibrosis',14:'No finding'
}

DESCRIPTIONS = {
    "Aortic enlargement":"Enlargement of the aorta.",
    "Atelectasis":"Collapsed lung tissue.",
    "Calcification":"Calcium deposits in tissues.",
    "Cardiomegaly":"Enlarged heart.",
    "Consolidation":"Fluid-filled lung tissue.",
    "ILD":"Interstitial lung disease.",
    "Infiltration":"Inflammatory lung shadows.",
    "Lung Opacity":"Diffuse hazy lung regions.",
    "Nodule/Mass":"Rounded lung lesion.",
    "Other lesion":"Unclassified abnormality.",
    "Pleural effusion":"Fluid around lungs.",
    "Pleural thickening":"Thickened pleural lining.",
    "Pneumothorax":"Collapsed lung due to air.",
    "Pulmonary fibrosis":"Lung scarring.",
    "No finding":"No abnormality detected."
}

# ==============================================================
# GRAD-CAM IMPLEMENTATION (ResNet50)
# ==============================================================
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1].conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax()
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = cam.detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam

def overlay_heatmap(img, cam):
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

# ==============================================================
# STREAMLIT UI
# ==============================================================
st.set_page_config(page_title="CliniScan", layout="wide")
st.title("🩺 CliniScan – Chest X-ray Analysis")

uploaded = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

# Zoom control
zoom = st.slider("Zoom Image (%)", 40, 150, 80)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    # ================= YOLO FIRST =================
    st.subheader("📦 YOLO Detection (Primary)")
    results = det_model(img_np, conf=0.30, iou=0.5)[0]
    yolo_labels = [det_model.names[int(b.cls[0])] for b in results.boxes]

    if yolo_labels:
        unique = list(set(yolo_labels))
        st.success(f"Detected: {', '.join(unique)}")

        for u in unique:
            st.write(f"**{u}:** {DESCRIPTIONS[u]}")

        annotated = results.plot()
        st.image(annotated, width=int(img_np.shape[1] * zoom / 100))

    # ================= FALLBACK =================
    else:
        st.warning("YOLO found no localized abnormality. Using classification + Grad-CAM.")

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            out = classifier(tensor)
            pred = out.argmax().item()

        label = CLASS_LABELS[pred]
        st.success(f"Predicted: {label}")
        st.write(DESCRIPTIONS[label])

        # ---- SHOW ORIGINAL IMAGE ----
        st.image(img_np, caption="Original X-ray", width=int(img_np.shape[1] * zoom / 100))

        # ---- GRAD-CAM ----
        cam = generate_gradcam(classifier, tensor)
        heatmap = overlay_heatmap(img_np, cam)

        st.subheader("🔥 Grad-CAM Visualization")
        st.image(heatmap, caption="Model Attention Map", width=int(img_np.shape[1] * zoom / 100))
