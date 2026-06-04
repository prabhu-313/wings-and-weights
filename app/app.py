import streamlit as st
import torch, timm, json, time
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
MEAN   = [0.485, 0.456, 0.406]
STD    = [0.229, 0.224, 0.225]

# ── Load class names ──────────────────────────────────────────────────────────
with open("classes.json") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load("DeiT-Tiny_phase2_best.pt", map_location=DEVICE)
    model.load_state_dict(state['model_state'])
    model.eval()
    return model

model = load_model()

# ── Transform ─────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def reshape_deit(tensor, h=14, w=14):
    r = tensor[:, 1:, :]
    r = r.reshape(r.size(0), h, w, r.size(2))
    return r.transpose(2, 3).transpose(1, 2)

@st.cache_resource
def load_cam(_model):
    return GradCAMPlusPlus(
        model=_model,
        target_layers=[_model.blocks[-1].norm1],
        reshape_transform=reshape_deit
    )

cam = load_cam(model)

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(img):
    img_t = tf(img).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        probs = torch.softmax(model(img_t), dim=1)[0]
    inf_ms = (time.time() - t0) * 1000
    top5_vals, top5_idx = torch.topk(probs, 5)

    orig_np = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    gcam    = cam(input_tensor=img_t,
                  targets=[ClassifierOutputTarget(top5_idx[0].item())])[0]
    if gcam.ndim == 3:
        gcam = gcam.mean(axis=0)
    cam_img = Image.fromarray(show_cam_on_image(orig_np, gcam, use_rgb=True))

    results = {CLASS_NAMES[i.item()].replace("_"," ").title(): float(v)
               for v, i in zip(top5_vals, top5_idx)}
    return results, cam_img, inf_ms

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🦋 Butterfly Recognition", layout="wide")
st.title("🦋 Butterfly Species Recognition")
st.markdown(f"**Model:** DeiT-Tiny &nbsp;|&nbsp; **Classes:** {NUM_CLASSES} species &nbsp;|&nbsp; **Top-1 Accuracy:** 98.00%")
st.divider()

uploaded = st.file_uploader("Upload a butterfly image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    with st.spinner("Identifying species..."):
        results, cam_img, inf_ms = predict(img)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Input Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("🔥 Grad-CAM Attention")
        st.image(cam_img, use_container_width=True)
        st.caption("Red = where the model is looking")

    with col3:
        st.subheader("🏆 Top-5 Predictions")
        for name, conf in results.items():
            st.progress(conf, text=f"{name}  —  {conf*100:.1f}%")
        st.caption(f"⚡ Inference time: {inf_ms:.1f} ms")

    top1      = list(results.keys())[0]
    top1_conf = list(results.values())[0]
    st.divider()
    st.success(f"✅ Predicted Species: **{top1}** ({top1_conf*100:.1f}% confidence)")