import os, time, argparse
from PIL import Image
import numpy as np
from openai import OpenAI
import joblib
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision import transforms
from torchvision.utils import save_image as imwrite
from utils.utils import print_args, load_restore_ckpt, load_embedder_ckpt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import base64
import gradio as gr

transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_feature(img_path):
    """Trích vector đặc trưng từ ảnh bằng CLIP"""
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.cpu().numpy().flatten()

# ========================
# Dataset + Classifier
# ========================
def build_dataset(root_dir):
    """Trích toàn bộ features cho dataset"""
    X, y, classes = [], [], sorted(os.listdir(root_dir))
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        for img_file in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_file)
            feat = extract_feature(img_path)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y), classes


def load_or_train_classifier(train_dir, test_dir, model_path="clip_degradation_classifier.pkl"):
    """Nếu có .pkl thì load, ngược lại train mới, nhưng luôn in classification_report trên test set"""
    if os.path.exists(model_path):
        print(f"✅ Found existing model at {model_path}, loading...")
        clf, classes = joblib.load(model_path)

        # Dù load sẵn thì vẫn evaluate lại
        _, _, classes_test = build_dataset(test_dir)
        X_test, y_test, _ = build_dataset(test_dir)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=classes))
    else:
        print("⚡ Training new model...")
        X_train, y_train, classes = build_dataset(train_dir)
        X_test, y_test, _ = build_dataset(test_dir)

        print("Train size:", X_train.shape, " Test size:", X_test.shape)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)

        # Save model
        joblib.dump((clf, classes), model_path)

        # Evaluate
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=classes))

    return clf, classes


def predict_image(img_path, clf, classes):
    """Dự đoán class degradation cho ảnh bất kỳ"""
    feat = extract_feature(img_path)
    pred = clf.predict([feat])[0]
    return classes[pred]



def generate_caption(img_path, category, img_id):
    """
    Generate an objective caption using the raw image as input.
    The LLM analyzes perceptual degradations directly from the image.
    """
    prompt = f"""
    You are an image degradation analysis assistant.

    Task:
    Analyze the image "{img_id}" and explain why it belongs to the category "{category}".
    Base your explanation only on what you observe in the image.

    Guidelines:
    1. Evaluate key perceptual properties: brightness, contrast, texture, sharpness, and clarity. For each, briefly explain how it appears in the image.
    2. Describe observable degradations in simple, objective terms (e.g., blurring, dimness, loss of detail, washed-out colors) and explain their impact on visibility.
    3. Determine the severity level of "{category}" in the image (e.g., low, moderate, strong) and justify your choice.
    4. Keep the explanation concise: 3–4 sentences, around 80-100 words.
    5. Conclude with a justification that clearly links the observed degradations to the "{category}" label.
    6. Do not mention technical details such as embeddings, features, or statistics.
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    try:
        image_base64 = encode_image(img_path)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
        )
        caption = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ LLM error on {img_id}: {e}")
        caption = ""

    return caption
     

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_on_image(image, prompt=None, concat=False):
    lq = image.convert('RGB')
    lq_tensor = transforms.ToTensor()(lq).unsqueeze(0).to(device)
    lq_em = transform_resize(lq).unsqueeze(0).to(device)

    # --- Predict or use prompt
    if prompt is None:
        # Lưu ảnh tạm vào bộ nhớ để sử dụng hàm có sẵn
        import io
        temp_path = 'temp_upload_image.png'
        lq.save(temp_path)
        pred_category = predict_image(temp_path, clf, classes)
        caption = generate_caption(temp_path, pred_category, 'uploaded_image')
        text_embedding, _, _ = embedder([caption], 'text_encoder')
    else:
        text_embedding, _, [pred_category] = embedder([prompt], 'text_encoder')
        caption = prompt

    # --- Restore image
    with torch.no_grad():
        restored = restorer(lq_tensor, text_embedding)
        if concat:
            restored = torch.cat((lq_tensor, restored), dim=3)

    restored_image = restored.squeeze(0).cpu()
    restored_pil = transforms.ToPILImage()(restored_image.clamp(0,1))
    return restored_pil, pred_category, caption

# --- Gradio Interface
iface = gr.Interface(
    fn=run_on_image,
    inputs=[
        gr.Image(type='pil', label='Input Image'),
        gr.Textbox(label='Prompt (optional)', placeholder='Leave empty for automatic degradation analysis'),
        gr.Checkbox(label='Concatenate Input + Output', value=False)
    ],
    outputs=[
        gr.Image(type='pil', label='Restored Image'),
        gr.Textbox(label='Predicted Category'),
        gr.Textbox(label='Caption / Explanation')
    ],
    title='OneRestore Gradio Demo',
    description='Upload an image and get the restored version using OneRestore. You can optionally provide a prompt or let the system automatically detect degradation.'
)

iface.launch(share=True)