import cv2
import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Settings
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_path = "./models/OpenAI-CLIP-ViT-Base-32.pt"
sam_model_path = "./models/MetaAI-SAM-ViT-Base-01ec64.pth"
clip_arch = "ViT-B-32"
clip_pretrained = ""  # Custom checkpoint
similarity_threshold = 0.1

input_image_path = "./images/Image - 01.png"
output_image_path = "./referral_segmented_output.png"
user_prompt = "eyes and glassess."

print("🚀 Starting segmentation process...")

# ---------- Load CLIP ----------
print("🔍 Loading CLIP model...")
model_clip, _, preprocess_clip = open_clip.create_model_and_transforms(
    clip_arch, pretrained=clip_pretrained, device=device,
)
model_clip.load_state_dict(torch.load(clip_model_path, map_location=device))
model_clip.eval()
tokenizer = open_clip.get_tokenizer(clip_arch)

# ---------- Load SAM ----------
print("🔍 Loading SAM model...")
sam = sam_model_registry["vit_b"](checkpoint=sam_model_path).to(device)
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)

# ---------- Load and preprocess image ----------
print(f"🖼️ Loading image from: {input_image_path}")
image = cv2.imread(input_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("🧠 Generating masks using SAM...")
masks = mask_generator.generate(image_rgb)
print(f"🪄 Total masks generated: {len(masks)}")

# ---------- Encode text prompt ----------
print(f"📝 Processing text prompt: \"{user_prompt}\"")
text_tokens = tokenizer([user_prompt]).to(device)
with torch.no_grad():
    text_features = model_clip.encode_text(text_tokens)

# ---------- Evaluate each mask ----------
print("🔎 Ranking masks with text using CLIP:")

segmentations = []
similarities = []

for ann in tqdm(masks, desc="Scoring masks", unit="mask"):
    segmentation = ann["segmentation"]
    masked_img = image_rgb.copy()
    masked_img[~segmentation] = 0

    ys, xs = np.where(segmentation)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    segment_crop = masked_img[y_min:y_max+1, x_min:x_max+1]
    h, w = segment_crop.shape[:2]

    # Resize keeping aspect ratio
    scale = 224 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_crop = cv2.resize(segment_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to 224x224
    padded_img = np.zeros((224, 224, 3), dtype=np.uint8)
    y_offset = (224 - new_h) // 2
    x_offset = (224 - new_w) // 2
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_crop

    pil_crop = Image.fromarray(padded_img)
    clip_input = preprocess_clip(pil_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(clip_input)
        similarity = (image_features @ text_features.T).squeeze().item()

    similarities.append(similarity)
    segmentations.append(segmentation)

# ---------- Softmax & Score Filtering ----------
similarities_tensor = torch.tensor(similarities)
softmax_scores = torch.softmax(similarities_tensor, dim=0).numpy()
scored_masks = list(zip(segmentations, softmax_scores))

for i, score in enumerate(softmax_scores):
    tqdm.write(f"📏 Mask {i:03}: Cosine similarity = {score:.4f}")

# ---------- Select masks above threshold ----------
print(f"📊 Selecting masks with similarity > {similarity_threshold}...")
filtered_masks = [seg for seg, sim in scored_masks if sim >= similarity_threshold]

# ---------- Save filtered result ----------
print("💾 Saving result...")

if filtered_masks:
    final_mask = np.any(np.stack(filtered_masks), axis=0)
    result = image_rgb.copy()
    result[~final_mask] = 0
    Image.fromarray(result).save(output_image_path)
    print(f"🎉 Segments above threshold saved to: {output_image_path}")
else:
    Image.fromarray(image_rgb).save(output_image_path)
    print(f"⚠️ No matched segment above threshold. Original image saved to: {output_image_path}")

# ---------- Save all masks overlay ----------
all_masks_overlay = np.zeros_like(image_rgb)
for ann in masks:
    m = ann['segmentation']
    color_mask = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
    all_masks_overlay[m] = color_mask
blended_all = cv2.addWeighted(image_rgb, 0.6, all_masks_overlay, 0.4, 0)
Image.fromarray(blended_all).save("all_generated_masks.png")
print("📸 All segments overlay saved to: all_generated_masks.png")