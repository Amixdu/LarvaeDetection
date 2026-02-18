import cv2
import numpy as np
import random
import os
import yaml
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
NUM_TRAIN_IMAGES = 800
NUM_VAL_IMAGES = 200
LARVAE_PER_IMAGE = (5, 40) # Random amount between 5 and 40
IMAGE_SIZE = 640           # Standard YOLO resolution

ASSETS_BG = "assets/backgrounds"
ASSETS_FG = "assets/foregrounds"
OUTPUT_DIR = "Larvae_Synthetic_Dataset"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def rotate_image_with_alpha(image, angle):
    """Rotates a transparent PNG and expands the canvas so it isn't cropped."""
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    rotated = cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def blend_transparent(bg, overlay, x, y):
    """Pastes a transparent PNG onto the background and returns the bounding box."""
    h, w = overlay.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    y1, y2 = max(0, y), min(bg_h, y + h)
    x1, x2 = max(0, x), min(bg_w, x + w)

    oy1, oy2 = max(0, -y), min(h, bg_h - y)
    ox1, ox2 = max(0, -x), min(w, bg_w - x)

    if y1 >= y2 or x1 >= x2:
        return bg, None

    overlay_cropped = overlay[oy1:oy2, ox1:ox2]
    
    alpha_s = overlay_cropped[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        bg[y1:y2, x1:x2, c] = (alpha_s * overlay_cropped[:, :, c] +
                               alpha_l * bg[y1:y2, x1:x2, c])
                               
    return bg, (x1, y1, x2 - x1, y2 - y1)

# ==========================================
# GENERATOR LOGIC
# ==========================================
def generate_split(split_name, num_images, bgs, fgs, out_dir):
    img_dir = Path(out_dir) / "images" / split_name
    lbl_dir = Path(out_dir) / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {split_name} set ({num_images} images)...")
    
    for i in tqdm(range(num_images)):
        # 1. Pick a random background
        bg_path = random.choice(bgs)
        bg = cv2.imread(str(bg_path))
        bg = cv2.resize(bg, (IMAGE_SIZE, IMAGE_SIZE))
        
        labels = []
        num_larvae = random.randint(*LARVAE_PER_IMAGE)
        
        for _ in range(num_larvae):
            # 2. Pick a random larva
            fg_path = random.choice(fgs)
            fg_filename = fg_path.name.lower()
            fg = cv2.imread(str(fg_path), cv2.IMREAD_UNCHANGED)
            
            # ==========================================
            # DYNAMIC SCALING LOGIC
            # ==========================================
            if "dark" in fg_filename:
                # Tiny, dark wiggling creatures
                scale = random.uniform(0.01, 0.035)
            elif "yellow" in fg_filename:
                # High contrast, standard size
                scale = random.uniform(0.04, 0.08)
            else:
                # Original translucent, standard size
                scale = random.uniform(0.04, 0.08)
            # ==========================================
                
            new_w = max(1, int(IMAGE_SIZE * scale))
            aspect_ratio = fg.shape[0] / fg.shape[1]
            new_h = max(1, int(new_w * aspect_ratio))
            fg_resized = cv2.resize(fg, (new_w, new_h))
            
            # Random Rotation
            angle = random.randint(0, 360)
            fg_rotated = rotate_image_with_alpha(fg_resized, angle)
            
            # Random Position
            x = random.randint(-new_w//2, IMAGE_SIZE - new_w//2)
            y = random.randint(-new_h//2, IMAGE_SIZE - new_h//2)
            
            # 3. Paste and get bounding box
            bg, bbox = blend_transparent(bg, fg_rotated, x, y)
            
            if bbox:
                bx, by, bw, bh = bbox
                
                x_center = (bx + bw / 2.0) / IMAGE_SIZE
                y_center = (by + bh / 2.0) / IMAGE_SIZE
                norm_w = bw / IMAGE_SIZE
                norm_h = bh / IMAGE_SIZE
                
                x_center, y_center = min(x_center, 1.0), min(y_center, 1.0)
                norm_w, norm_h = min(norm_w, 1.0), min(norm_h, 1.0)
                
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # 4. Save Outputs
        base_filename = f"syn_{split_name}_{i}"
        cv2.imwrite(str(img_dir / f"{base_filename}.jpg"), bg)
        with open(lbl_dir / f"{base_filename}.txt", "w") as f:
            f.write("\n".join(labels))

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    bgs = list(Path(ASSETS_BG).glob("*.jpg")) + list(Path(ASSETS_BG).glob("*.png"))
    fgs = list(Path(ASSETS_FG).glob("*.png"))
    
    if not bgs or not fgs:
        print("❌ ERROR: You must put background images in assets/backgrounds and transparent PNGs in assets/foregrounds")
        return

    generate_split("train", NUM_TRAIN_IMAGES, bgs, fgs, OUTPUT_DIR)
    generate_split("val", NUM_VAL_IMAGES, bgs, fgs, OUTPUT_DIR)

    yaml_data = {
        "path": f"../{OUTPUT_DIR}",
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "larva"
        }
    }
    
    yaml_path = Path(OUTPUT_DIR) / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    print(f"\n✅ Dataset successfully generated at: ./{OUTPUT_DIR}")
    print("🎯 You can now train YOLO with: yolo train data=Larvae_Synthetic_Dataset/data.yaml model=yolo11n.pt epochs=30 imgsz=640 batch=16")

if __name__ == "__main__":
    main()