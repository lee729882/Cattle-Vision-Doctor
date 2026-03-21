import os
import cv2
import glob
import random
import shutil
import math
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. Configuration & Paths
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
SOURCES_DIR = ASSETS_DIR / "sources"
MASKS_DIR = ASSETS_DIR / "masks"

DATA_DIR = ROOT_DIR / "data"
# Depending on your specific setup, raw images might be in data/images/train or data/raw/images.
# We will check data/raw/images first, then fallback to data/images/train.
RAW_IMG_DIR = DATA_DIR / "raw" / "images"
RAW_LBL_DIR = DATA_DIR / "raw" / "labels"

if not RAW_IMG_DIR.exists():
    RAW_IMG_DIR = DATA_DIR / "images" / "train"
    RAW_LBL_DIR = DATA_DIR / "labels" / "train"

AUG_DIR = DATA_DIR / "augmented"
AUG_IMG_DIR = AUG_DIR / "images"
AUG_LBL_DIR = AUG_DIR / "labels"

CLASS_ID_SPIDER_WEB = 80

def setup_directories():
    """Ensure all required directories exist."""
    print("[*] Checking directories...")
    for d in [MASKS_DIR, AUG_IMG_DIR, AUG_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not SOURCES_DIR.exists():
        SOURCES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[!] Please place contaminated CCTV images in {SOURCES_DIR}")
    if not RAW_IMG_DIR.exists():
        RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_LBL_DIR.exists():
        RAW_LBL_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Spider Web Masking (Phase 1)
# ==============================================================================
def create_masks():
    """Reads contaminated images, extracts bright spider webs, and saves as RGBA masks."""
    print("=" * 50)
    print("🕸️ Phase 1: Extracting Spider Web Masks")
    print("=" * 50)
    
    source_images = list(SOURCES_DIR.glob("*.jpg")) + list(SOURCES_DIR.glob("*.png")) + list(SOURCES_DIR.glob("*.jpeg"))
    if not source_images:
        print(f"[!] No source images found in {SOURCES_DIR}.")
        return

    processed_count = 0
    for img_path in source_images:
        mask_path = MASKS_DIR / f"{img_path.stem}_mask.png"
        if mask_path.exists():
            continue  # Skip if already masked
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # 1. Convert to grayscale & enhance contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 2. Thresholding to find bright regions (spider webs)
        # Using Adaptive Thresholding to handle uneven lighting
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, -10  # Negative C extracts bright lines
        )
        
        # 3. Morphology to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 4. Create RGBA image
        b, g, r = cv2.split(img)
        # Background becomes transparent, web keeps its original color
        rgba = cv2.merge([b, g, r, mask])
        
        cv2.imwrite(str(mask_path), rgba)
        processed_count += 1
        
    print(f"✅ Generated {processed_count} new masks in {MASKS_DIR}")

# ==============================================================================
# 3. Copy-Paste Synthesis & Auto-Labeling (Phase 2 & 3)
# ==============================================================================
def overlay_transparent(background, overlay, x, y, alpha_factor=1.0):
    """Overlays an RGBA image onto an RGB background at (x, y)."""
    bg_h, bg_w = background.shape[:2]
    h, w = overlay.shape[:2]

    # Calculate bounding box on background
    y1, y2 = max(0, y), min(bg_h, y + h)
    x1, x2 = max(0, x), min(bg_w, x + w)

    # Calculate bounding box on overlay
    y1o, y2o = max(0, -y), min(h, bg_h - y)
    x1o, x2o = max(0, -x), min(w, bg_w - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return background

    overlay_image = overlay[y1o:y2o, x1o:x2o]
    alpha = (overlay_image[:, :, 3] / 255.0) * alpha_factor
    alpha_3d = np.dstack((alpha, alpha, alpha))

    bg_region = background[y1:y2, x1:x2]
    fg_region = overlay_image[:, :, :3]

    background[y1:y2, x1:x2] = (alpha_3d * fg_region + (1 - alpha_3d) * bg_region).astype(np.uint8)
    return background

def rotate_and_scale_image(image, angle, scale):
    """Rotates and scales an RGBA image, keeping the entire image in view."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    
    # Calculate bounds of new image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    
    result = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return result

def augment_single_image(args):
    """Worker function to process a single image."""
    img_path, available_masks = args
    
    img = cv2.imread(str(img_path))
    if img is None:
        return False
        
    bg_h, bg_w = img.shape[:2]
    
    # Read existing labels
    label_path = RAW_LBL_DIR / f"{img_path.stem}.txt"
    labels_data = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            labels_data = [line.strip() for line in f.readlines()]
            
    # Number of masks to apply (1 to 3)
    num_masks = random.randint(1, 3)
    applied_masks_bboxes = []
    
    for _ in range(num_masks):
        mask_path = random.choice(available_masks)
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_img is None or mask_img.shape[2] != 4:
            continue
            
        # Random transformations
        angle = random.uniform(0, 360)
        scale = random.uniform(0.5, 1.5)
        alpha = random.uniform(0.3, 0.8)
        
        mask_transformed = rotate_and_scale_image(mask_img, angle, scale)
        mh, mw = mask_transformed.shape[:2]
        
        # Random position (allow partial off-screen)
        x = random.randint(-mw // 2, bg_w - mw // 2)
        y = random.randint(-mh // 2, bg_h - mh // 2)
        
        # Overlay
        img = overlay_transparent(img, mask_transformed, x, y, alpha_factor=alpha)
        
        # Calculate Bounding Box of the inserted mask
        # Find non-transparent pixels in the transformed mask
        alpha_channel = mask_transformed[:, :, 3]
        coords = cv2.findNonZero(alpha_channel)
        
        if coords is not None:
            ox, oy, w, h = cv2.boundingRect(coords)
            
            # Translate to background coordinates
            abs_x = x + ox
            abs_y = y + oy
            
            # Clip to image boundaries
            x1 = max(0, abs_x)
            y1 = max(0, abs_y)
            x2 = min(bg_w, abs_x + w)
            y2 = min(bg_h, abs_y + h)
            
            # If visible area is reasonable
            if x2 > x1 and y2 > y1:
                cx = ((x1 + x2) / 2.0) / bg_w
                cy = ((y1 + y2) / 2.0) / bg_h
                bw = (x2 - x1) / bg_w
                bh = (y2 - y1) / bg_h
                
                applied_masks_bboxes.append((CLASS_ID_SPIDER_WEB, cx, cy, bw, bh))

    # Save augmented image
    out_img_name = f"{img_path.stem}_aug.jpg"
    out_img_path = AUG_IMG_DIR / out_img_name
    cv2.imwrite(str(out_img_path), img)
    
    # Append labels and save
    out_lbl_path = AUG_LBL_DIR / f"{img_path.stem}_aug.txt"
    with open(out_lbl_path, 'w') as f:
        # Write existing labels
        for line in labels_data:
            f.write(f"{line}\n")
        # Write new spider web labels
        for bbox in applied_masks_bboxes:
            f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
            
    return True

# ==============================================================================
# 4. Multiprocessing Management (Phase 4)
# ==============================================================================
def main():
    print("=" * 50)
    print("🚀 Spider Web Data Augmentation System")
    print("=" * 50)
    
    setup_directories()
    
    # Phase 1: Create masks
    create_masks()
    
    # Prepare Phase 2 & 3
    available_masks = list(MASKS_DIR.glob("*.png"))
    if not available_masks:
        print("[!] No spider web masks available to overlay. Please add CCTV images to 'assets/sources/'.")
        return
        
    image_paths = list(RAW_IMG_DIR.glob("*.jpg")) + list(RAW_IMG_DIR.glob("*.png")) + list(RAW_IMG_DIR.glob("*.jpeg"))
    total_images = len(image_paths)
    
    if total_images == 0:
        print(f"[!] No raw images found in {RAW_IMG_DIR}.")
        return
        
    print(f"[*] Starting copy-paste augmentation for {total_images} images...")
    
    # Prepare arguments for multiprocessing
    # Each task gets the image path and the list of available masks
    tasks = [(img_path, available_masks) for img_path in image_paths]
    
    # Run multiprocessing
    num_cores = max(1, cpu_count() - 1) # Leave one core free for OS
    print(f"[*] Using {num_cores} CPU cores for processing.")
    
    processed_count = 0
    with Pool(processes=num_cores) as pool:
        for i, success in enumerate(pool.imap_unordered(augment_single_image, tasks), 1):
            if success:
                processed_count += 1
            if i % max(1, (total_images // 10)) == 0 or i == total_images:
                print(f"    -> Progress: {i}/{total_images} ({(i/total_images)*100:.1f}%)")

    print("=" * 50)
    print(f"✅ Finished! Successfully augmented {processed_count} images.")
    print(f"📁 Augmented Images: {AUG_IMG_DIR}")
    print(f"📁 Augmented Labels: {AUG_LBL_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()
