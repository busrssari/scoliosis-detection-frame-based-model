import cv2
import numpy as np
import os
import json
import pickle
import gc
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv

# ==========================================
# ⚙️ AYARLAR
# ==========================================
# JSON dosyanın yolu (Senin verdiğin içeriği bir dosyaya kaydet, örn: split.json)
JSON_SPLIT_PATH = "/mnt/c/Users/gtu/Desktop/scoliosis-detection/prepare-dataset/splits.json" 

OUTPUT_UNIFIED_PKL = "/mnt/c/Users/gtu/Desktop/scoliosis-detection/dataset_unified_64.pkl"
DEBUG_SAVE_IMAGES = True
OUTPUT_IMG_DEBUG_DIR = "/mnt/c/Users/gtu/Desktop/scoliosis-detection/dataset_64x64"

IMG_SIZE = (64, 64)
MODEL_NAME = "yolov8m-seg.pt"
TARGET_FPS = 30
SEQ_LEN = 64
STRIDE_TRAIN = 32
STRIDE_VAL = 32
STRIDE_TEST = 64

# ==========================================
# 🛠️ GÖRÜNTÜ İŞLEME FONKSİYONLARI (AYNI)
# ==========================================
# ... (Buradaki clean_mask, align_and_crop_centroid, resample_sequence, process_video 
# fonksiyonları az önceki kod ile BİREBİR AYNIDIR. Tekrar kopyalamana gerek yok, 
# sadece MAIN kısmı değişiyor.) ...

# Kolaylık olsun diye fonksiyonları import ettiğini veya yukarıya yapıştırdığını varsayıyorum.
# (Eğer tek dosya yapacaksan, az önceki koddaki fonksiyonları buraya yapıştır.)

def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def align_and_crop_centroid(mask, target_h, target_w):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols): return np.zeros((target_h, target_w), dtype=np.uint8)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped = mask[ymin:ymax+1, xmin:xmax+1]
    M = cv2.moments(cropped)
    if M["m00"] != 0: cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else: cX, cY = cropped.shape[1] // 2, cropped.shape[0] // 2
    h, w = cropped.shape
    scale = min((target_h * 0.85) / h, (target_w * 0.85) / w)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h <= 0 or new_w <= 0: return np.zeros((target_h, target_w), dtype=np.uint8)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    M_new = cv2.moments(resized)
    if M_new["m00"] != 0: n_cX, n_cY = int(M_new["m10"] / M_new["m00"]), int(M_new["m01"] / M_new["m00"])
    else: n_cX, n_cY = new_w // 2, new_h // 2
    final_img = np.zeros((target_h, target_w), dtype=np.uint8)
    target_cx, target_cy = target_w // 2, target_h // 2
    start_x, start_y = target_cx - n_cX, target_cy - n_cY
    end_x, end_y = start_x + new_w, start_y + new_h
    r_start_x, r_start_y = max(0, start_x), max(0, start_y)
    r_end_x, r_end_y = min(target_w, end_x), min(target_h, end_y)
    img_start_x, img_start_y = max(0, -start_x), max(0, -start_y)
    img_end_x, img_end_y = new_w - max(0, end_x - target_w), new_h - max(0, end_y - target_h)
    try: final_img[r_start_y:r_end_y, r_start_x:r_end_x] = resized[img_start_y:img_end_y, img_start_x:img_end_x]
    except: return cv2.resize(mask, (target_w, target_h))
    return final_img

def resample_sequence(frames, original_fps, target_fps):
    if len(frames) == 0: return frames
    num_frames = len(frames)
    duration = num_frames / original_fps
    target_num_frames = int(duration * target_fps)
    if target_num_frames == 0: return frames
    indices = np.linspace(0, num_frames - 1, target_num_frames).astype(int)
    return frames[indices]

def process_video(video_path, model, seq_len, stride, target_fps, save_folder=None):
    if not os.path.exists(video_path):
        print(f"❌ HATA: Video bulunamadı -> {video_path}")
        return []
        
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or np.isnan(original_fps): original_fps = 30.0
    tracker = sv.ByteTrack(frame_rate=target_fps)
    target_track_id = None 
    frames = []
    if DEBUG_SAVE_IMAGES and save_folder: os.makedirs(save_folder, exist_ok=True)
    frame_idx = 0
    last_valid_frame = None 
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model.predict(frame, classes=[0], verbose=False, retina_masks=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        selected_mask = None
        if len(detections) > 0:
            if target_track_id is None:
                areas = detections.area
                if len(areas) > 0:
                    largest_idx = np.argmax(areas)
                    target_track_id = detections.tracker_id[largest_idx]
            mask_idx = (detections.tracker_id == target_track_id)
            if not np.any(mask_idx):
                areas = detections.area
                if len(areas) > 0 and np.max(areas) > (frame.shape[0]*frame.shape[1]*0.05):
                     largest_idx = np.argmax(areas)
                     target_track_id = detections.tracker_id[largest_idx]
                     mask_idx = (detections.tracker_id == target_track_id)
            if np.any(mask_idx):
                raw_mask = detections.mask[mask_idx][0]
                mask_uint8 = (raw_mask * 255).astype(np.uint8)
                selected_mask = clean_mask(mask_uint8)
        if selected_mask is not None:
            processed_frame_uint8 = align_and_crop_centroid(selected_mask, IMG_SIZE[0], IMG_SIZE[1])
            processed_frame_uint8 = cv2.GaussianBlur(processed_frame_uint8, (3, 3), 0.5) # Soft Edge
            processed_frame = processed_frame_uint8.astype(np.float32) / 255.0
            frames.append(processed_frame)
            last_valid_frame = processed_frame
            if DEBUG_SAVE_IMAGES and save_folder and frame_idx % 5 == 0:
                save_path = os.path.join(save_folder, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(save_path, processed_frame_uint8)
        elif last_valid_frame is not None: frames.append(last_valid_frame) 
        frame_idx += 1
    cap.release()
    frames = np.array(frames)
    if len(frames) < seq_len // 2: return []
    if abs(original_fps - target_fps) > 1.0: frames = resample_sequence(frames, original_fps, target_fps)
    clips = []
    num_frames = len(frames)
    if num_frames < seq_len:
        padding = np.tile(frames[-1], (seq_len - num_frames, 1, 1))
        padded = np.vstack([frames, padding])
        clips.append(padded)
    else:
        for start in range(0, num_frames - seq_len + 1, stride):
            clip = frames[start : start + seq_len]
            clips.append(clip)
    return clips

# ==========================================
# 🚀 MAIN (JSON OKUMA GÜNCELLEMESİ)
# ==========================================
if __name__ == "__main__":
    print(f"🚀 JSON-BASED DATASET GENERATOR BAŞLIYOR")
    print(f"📂 Split JSON: {JSON_SPLIT_PATH}")
    
    # 1. JSON DOSYASINI OKU
    with open(JSON_SPLIT_PATH, 'r') as f:
        splits = json.load(f)

    try:
        model = YOLO(MODEL_NAME)
    except:
        model = YOLO("yolov8m-seg.pt")

    unified_data = {
        "train": {"sequences": [], "labels": []},
        "val":   {"sequences": [], "labels": []},
        "test":  {"sequences": [], "labels": []}
    }
    
    # 2. JSON İÇİNDEKİ LİSTEYE GÖRE İŞLE
    for split_name, files in splits.items():
        print(f"\n🔄 İşleniyor: {split_name.upper()} Seti ({len(files)} Video)")
        stride = STRIDE_TRAIN if split_name == "train" else (STRIDE_VAL if split_name == "val" else STRIDE_TEST)
        
        for v_path in tqdm(files):
            # Windows/Linux yol farkı varsa düzelt
            # v_path = v_path.replace('\\', '/') 
            
            # Etiketleme Mantığı (Dosya isminde 'hasta' geçiyor mu?)
            is_sick = ("hasta" in v_path.lower() or "scoliosis" in v_path.lower())
            label = 1 if is_sick else 0
            label_str = "HASTA" if is_sick else "SAGLIKLI"
            
            video_name = os.path.splitext(os.path.basename(v_path))[0]
            save_folder = os.path.join(OUTPUT_IMG_DEBUG_DIR, split_name, label_str, video_name)
            
            clips = process_video(v_path, model, SEQ_LEN, stride, TARGET_FPS, save_folder=save_folder)
            
            for c in clips:
                unified_data[split_name]["sequences"].append(c)
                unified_data[split_name]["labels"].append(label)
            
            gc.collect()

    # 3. KAYDET
    final_output = {}
    for key in unified_data:
        if len(unified_data[key]["labels"]) > 0:
            final_output[key] = {
                "sequences": np.array(unified_data[key]["sequences"], dtype=np.float32),
                "labels": np.array(unified_data[key]["labels"], dtype=np.longlong)
            }
            lbls = final_output[key]["labels"]
            pos = sum(lbls)
            neg = len(lbls) - pos
            print(f"📊 {key.upper()}: Toplam {len(lbls)} klip (Hasta: {pos}, Sağlıklı: {neg})")

    with open(OUTPUT_UNIFIED_PKL, "wb") as f:
        pickle.dump(final_output, f)

    print("🏁 İŞLEM TAMAM! JSON dosyasındaki plana sadık kalındı.")