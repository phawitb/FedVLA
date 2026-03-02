import cv2
import os
import numpy as np
import json

# --- 1. Input JSON Path ---
json_path = "/Users/phawit/Documents/Research/FedVLA/V3/reports/sim_eval_20260302_012644/summary.json"

# --- 2. Setup (สคริปต์จะคำนวณพาธอื่นจากตำแหน่ง JSON) ---
base_path = os.path.dirname(json_path)
output_dir = os.path.join(base_path, "combined_results")
final_output_file = os.path.join(output_dir, "research_summary_with_status.mp4")

with open(json_path, 'r') as f:
    data = json.load(f)

methods_config = [
    {'key': 'central', 'label': 'Centralize', 'color': (0, 255, 0)},
    {'key': 'fedavg',  'label': 'FedAvg',    'color': (0, 255, 255)},
    {'key': 'fedvla',  'label': 'FedVLA',    'color': (255, 255, 0)}
]
tasks = ['door-lock-v3', 'drawer-close-v3', 'window-open-v3']
ep_numbers = range(1, 11)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_and_preprocess(path):
    cap = cv2.VideoCapture(path)
    frames = []
    if not cap.isOpened(): return []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame, (480, 360))
        frames.append(frame)
    cap.release()
    return frames

# --- 3. Start Processing ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(final_output_file, fourcc, 24.0, (1440, 1080))

print(f"Reading JSON and creating video: {final_output_file}")

for ep_num in ep_numbers:
    ep_file = f"ep_{ep_num:03d}.mp4"
    print(f"Processing Episode {ep_num}...")
    
    grid_data = [] 
    max_frames = 0
    status_map = {} # เก็บสถานะ Success/Fail

    # โหลดวิดีโอและดึงสถานะจาก JSON
    for r_idx, task in enumerate(tasks):
        row_clips = []
        for c_idx, m_info in enumerate(methods_config):
            m_key = m_info['key']
            
            # ดึงข้อมูลจาก JSON
            try:
                ep_data = data[m_key]['tasks'][task]['details'][ep_num-1]
                is_success = ep_data['success']
                video_rel_path = ep_data['video'] # เช่น "videos/central/..."
                full_video_path = os.path.join(base_path, video_rel_path)
            except (KeyError, IndexError):
                is_success = None
                full_video_path = ""

            frames = load_and_preprocess(full_video_path)
            max_frames = max(max_frames, len(frames))
            
            row_clips.append(frames)
            status_map[(r_idx, c_idx)] = is_success
            
        grid_data.append(row_clips)

    if max_frames == 0: continue

    # สร้าง Grid เฟรมต่อเฟรม
    for i in range(max_frames):
        rows = []
        for r_idx, task_name in enumerate(tasks):
            cols = []
            for c_idx, m_info in enumerate(methods_config):
                video = grid_data[r_idx][c_idx]
                
                # Frame & Freeze Logic
                if i < len(video):
                    frame = video[i].copy()
                else:
                    frame = video[-1].copy() if video else np.zeros((360, 480, 3), dtype=np.uint8)

                # --- Draw Labels ---
                # 1. Method & EP (ด้านบน)
                cv2.putText(frame, f"{m_info['label']} | EP:{ep_num}", (15, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, m_info['color'], 2, cv2.LINE_AA)
                
                # 2. Task Name (ด้านล่าง)
                cv2.putText(frame, task_name, (15, 340), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # 3. Success/Fail Status (มุมขวาล่าง)
                status = status_map.get((r_idx, c_idx))
                if status is True:
                    text, color = "SUCCESS", (0, 255, 0)
                elif status is False:
                    text, color = "FAILED", (0, 0, 255)
                else:
                    text, color = "N/A", (128, 128, 128)
                
                # วาดพื้นหลังสีดำเล็กๆ ให้ตัวอักษรสถานะอ่านง่าย
                cv2.rectangle(frame, (330, 315), (465, 350), (0,0,0), -1)
                cv2.putText(frame, text, (340, 340), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                cols.append(frame)
            rows.append(np.hstack(cols))
        
        out.write(np.vstack(rows))

out.release()
print(f"\nFinished! Video with status overlays saved at: {final_output_file}")