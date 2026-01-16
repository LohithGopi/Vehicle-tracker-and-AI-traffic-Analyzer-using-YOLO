"""
Traffic Dashboard — Live overlays + Streamlit metrics
Features:
- YOLOv8 detection
- Vehicle tracking with CentroidTracker
- Bounding box width/height shrinking to match vehicle shape
- Live on-video dashboard: total vehicles, counts by type, speed violations
- Speed estimation (optional ppm calibration)
- Annotated video saved and downloadable

NOTE: Detection parameters (confidence, processing width, speed limit) are
hardcoded for a clean UI experience.
"""

import streamlit as st
import cv2
import tempfile
import time
import os
import numpy as np
from ultralytics import YOLO
from math import sqrt

# ---------------------------
# Centroid Tracker (No change needed)
# ---------------------------
class CentroidTracker:
    def __init__(self, max_distance=60, max_disappeared=30):
        self.next_object_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        oid = self.next_object_id
        self.next_object_id += 1
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        return oid

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, rects):
        input_centroids = []
        for (x1, y1, x2, y2, cls_name) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY, x1, y1, x2, y2, cls_name))

        if len(self.objects) == 0:
            outputs = []
            for cent in input_centroids:
                oid = self.register((cent[0], cent[1]))
                outputs.append((oid, cent[2], cent[3], cent[4], cent[5], cent[6]))
            return outputs

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype="float")
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                # Calculate Euclidean distance between centroids
                D[i, j] = sqrt((oc[0] - ic[0])**2 + (oc[1] - ic[1])**2)

        # Find the smallest distances and sort by row index
        rows = D.min(axis=1).argsort()
        # Find the column index corresponding to the minimum distance for each row
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        outputs = []

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            
            # Assignment successful
            oid = object_ids[row]
            ic = input_centroids[col]
            self.objects[oid] = (ic[0], ic[1]) # Update object centroid
            self.disappeared[oid] = 0 # Reset disappearance counter
            outputs.append((oid, ic[2], ic[3], ic[4], ic[5], ic[6]))
            assigned_rows.add(row)
            assigned_cols.add(col)

        # Handle unassigned objects (they have disappeared)
        unassigned_rows = set(range(0, D.shape[0])) - assigned_rows
        for row in unassigned_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        # Handle unassigned new detections (register them)
        unassigned_cols = set(range(0, D.shape[1])) - assigned_cols
        for col in unassigned_cols:
            ic = input_centroids[col]
            oid = self.register((ic[0], ic[1]))
            outputs.append((oid, ic[2], ic[3], ic[4], ic[5], ic[6]))

        return outputs

# ---------------------------
# Helpers
# ---------------------------
# Map COCO class IDs to vehicle names
COCO_VEHICLE_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}

# Factors to shrink the bounding box width/height for tighter fit
# Format: (width_factor, height_factor)
SHRINK_FACTORS = {
    "car": (0.8, 0.85),
    "motorcycle": (0.7, 0.7),
    "bus": (0.9, 0.9),
    "truck": (0.85, 0.9),
    "bicycle": (0.6, 0.6),
    "other": (0.7, 0.7)
}

def clamp(v, lo, hi):
    """Clamps a value within a given range."""
    return max(lo, min(hi, v))

def estimate_speed_pixels(prev_center, prev_time, cur_center, cur_time, ppm=None):
    """
    Estimates speed based on pixel movement over time.
    Returns speed in km/h.
    """
    dt = cur_time - prev_time
    if dt <= 0: return 0.0
    
    dx = cur_center[0] - prev_center[0]
    dy = cur_center[1] - prev_center[1]
    pix_distance = sqrt(dx*dx + dy*dy)
    
    # Using a placeholder conversion (0.02 meters/pixel) for demonstration
    meters = pix_distance * 0.02 if ppm is None else pix_distance / ppm 
    
    # Convert m/s to km/h (m/s * 3.6)
    return meters / dt * 3.6

def draw_dashboard(frame, records, overspeed_ids, width):
    """Draws a semi-transparent dashboard at the top of the video."""
    overlay = frame.copy()
    
    # Dashboard background
    cv2.rectangle(overlay, (0, 0), (width, 85), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Calculate stats
    total_vehicles = len(records)
    vehicle_types = [r['type'] for r in records.values()]
    n_cars = vehicle_types.count("car")
    n_trucks = vehicle_types.count("truck")
    n_buses = vehicle_types.count("bus")
    n_bikes = vehicle_types.count("motorcycle") + vehicle_types.count("bicycle")
    n_violations = len(overspeed_ids)
    
    # Draw Text Stats
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    
    # Row 1: Title & Total
    cv2.putText(frame, "TRAFFIC MONITORING SYSTEM", (20, 30), font, 0.7, yellow, 2)
    cv2.putText(frame, f"TOTAL: {total_vehicles}", (width - 150, 30), font, 0.7, white, 2)
    
    # Row 2: Type Counts
    stats_text = f"Car: {n_cars} | Truck: {n_trucks} | Bus: {n_buses} | Bike: {n_bikes}"
    cv2.putText(frame, stats_text, (20, 65), font, 0.55, white, 1)
    
    # Row 2 Right: Violations
    viol_text = f"VIOLATIONS: {n_violations}"
    cv2.putText(frame, viol_text, (width - 200, 65), font, 0.55, red, 2)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("🚦 Smart Traffic Analyzer")
st.markdown("Automated vehicle classification, tracking, and speed enforcement.")


# --- HARDCODED ANALYSIS PARAMETERS ---
min_conf = 0.35      # Default minimum detection confidence
process_width = 640  # Default width for faster processing
speed_limit = 60     # Default speed limit in km/h
# -------------------------------------


# Minimal Sidebar UI
with st.sidebar:
    st.header("Control Panel")
    uploaded = st.file_uploader("Upload Video File", type=["mp4","avi","mov"])
    start_btn = st.button("Start Analysis")
    st.markdown("---")
    st.info(f"Speed Limit set to **{speed_limit} km/h**")


tracker = CentroidTracker(max_distance=60)

@st.cache_resource
def load_yolo_model(): 
    """Loads the YOLOv8n model once."""
    return YOLO("yolov8n.pt")

yolo = load_yolo_model()

# Placeholders for the video display
video_placeholder = st.empty()

if uploaded and start_btn:
    # Setup temporary file and video capture/writer
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(uploaded.name)[1])
    tmp.write(uploaded.read()); tmp.flush()
    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(tempfile.gettempdir(),f"processed_traffic_{int(time.time())}.mp4")
    writer = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))

    # Tracking and metrics variables
    records,last_centers,seen_ids,overspeed_ids = {},{},set(),set()

    st.success("Initializing analysis engine...")
    progress_bar = st.progress(0)
    frame_idx = 0

    while True:
        ret,frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        if frame_count > 0:
            progress_bar.progress(frame_idx / frame_count)

        # Resize for faster processing logic (inference)
        h0,w0 = frame.shape[:2]
        scale = process_width/float(w0)
        small = cv2.resize(frame,(process_width,int(h0*scale)))
        rgb = small[:,:,::-1]

        # YOLOv8 Prediction
        results = yolo.predict(rgb,imgsz=process_width,conf=min_conf,verbose=False)
        detections = []
        
        try:
            res = results[0]
            if hasattr(res,"boxes") and res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                
                for (x1_s,y1_s,x2_s,y2_s),cls_id,conf in zip(boxes,classes,scores):
                    if conf<min_conf: continue
                    
                    # Rescale coordinates back to original frame size
                    x1_o=int(x1_s/scale); y1_o=int(y1_s/scale)
                    x2_o=int(x2_s/scale); y2_o=int(y2_s/scale)
                    
                    cls_name=COCO_VEHICLE_MAP.get(int(cls_id),"other")
                    w_factor, h_factor = SHRINK_FACTORS.get(cls_name, (0.7, 0.7))
                    
                    # Apply bounding box shrinking for both Width and Height
                    cx=(x1_o+x2_o)//2
                    cy=(y1_o+y2_o)//2
                    
                    orig_w = x2_o - x1_o
                    orig_h = y2_o - y1_o
                    
                    new_w = int(orig_w * w_factor)
                    new_h = int(orig_h * h_factor)
                    
                    # Clamp coordinates to frame boundaries
                    x1_new=clamp(cx-new_w//2,0,width-1)
                    x2_new=clamp(cx+new_w//2,0,width-1)
                    y1_new=clamp(cy-new_h//2,0,height-1)
                    y2_new=clamp(cy+new_h//2,0,height-1)
                    
                    if x2_new<=x1_new: x2_new=x1_new+1
                    if y2_new<=y1_new: y2_new=y1_new+1
                    
                    detections.append((x1_new,y1_new,x2_new,y2_new,cls_name))
        except Exception: 
             pass

        # Centroid Tracking Update
        tracked=tracker.update(detections)
        current_frame_time = time.time()
        
        # Process tracked objects
        for tid,x1,y1,x2,y2,cls_name in tracked:
            cx=(x1+x2)//2; cy=(y1+y2)//2
            
            if tid not in records: 
                records[tid]={"type":cls_name,"max_speed":0.0,"violations":set()}
                
            # Speed Estimation
            if tid in last_centers:
                (prev_cx,prev_cy),prev_t = last_centers[tid]
                speed_kmh = estimate_speed_pixels((prev_cx,prev_cy),prev_t,(cx,cy),current_frame_time)
            else: 
                speed_kmh=0.0
                
            last_centers[tid]=((cx,cy),current_frame_time)
            records[tid]["max_speed"]=max(records[tid]["max_speed"],speed_kmh)
            seen_ids.add(tid)
            
            # Violation Check
            is_overspeed = speed_kmh > speed_limit
            if is_overspeed:
                records[tid]["violations"].add("overspeed")
                overspeed_ids.add(tid)
            
            # --- Visualization ---
            has_violation = len(records[tid]["violations"]) > 0
            box_color = (0, 0, 255) if has_violation else (0, 255, 0)
            
            # 1. Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # 2. Label Background (for better readability)
            label = f"ID:{tid} {cls_name.upper()}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), box_color, -1)
            
            # 3. Label Text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 4. Speed Text
            speed_text = f"{speed_kmh:.1f} km/h"
            cv2.putText(frame, speed_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if is_overspeed:
                 cv2.putText(frame, "SPEED LIMIT!", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw the Dashboard on Top
        draw_dashboard(frame, records, overspeed_ids, width)

        writer.write(frame)
        video_placeholder.image(frame[:,:,::-1],caption=f"Processing Frame {frame_idx}",use_column_width=True)

    cap.release()
    writer.release()
    progress_bar.progress(1.0)
    
    st.success("Analysis Complete!")
    
    # Download Button
    with open(out_path,"rb") as f:
        st.download_button(
            label="⬇️ Download Processed Video",
            data=f.read(),
            file_name="traffic_analysis_output.mp4",
            mime="video/mp4"
        )
    
    try:
        os.unlink(tmp.name)
        os.unlink(out_path)
    except Exception:
        pass

else: 
    st.info("Upload a video to begin analysis.")
    st.write("The processed video will include an on-screen dashboard with vehicle counts and speed alerts.") 
