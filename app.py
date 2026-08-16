from flask import Flask, render_template, request, Response
from keras_preprocessing import image as kimage
from keras.models import load_model
from ultralytics import YOLO
from roug2 import calculate
import cv2
import numpy as np
import threading
import time
import json
import os
import serial

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

model_yolo = YOLO("best.pt")
model2     = load_model('model/Class2/model_Class2.h5')
model3     = load_model('model/Class3/model_Class3.h5')
model_vehicle = YOLO("yolov8x.pt")

keras_lock = threading.Lock()

import uuid
filename = f"{uuid.uuid4().hex}.png"
output_path = f"static/{filename}"

SERIAL_PORT = 'COM3'
SERIAL_BAUD = 9600
serial_lock = threading.Lock()
ser = None

def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        ser = None

def send_serial(ranks: list):

    global ser
    payload = ''.join(ranks).encode('ascii')


    if ser is None:
        return

    with serial_lock:
        try:
            ser.write(payload)
            ser.flush()
        except serial.SerialException as e:
            try:
                ser.close()
                ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
                time.sleep(2)
                ser.write(payload)
                ser.flush()
            except serial.SerialException as e2:
                print(f"[SERIAL] Reconnect failed: {e2}")

sse_clients = []
sse_lock    = threading.Lock()

def count_vehicles(filepath: str):
    print('working with vehicle count')

    img = cv2.imread(filepath)

    # Add confidence threshold here
    results = model_vehicle(img, conf=0.1)

    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
    count = 0

    for res in results:
        if res.boxes is None:
            continue

        cls_ids = res.boxes.cls.cpu().numpy()
        confs   = res.boxes.conf.cpu().numpy()

        for cls_id, conf in zip(cls_ids, confs):
            label = res.names[int(cls_id)].lower()

            # Debug print
            print(f"{label} ({conf:.2f})")

            if label in vehicle_classes and conf > 0.4:
                count += 1

    print("Vehicle count:", count)
    return count

def push_sse(payload: dict):
    data = f"data: {json.dumps(payload)}\n\n"
    with sse_lock:
        for q in sse_clients:
            q.append(data)

RANK_DURATION = {
    'a': 16,
    'b': 16,
    'c': 16,
    'd': 16,
}

cycle_thread  = None
cycle_stop    = threading.Event()
current_lanes = None
current_ranks = None

def signal_states_for_active(active_idx):
    return [4 if i == active_idx else 2 for i in range(4)]

def yellow_states_for(active_idx):
    states = [2] * 4
    states[active_idx] = 3
    return states

def run_cycle(lane_info, ranks):

    priority_order = ['a', 'b', 'c', 'd']
    rank_to_idx    = {str(r): idx for idx, r in enumerate(ranks)}

    while not cycle_stop.is_set():
        for rank in priority_order:
            if cycle_stop.is_set():
                break

            active_idx = rank_to_idx.get(rank)
            if active_idx is None:
                continue

            duration = RANK_DURATION[rank]

            green_states = signal_states_for_active(active_idx)

            for remaining in range(duration, 0, -1):
                if cycle_stop.is_set():
                    return
                push_sse({
                    "light_states": green_states,
                    "lanes":        lane_info,
                    "active_lane":  active_idx,
                    "rank":         rank,
                    "remaining":    remaining,
                    "total":        duration,
                    "phase":        "green",
                })
                time.sleep(1)

            yellow_states = yellow_states_for(active_idx)

            for remaining in range(3, 0, -1):
                if cycle_stop.is_set():
                    return
                push_sse({
                    "light_states": yellow_states,
                    "lanes":        lane_info,
                    "active_lane":  active_idx,
                    "rank":         rank,
                    "remaining":    remaining,
                    "total":        3,
                    "phase":        "yellow",
                })
                time.sleep(1)

def start_cycle(lane_info, ranks):
    """Stop any running cycle and immediately start a fresh one."""
    global cycle_thread, cycle_stop, current_lanes, current_ranks

    cycle_stop.set()
    if cycle_thread and cycle_thread.is_alive():
        cycle_thread.join(timeout=2)

    current_lanes = lane_info
    current_ranks = ranks
    cycle_stop    = threading.Event()
    cycle_thread  = threading.Thread(
        target=run_cycle, args=(lane_info, ranks), daemon=True
    )
    cycle_thread.start()

LABEL_TO_CLASS = {
    'Emergency vehicle': 1,
    'High Traffic':      2,
    'Medium Traffic':    3,
    'Low Traffic':       4,
}

def classify_image(filepath: str) -> dict:
    img_cv2      = cv2.imread(filepath)
    yolo_results = model_yolo(img_cv2, conf=0.15, verbose=False)
    vehicle_count = count_vehicles(filepath)
    detections   = []
    output_path  = None   

    for res in yolo_results:
        if res.boxes is not None and len(res.boxes) > 0:

            plotted_img = res.plot()   

            filename = os.path.basename(filepath)
            output_path = f"static/yolo_{filename}"
            cv2.imwrite(output_path, plotted_img)

            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                lbl    = res.names[cls_id]

                detections.append({
                    "label": lbl,
                    "conf": round(conf, 2)
                })

    if detections:
        return {
            "class_num":  1,
            "label":      "Emergency Vehicle",
            "emergency":  True,
            "detections": detections,
            "vehicle_count": vehicle_count,
            "image":      output_path  
        }

    img = kimage.load_img(filepath, target_size=(64, 64))
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    with keras_lock:
        r2 = model2.predict(arr, verbose=0)
        r3 = model3.predict(arr, verbose=0)

    if r2[0][0] == 0:
        label = "Low Traffic"
    elif r3[0][0] == 0:
        label = "Medium Traffic"
    else:
        label = "High Traffic"

    return {
        "class_num":  LABEL_TO_CLASS[label],
        "label":      label,
        "emergency":  False,
        "detections": [],
        "vehicle_count": vehicle_count,
        "image":      filepath   
    }


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=["POST"])
def upload():
    saved_paths = []

    for i in range(1, 5):
        img_file = request.files.get(f'image{i}')

        if not img_file or img_file.filename.strip() == "":
            print(f"[WARNING] image{i} missing or empty")
            saved_paths.append(None)
            continue

        path = f"static/{uuid.uuid4().hex}.png"
        img_file.save(path)

        print(f"[OK] Saved image{i}: {path}")
        saved_paths.append(path)

    lane_info = [None] * 4
    threads   = []

    def run(idx, path):
        try:
            lane_info[idx] = classify_image(path)
        except Exception as e:
            
            lane_info[idx] = {
                "class_num":  2,
                "label":      "High Traffic",
                "emergency":  False,
                "detections": [],
                "vehicle_count": 0,   # ✅ ADD THIS
                "image":      path
            }

    for i, path in enumerate(saved_paths):
        t = threading.Thread(target=run, args=(i, path))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    class_nums = [lane_info[i]["class_num"] for i in range(4)]
    ranks      = calculate(class_nums[0], class_nums[1], class_nums[2], class_nums[3])

    active_idx   = ranks.index('a')
    light_states = [4 if i == active_idx else 2 for i in range(4)]

    lane_info_json = [
        {
            "class_num":  lane_info[i]["class_num"],
            "label":      lane_info[i]["label"],
            "emergency":  lane_info[i]["emergency"],
            "detections": lane_info[i]["detections"],
            "vehicle_count": lane_info[i]["vehicle_count"],  
            "image":      lane_info[i]["image"],   
        }
    
        for i in range(4)
    ]
    start_cycle(lane_info_json, ranks)

    send_serial(ranks)

    return render_template(
        "traffic.html",
        light_states=light_states,
        lane_info=lane_info,
        ranks=ranks,
        active_lane=active_idx,
    )


@app.route('/stream')
def stream():
    client_queue = []
    with sse_lock:
        sse_clients.append(client_queue)

    def generate():
        try:
            while True:
                if client_queue:
                    yield client_queue.pop(0)
                else:
                    yield ": heartbeat\n\n"
                    time.sleep(1)
        except GeneratorExit:
            pass
        finally:
            with sse_lock:
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)

    return Response(generate(), mimetype='text/event-stream',
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


if __name__ == '__main__':
    app.run(debug=False, port=690)
