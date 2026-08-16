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
        #print(f"[SERIAL] Connected on {SERIAL_PORT} @ {SERIAL_BAUD}")
    except serial.SerialException as e:
        #print(f"[SERIAL] WARNING – could not open port: {e}")
        ser = None

def send_serial(ranks: list):

    global ser
    payload = ''.join(ranks).encode('ascii')

##    print(f"[SERIAL] final output ranks → {ranks}")
##    print(f"[SERIAL] binary payload     → {list(payload)}  hex={payload.hex()}")

    if ser is None:
        #print("[SERIAL] Port not open – skipping.")
        return

    with serial_lock:
        try:
            ser.write(payload)
            ser.flush()
            #print(f"[SERIAL] Sent → {payload.hex()}")
        except serial.SerialException as e:
            #print(f"[SERIAL] Write error: {e} – reconnecting…")
            try:
                ser.close()
                ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
                time.sleep(2)
                ser.write(payload)
                ser.flush()
                #print(f"[SERIAL] Reconnected & sent → {payload.hex()}")
            except serial.SerialException as e2:
                print(f"[SERIAL] Reconnect failed: {e2}")

sse_clients = []
sse_lock    = threading.Lock()

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

            # GREEN phase
            green_states = signal_states_for_active(active_idx)
            #print(f"[CYCLE] Lane {active_idx+1} (rank={rank}) → GREEN ({duration}s)  states={green_states}")

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

            # YELLOW phase
            yellow_states = yellow_states_for(active_idx)
            #print(f"[CYCLE] Lane {active_idx+1} (rank={rank}) → YELLOW (3s)  states={yellow_states}")

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
        #print("[CYCLE] Previous cycle stopped.")

    current_lanes = lane_info
    current_ranks = ranks
    cycle_stop    = threading.Event()
    cycle_thread  = threading.Thread(
        target=run_cycle, args=(lane_info, ranks), daemon=True
    )
    cycle_thread.start()
    #print("[CYCLE] New cycle started.")

# -------------------- CLASSIFICATION --------------------
LABEL_TO_CLASS = {
    'Emergency vehicle': 1,
    'High Traffic':      2,
    'Medium Traffic':    3,
    'Low Traffic':       4,
}

def classify_image(filepath: str) -> dict:
    img_cv2      = cv2.imread(filepath)
    yolo_results = model_yolo(img_cv2, conf=0.15, verbose=False)

    detections   = []
    output_path  = None   # ✅ store output image path

    for res in yolo_results:
        if res.boxes is not None and len(res.boxes) > 0:

            # ✅ DRAW BOUNDING BOXES
            plotted_img = res.plot()   # returns image with boxes

            # ✅ SAVE IMAGE
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

    # 🚨 If emergency detected
    if detections:
        return {
            "class_num":  1,
            "label":      "Emergency Vehicle",
            "emergency":  True,
            "detections": detections,
            "image":      output_path  # ✅ send image path
        }

    # ---------------- NON-YOLO PART ----------------
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
        "image":      filepath   # ✅ fallback: original image
    }

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=["POST"])
def upload():
    saved_paths = []
    for i in range(1, 5):
        img_file = request.files[f'image{i}']
        path = f"static/image{i}.png"
        img_file.save(path)
        saved_paths.append(path)

    # Classify all 4 images in parallel
    lane_info = [None] * 4
    threads   = []

    def run(idx, path):
        try:
            lane_info[idx] = classify_image(path)
        except Exception as e:
            #print(f"[ERROR] Lane {idx+1} classification failed: {e}")
            lane_info[idx] = {
                "class_num":  2,
                "label":      "High Traffic",
                "emergency":  False,
                "detections": [],
                "image":      path   # ✅ FIX: add this
            }

    for i, path in enumerate(saved_paths):
        t = threading.Thread(target=run, args=(i, path))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # Priority ranking
    class_nums = [lane_info[i]["class_num"] for i in range(4)]
    ranks      = calculate(class_nums[0], class_nums[1], class_nums[2], class_nums[3])

    active_idx   = ranks.index('a')
    light_states = [4 if i == active_idx else 2 for i in range(4)]

    # JSON-safe lane info for SSE
    lane_info_json = [
        {
            "class_num":  lane_info[i]["class_num"],
            "label":      lane_info[i]["label"],
            "emergency":  lane_info[i]["emergency"],
            "detections": lane_info[i]["detections"],
            "image":      lane_info[i]["image"],   # ✅ ADD THIS
        }
        for i in range(4)
    ]
    start_cycle(lane_info_json, ranks)

    # Send classification results to hardware once per new cycle
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




# -------------------- STARTUP --------------------
if __name__ == '__main__':
    init_serial()
    app.run(debug=False, port=2500)
