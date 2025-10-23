import os
import io
import json
import threading
import time
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np

# Import processing helpers from face1 (we'll reuse some functions)
from face1 import overlay_transparent, load_overlays, face_cascade

app = Flask(__name__)

# Global state shared between HTTP handlers and capture thread
state = {
    "overlays_dir": os.path.join(os.path.dirname(__file__), "overlays"),
    "snapshots_dir": os.path.join(os.path.dirname(__file__), "snapshots"),
    "overlays": [],
    "current_idx": 0,
    "overlay_lookup": {},
    "active_overlays": [],
    "show_face_box": False,
    "face_box_thickness": 1,
    "frame": None,
    "running": True,
    "nudge_mode": False,
    "nudge_overlay": None,
    "nudge_message": "",
    "fps": 0.0,
    "mode": "prod",
    "prod_frozen": False,
    "prod_frame": None,
    "prod_waiting": False,
    "last_snapshot_path": None,
}


def ensure_snapshot_dir():
    directory = state.get("snapshots_dir")
    if directory:
        os.makedirs(directory, exist_ok=True)


def next_snapshot_path():
    directory = state.get("snapshots_dir")
    if not directory:
        return None
    ensure_snapshot_dir()
    existing_numbers = []
    for name in os.listdir(directory):
        if not name.startswith("snap") or not name.lower().endswith(".jpg"):
            continue
        suffix = name[4:-4]
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    next_idx = max(existing_numbers) + 1 if existing_numbers else 0
    while True:
        candidate = os.path.join(directory, f"snap{next_idx:02d}.jpg")
        if not os.path.exists(candidate):
            return candidate
        next_idx += 1


def save_snapshot(frame):
    path = next_snapshot_path()
    if path is None:
        return
    try:
        cv2.imwrite(path, frame)
        state["last_snapshot_path"] = os.path.basename(path)
    except Exception:
        # Swallow errors but clear last_snapshot_path so status does not lie
        state["last_snapshot_path"] = None


def list_overlay_names():
    return [ov.get("name") for ov in state.get("overlays", [])]


def reload_overlays():
    directory = state.get("overlays_dir")
    overlays = load_overlays(directory) if directory and os.path.isdir(directory) else []
    state["overlays"] = overlays
    refresh_overlay_lookup()
    active = state.get("active_overlays", []) or []
    state["active_overlays"] = [name for name in active if get_overlay_entry(name)]
    if not overlays:
        state["current_idx"] = 0
    else:
        state["current_idx"] = state.get("current_idx", 0) % len(overlays)
    target = state.get("nudge_overlay")
    if target and not get_overlay_entry(target):
        state["nudge_overlay"] = None
        set_nudge_message("Nudge target unavailable; select another overlay.")


def persist_overlays_config():
    directory = state.get("overlays_dir")
    if not directory:
        return False
    cfg_path = os.path.join(directory, "config.json")
    data = {}
    for ov in state.get("overlays") or []:
        name = ov.get("name")
        cfg = ov.get("config", {}) or {}
        if not name:
            continue
        data[name] = {
            "scale": float(cfg.get("scale", 0.9)),
            "y_offset": float(cfg.get("y_offset", 0.55)),
            "x_anchor": float(cfg.get("x_anchor", 0.5)),
            "offset_x": int(cfg.get("offset_x", 0)),
            "offset_y": int(cfg.get("offset_y", 0)),
        }
        if "description" in cfg:
            data[name]["description"] = cfg["description"]
    tmp_path = cfg_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, cfg_path)
    refresh_overlay_lookup()
    return True


def refresh_overlay_lookup():
    overlays = state.get("overlays") or []
    state["overlay_lookup"] = {
        ov.get("name", "").lower(): ov for ov in overlays if ov.get("name")
    }


def get_overlay_entry(name):
    if not name:
        return None
    lookup = state.get("overlay_lookup") or {}
    return lookup.get(name.lower())


def get_overlay_config_snapshot(name):
    entry = get_overlay_entry(name)
    if entry is None:
        return None
    cfg = entry.get("config", {}) or {}
    return {
        "scale": float(cfg.get("scale", 0.9)),
        "y_offset": float(cfg.get("y_offset", 0.55)),
        "x_anchor": float(cfg.get("x_anchor", 0.5)),
        "offset_x": int(cfg.get("offset_x", 0)),
        "offset_y": int(cfg.get("offset_y", 0)),
    }


def set_nudge_message(text):
    state["nudge_message"] = text or ""


def ensure_nudge_target():
    target = state.get("nudge_overlay")
    entry = get_overlay_entry(target)
    if entry is not None:
        return target
    active = state.get("active_overlays") or []
    for name in reversed(active):
        if get_overlay_entry(name):
            state["nudge_overlay"] = name
            return name
    state["nudge_overlay"] = None
    return None


def add_overlay_by_name(name):
    reload_overlays()
    overlays = state.get("overlays") or []
    if not overlays:
        return None
    normalized = (name or "").lower().strip()
    if not normalized.endswith('.png'):
        normalized = f"{normalized}.png"
    entry = get_overlay_entry(normalized)
    if entry is None:
        return None
    actual_name = entry.get("name")
    active = state.setdefault("active_overlays", [])
    if actual_name in active:
        active.remove(actual_name)
    active.append(actual_name)
    if state.get("nudge_mode") and actual_name:
        state["nudge_overlay"] = actual_name
        set_nudge_message("")
    return entry


def clear_active_overlays():
    state["active_overlays"] = []
    state["current_idx"] = 0
    state["nudge_overlay"] = None
    set_nudge_message("")


def arm_snapshot_wait():
    state['prod_waiting'] = True
    state['prod_frozen'] = False
    state['prod_frame'] = None
    return True


def capture_loop():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    ensure_snapshot_dir()
    reload_overlays()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if state.get('mode') == 'dev':
            state['show_face_box'] = True

        # FPS calculation
        now = time.time()
        prev = state.get("_prev_time", None)
        if prev is None:
            state["_prev_time"] = now
        else:
            dt = now - prev
            if dt > 0:
                state["fps"] = round(1.0 / dt, 1)
            state["_prev_time"] = now

        # If in prod mode and frozen, skip capture processing and keep prod_frame
        if state.get('mode') == 'prod' and state.get('prod_frozen'):
            frozen = state.get('prod_frame')
            if frozen is not None:
                state['frame'] = frozen
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # choose best face by combined area+center heuristic
        from face1 import select_best_face

        chosen = select_best_face(faces, frame.shape[1], frame.shape[0])

        if chosen is not None:
            x, y, w, h = chosen
            if state["show_face_box"]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), state["face_box_thickness"])

            active_names = list(state.get("active_overlays", []))
            if active_names:
                for overlay_name in active_names:
                    entry = get_overlay_entry(overlay_name)
                    if entry is None:
                        continue
                    img = entry["image"]
                    cfg = entry.get("config", {})

                    scale = cfg.get("scale", 0.9)
                    y_offset = cfg.get("y_offset", 0.55)
                    x_anchor = cfg.get("x_anchor", 0.5)

                    overlay_w = int(w * scale)
                    aspect = img.shape[0] / img.shape[1]
                    overlay_h = int(overlay_w * aspect)

                    overlay_x = int(x + (w * x_anchor) - (overlay_w * x_anchor) + cfg.get("offset_x", 0))
                    overlay_y = int(y + int(h * y_offset) + cfg.get("offset_y", 0))

                    frame = overlay_transparent(frame, img, overlay_x, overlay_y, (overlay_w, overlay_h))

            # if we're in prod mode and waiting for a snapshot, freeze this frame now
            if state.get('mode') == 'prod' and state.get('prod_waiting') and not state.get('prod_frozen'):
                frozen_frame = frame.copy()
                state['prod_frame'] = frozen_frame
                state['prod_frozen'] = True
                state['prod_waiting'] = False
                save_snapshot(frozen_frame)

        # status
        if state.get('mode') != 'prod':
            active_names = state.get("active_overlays", [])
            if active_names:
                display = ", ".join(name.rsplit('.', 1)[0] for name in active_names)
                status = f"Overlays: {display}"
            else:
                status = "Overlays: none"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        state["frame"] = frame
        time.sleep(0.01)

    cap.release()


@app.route('/api/set_mode', methods=['POST'])
def api_set_mode():
    m = request.args.get('mode', request.form.get('mode', 'dev'))
    if m not in ('dev', 'prod'):
        return jsonify(success=False, error='invalid mode'), 400
    state['mode'] = m
    # changing modes clears any frozen/waiting prod state so live stream resumes
    state['prod_frozen'] = False
    state['prod_frame'] = None
    state['prod_waiting'] = False
    if m == 'dev':
        state['show_face_box'] = True
    else:
        state['show_face_box'] = False
    return jsonify(success=True, mode=m)


@app.route('/api/prod_freeze', methods=['POST'])
def api_prod_freeze():
    # direct freeze (manual): freeze the current frame immediately
    if state.get('frame') is None:
        return jsonify(success=False, error='no frame'), 400
    state['prod_frame'] = state['frame'].copy()
    state['prod_frozen'] = True
    state['prod_waiting'] = False
    return jsonify(success=True)


@app.route('/api/prod_reset', methods=['POST'])
def api_prod_reset():
    # reset to waiting-for-snapshot state
    state['prod_frozen'] = False
    state['prod_frame'] = None
    state['prod_waiting'] = False
    return jsonify(success=True)


@app.route('/api/prod_snapshot', methods=['POST'])
def api_prod_snapshot():
    if state.get('mode') != 'prod':
        return jsonify(success=False, error='snapshot only available in prod mode', mode=state.get('mode')), 409
    if state.get('prod_waiting'):
        return jsonify(success=True, waiting=True, mode=state.get('mode'))
    arm_snapshot_wait()
    return jsonify(success=True, waiting=True, mode=state.get('mode'))


@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    if state.get('mode') != 'prod':
        return jsonify(success=False, error='snapshot only available in prod mode', mode=state.get('mode')), 409
    if state.get('prod_waiting'):
        return jsonify(success=True, waiting=True, mode=state.get('mode'))
    arm_snapshot_wait()
    return jsonify(success=True, waiting=True, mode=state.get('mode'))


@app.route("/")
def index():
    return render_template("index.html")


def gen_frames():
    while True:
        frame = state.get("frame")
        if frame is None:
            time.sleep(0.01)
            continue
        # encode as jpeg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/toggle_overlay', methods=['POST'])
def api_toggle_overlay():
    if state.get('active_overlays'):
        clear_active_overlays()
    return jsonify(success=True, overlay=None, active_overlays=state.get('active_overlays', []))


@app.route('/api/next_overlay', methods=['POST'])
def api_next_overlay():
    overlays = state.get('overlays') or []
    added_name = None
    if overlays:
        state['current_idx'] = (state.get('current_idx', -1) + 1) % len(overlays)
        entry = overlays[state['current_idx']]
        added = add_overlay_by_name(entry.get('name'))
        if isinstance(added, dict):
            added_name = added.get('name')
    return jsonify(success=True, overlay=added_name, active_overlays=state.get('active_overlays', []))


@app.route('/api/prev_overlay', methods=['POST'])
def api_prev_overlay():
    overlays = state.get('overlays') or []
    added_name = None
    if overlays:
        state['current_idx'] = (state.get('current_idx', 0) - 1) % len(overlays)
        entry = overlays[state['current_idx']]
        added = add_overlay_by_name(entry.get('name'))
        if isinstance(added, dict):
            added_name = added.get('name')
    return jsonify(success=True, overlay=added_name, active_overlays=state.get('active_overlays', []))


@app.route('/api/overlay/reset', methods=['POST'])
def api_reset_overlays():
    clear_active_overlays()
    return jsonify(success=True, overlay=None, active_overlays=state.get('active_overlays', []))


@app.route('/api/overlay/<overlay_name>', methods=['POST'])
def api_set_overlay_name(overlay_name):
    overlay = add_overlay_by_name(overlay_name)
    if overlay is None:
        return jsonify(
            success=False,
            error='overlay not found',
            requested=overlay_name,
            available=list_overlay_names(),
        ), 404
    return jsonify(
        success=True,
        overlay=overlay.get('name') if isinstance(overlay, dict) else overlay,
        active_overlays=state.get('active_overlays', []),
    )


@app.route('/api/toggle_facebox', methods=['POST'])
def api_toggle_facebox():
    if state.get('mode') == 'dev':
        return jsonify(success=False, error='face box always on in dev mode'), 409
    state['show_face_box'] = not state['show_face_box']
    return jsonify(success=True, show_face_box=state['show_face_box'])


@app.route('/api/toggle_nudge', methods=['POST'])
def api_toggle_nudge():
    state['nudge_mode'] = not state.get('nudge_mode', False)
    if state['nudge_mode']:
        target = ensure_nudge_target()
        if target:
            set_nudge_message("")
        else:
            set_nudge_message("Activate an overlay to nudge.")
    else:
        state['nudge_overlay'] = None
        set_nudge_message("")
    target = state.get('nudge_overlay')
    return jsonify(
        success=True,
        nudge=state['nudge_mode'],
        target=target,
        config=get_overlay_config_snapshot(target) if target else None,
        message=state.get('nudge_message', ''),
        nudge_message=state.get('nudge_message', ''),
    )


@app.route('/api/nudge/<action>', methods=['POST'])
def api_nudge_action(action):
    if not state.get('nudge_mode'):
        set_nudge_message('Enable nudge mode first.')
        return jsonify(
            success=False,
            nudge=False,
            message=state.get('nudge_message', ''),
            nudge_message=state.get('nudge_message', ''),
        ), 409

    target = ensure_nudge_target()
    if not target:
        set_nudge_message('Select an overlay to edit.')
        return jsonify(
            success=False,
            nudge=True,
            target=None,
            message=state.get('nudge_message', ''),
            nudge_message=state.get('nudge_message', ''),
        ), 409

    entry = get_overlay_entry(target)
    if entry is None:
        set_nudge_message('Overlay not available; reload overlays.')
        return jsonify(
            success=False,
            nudge=True,
            target=None,
            message=state.get('nudge_message', ''),
            nudge_message=state.get('nudge_message', ''),
        ), 404

    cfg = entry.setdefault('config', {})
    action = (action or '').lower()

    pixel_step = 2
    scale_step = 0.02
    anchor_step = 0.01
    y_step = 0.01

    handled = True
    if action == 'offset_up':
        cfg['offset_y'] = int(cfg.get('offset_y', 0)) - pixel_step
        set_nudge_message(f"offset_y={cfg['offset_y']}")
    elif action == 'offset_down':
        cfg['offset_y'] = int(cfg.get('offset_y', 0)) + pixel_step
        set_nudge_message(f"offset_y={cfg['offset_y']}")
    elif action == 'offset_left':
        cfg['offset_x'] = int(cfg.get('offset_x', 0)) - pixel_step
        set_nudge_message(f"offset_x={cfg['offset_x']}")
    elif action == 'offset_right':
        cfg['offset_x'] = int(cfg.get('offset_x', 0)) + pixel_step
        set_nudge_message(f"offset_x={cfg['offset_x']}")
    elif action == 'scale_up':
        cfg['scale'] = round(float(cfg.get('scale', 0.9)) + scale_step, 3)
        set_nudge_message(f"scale={cfg['scale']}")
    elif action == 'scale_down':
        cfg['scale'] = round(max(0.01, float(cfg.get('scale', 0.9)) - scale_step), 3)
        set_nudge_message(f"scale={cfg['scale']}")
    elif action == 'anchor_left':
        cfg['x_anchor'] = round(max(0.0, float(cfg.get('x_anchor', 0.5)) - anchor_step), 3)
        set_nudge_message(f"x_anchor={cfg['x_anchor']}")
    elif action == 'anchor_right':
        cfg['x_anchor'] = round(min(1.0, float(cfg.get('x_anchor', 0.5)) + anchor_step), 3)
        set_nudge_message(f"x_anchor={cfg['x_anchor']}")
    elif action == 'y_offset_up':
        cfg['y_offset'] = round(float(cfg.get('y_offset', 0.55)) - y_step, 3)
        set_nudge_message(f"y_offset={cfg['y_offset']}")
    elif action == 'y_offset_down':
        cfg['y_offset'] = round(float(cfg.get('y_offset', 0.55)) + y_step, 3)
        set_nudge_message(f"y_offset={cfg['y_offset']}")
    elif action == 'save':
        ok = persist_overlays_config()
        set_nudge_message('Overlay config saved.' if ok else 'Failed to save overlay config.')
    elif action == 'reload':
        reload_overlays()
        target = ensure_nudge_target()
        if not target:
            return jsonify(
                success=True,
                nudge=True,
                target=None,
                config=None,
                message=state.get('nudge_message', ''),
                nudge_message=state.get('nudge_message', ''),
            )
        set_nudge_message('Reloaded overlays config.')
        entry = get_overlay_entry(target)
        cfg = entry.get('config', {}) if entry else {}
    else:
        handled = False

    if not handled:
        set_nudge_message(f'Unknown nudge action: {action}')
        return jsonify(
            success=False,
            nudge=True,
            target=target,
            config=get_overlay_config_snapshot(target),
            message=state.get('nudge_message', ''),
            nudge_message=state.get('nudge_message', ''),
        ), 400

    return jsonify(
        success=True,
        nudge=True,
        target=target,
        config=get_overlay_config_snapshot(target),
        message=state.get('nudge_message', ''),
        nudge_message=state.get('nudge_message', ''),
    )


@app.route('/api/status', methods=['GET'])
def api_status():
    active = list(state.get('active_overlays', []))
    primary = active[-1] if active else None
    target = state.get('nudge_overlay') if state.get('nudge_mode') else None
    return jsonify(
        fps=state.get('fps', 0.0),
        overlay=primary,
        overlays=list_overlay_names(),
        active_overlays=active,
        nudge=state.get('nudge_mode', False),
        nudge_target=target,
        nudge_config=get_overlay_config_snapshot(target) if target else None,
        nudge_message=state.get('nudge_message', ''),
        mode=state.get('mode','prod'),
        prod_frozen=state.get('prod_frozen', False),
        prod_waiting=state.get('prod_waiting', False),
        last_snapshot=state.get('last_snapshot_path'),
    )


if __name__ == '__main__':
    # start capture thread
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000)
