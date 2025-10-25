import os
import json
import cv2
import numpy as np


def _resolve_haarcascade():
    candidates = []
    data_path = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_path:
        candidates.append(os.path.join(data_path, "haarcascade_frontalface_default.xml"))
    # legacy Ubuntu path retained for compatibility with older installs
    candidates.append("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("Could not locate haarcascade_frontalface_default.xml; install opencv-data")


# Haar cascade face detector (bundled with OpenCV)
face_cascade = cv2.CascadeClassifier(_resolve_haarcascade())


def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlay `overlay` onto `background` at position (x, y) with alpha channel.

    background: BGR image (numpy array)
    overlay: BGRA image (numpy array) or BGR (will be treated as fully opaque)
    x, y: top-left coords in background where overlay will be placed
    overlay_size: (w, h) to resize overlay to before blending
    """
    bg_h, bg_w = background.shape[:2]

    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

    # Ensure overlay has alpha channel
    if overlay.shape[2] == 3:
        # add fully opaque alpha channel
        overlay = np.concatenate(
            [overlay, 255 * np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)],
            axis=2,
        )

    h, w = overlay.shape[0], overlay.shape[1]

    if x >= bg_w or y >= bg_h:
        # completely outside
        return background

    # Clip overlay region to background bounds
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, bg_w)
    y2 = min(y + h, bg_h)

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # Extract regions
    alpha_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (
            alpha_overlay * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]
            + alpha_background * background[y1:y2, x1:x2, c]
        )

    return background



def load_overlays(overlays_dir):
    """Load PNGs from overlays_dir and read config.json if present.

    Returns list of dicts: {"name", "image", "config"}
    """
    cfg_path = os.path.join(overlays_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Warning: couldn't load overlays config: {e}")

    overlays = []
    for fname in sorted(os.listdir(overlays_dir)):
        if not fname.lower().endswith(".png"):
            continue
        path = os.path.join(overlays_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not load overlay image {path}")
            continue
        conf = cfg.get(fname, {"scale": 0.9, "y_offset": 0.55, "x_anchor": 0.5})
        # ensure optional nudge offsets exist
        conf.setdefault("offset_x", 0)
        conf.setdefault("offset_y", 0)
        overlays.append({
            "name": fname,
            "image": img,
            "config": conf,
        })

    return overlays


def select_best_face(faces, frame_w, frame_h, weight_area=0.6, weight_center=0.4):
    """Select the best face from an array/list of (x,y,w,h) by combining
    normalized area and proximity to frame center.

    weights should sum to 1.0 (defaults 0.6 area, 0.4 center).
    Returns (x,y,w,h) or None if faces is empty.
    """
    if faces is None or len(faces) == 0:
        return None

    # Convert to list to be safe (faces may be numpy array)
    face_list = list(faces)

    areas = [w * h for (_, _, w, h) in face_list]
    max_area = max(areas) if areas else 1

    # frame center
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    # max possible distance is corner to center
    max_dist = (cx ** 2 + cy ** 2) ** 0.5

    best = None
    best_score = -1
    for (x, y, w, h), area in zip(face_list, areas):
        # area score normalized
        area_score = area / max_area if max_area > 0 else 0

        # center proximity score (1 = center, 0 = far corner)
        fx = x + w / 2.0
        fy = y + h / 2.0
        dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
        center_score = 1.0 - (dist / max_dist) if max_dist > 0 else 0

        score = weight_area * area_score + weight_center * center_score
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


def segment_person(frame, face_bbox, expand_x=2.5, expand_y=3.0, iter_count=1, center_y_bias=0.4):
    """Segment out the person around a face bounding box using GrabCut.

    This expands the face bbox to include shoulders, chest and top of head by
    default (expand_x=2.5, expand_y=3.0) and shifts the rectangle downward
    a bit (center_y_bias) so the body is included.

    frame: BGR image
    face_bbox: (x,y,w,h)
    returns mask (uint8 0/1) same height/width as frame
    """
    if face_bbox is None:
        return None

    h, w = frame.shape[:2]
    x, y, fw, fh = face_bbox

    # compute expanded rectangle centered slightly lower than face center
    cx = x + fw / 2.0
    cy = y + fh / 2.0 + fh * center_y_bias

    new_w = int(fw * expand_x)
    new_h = int(fh * expand_y)
    rx = int(cx - new_w / 2.0)
    ry = int(cy - new_h / 2.0)

    # Clip
    rx = max(0, rx)
    ry = max(0, ry)
    new_w = min(w - rx, new_w)
    new_h = min(h - ry, new_h)

    rect = (rx, ry, new_w, new_h)

    # Initialize mask and models
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    except Exception:
        # grabCut may fail on some builds; fall back to simple ellipse mask
        mm = np.zeros(frame.shape[:2], np.uint8)
        rrw = int(new_w * 0.55)
        rrh = int(new_h * 0.65)
        center = (int(cx), int(cy))
        cv2.ellipse(mm, center, (rrw, rrh), 0, 0, 360, 1, -1)
        return mm

    # mask values: 0,2 = background, 1,3 = foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Smooth edges and threshold
    mask2 = cv2.GaussianBlur(mask2.astype('float32'), (7, 7), 0)
    mask2 = (mask2 > 0.5).astype('uint8')

    return mask2


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    overlays_dir = os.path.join(os.path.dirname(__file__), "overlays")
    overlays = load_overlays(overlays_dir) if os.path.isdir(overlays_dir) else []
    if not overlays:
        print(f"No overlays loaded from {overlays_dir}. Place PNGs there.")

    current_idx = 0
    show_overlay = True
    nudge_mode = False
    show_face_box = True
    face_box_thickness = 1
    smoothed_face = None
    missing_frames = 0

    cfg_path = os.path.join(overlays_dir, "config.json")

    def save_overlays_config():
        # write current overlays' config back to config.json
        to_save = {}
        for ov in overlays:
            # copy only relevant keys
            c = ov.get("config", {})
            to_save[ov["name"]] = {
                "scale": c.get("scale", 0.9),
                "y_offset": c.get("y_offset", 0.55),
                "x_anchor": c.get("x_anchor", 0.5),
                "offset_x": round(float(c.get("offset_x", 0.0)), 3),
                "offset_y": round(float(c.get("offset_y", 0.0)), 3),
                # preserve description if present
                **({"description": c.get("description")} if c.get("description") else {}),
            }
        try:
            with open(cfg_path, "w") as f:
                json.dump(to_save, f, indent=2)
            print(f"Saved overlay config to {cfg_path}")
        except Exception as e:
            print(f"Error saving config: {e}")

    print("Controls: left/right arrows to change overlay, space to toggle overlay, q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # If multiple faces detected, pick the largest (closest) by area
        chosen = None
        if len(faces) > 0:
            # faces is a list of (x,y,w,h)
            chosen = max(faces, key=lambda r: r[2] * r[3])

        smoothing_alpha = 0.6
        hold_frames = 4
        max_jump_ratio = 0.22

        if chosen is not None:
            if smoothed_face is None:
                smoothed_face = tuple(float(v) for v in chosen)
            else:
                sx, sy, sw, sh = smoothed_face
                cx_prev = sx + sw / 2.0
                cy_prev = sy + sh / 2.0
                cx_new = chosen[0] + chosen[2] / 2.0
                cy_new = chosen[1] + chosen[3] / 2.0
                jump_x = abs(cx_new - cx_prev)
                jump_y = abs(cy_new - cy_prev)
                if (
                    jump_x > frame.shape[1] * max_jump_ratio
                    or jump_y > frame.shape[0] * max_jump_ratio
                ):
                    smoothed_face = tuple(float(v) for v in chosen)
                else:
                    smoothed_face = tuple(
                        (1.0 - smoothing_alpha) * smoothed_face[i] + smoothing_alpha * chosen[i]
                        for i in range(4)
                    )
            missing_frames = 0
        else:
            if smoothed_face is not None:
                missing_frames += 1
                if missing_frames > hold_frames:
                    smoothed_face = None
            else:
                missing_frames = 0

        if smoothed_face is not None:
            x, y, w, h = [int(round(v)) for v in smoothed_face]
            if w <= 1 or h <= 1:
                smoothed_face = None
                missing_frames = 0
                continue
            # Draw face rectangle for debugging (toggleable)
            if show_face_box:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), face_box_thickness)

            if overlays and show_overlay:
                ov = overlays[current_idx]
                img = ov["image"]
                cfg = ov.get("config", {})

                scale = cfg.get("scale", 0.9)
                y_offset = cfg.get("y_offset", 0.55)
                x_anchor = cfg.get("x_anchor", 0.5)  # 0:left, 0.5:center, 1:right

                overlay_w = int(w * scale)
                aspect = img.shape[0] / img.shape[1]
                overlay_h = int(overlay_w * aspect)

                offset_x_pct = float(cfg.get("offset_x", 0.0))
                offset_y_pct = float(cfg.get("offset_y", 0.0))
                offset_x_px = int(round((offset_x_pct / 100.0) * (w / 2.0)))
                offset_y_px = int(round((offset_y_pct / 100.0) * (h / 2.0)))

                overlay_x = int(x + (w * x_anchor) - (overlay_w * x_anchor) + offset_x_px)
                overlay_y = int(y + int(h * y_offset) + offset_y_px)

                frame = overlay_transparent(frame, img, overlay_x, overlay_y, (overlay_w, overlay_h))

                if nudge_mode:
                    # show current config for the active overlay
                    info = (
                        f"scale={scale:.2f} y_off={y_offset:.2f} x_anchor={x_anchor:.2f} "
                        f"offx={cfg.get('offset_x', 0)}% offy={cfg.get('offset_y', 0)}%"
                    )
                    cv2.putText(frame, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # Draw overlay name/status
        if overlays:
            status = f"Overlay: {overlays[current_idx]['name']} [{'ON' if show_overlay else 'OFF'}]"
        else:
            status = "No overlays"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pumpkin AI", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key == 32:  # space
            show_overlay = not show_overlay
        elif key & 0xFF == ord('f'):
            show_face_box = not show_face_box
            print("Face box" + (" ON" if show_face_box else " OFF"))
        elif key & 0xFF == ord('n'):
            # toggle nudge mode
            nudge_mode = not nudge_mode
            print("Nudge mode" + (" ON" if nudge_mode else " OFF"))
            if nudge_mode:
                print("Nudge keys: arrows move (pixels), +/- scale, a/d anchor, w/s y_offset, s to save, n to exit")
        elif key == 81 or key == 2424832:  # left arrow (different values across platforms)
            if overlays:
                if nudge_mode:
                    # move offset left
                    overlays[current_idx]["config"]["offset_x"] = round(overlays[current_idx]["config"].get("offset_x", 0.0) - 5.0, 3)
                else:
                    current_idx = (current_idx - 1) % len(overlays)
        elif key == 83 or key == 2555904:  # right arrow
            if overlays:
                if nudge_mode:
                    overlays[current_idx]["config"]["offset_x"] = round(overlays[current_idx]["config"].get("offset_x", 0.0) + 5.0, 3)
                else:
                    current_idx = (current_idx + 1) % len(overlays)
        elif key == 82 or key == 2490368:  # up arrow
            if overlays and nudge_mode:
                overlays[current_idx]["config"]["offset_y"] = round(overlays[current_idx]["config"].get("offset_y", 0.0) - 5.0, 3)
        elif key == 84 or key == 2621440:  # down arrow
            if overlays and nudge_mode:
                overlays[current_idx]["config"]["offset_y"] = round(overlays[current_idx]["config"].get("offset_y", 0.0) + 5.0, 3)
        elif overlays and nudge_mode and (key & 0xFF in (ord('+'), ord('='))):
            c = overlays[current_idx]["config"]
            c["scale"] = round(c.get("scale", 1.0) + 0.02, 3)
        elif overlays and nudge_mode and (key & 0xFF in (ord('-'), ord('_'))):
            c = overlays[current_idx]["config"]
            c["scale"] = round(max(0.01, c.get("scale", 1.0) - 0.02), 3)
        elif overlays and nudge_mode and (key & 0xFF == ord('a')):
            c = overlays[current_idx]["config"]
            c["x_anchor"] = max(0.0, round(c.get("x_anchor", 0.5) - 0.01, 3))
        elif overlays and nudge_mode and (key & 0xFF == ord('d')):
            c = overlays[current_idx]["config"]
            c["x_anchor"] = min(1.0, round(c.get("x_anchor", 0.5) + 0.01, 3))
        elif overlays and nudge_mode and (key & 0xFF == ord('w')):
            c = overlays[current_idx]["config"]
            c["y_offset"] = round(c.get("y_offset", 0.5) - 0.01, 3)
        elif overlays and nudge_mode and (key & 0xFF == ord('s')):
            # save current overlay configs
            save_overlays_config()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

