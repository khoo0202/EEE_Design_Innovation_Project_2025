import cv2

_DEF_FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colors for OpenCV
COLOR_VAPING  = (255, 0, 0)    # blue
COLOR_SMOKING = (0, 0, 255)    # red
COLOR_NONE    = (0, 200, 0)    # green
COLOR_DEFAULT = (0, 255, 0)

def _draw_label_block(img, x, y, lines, font_scale=0.5, thickness=1, pad=4,
                      fg=(255, 255, 255), bg=(0, 0, 0)):
    # measure text sizes
    max_w = 0
    line_h = 0
    for t in lines:
        (tw, th), _ = cv2.getTextSize(t, _DEF_FONT, font_scale, thickness)
        max_w = max(max_w, tw)
        line_h = max(line_h, th)
    box_w = max_w + pad * 2
    box_h = line_h * len(lines) + pad * (len(lines) + 1)

    H, W = img.shape[:2]
    x = max(0, min(W - box_w - 1, x))
    y = max(0, min(H - box_h - 1, y))

    # background rectangle
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), bg, -1)

    # draw text lines
    cy = y + pad + line_h
    for t in lines:
        cv2.putText(img, t, (x + pad, cy - 2), _DEF_FONT, font_scale, fg, thickness, cv2.LINE_AA)
        cy += line_h + pad

def _pick_color_from_probs(probs: dict):
    if not probs:
        return COLOR_DEFAULT
    vaping  = float(probs.get("vaping", 0.0))
    smoking = float(probs.get("smoking", 0.0))
    none_p  = float(probs.get("none", 0.0))
    if smoking >= vaping and smoking >= none_p:
        return COLOR_SMOKING
    if vaping >= smoking and vaping >= none_p:
        return COLOR_VAPING
    return COLOR_NONE

def draw_tracks(frame, tracks, info=None):
    H, W = frame.shape[:2]
    for tid, (x1, y1, x2, y2) in tracks:
        probs = info.get(tid, {}).get("probs") if info else None
        color = _pick_color_from_probs(probs)

        # draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 6)), _DEF_FONT, 0.6, color, 2, cv2.LINE_AA)

        # build label lines
        lines = [f"ID {tid}"]
        if probs:
            vaping  = int(round(probs.get("vaping", 0.0)  * 100))
            smoking = int(round(probs.get("smoking", 0.0) * 100))
            none_p  = int(round(probs.get("none", 0.0)    * 100))
            lines.append(f"Vape {vaping}%  Smoke {smoking}%  None {none_p}%")

        # place overlay near bbox
        bx = x2 + 6
        by = max(0, y1 - 2)
        approx_w = 200
        if bx + approx_w > W:  # near right edge → place left
            bx = max(0, x1 - approx_w - 6)
            by = max(0, y1 - 24)

        _draw_label_block(frame, bx, by, lines, font_scale=0.55, thickness=1, pad=5,
                          fg=(255, 255, 255), bg=(0, 0, 0))
    return frame
