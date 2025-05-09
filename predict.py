from ultralytics import YOLO
import cv2
import numpy as np
import math
import os


def rotate_point(px, py, cx, cy, theta_deg):
    """
    Rotate a point (px, py) counterclockwise around (cx, cy) by theta_deg degrees.
    Used to align coordinates with the ellipse's major axis.
    """
    tx = px - cx
    ty = py - cy
    rad = math.radians(-theta_deg)
    cos_ = math.cos(rad)
    sin_ = math.sin(rad)
    rx = tx * cos_ - ty * sin_
    ry = tx * sin_ + ty * cos_
    return rx, ry


def angle_0_360_downwards(a_rad):
    """
    Normalize angle from [-pi, pi] to [0, 360) degrees.
    Uses 'downward' direction (270°) as 0°.
    """
    shifted = a_rad + math.pi / 2
    deg = math.degrees(shifted)
    return deg % 360


def draw_polygon(image, points, color, thickness=2, is_closed=True):
    """
    Draws a polygon with the given points on the image.
    """
    if len(points) >= 2:
        pts_np = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts_np], is_closed, color, thickness)


def process_yolo_result(result, image, output_file=None):
    """
    Processes a YOLO segmentation result:
    - Extracts pointer and scale polygons.
    - Fits an ellipse to the scale arc.
    - Computes pointer angle and percentage relative to the arc.
    - Optionally saves the annotated image.
    """
    h, w = image.shape[:2]
    pointer_boxes = []
    scale_boxes = []

    if result.masks is not None:
        segments = result.masks.xy
        classes = result.boxes.cls.cpu().numpy()

        for seg, cls in zip(segments, classes):
            class_id = int(cls)
            poly = [(x, y) for x, y in seg]
            if class_id == 0:
                pointer_boxes.append(poly)
            elif class_id == 1:
                scale_boxes.append(poly)
    else:
        print("No segmentation masks found. Make sure you are using a segment model.")
        return

    for poly in pointer_boxes:
        draw_polygon(image, poly, (255, 0, 0), thickness=1)
    for poly in scale_boxes:
        draw_polygon(image, poly, (0, 255, 0), thickness=1)

    scale_points = [pt for poly in scale_boxes for pt in poly]
    pointer_points = [pt for poly in pointer_boxes for pt in poly]

    if len(scale_points) < 5:
        if output_file:
            cv2.imwrite(output_file, image)
        return

    # === Fit ellipse from scale points ===
    scale_int = np.array(scale_points, dtype=np.int32)
    ellipse = cv2.fitEllipse(scale_int)
    (cx, cy), (ma, MA), theta_deg = ellipse
    cv2.ellipse(image, ellipse, (0, 255, 255), 2)
    cv2.circle(image, (int(cx), int(cy)), 5, (0, 255, 255), -1)

    # === Convert all scale points to angles ===
    scale_rot = []
    for sx, sy in scale_points:
        rx, ry = rotate_point(sx, sy, cx, cy, theta_deg)
        raw_angle = math.atan2(ry, rx)
        if raw_angle < 0:
            raw_angle += 2 * math.pi
        scale_rot.append((raw_angle, (sx, sy)))

    # === Find largest angle gap ===
    scale_rot.sort(key=lambda x: x[0])
    n = len(scale_rot)
    max_gap = 0.0
    max_idx = 0
    for i in range(n - 1):
        gap = scale_rot[i + 1][0] - scale_rot[i][0]
        if gap > max_gap:
            max_gap = gap
            max_idx = i
    wrap_gap = (scale_rot[0][0] + 2 * math.pi) - scale_rot[-1][0]
    if wrap_gap > max_gap:
        max_gap = wrap_gap
        max_idx = n - 1

    arc_start_idx = (max_idx + 1) % n
    arc_end_idx = (arc_start_idx - 1) % n
    arc_start_angle, arc_start_pt = scale_rot[arc_start_idx]
    arc_end_angle, arc_end_pt = scale_rot[arc_end_idx]

    cv2.circle(image, (int(arc_start_pt[0]), int(arc_start_pt[1])), 7, (255, 0, 255), -1)
    cv2.putText(image, "Start", (int(arc_start_pt[0]) + 5, int(arc_start_pt[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.circle(image, (int(arc_end_pt[0]), int(arc_end_pt[1])), 7, (0, 0, 255), -1)
    cv2.putText(image, "End", (int(arc_end_pt[0]) + 5, int(arc_end_pt[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if arc_end_angle < arc_start_angle:
        arc_end_angle += 2 * math.pi

    # === Extract arc span ===
    arc_list = []
    idx = arc_start_idx
    while True:
        arc_list.append(scale_rot[idx])
        if idx == arc_end_idx:
            break
        idx = (idx + 1) % n

    # === Pointer angle calculation ===
    if len(pointer_points) > 0:
        pointer_tip_raw = min(pointer_points, key=lambda p: (p[1], p[0]))
        cv2.circle(image, (int(pointer_tip_raw[0]), int(pointer_tip_raw[1])), 7, (255, 0, 0), -1)
        rx, ry = rotate_point(pointer_tip_raw[0], pointer_tip_raw[1], cx, cy, theta_deg)
        pointer_angle = math.atan2(ry, rx)
        if pointer_angle < 0:
            pointer_angle += 2 * math.pi

        arc_span = arc_end_angle - arc_start_angle
        if pointer_angle < arc_start_angle:
            ratio = 0.0
        elif pointer_angle > arc_end_angle:
            ratio = 1.0
        else:
            ratio = (pointer_angle - arc_start_angle) / (arc_span + 1e-12)

        pointer_angle_disp = angle_0_360_downwards(pointer_angle)
        if pointer_angle < arc_start_angle:
            pointer_angle_disp = 0.0

        # === Annotate percentage and angle ===
        cv2.putText(image, f"Pointer: {ratio * 100:.2f}%", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"PointerAngle: {pointer_angle_disp:.1f} deg", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)

    # === Save output ===
    if output_file:
        cv2.imwrite(output_file, image)
        print(f"Saved result: {output_file}")


def main(model_path, image_folder, output_folder):
    """
    Loads a YOLO segmentation model and processes all images in a folder.
    Saves annotated images with calculated pointer percentages.
    """
    model = YOLO(model_path)
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        base = os.path.splitext(img_file)[0]
        out_img_path = os.path.join(output_folder, f"{base}_ellipse_vis.jpg")
        results = model.predict(source=img_path, conf=0.25, imgsz=640, save=False)
        result = results[0]
        image = result.orig_img.copy()
        process_yolo_result(result, image, out_img_path)


# === Example call ===
main(
    model_path="./scale_segment.pt",
    image_folder="./data_valid",
    output_folder="./scale_results"
)
