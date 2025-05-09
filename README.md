# Gauge Pointer Reading via Scale Segmentation
This project provides a robust solution for automatic reading of analog gauges using deep learning and geometric analysis. It supports circular instruments such as pressure gauges, water meters, and electric meters by detecting both the pointer and the scale lines, then calculating the pointer's relative position as a percentage.

![image](https://github.com/user-attachments/assets/6fbcba94-8777-45fb-9f52-c9ef55adb7a8)

## 🔍 Background
Analog meter reading methods typically fall into three categories:

OCR-based digit recognition
Detect printed numbers and read values directly.
✅ Suitable for digital meters
❌ Not reliable for small or occluded analog digits

Pointer angle + value mapping
Compute pointer angle and map it to known value ranges.
✅ Simple and fast
❌ Requires manual calibration of start/end angles

Scale segmentation + geometric fitting (this project)
Segment the pointer and tick marks, fit an ellipse to the arc, and compute pointer position by angle ratio.
✅ No manual setup, robust to tilt or partial occlusion
❌ Requires training data with segmentation masks

## ✅ Project Workflow
1. Use YOLOv11-Segment to detect:

2. Aggregate all scale points and fit an ellipse

3. Detect the largest angle gap to define arc start/end

4. Compute pointer angle relative to arc

## ⚠️ Limitations
Only supports one pointer + one scale arc
Not designed for multi-pointer or multi-dial gauges
Start/end rules are empirical (can be customized per dataset)

## 📐 Geometric Analysis
Fitting the scale points into an ellipse (instead of a circle) accounts for camera tilt, lens distortion, and real-world positioning variance.
```
scale_int = np.array(scale_points, dtype=np.int32)
ellipse = cv2.fitEllipse(scale_int)
(cx, cy), (ma, MA), theta_deg = ellipse
```
Then sort angles, find the largest gap, and define the arc:
```
scale_rot.sort(key=lambda x: x[0])
```
Compute pointer angle and map to percentage:
```
rx, ry = rotate_point(pointer_tip[0], pointer_tip[1], cx, cy, theta_deg)
pointer_angle = math.atan2(ry, rx)
ratio = (pointer_angle - arc_start_angle) / (arc_span + 1e-12)
```


