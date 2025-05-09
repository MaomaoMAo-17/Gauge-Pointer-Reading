# 仪表盘指针识别与刻度百分比估计
本项目是一种针对模拟仪表（如压力表、水表、电表等）的自动识别方案，利用深度学习进行指针和刻度线分割，通过几何分析计算指针在有效刻度范围内的位置百分比。

![figure_zh](https://github.com/user-attachments/assets/764fe938-a274-4152-bf87-ecef34ee5882)

## 🔍 背景方法对比
### 1. 数字识别（OCR）
读取表盘上的数字（适用于电子仪表）
缺点：刻度数字小且易遮挡，鲁棒性差

### 2. 指针角度 + 值映射
检测指针角度，映射到量程
缺点：需人工设定起止角，受角度影响大

### 3. 刻度分割 + 几何拟合（本项目）
分割刻度线和指针，拟合几何结构，计算百分比
优点：无需预设、鲁棒性好、通用性强

## 处理步骤：

1. 拟合椭圆重建刻度弧形
2. 计算所有刻度点角度，找到最大跳变定义起止点
3. 计算指针顶点角度 → 映射为百分比
4. 可视化结果（椭圆 + 起止点 + 角度百分比）

## ⚠️ 当前限制
当前仅支持 一个指针 + 一个刻度圈
不适用于多表盘或双指针结构
起止点判断基于经验规则，可扩展调整

## 🔍 推理代码示例
```
if result.masks is not None:
    for seg, cls in zip(result.masks.xy, result.boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            pointer_boxes.append(seg)
        elif int(cls) == 1:
            scale_boxes.append(seg)
```

## 📦 预训练模型

可以下载训练好的预训练模型：

🔗 [Download scale_segment.pt](https://huggingface.co/miyoshi4417/scale_segment/resolve/main/scale_segment.pt)

或
```
链接: https://pan.baidu.com/s/1A7XuUHMqnEvxv6KWA7m8dQ?pwd=jv44 提取码: jv44 
```

## 📐 几何计算说明
刻度线通常为圆形，但图像中因拍摄角度会变为椭圆。使用 cv2.fitEllipse 拟合更稳定：
```
ellipse = cv2.fitEllipse(np.array(scale_points, dtype=np.int32))
(cx, cy), (_, _), theta_deg = ellipse
```

角度排序后找最大跳变作为刻度弧断口：
```
scale_rot.sort(key=lambda x: x[0])
```

指针位置角度映射为百分比：
```
rx, ry = rotate_point(pointer_tip[0], pointer_tip[1], cx, cy, theta_deg)
pointer_angle = math.atan2(ry, rx)
ratio = (pointer_angle - arc_start_angle) / (arc_span + 1e-12)
```
