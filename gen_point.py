import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box

def generate_pixels(image):
    img = cv2.imread(image)
    img_copy = img.copy()
    points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click 2 points for calibration", img_copy)

    cv2.imshow("Click 2 points for calibration", img_copy)
    cv2.setMouseCallback("Click 2 points for calibration", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"标定线两端点像素坐标: {points}")
    (x1, y1), (x2, y2) = points
    pixel_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print(f"标定线像素长度: {pixel_length:.2f} pixels")
    real_length_cm = float(input("请输入该线段的真实长度 (单位 cm): "))
    per_pixel = real_length_cm / pixel_length
    print(f"换算系数: 1 pixel = {per_pixel:.4f} cm")
    return per_pixel

def generate_shape(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # 二值化
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大轮廓
    cnt = max(contours, key=cv2.contourArea)

    # 多边形逼近
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx = cv2.flip(approx, 0)  # 翻转点数组

    # 提取 x 和 y 坐标（像素）
    outline_x_pixel = approx[:, 0, 0].tolist()
    outline_y_pixel = (img.shape[0] - approx[:, 0, 1]).tolist()
    pixel_per_cm = generate_pixels(image)
    outline_x_real = [x * pixel_per_cm for x in outline_x_pixel]
    outline_y_real = [y * pixel_per_cm for y in outline_y_pixel]
    print("outline_x_real =", [round(x) for x in outline_x_real])
    print("outline_y_real =", [round(y) for y in outline_y_real])
    return [round(x) for x in outline_x_real] , [round(y) for y in outline_y_real]

def fill_shape(image, square_len):
    outline_x, outline_y = generate_shape(image)
    contour = Polygon(zip(outline_x, outline_y))

    # === 2. 正方形参数 ===
    a = square_len  # 正方形边长cm

    # === 3. 在外接矩形内生成正方形格点 ===
    minx, miny, maxx, maxy = contour.bounds
    squares = []
    intersected = []
    centers = []
    intersected_shapes = []

    y = miny
    row = 0
    while y < maxy + a:
        x = minx
        while x < maxx + a:
            square = box(x, y, x + a, y + a)  # 创建正方形
            center_x = x + a / 2
            center_y = y + a / 2
            if contour.contains(square):
                squares.append(square)
                centers.append((center_x, center_y))
            elif contour.intersects(square):
                part = contour.intersection(square)
                intersected.append((part, (center_x, center_y)))
                intersected_shapes.append(part)
            x += a
        y += a
        row += 1

    # === 4. 可视化 ===
    fig, ax = plt.subplots(figsize=(12, 12))

    # 绘制轮廓
    cx, cy = contour.exterior.xy
    ax.plot(cx, cy, 'k-', lw=2)

    # 绘制完整正方形
    for idx, (square, center) in enumerate(zip(squares, centers), start=1):
        sx, sy = square.exterior.xy
        ax.fill(sx, sy, edgecolor='blue', facecolor='lightblue', alpha=0.5)
        ax.text(center[0], center[1], str(idx), color='black', fontsize=6, ha='center', va='center')

    # 绘制裁剪单元
    offset = len(squares)
    for idx, (part, center) in enumerate(intersected, start=1):
        if part.is_empty:
            continue
        if part.geom_type == 'Polygon':
            parts = [part]
        else:
            parts = list(part._geom)
        for subpart in parts:
            sx, sy = subpart.exterior.xy
            ax.fill(sx, sy, edgecolor='green', facecolor='lightgreen', alpha=0.5)
        ax.text(center[0], center[1], str(offset + idx), color='red', fontsize=6, ha='center', va='center')

    ax.set_aspect('equal')
    plt.title(f'Full squares: {len(squares)}, Clipped squares: {len(intersected)}')
    plt.show()

    # === 5. 输出信息 ===
    print("\n== 完整正方形中心点 ==")
    for idx, (x, y) in enumerate(centers, start=1):
        print(f"{idx}: ({x}, {y})")

    print("\n== 裁剪正方形中心点 ==")
    for idx, (part, (x, y)) in enumerate(intersected, start=len(centers) + 1):
        print(f"{idx}: ({x}, {y})")


if __name__ == '__main__':
    fill_shape("test.png", 80)
