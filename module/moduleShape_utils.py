import matplotlib.pyplot as plt
import numpy as np


def check_complex_contour(points_to_check, verbose=True):
    """
    接收一组可能无序的轮廓点，通过排序、简化和几何判断，
    检查其是否包含至少三条相邻的宏观边且两两垂直。

    Args:
        points_to_check (list): 一个包含 (x, y) 元组的列表，代表无序的轮廓点。
        verbose (bool): 是否打印详细的中间步骤信息。

    Returns:
        bool: 如果满足条件，返回 True，否则返回 False。
    """

    # --- 内部辅助函数 1: 最近邻排序 ---
    def _sort_points_nearest_neighbor(points):
        if not points:
            return []
        remaining_points = points[:]
        path = [remaining_points.pop(0)]
        current_point = path[0]
        while remaining_points:
            nearest_point = min(
                remaining_points,
                key=lambda p: (current_point[0] - p[0]) ** 2 + (current_point[1] - p[1]) ** 2
            )
            path.append(nearest_point)
            current_point = nearest_point
            remaining_points.remove(nearest_point)
        return path

    # --- 内部辅助函数 2: 轮廓简化 ---
    def _simplify_contour(points):
        if len(points) < 3:
            return points
        simplified = [points[0]]
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i - 1], points[i], points[i + 1]
            cross_product = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
            if cross_product != 0:
                simplified.append(p2)
        simplified.append(points[-1])

        # 闭合回路检查
        if len(simplified) > 2:
            p1, p2, p3 = simplified[-2], simplified[-1], simplified[0]
            cross_product = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
            if cross_product == 0:
                simplified.pop(-1)
        return simplified

    # --- 内部辅助函数 3: 垂直性判断 ---
    def _has_perpendicular_sides(contour_points):
        if len(contour_points) < 4:
            return False
        points = contour_points + contour_points[:3]
        for i in range(len(contour_points)):  # 遍历闭合的轮廓
            p1, p2, p3, p4 = points[i], points[i + 1], points[i + 2], points[i + 3]
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            v3 = (p4[0] - p3[0], p4[1] - p3[1])
            if any(v == (0, 0) for v in [v1, v2, v3]):
                continue
            dot_product1 = np.dot(v1, v2)
            dot_product2 = np.dot(v2, v3)
            if dot_product1 == 0 and dot_product2 == 0:
                if verbose:
                    print(f"找到满足条件的角点序列: P1{p1}, P2{p2}, P3{p3}, P4{p4}")
                return True
        return False

    # --- 主函数执行流程 ---
    if verbose: print("--- 步骤 1: 排序原始点 ---")
    sorted_path = _sort_points_nearest_neighbor(points_to_check)
    if verbose: print(f"排序完成，共 {len(sorted_path)} 个点。")

    if verbose: print("\n--- 步骤 2: 简化轮廓（合并共线点） ---")
    simplified_path = _simplify_contour(sorted_path)
    if verbose: print(f"简化完成，剩余 {len(simplified_path)} 个角点。")
    if verbose: print("角点路径:", simplified_path)

    if verbose: print("\n--- 步骤 3: 在角点轮廓上进行最终判断 ---")
    final_result = _has_perpendicular_sides(simplified_path)

    print(f"结果: {final_result}")
    return final_result


def get_contour_and_plot(points):
    """
    此函数从一系列正方形中心点中识别出最外层的轮廓点，
    并直接显示其可视化图像。

    Args:
      points: 一个包含元组的列表，每个元组代表一个正方形中心的(x, y)坐标。
    """

    point_set = set(points)
    contour_points = []

    # 遍历每个点以确定它是否在轮廓上
    for x, y in points:
        is_contour = False
        # 检查四个方向的邻居是否存在
        if (x + 80, y) not in point_set:
            is_contour = True
        if (x - 80, y) not in point_set:
            is_contour = True
        if (x, y + 80) not in point_set:
            is_contour = True
        if (x, y - 80) not in point_set:
            is_contour = True

        if is_contour:
            contour_points.append((x, y))

    # --- 开始绘图 ---
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 将所有中心点绘制为蓝点
    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]
    ax.scatter(all_x, all_y, c='blue', s=15, label='所有正方形中心')

    # 为轮廓点绘制红色方框
    for x, y in contour_points:
        # 计算正方形的左下角坐标
        bottom_left_x = x - 40
        bottom_left_y = y - 40
        rectangle = plt.Rectangle((bottom_left_x, bottom_left_y), 80, 80,
                                  facecolor='none', edgecolor='red', linewidth=2)
        ax.add_patch(rectangle)

    # 创建自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=6, label='所有正方形中心'),
        Line2D([0], [0], color='red', lw=2, label='轮廓线')
    ]
    ax.legend(handles=legend_elements)

    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.title("由正方形区域构成的轮廓")
    plt.grid(True)
    plt.axis('equal')

    # 直接显示图像而不是保存
    plt.show()
    return contour_points


def calculate_slope(x1, y1, x2, y2):
    """
    计算并返回连接两点的直线的斜率。

    参数:
    x1 (int or float): 第一个点的 x 坐标。
    y1 (int or float): 第一个点的 y 坐标。
    x2 (int or float): 第二个点的 x 坐标。
    y2 (int or float): 第二个点的 y 坐标。

    返回:
    float or None: 如果斜率存在，则返回斜率 (浮点数)。
                   如果是一条垂直线 (斜率未定义), 则返回 None。
    """
    # 检查是否为垂直线
    if x2 == x1:
        print("错误：这是一条垂直线，斜率未定义。")
        return None

    # 计算斜率
    slope = (y2 - y1) / (x2 - x1)

    return round(slope, 1)


if __name__ == '__main__':
    points = [(1720, 840), (1800, 840), (1880, 840), (1960, 840), (1720, 920), (1800, 920), (1880, 920), (1640, 1000), (1720, 1000), (1800, 1000), (1560, 1080), (1640, 1080), (1720, 1080), (1560, 1160), (1640, 1160), (1560, 1240), (1960, 920), (1880, 1000), (1960, 1000), (1800, 1080), (1880, 1080), (1720, 1160), (1800, 1160), (1640, 1240), (1720, 1240)]
    contour_points = get_contour_and_plot(points)
    print(len(contour_points))
    print(f"轮廓点: {contour_points}")
    check_complex_contour(contour_points)







