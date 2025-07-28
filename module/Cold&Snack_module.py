import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import collections

# 1. 初始数据和参数定义
original_coords = [(120, 1001), (200, 1001), (520, 1001), (40, 1081), (120, 1081), (200, 1081), (280, 1081), (360, 1081), (440, 1081), (520, 1081), (40, 1161), (120, 1161), (200, 1161), (280, 1161), (360, 1161), (440, 1161), (520, 1161), (40, 1241), (120, 1241), (200, 1241), (280, 1241), (360, 1241), (440, 1241), (520, 1241), (600, 1081), (600, 1161), (40, 1321), (120, 1321), (200, 1321), (280, 1321), (360, 1321), (440, 1321)]
SQUARE_SIDE = 80
# 定义分割通道的斜率，0 代表水平通道
SLOPE = 100

# 2. 坐标转换和辅助函数
X_MIN, Y_MIN = 1560, 840


def to_grid(coords):
    return (coords[0] - X_MIN) // SQUARE_SIDE, (coords[1] - Y_MIN) // SQUARE_SIDE


def to_original(grid_coords):
    return grid_coords[0] * SQUARE_SIDE + X_MIN, grid_coords[1] * SQUARE_SIDE + Y_MIN


grid_coords = {to_grid(c) for c in original_coords}


def is_contiguous(points_set):
    if not points_set:
        return True
    # 使用 next(iter(points_set)) 来安全地获取第一个元素
    start_node = next(iter(points_set))
    q = collections.deque([start_node])
    visited = {start_node}
    while q:
        x, y = q.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            if neighbor in points_set and neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return len(visited) == len(points_set)


# 3. 遗传算法核心
def partition_and_evaluate(chromosome_c, all_points, slope):
    """根据给定的斜率和分割值c，评估布局的适应度"""
    areas = {'Area1': set(), 'Channel': set(), 'Area2': set()}

    for x, y in all_points:
        # 使用 y - slope * x = c 的分割线方程
        val = y - slope * x
        # 由于坐标是整数，比较时可能存在浮点误差，进行取整比较
        if round(val) < chromosome_c:
            areas['Area1'].add((x, y))
        elif round(val) == chromosome_c:
            areas['Channel'].add((x, y))
        else:  # round(val) > chromosome_c
            areas['Area2'].add((x, y))

    fitness = 0

    # a. 面积均衡惩罚 (两个功能区面积差异越小越好)
    fitness += abs(len(areas['Area1']) - len(areas['Area2'])) * 10

    # b. 连续性惩罚 (确保每个区域自身是连通的)
    if not is_contiguous(areas['Area1']): fitness += 1000
    if not is_contiguous(areas['Area2']): fitness += 1000

    # c. 有效通道惩罚 (必须存在一个通道)
    if not areas['Channel']:
        fitness += 5000
    elif not is_contiguous(areas['Channel']):  # 通道自身也需要是连续的
        fitness += 1000

    return fitness, areas


def run_genetic_algorithm(slope):
    """执行遗传算法来找到最佳分割值c"""
    # 根据斜率计算分割值的取值范围
    partition_values = [round(y - slope * x) for x, y in grid_coords]
    min_c, max_c = min(partition_values), max(partition_values)

    # GA 参数
    POPULATION_SIZE = 50
    GENERATIONS = 40
    ELITE_SIZE = 5
    MUTATION_RATE = 0.1

    # 初始化种群 (每个个体就是一个分割值 c)
    population = [random.randint(min_c, max_c) for _ in range(POPULATION_SIZE)]

    best_overall_fitness = float('inf')
    best_solution = (None, (float('inf'), None))

    for gen in range(GENERATIONS):
        # 评估当前种群中每个个体的适应度
        evaluations = [partition_and_evaluate(chromo, grid_coords, slope) for chromo in population]

        # 将种群和评估结果打包并排序
        sorted_population = sorted(zip(population, evaluations), key=lambda x: x[1][0])

        # 记录全局最优解
        if sorted_population[0][1][0] < best_overall_fitness:
            best_overall_fitness = sorted_population[0][1][0]
            best_solution = sorted_population[0]

        # 选择精英个体直接进入下一代
        elites = [p for p, (f, a) in sorted_population[:ELITE_SIZE]]
        next_generation = elites[:]

        # 通过交叉和变异产生新的后代
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)

            # 交叉 (取平均值)
            child = (parent1 + parent2) // 2

            # 变异
            if random.random() < MUTATION_RATE:
                child += random.choice([-1, 1])

            # 确保子代在有效范围内
            child = max(min_c, min(child, max_c))
            next_generation.append(child)

        population = next_generation

    best_chromosome, (best_fitness, best_areas) = best_solution
    return best_chromosome, best_fitness, best_areas


# 4. 执行并可视化
best_chromo_c, best_fitness, best_areas = run_genetic_algorithm(SLOPE)

print(f"找到的最优分割值 c: {best_chromo_c}")
print(f"对应的最低适应度分数: {best_fitness}")

final_coords = {
    'cold': [to_original(c) for c in best_areas['Area1']],
    'snack': [to_original(c) for c in best_areas['Area2']],
    '通道': [to_original(c) for c in best_areas['Channel']],
}

# 5. 绘图
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 8))

colors = {
    'cold': '#1f77b4',
    'snack': '#ff7f0e',
    '通道': '#d3d3d3',
}
labels = {
    'cold': f"Cold Area (面积: {len(final_coords['cold'])})",
    'snack': f"Snack Area (面积: {len(final_coords['snack'])})",
    '通道': f"通道 (面积: {len(final_coords['通道'])})",
}

for name, coords_list in final_coords.items():
    for x, y in coords_list:
        rect = patches.Rectangle((x, y), SQUARE_SIDE, SQUARE_SIDE, linewidth=1, edgecolor='white',
                                 facecolor=colors[name])
        ax.add_patch(rect)

ax.autoscale_view()
ax.set_aspect('equal', adjustable='box')
plt.title(f'Cold & Snack Module Layout (Slope = {SLOPE})', fontsize=20)

legend_patches = [patches.Patch(color=colors[name], label=labels[name]) for name in colors]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel("X 坐标")
plt.ylabel("Y 坐标")

plt.show()

# 最终输出坐标
for name, coords_list in final_coords.items():
    print(f"\n{name} (面积: {len(coords_list)}):")
    sorted_coords = sorted(coords_list, key=lambda c: (c[1], c[0]))
    print(sorted_coords)
    print("-" * 30)