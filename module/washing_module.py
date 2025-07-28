import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import collections

# 1. 初始数据和参数定义
original_coords = [(840, 41), (920, 41), (1000, 41), (1080, 41), (1160, 41), (1240, 41), (840, 121), (920, 121), (1000, 121), (1080, 121), (1160, 121), (1240, 121), (1320, 121), (920, 201), (1000, 201), (1080, 201), (1160, 201), (1240, 201), (1320, 201), (1400, 201), (1480, 201), (920, 281), (1000, 281), (1080, 281), (1160, 281), (1240, 281), (1320, 281), (1400, 281), (920, 361), (1000, 361), (1080, 361), (1160, 361), (1240, 361), (1320, 361), (1400, 361), (1480, 361), (920, 441), (1000, 441), (1080, 441), (1160, 441), (1240, 441), (1320, 441), (1400, 441), (1480, 441), (840, 521), (920, 521), (1000, 521), (1080, 521), (1160, 521), (1240, 521), (1320, 521), (1400, 521), (840, 601), (920, 601), (1000, 601), (1080, 601), (1160, 601), (1240, 601), (1320, 601), (1400, 601), (920, 681), (1000, 681), (1080, 681), (1160, 681), (1240, 681), (1320, 681), (1400, 681), (840, 761), (920, 761), (1000, 761), (1080, 761), (1160, 761), (1240, 761), (1320, 761), (840, 841), (920, 841), (1000, 841), (1080, 841), (1160, 841), (920, 921), (1000, 921), (1080, 921)]


SQUARE_SIDE = 80
# 定义分割通道的斜率，0 代表水平通道
SLOPE = 100

# 2. 坐标转换和辅助函数
X_MIN, Y_MIN = 40, 40


def to_grid(coords):
    return (coords[0] - X_MIN) // SQUARE_SIDE, (coords[1] - Y_MIN) // SQUARE_SIDE


def to_original(grid_coords):
    return grid_coords[0] * SQUARE_SIDE + X_MIN, grid_coords[1] * SQUARE_SIDE + Y_MIN


grid_coords = {to_grid(c) for c in original_coords}


def is_contiguous(points_set):
    if not points_set:
        return True
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
    """根据给定的斜率和两个分割值c1, c2，评估布局的适应度"""
    c1, c2 = sorted(chromosome_c)
    areas = {'Area1': set(), 'Channel1': set(), 'Area2': set(), 'Channel2': set(), 'Area3': set()}

    for x, y in all_points:
        # 使用 y - slope * x = c 的分割线方程
        val = round(y - slope * x)

        if val < c1:
            areas['Area1'].add((x, y))
        elif val == c1:
            areas['Channel1'].add((x, y))
        elif c1 < val < c2:
            areas['Area2'].add((x, y))
        elif val == c2:
            areas['Channel2'].add((x, y))
        else:  # val > c2
            areas['Area3'].add((x, y))

    fitness = 0

    # a. 面积均衡惩罚 (目标是三个功能区的面积尽可能接近 1:1:1)
    func_areas = [areas['Area1'], areas['Area2'], areas['Area3']]
    non_empty_areas = [a for a in func_areas if a]
    if len(non_empty_areas) < 3:
        fitness += 5000  # 硬惩罚：必须要有三个区域
    else:
        avg_area = sum(len(a) for a in non_empty_areas) / len(non_empty_areas)
        for a in non_empty_areas:
            # 使用方差作为惩罚项，鼓励面积均等
            fitness += (len(a) - avg_area) ** 2

    # b. 连续性惩罚 (硬约束)
    for name, area_set in areas.items():
        if area_set and not is_contiguous(area_set):
            fitness += 1000

    # c. 有效通道惩罚
    if not areas['Channel1'] or not areas['Channel2']:
        fitness += 5000

    return fitness, areas


def run_genetic_algorithm(slope):
    """执行遗传算法来找到最佳的两个分割值 c1, c2"""
    # 根据斜率计算分割值的候选范围
    candidate_values = sorted(list(set(round(y - slope * x) for x, y in grid_coords)))

    # GA 参数
    POPULATION_SIZE = 100
    GENERATIONS = 50
    ELITE_SIZE = 10
    MUTATION_RATE = 0.2

    # 初始化种群 (每个个体是(c1, c2)元组)
    population = []
    for _ in range(POPULATION_SIZE):
        c1 = random.choice(candidate_values)
        c2 = random.choice(candidate_values)
        if c1 == c2:
            idx = candidate_values.index(c1)
            if idx + 1 < len(candidate_values):
                c2 = candidate_values[idx + 1]
            else:
                c2 = candidate_values[idx - 1]
        population.append(tuple(sorted((c1, c2))))

    best_overall_fitness = float('inf')
    best_solution = (None, (float('inf'), None))

    for gen in range(GENERATIONS):
        # 评估种群
        evaluations = [partition_and_evaluate(chromo, grid_coords, slope) for chromo in population]

        # 排序并选出精英
        sorted_population = sorted(zip(population, evaluations), key=lambda x: x[1][0])

        # 记录全局最优解
        if sorted_population[0][1][0] < best_overall_fitness:
            best_overall_fitness = sorted_population[0][1][0]
            best_solution = sorted_population[0]

        elites = [p for p, (f, a) in sorted_population[:ELITE_SIZE]]

        # 生成下一代
        next_generation = elites[:]

        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)

            # 交叉 (交换一个c值)
            child = tuple(sorted((parent1[0], parent2[1])))

            # 变异
            if random.random() < MUTATION_RATE:
                child_list = list(child)
                idx_to_mutate = random.randint(0, 1)
                child_list[idx_to_mutate] = random.choice(candidate_values)
                child = tuple(sorted(child_list))

            # 避免两个c值相等
            if child[0] == child[1]:
                next_generation.append(random.choice(elites))  # 如果相等则从精英中随机选一个
            else:
                next_generation.append(child)

        population = next_generation

    best_chromosome, (best_fitness, best_areas) = best_solution
    return best_chromosome, best_fitness, best_areas


# 4. 执行并可视化
best_chromo, best_fitness, best_areas = run_genetic_algorithm(SLOPE)

print(f"找到的最优分割值 (c1, c2): {best_chromo}")
print(f"对应的最低适应度分数: {best_fitness}")

# 转换回原始坐标
final_coords = {
    '洗杯': [to_original(c) for c in best_areas['Area1']],
    '洗碗': [to_original(c) for c in best_areas['Area2']],
    '洗毛巾': [to_original(c) for c in best_areas['Area3']],
    '通道1': [to_original(c) for c in best_areas['Channel1']],
    '通道2': [to_original(c) for c in best_areas['Channel2']],
}

# 5. 绘图
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 10))

colors = {
    '洗杯': '#1f77b4',
    '洗碗': '#ff7f0e',
    '洗毛巾': '#2ca02c',
    '通道': '#d3d3d3',
}
labels = {
    '洗杯': f"洗杯区 (面积: {len(final_coords['洗杯'])})",
    '洗碗': f"洗碗区 (面积: {len(final_coords['洗碗'])})",
    '洗毛巾': f"洗毛巾区 (面积: {len(final_coords['洗毛巾'])})",
    '通道': f"通道 (总面积: {len(final_coords['通道1']) + len(final_coords['通道2'])})",
}

for name, coords_list in final_coords.items():
    area_name = '通道' if '通道' in name else name
    for x, y in coords_list:
        rect = patches.Rectangle((x, y), SQUARE_SIDE, SQUARE_SIDE, linewidth=1, edgecolor='white',
                                 facecolor=colors[area_name])
        ax.add_patch(rect)

ax.autoscale_view()
ax.set_aspect('equal', adjustable='box')
plt.title(f'Washing Module Layout (Slope = {SLOPE})', fontsize=20)

legend_patches = [patches.Patch(color=colors[name], label=labels[name]) for name in
                  ['洗杯', '洗碗', '洗毛巾', '通道']]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel("X 坐标")
plt.ylabel("Y 坐标")

plt.show()

# 最终输出坐标
for name, coords_list in final_coords.items():
    if '通道' not in name:
        print(f"\n{name} (面积: {len(coords_list)}):")
        sorted_coords = sorted(coords_list, key=lambda c: (c[1], c[0]))
        print(sorted_coords)
        print("-" * 30)