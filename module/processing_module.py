import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import collections

# 1. 初始数据和参数定义
original_coords = [(40, 41), (120, 41), (200, 41), (280, 41), (360, 41), (440, 41), (520, 41), (40, 121), (120, 121), (200, 121), (280, 121), (360, 121), (440, 121), (520, 121), (600, 121), (40, 201), (120, 201), (200, 201), (280, 201), (360, 201), (440, 201), (520, 201), (600, 201), (680, 201), (40, 281), (120, 281), (200, 281), (280, 281), (360, 281), (440, 281), (520, 281), (600, 281), (680, 281), (40, 361), (120, 361), (200, 361), (280, 361), (360, 361), (440, 361), (520, 361), (600, 361), (680, 361), (40, 441), (120, 441), (200, 441), (280, 441), (360, 441), (440, 441), (520, 441), (600, 441), (680, 441), (40, 521), (120, 521), (200, 521), (280, 521), (360, 521), (440, 521), (520, 521), (600, 521), (40, 601), (120, 601), (200, 601), (280, 601), (360, 601), (440, 601), (520, 601), (600, 601), (40, 681), (120, 681), (200, 681), (280, 681), (360, 681), (440, 681), (520, 681), (600, 681), (680, 681), (40, 761), (120, 761), (200, 761), (280, 761), (360, 761), (440, 761), (520, 761), (600, 761), (40, 841), (280, 841), (360, 841), (440, 841), (600, 841)]

SQUARE_SIDE = 80

# 目标面积
TARGET_AREAS = {'MD': 31, 'PC': 40, 'CG': 18}

# 分割线斜率 (0 表示水平线)
SLOPE = 100
# 2. 坐标转换和辅助函数
X_MIN, Y_MIN = 600, 280


def to_grid(coords):
    return (coords[0] - X_MIN) // SQUARE_SIDE, (coords[1] - Y_MIN) // SQUARE_SIDE


def to_original(grid_coords):
    return grid_coords[0] * SQUARE_SIDE + X_MIN, grid_coords[1] * SQUARE_SIDE + Y_MIN


grid_coords = {to_grid(c) for c in original_coords}


def is_contiguous(points_set):
    if not points_set:
        return True
    q = collections.deque([next(iter(points_set))])
    visited = {next(iter(points_set))}
    while q:
        x, y = q.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            if neighbor in points_set and neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return len(visited) == len(points_set)


# 3. 遗传算法核心
def partition_and_evaluate(chromosome, all_points, slope):
    c1, c2 = sorted(chromosome)

    # 区域划分逻辑：
    # 使用 y - slope * x = c 作为分割线。当 slope = 0 时，即使用水平线 y=c 进行分割。
    # 假设区域顺序从y值小到大为：粗加工(CG) -> 配菜(PC) -> 明档(MD)
    areas = {'CG': set(), 'PC': set(), 'MD': set(), 'CH1': set(), 'CH2': set()}

    for x, y in all_points:
        # 根据斜率计算分割依据值
        val = y - slope * x

        if val < c1:
            areas['CG'].add((x, y))
        elif val == c1:
            areas['CH1'].add((x, y))
        elif c1 < val < c2:
            areas['PC'].add((x, y))
        elif val == c2:
            areas['CH2'].add((x, y))
        else:  # val > c2
            areas['MD'].add((x, y))

    # 计算适应度 (越小越好)
    fitness = 0

    # a. 面积惩罚
    fitness += abs(len(areas['MD']) - TARGET_AREAS['MD']) * 5
    fitness += abs(len(areas['PC']) - TARGET_AREAS['PC']) * 5
    fitness += abs(len(areas['CG']) - TARGET_AREAS['CG']) * 5

    # b. 连续性惩罚 (硬约束)
    if not is_contiguous(areas['MD']): fitness += 1000
    if not is_contiguous(areas['PC']): fitness += 1000
    if not is_contiguous(areas['CG']): fitness += 1000

    return fitness, areas


def run_genetic_algorithm(slope):
    # 确定c的取值范围
    partition_values = [y - slope * x for x, y in grid_coords]
    min_c, max_c = min(partition_values), max(partition_values)

    # GA 参数
    POPULATION_SIZE = 100
    GENERATIONS = 50
    ELITE_SIZE = 10
    MUTATION_RATE = 0.1

    # 初始化种群
    population = []
    for _ in range(POPULATION_SIZE):
        c1 = random.randint(min_c, max_c)
        c2 = random.randint(min_c, max_c)
        if c1 == c2: c2 += 1
        population.append(tuple(sorted((c1, c2))))

    for gen in range(GENERATIONS):
        # 评估种群
        fitness_scores = [partition_and_evaluate(chromo, grid_coords, slope)[0] for chromo in population]

        # 排序并选出精英
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        elites = [p for p, f in sorted_population[:ELITE_SIZE]]

        # 生成下一代
        next_generation = elites[:]

        while len(next_generation) < POPULATION_SIZE:
            # 选择父母
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)

            # 交叉
            child = (parent1[0], parent2[1])

            # 变异
            if random.random() < MUTATION_RATE:
                child = (child[0] + random.choice([-1, 1]), child[1])
            if random.random() < MUTATION_RATE:
                child = (child[0], child[1] + random.choice([-1, 1]))

            # 保证 c1 < c2 且在范围内
            c1, c2 = sorted(child)
            c1 = max(min_c, c1)
            c2 = min(max_c, c2)
            if c1 == c2:  # 如果交叉或变异后相等，则从精英中随机选择一个个体
                next_generation.append(random.choice(elites))
            else:
                next_generation.append((c1, c2))

        population = next_generation

    # 对最后一袋进行评估和排序
    final_fitness_scores = [partition_and_evaluate(chromo, grid_coords, slope)[0] for chromo in population]
    final_sorted_population = sorted(zip(population, final_fitness_scores), key=lambda x: x[1])

    best_chromosome = final_sorted_population[0][0]
    best_fitness, best_areas = partition_and_evaluate(best_chromosome, grid_coords, slope)

    print(f"找到的最优分割值 (c1, c2): {best_chromosome}")
    print(f"对应的最低适应度分数: {best_fitness}")

    return best_chromosome, best_fitness, best_areas


# 4. 执行并可视化
best_chromo, best_fitness, best_areas = run_genetic_algorithm(SLOPE)

# 转换回原始坐标
final_coords = {
    '明档': [to_original(c) for c in best_areas['MD']],
    '配菜间': [to_original(c) for c in best_areas['PC']],
    '粗加工间': [to_original(c) for c in best_areas['CG']],
    '通道1': [to_original(c) for c in best_areas['CH1']],
    '通道2': [to_original(c) for c in best_areas['CH2']],
}

# 5. 绘图
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 16))

colors = {
    '明档': '#ff6347',
    '配菜间': '#4682b4',
    '粗加工间': '#32cd32',
    '通道': '#d3d3d3',
}
labels = {
    '明档': f"明档 (面积: {len(final_coords['明档'])})",
    '配菜间': f"配菜间 (面积: {len(final_coords['配菜间'])})",
    '粗加工间': f"粗加工间 (面积: {len(final_coords['粗加工间'])})",
    '通道': f"通道 (面积: {len(final_coords['通道1']) + len(final_coords['通道2'])})",
}

all_plotted_coords = set()

# 绘制功能区
for name, coords_list in final_coords.items():
    area_name = '通道' if '通道' in name else name
    for x, y in coords_list:
        if (x, y) not in all_plotted_coords:
            rect = patches.Rectangle((x, y), SQUARE_SIDE, SQUARE_SIDE, linewidth=1, edgecolor='white',
                                     facecolor=colors[area_name])
            ax.add_patch(rect)
            all_plotted_coords.add((x, y))

ax.autoscale_view()
ax.set_aspect('equal', adjustable='box')
plt.title('厨房功能区优化布局 (水平分割)', fontsize=20)

# 创建图例
legend_patches = [patches.Patch(color=colors[name], label=labels[name]) for name in ['明档', '配菜间', '粗加工间']]
legend_patches.append(patches.Patch(color=colors['通道'], label=labels['通道']))
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.grid(False)
plt.show()
# 保存图像
# plt.savefig("kitchen_layout_horizontal.png", dpi=300, bbox_inches='tight')

# 最终输出坐标
for name, coords_list in final_coords.items():
    if '通道' not in name:
        print(
            f"\n{name} (目标面积: {TARGET_AREAS[name.replace('间', '')] if name.replace('间', '') in TARGET_AREAS else 'N/A'}, 实际面积: {len(coords_list)}):")
        sorted_coords = sorted(coords_list, key=lambda c: (c[1], c[0]))
        print(sorted_coords)
        print("-" * 30)