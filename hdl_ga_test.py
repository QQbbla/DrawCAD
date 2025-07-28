import random
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import numpy as np
import time

# --- 1. 数据准备和配置 ---

MODULE_SPACING = 0.0
SQUARE_SIDE = 80

# --- 遗传算法参数 ---
POPULATION_SIZE = 100
GENERATIONS = 400
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5

# --- 【最终策略】适应度函数权重 ---
# 显著提高所有与形状相关的权重，降低邻近度权重
PROXIMITY_WEIGHT = 4.0
RECTANGULARITY_WEIGHT = 10.0
HOLE_PENALTY_WEIGHT = 2.0
# 【重新引入】共享边界惩罚，用于惩罚犬牙交错的边界
SHARED_BORDER_WEIGHT = 3.0
# 硬约束惩罚
DISCONNECTED_PENALTY = -1000.0

# ... [模块定义、目标点定义、坐标数据保持不变] ...
MODULE_AREAS = {'Processing': 89, 'Cold & Staples': 32, 'Storage': 40, 'Washing': 82, }
FIXED_POINTS = {'Storage': [340, 356], 'Processing': [154, 177], 'Washing': [120, 143], 'Cold & Staples': [300, 307]}
COORDINATE_DATA = """
1: (40, 41)2: (120, 41)3: (200, 41)4: (280, 41)5: (360, 41)6: (440, 41)7: (520, 41)8: (600, 41)9: (680, 41)10: (760, 41)11: (840, 41)12: (920, 41)13: (1000, 41)14: (1080, 41)15: (1160, 41)16: (1240, 41)17: (1320, 41)18: (1400, 41)19: (1480, 41)20: (1560, 41)21: (1640, 41)22: (1720, 41)23: (1800, 41)24: (1880, 41)25: (1960, 41)26: (2040, 41)27: (2120, 41)28: (2200, 41)29: (40, 121)30: (120, 121)31: (200, 121)32: (280, 121)33: (360, 121)34: (440, 121)35: (520, 121)36: (600, 121)37: (680, 121)38: (760, 121)39: (840, 121)40: (920, 121)41: (1000, 121)42: (1080, 121)43: (1160, 121)44: (1240, 121)45: (1320, 121)46: (1400, 121)47: (1480, 121)48: (1560, 121)49: (1640, 121)50: (1720, 121)51: (1800, 121)52: (1880, 121)53: (1960, 121)54: (2040, 121)55: (2120, 121)56: (2200, 121)57: (40, 201)58: (120, 201)59: (200, 201)60: (280, 201)61: (360, 201)62: (440, 201)63: (520, 201)64: (600, 201)65: (680, 201)66: (760, 201)67: (840, 201)68: (920, 201)69: (1000, 201)70: (1080, 201)71: (1160, 201)72: (1240, 201)73: (1320, 201)74: (1400, 201)75: (1480, 201)76: (1560, 201)77: (1640, 201)78: (1720, 201)79: (1800, 201)80: (1880, 201)81: (1960, 201)82: (2040, 201)83: (2120, 201)84: (2200, 201)85: (40, 281)86: (120, 281)87: (200, 281)88: (280, 281)89: (360, 281)90: (440, 281)91: (520, 281)92: (600, 281)93: (680, 281)94: (760, 281)95: (840, 281)96: (920, 281)97: (1000, 281)98: (1080, 281)99: (1160, 281)100: (1240, 281)101: (1320, 281)102: (1400, 281)103: (1480, 281)104: (1560, 281)105: (1640, 281)106: (1720, 281)107: (1800, 281)108: (40, 361)109: (120, 361)110: (200, 361)111: (280, 361)112: (360, 361)113: (440, 361)114: (520, 361)115: (600, 361)116: (680, 361)117: (760, 361)118: (840, 361)119: (920, 361)120: (1000, 361)121: (1080, 361)122: (1160, 361)123: (1240, 361)124: (1320, 361)125: (1400, 361)126: (1480, 361)127: (1560, 361)128: (1640, 361)129: (1720, 361)130: (1800, 361)131: (40, 441)132: (120, 441)133: (200, 441)134: (280, 441)135: (360, 441)136: (440, 441)137: (520, 441)138: (600, 441)139: (680, 441)140: (760, 441)141: (840, 441)142: (920, 441)143: (1000, 441)144: (1080, 441)145: (1160, 441)146: (1240, 441)147: (1320, 441)148: (1400, 441)149: (1480, 441)150: (1560, 441)151: (1640, 441)152: (1720, 441)153: (1800, 441)154: (40, 521)155: (120, 521)156: (200, 521)157: (280, 521)158: (360, 521)159: (440, 521)160: (520, 521)161: (600, 521)162: (680, 521)163: (760, 521)164: (840, 521)165: (920, 521)166: (1000, 521)167: (1080, 521)168: (1160, 521)169: (1240, 521)170: (1320, 521)171: (1400, 521)172: (1480, 521)173: (1560, 521)174: (1640, 521)175: (1720, 521)176: (1800, 521)177: (40, 601)178: (120, 601)179: (200, 601)180: (280, 601)181: (360, 601)182: (440, 601)183: (520, 601)184: (600, 601)185: (680, 601)186: (760, 601)187: (840, 601)188: (920, 601)189: (1000, 601)190: (1080, 601)191: (1160, 601)192: (1240, 601)193: (1320, 601)194: (1400, 601)195: (1480, 601)196: (1560, 601)197: (1640, 601)198: (1720, 601)199: (1800, 601)200: (1880, 601)201: (40, 681)202: (120, 681)203: (200, 681)204: (280, 681)205: (360, 681)206: (440, 681)207: (520, 681)208: (600, 681)209: (680, 681)210: (760, 681)211: (840, 681)212: (920, 681)213: (1000, 681)214: (1080, 681)215: (1160, 681)216: (1240, 681)217: (1320, 681)218: (1400, 681)219: (1480, 681)220: (1560, 681)221: (1640, 681)222: (1720, 681)223: (1800, 681)224: (1880, 681)225: (40, 761)226: (120, 761)227: (200, 761)228: (280, 761)229: (360, 761)230: (440, 761)231: (520, 761)232: (600, 761)233: (680, 761)234: (760, 761)235: (840, 761)236: (920, 761)237: (1000, 761)238: (1080, 761)239: (1160, 761)240: (1240, 761)241: (1320, 761)242: (1400, 761)243: (1480, 761)244: (1560, 761)245: (1640, 761)246: (1720, 761)247: (1800, 761)248: (1880, 761)249: (40, 841)250: (120, 841)251: (200, 841)252: (280, 841)253: (360, 841)254: (440, 841)255: (520, 841)256: (600, 841)257: (680, 841)258: (760, 841)259: (840, 841)260: (920, 841)261: (1000, 841)262: (1080, 841)263: (1160, 841)264: (1240, 841)265: (1320, 841)266: (1400, 841)267: (1480, 841)268: (1560, 841)269: (1640, 841)270: (1720, 841)271: (40, 921)272: (120, 921)273: (200, 921)274: (280, 921)275: (360, 921)276: (440, 921)277: (520, 921)278: (600, 921)279: (680, 921)280: (760, 921)281: (840, 921)282: (920, 921)283: (1000, 921)284: (1080, 921)285: (1160, 921)286: (1240, 921)287: (1320, 921)288: (1400, 921)289: (1480, 921)290: (1560, 921)291: (1640, 921)292: (1720, 921)293: (40, 1001)294: (120, 1001)295: (200, 1001)296: (280, 1001)297: (360, 1001)298: (440, 1001)299: (520, 1001)300: (40, 1081)301: (120, 1081)302: (200, 1081)303: (280, 1081)304: (360, 1081)305: (440, 1081)306: (520, 1081)307: (40, 1161)308: (120, 1161)309: (200, 1161)310: (280, 1161)311: (360, 1161)312: (440, 1161)313: (520, 1161)314: (40, 1241)315: (120, 1241)316: (200, 1241)317: (280, 1241)318: (360, 1241)319: (440, 1241)320: (520, 1241)321: (2280, 41)322: (2280, 121)323: (2280, 201)324: (1880, 281)325: (1960, 281)326: (2040, 281)327: (2120, 281)328: (2200, 281)329: (2280, 281)330: (1880, 361)331: (1880, 441)332: (1880, 521)333: (1960, 521)334: (1960, 601)335: (1960, 681)336: (1960, 761)337: (1800, 841)338: (1880, 841)339: (1960, 841)340: (1800, 921)341: (600, 1001)342: (680, 1001)343: (760, 1001)344: (840, 1001)345: (920, 1001)346: (1000, 1001)347: (1080, 1001)348: (1160, 1001)349: (1240, 1001)350: (1320, 1001)351: (1400, 1001)352: (1480, 1001)353: (1560, 1001)354: (1640, 1001)355: (1720, 1001)356: (1800, 1001)357: (600, 1081)358: (600, 1161)359: (600, 1241)360: (40, 1321)361: (120, 1321)362: (200, 1321)363: (280, 1321)364: (360, 1321)365: (440, 1321)366: (520, 1321)367: (600, 1321)
"""

# --- 2. 核心功能函数 ---

# 全局变量，程序启动时初始化一次
ID_TO_COORDS, ID_TO_NEIGHBORS, COORDS_TO_ID = {}, {}, {}


def setup_floor_plan(coordinate_text):
    """解析坐标文本，构建ID、坐标、邻居的互相映射关系"""
    global ID_TO_COORDS, ID_TO_NEIGHBORS, COORDS_TO_ID
    pattern = re.compile(r'(\d+):\s*\((\d+),\s*(\d+)\)')
    matches = pattern.findall(coordinate_text)
    for match in matches:
        square_id, x, y = map(int, match)
        ID_TO_COORDS[square_id] = (x, y)

    COORDS_TO_ID = {v: k for k, v in ID_TO_COORDS.items()}

    for square_id, (x, y) in ID_TO_COORDS.items():
        neighbors = []
        for dx, dy in [(0, SQUARE_SIDE), (0, -SQUARE_SIDE), (SQUARE_SIDE, 0), (-SQUARE_SIDE, 0)]:
            neighbor_coord = (x + dx, y + dy)
            if neighbor_coord in COORDS_TO_ID:
                neighbors.append(COORDS_TO_ID[neighbor_coord])
        ID_TO_NEIGHBORS[square_id] = neighbors


# 程序开始时执行初始化
setup_floor_plan(COORDINATE_DATA)
ALL_SQUARE_IDS = list(ID_TO_COORDS.keys())
fixed_point_ids = {id for ids in FIXED_POINTS.values() for id in ids}
AVAILABLE_SQUARES = [sq for sq in ALL_SQUARE_IDS if sq not in fixed_point_ids]


def get_buffer_zone(square_ids, distance):
    if not square_ids or distance <= 0: return set(square_ids)
    buffer = set(square_ids)
    queue = deque([(sq, 0) for sq in square_ids])
    visited = set(square_ids)

    while queue:
        current_sq, dist = queue.popleft()
        if dist >= distance:
            continue
        for neighbor in ID_TO_NEIGHBORS.get(current_sq, []):
            if neighbor not in visited:
                visited.add(neighbor)
                buffer.add(neighbor)
                queue.append((neighbor, dist + 1))
    return buffer


# ... [create_individual, is_module_continuous, mutate, crossover, selection, etc. 省略未修改的函数] ...
def create_individual():
    max_retries = 300
    for _ in range(max_retries):
        chromosome = {}
        squares_for_this_try = set(AVAILABLE_SQUARES)
        sorted_modules = sorted(MODULE_AREAS.items(), key=lambda item: item[1], reverse=True)
        all_modules_placed = True
        for module_name, area in sorted_modules:
            module_squares = []
            potential_seeds = []
            if module_name in FIXED_POINTS:
                for fp in FIXED_POINTS[module_name]:
                    for neighbor in ID_TO_NEIGHBORS.get(fp, []):
                        if neighbor in squares_for_this_try:
                            potential_seeds.append(neighbor)
            if not potential_seeds:
                if not squares_for_this_try:
                    all_modules_placed = False;
                    break
                seed = random.choice(list(squares_for_this_try))
            else:
                seed = random.choice(potential_seeds)
            module_squares.append(seed)
            current_placement_options = squares_for_this_try.copy()
            current_placement_options.remove(seed)
            for _ in range(area - 1):
                frontier = {neighbor for sq in module_squares for neighbor in ID_TO_NEIGHBORS.get(sq, []) if
                            neighbor in current_placement_options}
                if not frontier:
                    all_modules_placed = False;
                    break
                new_square = random.choice(list(frontier))
                module_squares.append(new_square)
                current_placement_options.remove(new_square)
            if not all_modules_placed: break
            for sq in module_squares: chromosome[sq] = module_name
            buffer_to_remove = get_buffer_zone(module_squares, MODULE_SPACING)
            squares_for_this_try -= buffer_to_remove
        if all_modules_placed: return chromosome
    raise RuntimeError(f"在 {max_retries} 次尝试后仍无法生成一个满足间隔约束的初始布局。")


def is_module_continuous(module_squares_set, id_to_neighbors_map):
    if not module_squares_set: return True
    start_node = next(iter(module_squares_set))
    visited, queue = set(), deque([start_node]);
    visited.add(start_node)
    while queue:
        current_sq = queue.popleft()
        for neighbor in id_to_neighbors_map.get(current_sq, []):
            if neighbor in module_squares_set and neighbor not in visited:
                visited.add(neighbor);
                queue.append(neighbor)
    return visited == module_squares_set


def mutate(chromosome):
    if random.random() > MUTATION_RATE: return chromosome
    new_chromosome = chromosome.copy()
    module_to_mutate = random.choice(list(MODULE_AREAS.keys()))
    current_module_squares = {sq for sq, mod in new_chromosome.items() if mod == module_to_mutate}
    if not current_module_squares: return new_chromosome
    removable_squares = []
    for sq in current_module_squares:
        if is_module_continuous(current_module_squares - {sq}, ID_TO_NEIGHBORS):
            removable_squares.append(sq)
    if not removable_squares: return new_chromosome
    sq_to_remove = random.choice(removable_squares)
    occupied_or_buffered = set()
    for mod_name, area in MODULE_AREAS.items():
        if mod_name != module_to_mutate:
            other_module_squares = {sq for sq, mod in new_chromosome.items() if mod == mod_name}
            occupied_or_buffered.update(get_buffer_zone(other_module_squares, MODULE_SPACING))
    occupied_or_buffered.update(current_module_squares - {sq_to_remove})
    potential_add_squares = []
    for sq in current_module_squares:
        for neighbor in ID_TO_NEIGHBORS.get(sq, []):
            if neighbor not in occupied_or_buffered and neighbor in ALL_SQUARE_IDS:
                potential_add_squares.append(neighbor)
    if sq_to_remove in potential_add_squares: potential_add_squares.remove(sq_to_remove)
    if not potential_add_squares: return new_chromosome
    valid_add_squares = []
    for sq_to_add in potential_add_squares:
        if is_module_continuous(current_module_squares - {sq_to_remove} | {sq_to_add}, ID_TO_NEIGHBORS):
            valid_add_squares.append(sq_to_add)
    if not valid_add_squares: return new_chromosome
    sq_to_add = random.choice(valid_add_squares)
    del new_chromosome[sq_to_remove];
    new_chromosome[sq_to_add] = module_to_mutate
    return new_chromosome


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        child1, child2 = parent1.copy(), parent2.copy()
        common_squares = list(set(parent1.keys()) & set(parent2.keys()))
        if len(common_squares) < 2: return parent1.copy(), parent2.copy()
        split_point = random.randint(1, len(common_squares) - 1)
        for i in range(split_point, len(common_squares)):
            sq = common_squares[i];
            val1, val2 = parent1.get(sq), parent2.get(sq)
            child1[sq] = val2;
            child2[sq] = val1
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


def selection(population_with_fitness):
    actual_tournament_size = min(TOURNAMENT_SIZE, len(population_with_fitness))
    if actual_tournament_size == 0: return {}
    tournament = random.sample(population_with_fitness, actual_tournament_size)
    tournament.sort(key=lambda x: x[1]['total'], reverse=True)
    return tournament[0][0]


def manhattan_distance(p1, p2): return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def calculate_avg_distance(module_squares, target_points):
    if not module_squares or not target_points: return 1e9
    total_dist = 0
    valid_target_coords = [ID_TO_COORDS[tp] for tp in target_points if tp in ID_TO_COORDS]
    if not valid_target_coords: return 1e9
    for sq_id in module_squares:
        sq_coord = ID_TO_COORDS[sq_id]
        min_dist_to_target = min(manhattan_distance(sq_coord, tc) for tc in valid_target_coords)
        total_dist += min_dist_to_target
    return total_dist / len(module_squares)


def calculate_rectangularity(square_ids):
    if not square_ids: return 0.0
    coords = [ID_TO_COORDS[sq_id] for sq_id in square_ids]
    min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
    min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)
    bounding_box_width = (max_x - min_x) / SQUARE_SIDE + 1
    bounding_box_height = (max_y - min_y) / SQUARE_SIDE + 1
    bounding_box_area = bounding_box_width * bounding_box_height
    return len(square_ids) / bounding_box_area if bounding_box_area > 0 else 0.0


# --- 形状惩罚函数 ---

def calculate_hole_penalty(module_squares):
    """计算一个模块内部的“空洞”数量"""
    if len(module_squares) < 3: return 0
    coords = [ID_TO_COORDS[sq_id] for sq_id in module_squares]
    min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
    min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)
    hole_count = 0
    for x in range(min_x, max_x + 1, SQUARE_SIDE):
        for y in range(min_y, max_y + 1, SQUARE_SIDE):
            coord = (x, y)
            if coord in COORDS_TO_ID and COORDS_TO_ID[coord] not in module_squares:
                hole_count += 1
    return hole_count


def calculate_shared_border_penalty(chromosome):
    """【新实现】计算模块间共享边界的总长度，用于惩罚犬牙交错的边界"""
    penalty = 0
    # 遍历布局中的每一个被分配的方块
    for sq_id, module_name in chromosome.items():
        # 寻找该方块的邻居
        for neighbor_id in ID_TO_NEIGHBORS.get(sq_id, []):
            # 检查邻居是否也存在于布局中，并且属于一个 *不同* 的模块
            neighbor_module = chromosome.get(neighbor_id)
            if neighbor_module and neighbor_module != module_name:
                penalty += 1
    # 每个共享边界都被从两侧计算了一次，所以最终结果除以2
    return penalty / 2


# --- 【最终版】核心适应度函数 ---
def calculate_fitness(chromosome):
    """
    计算一个布局的适应度总分，采用“三位一体”的形状惩罚策略。
    """
    total_fitness = 0.0
    proximity_score = 0.0
    rectangularity_score = 0.0
    total_hole_penalty = 0

    # 硬约束：检查方块重叠
    sq_ids = list(chromosome.keys())
    if len(sq_ids) != len(set(sq_ids)):
        total_fitness += DISCONNECTED_PENALTY * 10  # 对重叠施加极高惩罚

    # 遍历每个模块，计算其得分和惩罚
    for module_name, area_required in MODULE_AREAS.items():
        module_squares = {sq for sq, mod in chromosome.items() if mod == module_name}

        # 硬约束检查：面积和连续性
        if len(module_squares) != area_required or not is_module_continuous(module_squares, ID_TO_NEIGHBORS):
            total_fitness += DISCONNECTED_PENALTY
            continue  # 对于严重错误的个体，直接进入下一个模块的检查

        # 邻近度得分
        target_points = FIXED_POINTS.get(module_name, [])
        avg_dist = calculate_avg_distance(module_squares, target_points)
        MAX_POSSIBLE_AVG_DIST = 3000
        normalized_proximity = 1.0 - (avg_dist / MAX_POSSIBLE_AVG_DIST)
        proximity_score += max(0, normalized_proximity)

        # 形状得分/惩罚 Part 1: 矩形度
        rectangularity_score += calculate_rectangularity(module_squares)

        # 形状得分/惩罚 Part 2: 空洞
        total_hole_penalty += calculate_hole_penalty(module_squares)

    # 形状得分/惩罚 Part 3: 模块间共享边界
    shared_border_penalty = calculate_shared_border_penalty(chromosome)

    # 加权计算总分
    total_fitness += (proximity_score * PROXIMITY_WEIGHT)
    total_fitness += (rectangularity_score * RECTANGULARITY_WEIGHT)
    total_fitness -= (total_hole_penalty * HOLE_PENALTY_WEIGHT)
    total_fitness -= (shared_border_penalty * SHARED_BORDER_WEIGHT)  # 减去共享边界惩罚

    scores = {
        "total": total_fitness,
        "proximity": proximity_score,
        "rectangularity": rectangularity_score,
        "hole_penalty": total_hole_penalty,
        "shared_border_penalty": shared_border_penalty  # 返回所有分数用于监控
    }
    return scores


def visualize_layout(chromosome, generation, fitness_scores):
    """可视化最终的布局方案"""
    fig, ax = plt.subplots(figsize=(22, 17))
    ax.set_aspect('equal')
    colors = {'Processing': 'orangered', 'Cold & Staples': 'deepskyblue', 'Storage': 'gold', 'Washing': 'limegreen'}

    for square_id, (x, y) in ID_TO_COORDS.items():
        rect = patches.Rectangle((x - SQUARE_SIDE / 2, y - SQUARE_SIDE / 2), SQUARE_SIDE, SQUARE_SIDE,
                                 linewidth=0.5, edgecolor='black', facecolor='#DDDDDD')
        ax.add_patch(rect)

    for square_id, module in chromosome.items():
        x, y = ID_TO_COORDS[square_id]
        rect = patches.Rectangle((x - SQUARE_SIDE / 2, y - SQUARE_SIDE / 2), SQUARE_SIDE, SQUARE_SIDE,
                                 linewidth=1, edgecolor='black', facecolor=colors.get(module, 'white'))
        ax.add_patch(rect)
        ax.text(x, y, str(square_id), ha='center', va='center', fontsize=5, color='white')

    for module_name, points in FIXED_POINTS.items():
        if point_id in ID_TO_COORDS:
            x, y = ID_TO_COORDS[point_id]
            label_text = f"TARGET\n{module_name}"
            ax.text(x, y, label_text, ha='center', va='center', fontsize=8, color='white', weight='bold',
                    bbox=dict(facecolor=colors.get(module_name, 'darkviolet'), alpha=0.9, boxstyle='round,pad=0.2'))

    # 更新图表标题以显示所有关键分数
    title_text = (f"布局优化 - {generation}\n"
                  f"总分: {fitness_scores['total']:.2f} | "
                  f"邻近度: {fitness_scores['proximity']:.3f} | "
                  f"矩形度: {fitness_scores['rectangularity']:.3f}\n"
                  f"空洞惩罚: {fitness_scores['hole_penalty']:.0f} | "
                  f"边界惩罚: {fitness_scores['shared_border_penalty']:.0f}")
    plt.title(title_text, fontsize=16)
    plt.xlabel("X Coordinate (cm)")
    plt.ylabel("Y Coordinate (cm)")
    ax.autoscale_view()
    plt.grid(True, linestyle='--', alpha=0.2)

    legend_patches = [patches.Patch(color=color, label=name) for name, color in colors.items()]
    legend_patches.append(patches.Patch(color='#DDDDDD', label='Main Corridor / Aisle'))
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# --- 3. 主程序入口 ---
if __name__ == '__main__':
    print("重要提示：启动优化，采用【三位一体】最终策略！")
    print(f"  - 形状权重: 矩形度({RECTANGULARITY_WEIGHT}), 空洞({HOLE_PENALTY_WEIGHT}), 边界({SHARED_BORDER_WEIGHT})")
    print(f"  - 位置权重: 邻近度({PROXIMITY_WEIGHT})")

    try:
        start_time = time.time()
        population = [create_individual() for _ in range(POPULATION_SIZE)]
        print(f"\n初始化 {POPULATION_SIZE} 个随机布局成功！耗时 {time.time() - start_time:.2f} 秒。")

        best_fitness_so_far = -float('inf')
        best_chromosome = None
        best_scores = {}

        print("开始遗传算法迭代优化...")
        for gen in range(GENERATIONS):
            population_with_fitness = [(chrom, calculate_fitness(chrom)) for chrom in population]
            population_with_fitness.sort(key=lambda x: x[1]['total'], reverse=True)

            current_best_chromosome, current_best_scores = population_with_fitness[0]

            if current_best_scores['total'] > best_fitness_so_far:
                best_fitness_so_far = current_best_scores['total']
                best_chromosome = current_best_chromosome
                best_scores = current_best_scores
                # 更新日志输出
                print(f"第 {gen + 1:>3}/{GENERATIONS} 代 | 新最优解! 总分: {best_scores['total']:.2f} "
                      f"(邻: {best_scores['proximity']:.2f}, "
                      f"矩: {best_scores['rectangularity']:.2f}, "
                      f"空: {best_scores['hole_penalty']:.0f}, "
                      f"边: {best_scores['shared_border_penalty']:.0f})")

            if (gen + 1) % 50 == 0:
                avg_fitness = np.mean([s['total'] for c, s in population_with_fitness])
                print(f"--- 第 {gen + 1} 代 / {GENERATIONS} --- 当前种群平均适应度: {avg_fitness:.2f} ---")

            next_generation = []
            if best_chromosome:
                next_generation.append(best_chromosome)

            while len(next_generation) < POPULATION_SIZE:
                parent1 = selection(population_with_fitness)
                parent2 = selection(population_with_fitness)
                if not parent1 or not parent2: continue
                child1, child2 = crossover(parent1, parent2)
                next_generation.append(mutate(child1))
                if len(next_generation) < POPULATION_SIZE:
                    next_generation.append(mutate(child2))

            population = next_generation

        print("\n优化完成!")
        print(f"找到的最优布局总适应度为: {best_scores['total']:.2f}")
        print(f"  - 最终邻近度得分: {best_scores['proximity']:.3f}")
        print(f"  - 最终矩形度得分: {best_scores['rectangularity']:.3f}")
        print(f"  - 最终空洞惩罚值: {best_scores['hole_penalty']:.0f}")
        print(f"  - 最终边界惩罚值: {best_scores['shared_border_penalty']:.0f}")

        if best_chromosome:
            for module_name in MODULE_AREAS.keys():
                module_squares = {sq for sq, mod in best_chromosome.items() if mod == module_name}
                print(f"\n模块 '{module_name}' (面积: {len(module_squares)}), 方块ID列表:")
                print(sorted(list(module_squares)))

            visualize_layout(best_chromosome, f"Final Result (Gen {GENERATIONS})", best_scores)

    except RuntimeError as e:
        print(f"\n程序错误：{e}")