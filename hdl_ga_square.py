import random
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import numpy as np
import time

# --- 1. 数据准备和配置 ---

# 模块间的最小间隔（单位：方块数）
MODULE_SPACING = 1.5
SQUARE_SIDE = 80

# 遗传算法参数
POPULATION_SIZE = 100
GENERATIONS = 400
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5

# 【修改】适应度函数权重，替代了 PRIORITY_FACTOR
# 邻近度得分的权重（更高的优先级）
PROXIMITY_WEIGHT = 10.0
# 矩形度得分的权重（次要优先级）
RECTANGULARITY_WEIGHT = 10.0
# 【新增】模块不连续的惩罚值 (极高惩罚，确保连续性是硬约束)
DISCONNECTED_PENALTY = -1000.0

# 功能模块定义
MODULE_AREAS = {
    'Processing': 89,
    'Cold & Staples': 32,
    'Storage': 40,
    'Washing': 82,
}

FIXED_POINTS = {
    'Storage': [340, 356],  # 仓储模块 -> 邻近进货口
    'Processing': [154, 177],  # 加工出品模块 -> 邻近目标点
    'Washing': [120, 143],  # 清洗模块 -> 邻近目标点
    'Cold & Staples': [300, 307]  # 凉菜及主食间 -> 邻近目标点
}

# 坐标原始文本数据
COORDINATE_DATA = """
1: (40, 41)2: (120, 41)3: (200, 41)4: (280, 41)5: (360, 41)6: (440, 41)7: (520, 41)8: (600, 41)9: (680, 41)10: (760, 41)11: (840, 41)12: (920, 41)13: (1000, 41)14: (1080, 41)15: (1160, 41)16: (1240, 41)17: (1320, 41)18: (1400, 41)19: (1480, 41)20: (1560, 41)21: (1640, 41)22: (1720, 41)23: (1800, 41)24: (1880, 41)25: (1960, 41)26: (2040, 41)27: (2120, 41)28: (2200, 41)29: (40, 121)30: (120, 121)31: (200, 121)32: (280, 121)33: (360, 121)34: (440, 121)35: (520, 121)36: (600, 121)37: (680, 121)38: (760, 121)39: (840, 121)40: (920, 121)41: (1000, 121)42: (1080, 121)43: (1160, 121)44: (1240, 121)45: (1320, 121)46: (1400, 121)47: (1480, 121)48: (1560, 121)49: (1640, 121)50: (1720, 121)51: (1800, 121)52: (1880, 121)53: (1960, 121)54: (2040, 121)55: (2120, 121)56: (2200, 121)57: (40, 201)58: (120, 201)59: (200, 201)60: (280, 201)61: (360, 201)62: (440, 201)63: (520, 201)64: (600, 201)65: (680, 201)66: (760, 201)67: (840, 201)68: (920, 201)69: (1000, 201)70: (1080, 201)71: (1160, 201)72: (1240, 201)73: (1320, 201)74: (1400, 201)75: (1480, 201)76: (1560, 201)77: (1640, 201)78: (1720, 201)79: (1800, 201)80: (1880, 201)81: (1960, 201)82: (2040, 201)83: (2120, 201)84: (2200, 201)85: (40, 281)86: (120, 281)87: (200, 281)88: (280, 281)89: (360, 281)90: (440, 281)91: (520, 281)92: (600, 281)93: (680, 281)94: (760, 281)95: (840, 281)96: (920, 281)97: (1000, 281)98: (1080, 281)99: (1160, 281)100: (1240, 281)101: (1320, 281)102: (1400, 281)103: (1480, 281)104: (1560, 281)105: (1640, 281)106: (1720, 281)107: (1800, 281)108: (40, 361)109: (120, 361)110: (200, 361)111: (280, 361)112: (360, 361)113: (440, 361)114: (520, 361)115: (600, 361)116: (680, 361)117: (760, 361)118: (840, 361)119: (920, 361)120: (1000, 361)121: (1080, 361)122: (1160, 361)123: (1240, 361)124: (1320, 361)125: (1400, 361)126: (1480, 361)127: (1560, 361)128: (1640, 361)129: (1720, 361)130: (1800, 361)131: (40, 441)132: (120, 441)133: (200, 441)134: (280, 441)135: (360, 441)136: (440, 441)137: (520, 441)138: (600, 441)139: (680, 441)140: (760, 441)141: (840, 441)142: (920, 441)143: (1000, 441)144: (1080, 441)145: (1160, 441)146: (1240, 441)147: (1320, 441)148: (1400, 441)149: (1480, 441)150: (1560, 441)151: (1640, 441)152: (1720, 441)153: (1800, 441)154: (40, 521)155: (120, 521)156: (200, 521)157: (280, 521)158: (360, 521)159: (440, 521)160: (520, 521)161: (600, 521)162: (680, 521)163: (760, 521)164: (840, 521)165: (920, 521)166: (1000, 521)167: (1080, 521)168: (1160, 521)169: (1240, 521)170: (1320, 521)171: (1400, 521)172: (1480, 521)173: (1560, 521)174: (1640, 521)175: (1720, 521)176: (1800, 521)177: (40, 601)178: (120, 601)179: (200, 601)180: (280, 601)181: (360, 601)182: (440, 601)183: (520, 601)184: (600, 601)185: (680, 601)186: (760, 601)187: (840, 601)188: (920, 601)189: (1000, 601)190: (1080, 601)191: (1160, 601)192: (1240, 601)193: (1320, 601)194: (1400, 601)195: (1480, 601)196: (1560, 601)197: (1640, 601)198: (1720, 601)199: (1800, 601)200: (1880, 601)201: (40, 681)202: (120, 681)203: (200, 681)204: (280, 681)205: (360, 681)206: (440, 681)207: (520, 681)208: (600, 681)209: (680, 681)210: (760, 681)211: (840, 681)212: (920, 681)213: (1000, 681)214: (1080, 681)215: (1160, 681)216: (1240, 681)217: (1320, 681)218: (1400, 681)219: (1480, 681)220: (1560, 681)221: (1640, 681)222: (1720, 681)223: (1800, 681)224: (1880, 681)225: (40, 761)226: (120, 761)227: (200, 761)228: (280, 761)229: (360, 761)230: (440, 761)231: (520, 761)232: (600, 761)233: (680, 761)234: (760, 761)235: (840, 761)236: (920, 761)237: (1000, 761)238: (1080, 761)239: (1160, 761)240: (1240, 761)241: (1320, 761)242: (1400, 761)243: (1480, 761)244: (1560, 761)245: (1640, 761)246: (1720, 761)247: (1800, 761)248: (1880, 761)249: (40, 841)250: (120, 841)251: (200, 841)252: (280, 841)253: (360, 841)254: (440, 841)255: (520, 841)256: (600, 841)257: (680, 841)258: (760, 841)259: (840, 841)260: (920, 841)261: (1000, 841)262: (1080, 841)263: (1160, 841)264: (1240, 841)265: (1320, 841)266: (1400, 841)267: (1480, 841)268: (1560, 841)269: (1640, 841)270: (1720, 841)271: (40, 921)272: (120, 921)273: (200, 921)274: (280, 921)275: (360, 921)276: (440, 921)277: (520, 921)278: (600, 921)279: (680, 921)280: (760, 921)281: (840, 921)282: (920, 921)283: (1000, 921)284: (1080, 921)285: (1160, 921)286: (1240, 921)287: (1320, 921)288: (1400, 921)289: (1480, 921)290: (1560, 921)291: (1640, 921)292: (1720, 921)293: (40, 1001)294: (120, 1001)295: (200, 1001)296: (280, 1001)297: (360, 1001)298: (440, 1001)299: (520, 1001)300: (40, 1081)301: (120, 1081)302: (200, 1081)303: (280, 1081)304: (360, 1081)305: (440, 1081)306: (520, 1081)307: (40, 1161)308: (120, 1161)309: (200, 1161)310: (280, 1161)311: (360, 1161)312: (440, 1161)313: (520, 1161)314: (40, 1241)315: (120, 1241)316: (200, 1241)317: (280, 1241)318: (360, 1241)319: (440, 1241)320: (520, 1241)321: (2280, 41)322: (2280, 121)323: (2280, 201)324: (1880, 281)325: (1960, 281)326: (2040, 281)327: (2120, 281)328: (2200, 281)329: (2280, 281)330: (1880, 361)331: (1880, 441)332: (1880, 521)333: (1960, 521)334: (1960, 601)335: (1960, 681)336: (1960, 761)337: (1800, 841)338: (1880, 841)339: (1960, 841)340: (1800, 921)341: (600, 1001)342: (680, 1001)343: (760, 1001)344: (840, 1001)345: (920, 1001)346: (1000, 1001)347: (1080, 1001)348: (1160, 1001)349: (1240, 1001)350: (1320, 1001)351: (1400, 1001)352: (1480, 1001)353: (1560, 1001)354: (1640, 1001)355: (1720, 1001)356: (1800, 1001)357: (600, 1081)358: (600, 1161)359: (600, 1241)360: (40, 1321)361: (120, 1321)362: (200, 1321)363: (280, 1321)364: (360, 1321)365: (440, 1321)366: (520, 1321)367: (600, 1321)
"""

def setup_floor_plan(coordinate_text):
    id_to_coords = {}
    id_to_neighbors = {}
    pattern = re.compile(r'(\d+):\s*\((\d+),\s*(\d+)\)')
    matches = pattern.findall(coordinate_text)
    for match in matches:
        square_id, x, y = map(int, match)
        id_to_coords[square_id] = (x, y)
    coords_to_id = {v: k for k, v in id_to_coords.items()}
    for square_id, (x, y) in id_to_coords.items():
        neighbors = []
        for dx, dy in [(0, SQUARE_SIDE), (0, -SQUARE_SIDE), (SQUARE_SIDE, 0), (-SQUARE_SIDE, 0)]:
            neighbor_coord = (x + dx, y + dy)
            if neighbor_coord in coords_to_id:
                neighbors.append(coords_to_id[neighbor_coord])
        id_to_neighbors[square_id] = neighbors
    return id_to_coords, id_to_neighbors


ID_TO_COORDS, ID_TO_NEIGHBORS = setup_floor_plan(COORDINATE_DATA)
ALL_SQUARE_IDS = list(ID_TO_COORDS.keys())
fixed_point_ids = {id for ids in FIXED_POINTS.values() for id in ids}
AVAILABLE_SQUARES = [sq for sq in ALL_SQUARE_IDS if sq not in fixed_point_ids]


def get_buffer_zone(square_ids, distance):
    if not square_ids: return set()
    buffer, queue, visited = set(square_ids), deque([(sq, 0) for sq in square_ids]), set(square_ids)
    while queue:
        current_sq, dist = queue.popleft()
        if dist >= distance: continue
        for neighbor in ID_TO_NEIGHBORS.get(current_sq, []):
            if neighbor not in visited:
                visited.add(neighbor)
                buffer.add(neighbor)
                queue.append((neighbor, dist + 1))
    return buffer


def create_individual():
    max_retries = 300
    for _ in range(max_retries):
        chromosome = {}
        squares_for_this_try = set(AVAILABLE_SQUARES)
        # 优先放置大模块，有助于减少碎片化
        sorted_modules = sorted(MODULE_AREAS.items(), key=lambda item: item[1], reverse=True)

        all_modules_placed = True
        for module_name, area in sorted_modules:
            module_squares = []

            # Try to place module near its fixed points if any
            potential_seeds = []
            if module_name in FIXED_POINTS:
                for fp in FIXED_POINTS[module_name]:
                    # Find available neighbors around fixed points
                    for neighbor in ID_TO_NEIGHBORS.get(fp, []):
                        if neighbor in squares_for_this_try:
                            potential_seeds.append(neighbor)

            # If no suitable seeds near fixed points, pick a random available square
            if not potential_seeds:
                if not squares_for_this_try:
                    all_modules_placed = False
                    break
                seed = random.choice(list(squares_for_this_try))
            else:
                seed = random.choice(potential_seeds)

            module_squares.append(seed)
            current_placement_options = squares_for_this_try.copy()
            current_placement_options.remove(seed)

            for _ in range(area - 1):
                # Ensure new square is adjacent to existing module squares
                frontier = {neighbor for sq in module_squares for neighbor in ID_TO_NEIGHBORS.get(sq, []) if
                            neighbor in current_placement_options}
                if not frontier:
                    all_modules_placed = False
                    break
                new_square = random.choice(list(frontier))
                module_squares.append(new_square)
                current_placement_options.remove(new_square)

            if not all_modules_placed: break

            for sq in module_squares: chromosome[sq] = module_name

            # Remove placed module squares and their buffer zone from available squares
            buffer_to_remove = get_buffer_zone(module_squares, MODULE_SPACING)
            squares_for_this_try -= buffer_to_remove

        if all_modules_placed: return chromosome

    raise RuntimeError(f"在 {max_retries} 次尝试后仍无法生成一个满足间隔约束的初始布局。\n"
                       "可能原因：地图空间不足、模块总面积过大或间隔要求过高。")


def is_move_valid(square_to_add, target_module, chromosome):
    # Check for buffer zone violations with other modules
    for module_name in MODULE_AREAS:
        if module_name == target_module: continue  # Don't check against itself
        module_squares = {sq for sq, mod in chromosome.items() if mod == module_name}
        if square_to_add in get_buffer_zone(module_squares, MODULE_SPACING):
            return False
    return True


def mutate(chromosome):
    if random.random() > MUTATION_RATE:
        return chromosome

    # Create a deep copy to avoid modifying the original chromosome if mutation fails
    new_chromosome = chromosome.copy()

    module_to_mutate = random.choice(list(MODULE_AREAS.keys()))
    current_module_squares = {sq for sq, mod in new_chromosome.items() if mod == module_to_mutate}

    if not current_module_squares:  # Cannot mutate an empty module
        return new_chromosome

    # Find squares on the border of the module that can be removed
    removable_squares = []
    for sq in current_module_squares:
        # A square is removable if removing it doesn't disconnect the module
        temp_module_squares = current_module_squares - {sq}
        if is_module_continuous(temp_module_squares, ID_TO_NEIGHBORS):
            removable_squares.append(sq)

    if not removable_squares:
        return new_chromosome  # No safe squares to remove

    sq_to_remove = random.choice(removable_squares)

    # Find potential squares to add, which are adjacent to the module and in aisle
    # and don't violate buffer zones of other modules
    potential_add_squares = []
    # Identify all squares currently occupied by other modules or their buffer zones
    occupied_or_buffered = set()
    for mod_name, area in MODULE_AREAS.items():
        if mod_name != module_to_mutate:
            other_module_squares = {sq for sq, mod in new_chromosome.items() if mod == mod_name}
            occupied_or_buffered.update(get_buffer_zone(other_module_squares, MODULE_SPACING))

    # Also exclude squares in the module being mutated (except the one being removed)
    occupied_or_buffered.update(current_module_squares - {sq_to_remove})

    for sq in current_module_squares:
        for neighbor in ID_TO_NEIGHBORS.get(sq, []):
            if neighbor not in occupied_or_buffered and neighbor in ALL_SQUARE_IDS:
                potential_add_squares.append(neighbor)

    # Remove the square that is about to be removed from the potential add list (if it was there)
    if sq_to_remove in potential_add_squares:
        potential_add_squares.remove(sq_to_remove)

    if not potential_add_squares:
        return new_chromosome  # No suitable squares to add

    # Filter potential_add_squares to ensure they maintain contiguity if added
    valid_add_squares = []
    for sq_to_add in potential_add_squares:
        # Temporarily add the square and check if the module remains continuous with the new square
        temp_module_squares_for_contiguity_check = current_module_squares - {sq_to_remove} | {sq_to_add}
        if is_module_continuous(temp_module_squares_for_contiguity_check, ID_TO_NEIGHBORS):
            valid_add_squares.append(sq_to_add)

    if not valid_add_squares:
        return new_chromosome  # No valid squares to add that maintain contiguity

    sq_to_add = random.choice(valid_add_squares)

    # Apply the mutation
    del new_chromosome[sq_to_remove]
    new_chromosome[sq_to_add] = module_to_mutate

    return new_chromosome


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def calculate_avg_distance(module_squares, target_points):
    if not module_squares: return 1e9  # Penalize empty modules
    total_dist = 0
    # Ensure all target points actually exist in ID_TO_COORDS
    valid_target_coords = [ID_TO_COORDS[tp] for tp in target_points if tp in ID_TO_COORDS]

    # If there are no valid target points, this module should be heavily penalized.
    # This might indicate an issue with fixed_points data or map.
    if not valid_target_coords: return 1e9

    for sq_id in module_squares:
        if sq_id not in ID_TO_COORDS:
            # This should ideally not happen if chromosome is well-formed
            continue
        sq_coord = ID_TO_COORDS[sq_id]
        min_dist_to_target = min([manhattan_distance(sq_coord, tc) for tc in valid_target_coords])
        total_dist += min_dist_to_target

    return total_dist / len(module_squares)  # Return average distance directly


def selection(population_with_fitness):
    # Ensure tournament size does not exceed population size
    actual_tournament_size = min(TOURNAMENT_SIZE, len(population_with_fitness))

    # Handle edge case where population is too small (e.g., in very early generations or small POPULATION_SIZE)
    if actual_tournament_size == 0:
        return {}  # Return an empty chromosome if no population, or raise an error as appropriate

    tournament = random.sample(population_with_fitness, actual_tournament_size)
    tournament.sort(key=lambda x: x[1]['total'], reverse=True)  # Sort by total fitness

    # Always return the chromosome part of the best individual in the tournament
    return tournament[0][0]


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Identify common squares in both parents
        common_squares = list(set(parent1.keys()) & set(parent2.keys()))

        # If no common squares, or only one, simple swap might not be meaningful for a 'split'
        if not common_squares:
            return parent1.copy(), parent2.copy()

            # If there's only one common square, we can't choose a split_point between 1 and 0.
        # In this case, just return copies or perform a very specific 1-square swap if desired.
        if len(common_squares) == 1:
            # Option 1: Don't perform the swap if only one common square (effectively, no "split point")
            return parent1.copy(), parent2.copy()  # <-- UNCOMMENTED THIS LINE

        # Choose a random split point within the common squares
        split_point = random.randint(1, len(common_squares) - 1)

        # Swap assignments for squares after the split point
        for i in range(split_point, len(common_squares)):
            sq = common_squares[i]
            # Swap assignments for the chosen square between child1 and child2
            val1 = parent1.get(
                sq)  # Use .get() to handle cases where sq might not be in a parent (though common_squares implies it is)
            val2 = parent2.get(sq)
            child1[sq] = val2
            child2[sq] = val1

        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


def is_module_continuous(module_squares_set, id_to_neighbors_map):
    """
    Checks if a set of squares forms a continuous block.
    Args:
        module_squares_set (set): A set of square IDs belonging to a module.
        id_to_neighbors_map (dict): Mapping from square ID to its neighbors.
    Returns:
        bool: True if the module squares are continuous, False otherwise.
    """
    if not module_squares_set:
        return True  # An empty module is trivially continuous

    start_node = next(iter(module_squares_set))  # Pick an arbitrary starting node

    visited = set()
    queue = deque([start_node])
    visited.add(start_node)

    while queue:
        current_sq = queue.popleft()
        for neighbor in id_to_neighbors_map.get(current_sq, []):
            if neighbor in module_squares_set and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited == module_squares_set


def calculate_fitness(chromosome):
    """
    【重大修改】适应度函数采用权重加和策略，并引入不连续惩罚。
    """
    total_fitness = 0.0

    # Initialize scores to 0 for aggregation
    proximity_score = 0.0
    rectangularity_score = 0.0

    # Keep track of assigned squares to check for overlaps
    assigned_squares_count = {}  # {square_id: count}

    for module_name, area_required in MODULE_AREAS.items():
        module_squares = {sq for sq, mod in chromosome.items() if mod == module_name}

        # --- 1. 模块连续性 (硬约束 - 高度惩罚) ---
        if not is_module_continuous(module_squares, ID_TO_NEIGHBORS):
            total_fitness += DISCONNECTED_PENALTY
            # If a module is disconnected, its other scores are likely irrelevant,
            # but we continue to calculate for debugging/display purposes.

        # Check if the module has the required area
        if len(module_squares) != area_required:
            total_fitness += DISCONNECTED_PENALTY  # Penalize if area requirement not met

        # Track square assignments for overlap detection
        for sq_id in module_squares:
            assigned_squares_count[sq_id] = assigned_squares_count.get(sq_id, 0) + 1

        # --- 2. 模块靠近目标点 (高优先级) ---
        target_points = FIXED_POINTS.get(module_name, [])
        if target_points and module_squares:
            avg_dist = calculate_avg_distance(module_squares, target_points)

            MAX_POSSIBLE_AVG_DIST = 3000  # This needs to be estimated from your map
            normalized_proximity = 1.0 - (avg_dist / MAX_POSSIBLE_AVG_DIST)
            proximity_score += max(0, normalized_proximity)  # Ensure score is not negative

        # --- 3. 模块形状规整 (次要优先级) ---
        rectangularity = calculate_rectangularity(module_squares)
        rectangularity_score += rectangularity  # Rectangularity is already 0-1, 1 is best

    # --- 4. 检查重叠 (硬约束 - 高度惩罚) ---
    # Squares assigned to more than one module
    overlap_penalty_multiplier = 0
    for sq_id, count in assigned_squares_count.items():
        if count > 1:
            overlap_penalty_multiplier += 1

    total_fitness += (proximity_score * PROXIMITY_WEIGHT)
    total_fitness += (rectangularity_score * RECTANGULARITY_WEIGHT)

    if overlap_penalty_multiplier > 0:
        total_fitness += (overlap_penalty_multiplier * DISCONNECTED_PENALTY * 2)  # Heavier penalty for overlaps

    # Check for modules not meeting their area requirements or being empty
    for module_name, area_required in MODULE_AREAS.items():
        module_squares = {sq for sq, mod in chromosome.items() if mod == module_name}
        if len(module_squares) != area_required:
            total_fitness += DISCONNECTED_PENALTY  # Apply penalty if area not met

    # Return a dictionary for better insight into individual score components
    scores = {
        "total": total_fitness,
        "proximity": proximity_score,
        "rectangularity": rectangularity_score
    }
    return scores


def visualize_layout(chromosome, generation, fitness_scores):
    fig, ax = plt.subplots(figsize=(22, 17))
    ax.set_aspect('equal')
    colors = {'Processing': 'orangered', 'Cold & Staples': 'deepskyblue', 'Storage': 'gold', 'Washing': 'limegreen'}

    # Draw all available squares as background
    for square_id, (x, y) in ID_TO_COORDS.items():
        rect = patches.Rectangle((x - SQUARE_SIDE / 2, y - SQUARE_SIDE / 2), SQUARE_SIDE, SQUARE_SIDE,
                                 linewidth=0.5, edgecolor='black', facecolor='#DDDDDD')
        ax.add_patch(rect)

    # Draw modules
    for square_id, module in chromosome.items():
        x, y = ID_TO_COORDS[square_id]
        rect = patches.Rectangle((x - SQUARE_SIDE / 2, y - SQUARE_SIDE / 2), SQUARE_SIDE, SQUARE_SIDE,
                                 linewidth=1, edgecolor='black', facecolor=colors.get(module, 'white'))
        ax.add_patch(rect)
        # Display square ID only for placed squares
        ax.text(x, y, str(square_id), ha='center', va='center', fontsize=5, color='white')

    # Draw fixed points/targets
    for module_name, points in FIXED_POINTS.items():
        for point_id in points:
            if point_id in ID_TO_COORDS:
                x, y = ID_TO_COORDS[point_id]
                label_text = f"TARGET\n{module_name}"
                ax.text(x, y, label_text, ha='center', va='center', fontsize=8, color='white', weight='bold',
                        bbox=dict(facecolor=colors.get(module_name, 'darkviolet'), alpha=0.9, boxstyle='round,pad=0.2'))

    title_text = f"Layout Optimization - Generation {generation}\n" \
                 f"Total Fitness: {fitness_scores['total']:.2f} | " \
                 f"Proximity Score: {fitness_scores['proximity']:.4f} | " \
                 f"Rectangularity Score: {fitness_scores['rectangularity']:.4f}"
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


def calculate_rectangularity(square_ids):
    if not square_ids: return 0

    # Convert square IDs to coordinates
    coords = [ID_TO_COORDS[sq_id] for sq_id in square_ids if sq_id in ID_TO_COORDS]
    if not coords: return 0

    # Find min/max X and Y coordinates to define the bounding box
    min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
    min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)

    bounding_box_squares_width = (max_x - min_x) / SQUARE_SIDE + 1
    bounding_box_squares_height = (max_y - min_y) / SQUARE_SIDE + 1
    bounding_box_area_in_squares = bounding_box_squares_width * bounding_box_squares_height

    return len(square_ids) / bounding_box_area_in_squares if bounding_box_area_in_squares > 0 else 0


# --- 主程序 ---
if __name__ == '__main__':
    print("重要提示：启动优化，严格遵循以下优先级：")
    print("  1. 模块连续性 & 模块面积 & 模块间距 (硬约束 - 高度惩罚，确保可行性)")
    print("  2. 模块靠近目标点 (核心优化目标 - 高权重)")
    print("  3. 模块形状规整 (次要优化目标 - 较低权重)")

    try:
        start_time = time.time()
        population = [create_individual() for _ in range(POPULATION_SIZE)]
        print(f"初始化成功！耗时 {time.time() - start_time:.2f} 秒。")

        best_fitness_so_far = -float('inf')
        best_chromosome = None
        best_scores = {}

        print("开始遗传算法优化...")
        for gen in range(GENERATIONS):
            population_with_fitness = [(chrom, calculate_fitness(chrom)) for chrom in population]
            population_with_fitness.sort(key=lambda x: x[1]['total'], reverse=True)

            current_best_chromosome, current_best_scores = population_with_fitness[0]

            if current_best_scores['total'] > best_fitness_so_far:
                best_fitness_so_far = current_best_scores['total']
                best_chromosome = current_best_chromosome
                best_scores = current_best_scores
                print(f"第 {gen + 1}/{GENERATIONS} 代 | 新最优解! 总分: {best_scores['total']:.2f} "
                      f"(邻近度: {best_scores['proximity']:.4f}, 矩形度: {best_scores['rectangularity']:.4f})")

            # Print average fitness every 50 generations to see population diversity/convergence
            if (gen + 1) % 50 == 0:
                avg_fitness = np.mean([s['total'] for c, s in population_with_fitness])
                print(f"--- 第 {gen + 1} 代 / {GENERATIONS} --- 当前种群平均适应度: {avg_fitness:.2f} ---")

            # Elitism: carry over the best individual to the next generation
            next_generation = [best_chromosome]

            # Fill the rest of the next generation using selection and crossover/mutation
            while len(next_generation) < POPULATION_SIZE:
                parent1 = selection(population_with_fitness)  # Pass population with detailed fitness
                parent2 = selection(population_with_fitness)

                child1, child2 = crossover(parent1, parent2)  # Crossover returns two children

                # Apply mutation to children
                next_generation.append(mutate(child1))
                if len(next_generation) < POPULATION_SIZE:  # Ensure we don't exceed population size
                    next_generation.append(mutate(child2))

            population = next_generation

        print("\n优化完成!")
        print(f"找到的最优布局总适应度为: {best_scores['total']:.2f}")
        print(f"  - 最终邻近度得分 (高权重): {best_scores['proximity']:.2f}")
        print(f"  - 最终矩形度得分 (低权重): {best_scores['rectangularity']:.2f}")

        if best_chromosome:
            for module_name in MODULE_AREAS.keys():
                module_squares = [sq for sq, mod in best_chromosome.items() if mod == module_name]
                print(f"{module_name} 模块（面积: {len(module_squares)}）:")
                coords_list = []
                for sq_id in sorted(module_squares):
                    coords_list.append(ID_TO_COORDS[sq_id])
                print(coords_list)
            visualize_layout(best_chromosome, f"Final Result (Gen {GENERATIONS})", best_scores)

    except RuntimeError as e:
        print(f"\n错误：{e}")