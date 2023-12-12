from Sol import *
import matplotlib.pyplot as plt


# TODO: 调优策略: 1. 电量和路程惩罚项调优、2. 局部搜索算法调整

# def cost_route(r, vehicle_type=1, depart_time=0, penalty=False, penalty_lam=0):
#     """
#     计算单独一条路的cost。包含
#     1. 车辆固定成本
#     2. travel cost
#     3. waiting cost
#     4. charging cost
#     :param r: 如[0, 1, 0]
#     :param vehicle_type: 1或2,默认为1
#     :param depart_time: 出发时间,默认为0
#     :return: 这条路的成本
#     TODO: 重新实现目标函数，加入惩罚项
#     """
#     total_fix_cost = 0
#     # if vehicle_type == 1:
#     #     total_fix_cost = df_vehicle.iloc[0][8]
#     # else:
#     #     total_fix_cost = df_vehicle.iloc[1][8]
#     total_fix_cost = df_vehicle.loc[vehicle_type, 'vehicle_cost']
#
#     total_travel_cost = 0
#     for i in range(len(r)-1):
#         from_point = r[i]
#         to_point = r[i+1]
#         total_travel_cost += df_distance.loc[(from_point, to_point), 'distance'] * df_vehicle.loc[vehicle_type, 'unit_trans_cost'] / 1000
#
#     total_waiting_cost = 0
#     path_time = depart_time
#     path_time_list = [depart_time]  # 注意这个list存储的均为到达时间
#     for i in range(len(r)-1):
#         from_point = r[i]
#         to_point = r[i+1]
#         path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
#         path_time_list.append(path_time)
#         if to_point != 0:
#             total_waiting_cost += max((df_nodes.loc[to_point, 'first_receive_tm'] - path_time), 0) * waiting_cost
#             path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
#         if to_point == 0:
#             path_time += 60
#         else:
#             path_time += service_time
#         total_waiting_cost += (r.count(0) - 2) * waiting_cost * 60
#
#     total_charging_cost = 0
#     for i in range(len(r)):
#         if r[i] in index_recharge:
#             total_charging_cost += charging_cost * service_time
#
#     total_cost = total_fix_cost + total_travel_cost + total_waiting_cost + total_charging_cost
#
#     return total_cost

def find_nearest_recharge_station(node_i, node_j):
    """
    Find the recharge station with the minimum sum of distances to two customer nodes.
    :param df_distance: DataFrame containing distances between nodes.
    :param customer_nodes: Tuple of two customer node indices.
    :param recharge_stations: List of indices of recharge stations.
    :return: Index of the nearest recharge station.
    """
    min_distance = float('inf')
    nearest_station = None

    # Iterate over each recharge station
    for station in index_recharge:
        # Calculate the sum of distances from the station to both customer nodes
        distance_to_customers = df_distance.loc[(node_i, station), 'distance'] + df_distance.loc[(station, node_j), 'distance']

        # Check if this is the shortest distance found so far
        if distance_to_customers < min_distance:
            min_distance = distance_to_customers
            nearest_station = station

    return nearest_station

def expand(r, vehicle_type, i, extension=1, R=None, W=None, V=None, T=None, T_w=None, d=None, f=None):
    """
    判断i节点处是否可以进行类型为extension的拓展
    :param r: 路径
    :param i: 节点
    :param extension: 拓展类型
    :return: 是否可以拓展、拓展一步增加的成本、更新后的R/W/V/T/T_w/d/f
    """
    if (i+2) < len(r):

        if extension == 1:
            T[i+1] = max(T[i] + df_distance.loc[(r[i], r[i+1]), "spend_tm"], df_nodes.loc[r[i+1], "first_receive_tm"]) + service_time
            T_w[i+1] = T_w[i] + max(0, df_nodes.loc[r[i+1], "first_receive_tm"] - T[i] - df_distance.loc[(r[i], r[i+1]), "spend_tm"])
            W[i+1] = W[i] + df_nodes.loc[r[i+1], "pack_total_weight"]
            V[i+1] = V[i] + df_nodes.loc[r[i+1], "pack_total_volume"]
            R[i+1] = R[i] + df_distance.loc[(r[i], r[i+1]), "distance"]
            d[i+1] = d[i] + df_distance.loc[(r[i], r[i+1]), "distance"]
            time_wait = max(0, df_nodes.loc[r[i+1], "first_receive_tm"] - T[i] - df_distance.loc[(r[i], r[i+1]), "spend_tm"])
            f[i+1] = f[i] + waiting_cost * time_wait + df_distance.loc[(r[i], r[i+1]), 'distance'] * df_vehicle.loc[vehicle_type, 'unit_trans_cost'] / 1000
            delta_cost = f[i+1] - f[i]
            premise1 = (T[i+1] >= service_time) and (T[i+1] <= service_time + df_nodes.loc[r[i+1], "last_receive_tm"]) and (T[i+1] <= time_horizon)
            premise2 = (W[i+1] <= df_vehicle.loc[vehicle_type, "max_weight"])
            premise3 = (V[i+1] <= df_vehicle.loc[vehicle_type, "max_volume"])
            premise4 = R[i+1] <= df_vehicle.loc[vehicle_type, "driving_range"]
            premise = premise1 and premise2 and premise3 and premise4
            if not premise:
                return {'logic': False, 'delta_cost': delta_cost, 'extension': extension, 'charge': None, 'R': R, 'W': W, 'V': V, 'T': T, 'T_w': T_w, 'd': d, 'f': f}
            else:
                return {'logic': True, 'delta_cost': delta_cost, 'extension': extension, 'charge': None, 'R': R, 'W': W, 'V': V, 'T': T, 'T_w': T_w, 'd': d, 'f': f}

        if extension == 2:
            k = find_nearest_recharge_station(r[i], r[i+1])
            print(f"station: {k}")
            T[i+1] = max(T[i] + df_distance.loc[(r[i], k), "spend_tm"] + service_time + df_distance.loc[(k, r[i+1]), "spend_tm"], df_nodes.loc[r[i+1], "first_receive_tm"]) + service_time
            T_w[i+1] = T_w[i] + max(0, df_nodes.loc[r[i+1], "first_receive_tm"] - T[i] - df_distance.loc[(r[i], k), "spend_tm"] - service_time - df_distance.loc[(k, r[i+1]), "spend_tm"])
            W[i+1] = W[i] + df_nodes.loc[r[i+1], "pack_total_weight"]
            V[i+1] = V[i] + df_nodes.loc[r[i+1], "pack_total_volume"]
            R[i+1] = df_distance.loc[(k, r[i+1]), "distance"]
            d[i+1] = d[i] + df_distance.loc[(r[i], k), "distance"] + df_distance.loc[(k, r[i+1]), "distance"]
            time_wait = max(0, df_nodes.loc[r[i+1], "first_receive_tm"] - T[i] - df_distance.loc[(r[i], k), "spend_tm"] - service_time - df_distance.loc[(k, r[i+1]), "spend_tm"])
            f[i+1] = f[i] + waiting_cost * time_wait + (df_distance.loc[(r[i], k), 'distance'] + df_distance.loc[(k, r[i+1]), 'distance']) * df_vehicle.loc[vehicle_type, 'unit_trans_cost'] / 1000 + charging_cost * service_time
            delta_cost = f[i + 1] - f[i]
            premise1 = (T[i+1] >= service_time) and (T[i+1] <= service_time + df_nodes.loc[r[i+1], "last_receive_tm"]) and (T[i+1] <= time_horizon)
            premise2 = (W[i+1] <= df_vehicle.loc[vehicle_type, "max_weight"])
            premise3 = (V[i+1] <= df_vehicle.loc[vehicle_type, "max_volume"])
            premise4 = R[i+1] <= df_vehicle.loc[vehicle_type, "driving_range"]
            premise5 = R[i] + df_distance.loc[(r[i], k), "distance"] <= df_vehicle.loc[vehicle_type, "driving_range"]  # 还需要可以走到最左充电站
            premise = premise1 and premise2 and premise3 and premise4 and premise5
            if not premise:
                return {'logic': False, 'delta_cost': delta_cost, 'extension': extension, 'charge': k, 'R': R, 'W': W, 'V': V, 'T': T, 'T_w': T_w, 'd': d, 'f': f}
            else:
                return {'logic': True, 'delta_cost': delta_cost, 'extension': extension, 'charge': k, 'R': R, 'W': W, 'V': V, 'T': T, 'T_w': T_w, 'd': d, 'f': f}

        if extension == 3:
            T[i+1] = max(T[i] + df_distance.loc[(r[i], 0), "spend_tm"] + service_time_depot + df_distance.loc[(0, r[i+1]), "spend_tm"], df_nodes.loc[r[i+1], "first_receive_tm"]) + service_time
            T_w[i+1] = T_w[i] + max(0, df_nodes.loc[r[i+1], "first_receive_tm"] - T[i] - df_distance.loc[(r[i], 0), "spend_tm"] - service_time_depot - df_distance.loc[(0, r[i+1]), "spend_tm"])
            W[i+1] = df_nodes.loc[r[i+1], "pack_total_weight"]
            V[i+1] = df_nodes.loc[r[i+1], "pack_total_volume"]
            R[i+1] = df_distance.loc[(0, r[i+1]), "distance"]
            d[i+1] = d[i] + df_distance.loc[(r[i], 0), "distance"] + df_distance.loc[(0, r[i + 1]), "distance"]
            time_wait = max(0, df_nodes.loc[r[i+1], "first_receive_tm"] - T[i] - df_distance.loc[(r[i], 0), "spend_tm"] - service_time_depot - df_distance.loc[(0, r[i+1]), "spend_tm"]) + service_time_depot
            f[i+1] = f[i] + waiting_cost * time_wait + (df_distance.loc[(r[i], 0), 'distance'] + df_distance.loc[(0, r[i+1]), 'distance']) * df_vehicle.loc[vehicle_type, 'unit_trans_cost'] / 1000
            delta_cost = f[i + 1] - f[i]
            premise1 = (T[i+1] >= service_time) and (T[i+1] <= service_time + df_nodes.loc[r[i+1], "last_receive_tm"]) and (T[i+1] <= time_horizon)
            premise2 = (W[i+1] <= df_vehicle.loc[vehicle_type, "max_weight"])
            premise3 = (V[i+1] <= df_vehicle.loc[vehicle_type, "max_volume"])
            premise4 = R[i+1] <= df_vehicle.loc[vehicle_type, "driving_range"]
            premise5 = R[i] + df_distance.loc[(r[i], 0), "distance"] <= df_vehicle.loc[vehicle_type, "driving_range"]  # 可以到达depot
            premise = premise1 and premise2 and premise3 and premise4 and premise5
            if not premise:
                return {'logic': False, 'delta_cost': delta_cost, 'extension': extension, 'charge': None, 'R': R, 'W': W, 'V': V, 'T': T, 'T_w': T_w, 'd': d, 'f': f}
            else:
                return {'logic': True, 'delta_cost': delta_cost, 'extension': extension, 'charge': None, 'R': R, 'W': W, 'V': V, 'T': T, 'T_w': T_w, 'd': d, 'f': f}

    elif (i+2) == len(r):  # 有可能回不去需要找最左充电站，但是只考虑直接回去情况
        delta_cost = 0
        premise1 = T[i] + df_distance.loc[(r[i], 0), "spend_tm"] <= time_horizon
        premise2 = R[i] + df_distance.loc[(r[i], 0), "distance"] <= df_vehicle.loc[vehicle_type, "driving_range"]
        premise = premise1 and premise2
        if not premise:
            return {'logic': False, 'delta_cost': delta_cost, 'extension': extension, 'R': R, 'W': W, 'V': V, 'T': T, 'T_w': T_w, 'd': d, 'f': f}
        else:
            return {'logic': True, 'delta_cost': delta_cost, 'extension': extension, 'R': R, 'W': W, 'V': V, 'T': T, 'T_w': T_w, 'd': d, 'f': f}

# expand(test_route, vehicle_types[10], 1, , R=None, W=None, V=None, T=None, T_w=None, d=None, f=None)


# def time_checker(routes, departure_times):
#     time_table = routes[:]
#     for i in range(len(time_table)):
#         time_table[i][0] = departure_times[i]
#         over_time = False
#
#         if len(route) <= 2:
#             continue
#
#         for j in range(len(time_table[i])):
#             time_table[i][j+1] = time_table[i][j] + df_distance.loc[(r[i][j], r[i][j+1]), 'spend_tm']
#             if time_table[i][j+1] > df_nodes.loc[r[i][j+1], 'last_receive_tm']:
#                 over_time = True
#         if over_time:
#             print("Warning!!!")



def labeling(origin_r, vehicle_type, departure_time, if_must=True):
    """
    TODO: 标签算法（见论文5.5.7），
    :param r: 配送顺序r（r中可能还包含充电站节点，并不一定像论文中都是客户节点）
    :param vehicle_type: r上原本的配送车型
    :param departure_time: r原本的出发时间
    :return: r_ret为返回的路径（决定了充电站、返回配送中心）, vehicle_type_ret为决定的配送车型
    """

    # time_checker(r, departure_times)
    ## 移除所有的充电站
    r = origin_r[:]
    r = [x for x in r if x <= 1000 and x != 0]

    r = [0] + r + [0]  # 在首尾添加0

    R = [None] * len(r)  # 车辆离开节点时的里程状态
    W = [None] * len(r)  # 车辆离开节点时的载重状态，至多还可以装载多少重量
    V = [None] * len(r)  # 车辆离开节点时的容积状态，至多还可以装载多少容积
    T = [None] * len(r)  # 车辆离开节点时的时间
    T_w = [None] * len(r)  # 车辆离开节点时的累积等待时间
    d = [None] * len(r)  # 车辆离开节点时的累积行驶里程
    f = [None] * len(r)  # 车辆离开节点时的累积成本
    expand_list = []  # 内部子list，记录拓展节点和拓展类型
    charge = 0
    R[0] = 0
    W[0] = 0
    V[0] = 0
    T[0] = departure_time
    T_w[0] = 0
    d[0] = 0
    f[0] = 0  # 初始化

    for i in range(len(r)-2):  # 最后一个节点不需要拓展

        print(i)
        dic_list = []
        expand_11 = {'logic': False}
        expand_12 = {'logic': False}
        expand_13 = {'logic': False}

        expand_1 = expand(r, vehicle_type, i, 1, R[:], W[:], V[:], T[:], T_w[:], d[:], f[:], if_must)
        if expand_1['logic']:
            expand_11 = expand(r, vehicle_type, i+1, 1, expand_1['R'][:], expand_1['W'][:], expand_1['V'][:], expand_1['T'][:], expand_1['T_w'][:], expand_1['d'][:], expand_1['f'][:], if_must)
            expand_12 = expand(r, vehicle_type, i+1, 2, expand_1['R'][:], expand_1['W'][:], expand_1['V'][:], expand_1['T'][:], expand_1['T_w'][:], expand_1['d'][:], expand_1['f'][:], if_must)
            expand_13 = expand(r, vehicle_type, i+1, 3, expand_1['R'][:], expand_1['W'][:], expand_1['V'][:], expand_1['T'][:], expand_1['T_w'][:], expand_1['d'][:], expand_1['f'][:], if_must)
        dic_list.append(expand_1)

        expand_2 = expand(r, vehicle_type, i, 2, R[:], W[:], V[:], T[:], T_w[:], d[:], f[:], if_must)
        dic_list.append(expand_2)

        if r[i]!=0:
            expand_3 = expand(r, vehicle_type, i, 3, R[:], W[:], V[:], T[:], T_w[:], d[:], f[:], if_must)
        else:
            expand_3 = {'logic': False}
        dic_list.append(expand_3)

        if not (expand_12['logic'] or expand_13['logic']):

            feasible_dicts = [d for d in dic_list[-2:] if d['logic']]

            if len(feasible_dicts) == 0:
                print("infeasible")
                return origin_r, vehicle_type, departure_time

            min_cost_dict = min(feasible_dicts, key=lambda x: x['delta_cost'])
            extension_type = min_cost_dict['extension']
            charge = min_cost_dict['charge']
            expand_list.append([i, extension_type, charge])

            R = min_cost_dict['R']
            W = min_cost_dict['W']
            V = min_cost_dict['V']
            T = min_cost_dict['T']
            T_w = min_cost_dict['T_w']
            d = min_cost_dict['d']
            f = min_cost_dict['f']

        else:
            feasible_dicts = [d for d in dic_list if d['logic']]

            if not feasible_dicts:
                print("infeasible")
                return origin_r, vehicle_type, departure_time

            min_cost_dict = min(feasible_dicts, key=lambda x: x['delta_cost'])
            extension_type = min_cost_dict['extension']
            charge = min_cost_dict['charge']
            expand_list.append([i, extension_type, charge])

            R = min_cost_dict['R']
            W = min_cost_dict['W']
            V = min_cost_dict['V']
            T = min_cost_dict['T']
            T_w = min_cost_dict['T_w']
            d = min_cost_dict['d']
            f = min_cost_dict['f']

    expand_0 = expand(r, vehicle_type, len(r)-2, 1, R[:], W[:], V[:], T[:], T_w[:], d[:], f[:], if_must)
    if not expand_0['logic']:
        return origin_r, vehicle_type, departure_time

    index = 0  # 记录已经插入了几个节点，用于确定插入具体位置
    for expand_operation in expand_list:
        if expand_operation[1] == 2:
            r.insert(expand_operation[0]+1+index, charge)
            index += 1
        elif expand_operation[1] == 3:
            r.insert(expand_operation[0]+1+index, 0)
            index += 1

    r_ret = r
    vehicle_type_ret = vehicle_type
    departure_time_ret = departure_time
    return r_ret, vehicle_type_ret, departure_time_ret


def local_search(sol: Sol, neighbor_structure, lam) -> Sol:
    # 爬山算法
    best_sol = sol
    iter = 0
    while True:
        neighbor_ls = best_sol.neighbor(neighbor_structure)
        find_better = False
        best_sol_value = best_sol.cost_val + lam * best_sol.penalty_val
        for n in neighbor_ls:
            n_val = n.cost_val + lam * n.penalty_val
            if n_val < best_sol_value:
                find_better = True
                best_sol_value = n_val
                best_sol = n
        if not find_better:
            break
        iter += 1
        if iter > MAX_LOCAL_ITER_NUM:
            break
    return best_sol


def vns(sol: Sol, lam: float):
    """
    变邻域搜索
    :param sol:
    :return: sol变邻域搜索之后的解, best_sol_changed: 搜索结果和原先解不同
    """
    neighbor_structures = ['2opt*', 'relocate', 'swap']
    neighbor_structures_local = ['2opt*', 'relocate', 'swap']
    i = 0
    best_sol = sol
    best_sol_changed = False
    while i < len(neighbor_structures):
        # Shaking: 将x'进行扰动
        x_shaked = random.sample(best_sol.neighbor(method=neighbor_structures[i]), 1)[0]
        # 对x'进行局部vnd搜索
        j = 0
        find_local_best = False
        while j < len(neighbor_structures_local):
            x_local = local_search(x_shaked, neighbor_structures_local[j], lam)
            print("  local search of " + neighbor_structures_local[j] + " :", x_local.cost_val + lam * x_local.penalty_val)
            if x_local.cost_val + lam * x_local.penalty_val < x_shaked.cost_val + lam * x_shaked.penalty_val:
                x_shaked = x_local
                j = 0
            else:
                j += 1
            if x_local.cost_val + lam * x_local.penalty_val < best_sol.cost_val + lam * best_sol.penalty_val:
                best_sol = x_local
                find_local_best = True
                best_sol_changed = True
                print("  It is a better solution:")
                print("    Cost:", x_local.cost_val, "Penalty:", x_local.penalty_val)
                break
        if find_local_best:
            i = 0
        else:
            i = i + 1
    return best_sol, best_sol_changed


def main():
    # method1读取方式
    # sol = Sol()
    # sol.initialization('method1')
    # method2读取方式
    ff = open("init_sol.bin", "rb")
    sol = pickle.load(ff)
    ff.close()

    best_sol = sol
    lam = lam0  # 惩罚因子
    continue_infeasible_times = 0
    continue_feasible_times = 0
    init_cost = best_sol.cost_val
    init_penalty_cost = best_sol.cost_val+lam0*best_sol.penalty_val
    
    # 初始化成本列表
    costs = []
    penalty_costs = []
    iterations = []
    
    for iternum in range(100):
        iterations.append(iternum+1)
        # 对best_sol进行vns搜索
        print("iter", iternum + 1, ":")
        searched_sol, sol_changed = vns(best_sol, lam)
        print("  vns of iter", iternum + 1, "ended.")

        # 对S进行可行性检验，更新惩罚系数
        feasible = searched_sol.feasible()
        if not feasible:
            # 如果不可行，则更新惩罚系数
            continue_infeasible_times += 1
            continue_feasible_times = 0
            if continue_infeasible_times >= eta_penalty:
                print("Too much infeasible sol, increase penalty lambda.")
                lam *= 10
                continue_infeasible_times = 0
        else:
            # 如果可行，则更新惩罚系数
            continue_infeasible_times = 0
            continue_feasible_times += 1
            if continue_feasible_times > eta_penalty:
                print("Too much feasible sol, decrease penalty lambda.")
                lam /= 10
                continue_feasible_times = 0
        # 如果可行而且搜索到的解比原先解要好，更新解
        if feasible and sol_changed:
            best_sol = searched_sol
        # 如果迭代次数到达一定程度，则采用标签优化算法
        if iternum >= 0:
            for r_ind, r in enumerate(best_sol.routes):
                old_cost = cost_route(r, best_sol.vehicle_types[r_ind], best_sol.departure_times[r_ind])
                old_penalty = penalty_route(r, best_sol.vehicle_types[r_ind], best_sol.departure_times[r_ind])
                best_sol.routes[r_ind], best_sol.vehicle_types[r_ind], best_sol.departure_times[r_ind] = labeling(r, vehicle_type=best_sol.vehicle_types[r_ind], departure_time=best_sol.departure_times[r_ind])
                new_cost = cost_route(best_sol.routes[r_ind], best_sol.vehicle_types[r_ind], best_sol.departure_times[r_ind])
                new_penalty = penalty_route(best_sol.routes[r_ind], best_sol.vehicle_types[r_ind], best_sol.departure_times[r_ind])

                best_sol.cost_val = best_sol.cost_val + new_cost - old_cost
                best_sol.penalty_val = best_sol.penalty_val + new_penalty - old_penalty
        #1 BESTSOL的cost下降
        #2 bestsol.cost加lam0乘pena和初始解sol的cost加lam乘pena对比
        costs.append(best_sol.cost_val)
        penalty_costs.append(best_sol.cost_val+lam0*best_sol.penalty_val)
        if best_sol.penalty_val <= 0:
            if iternum > 15 or init_penalty_cost - (best_sol.cost_val + lam * best_sol.penalty_val) / init_penalty_cost > 0.25:
                break
    # 创建图形
    plt.figure()
    # 绘制基准线（200000）
    plt.axhline(y=init_cost, color='r', linestyle='--')
    # 绘制成本函数曲线
    plt.plot(iterations, costs, marker='o')  # 使用'o'标记每个点
    
    # 添加标题和轴标签
    plt.title('Cost Function over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    
    # 显示图例
    plt.legend(['Cost']) 

    # 保存图形到文件
    plt.savefig("cost_function_plot.png", dpi=300)  # 保存为PNG文件，高分辨率
    # 显示图形
    # plt.show()
    
    
    # 创建图形
    plt.figure()
    # 绘制基准线（200000）
    plt.axhline(y=init_penalty_cost, color='r', linestyle='--')
    # 绘制成本函数曲线
    plt.plot(iterations, penalty_costs, marker='o')  # 使用'o'标记每个点
    
    # 添加标题和轴标签
    plt.title('Penalty Cost Function over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Penalty Cost')
    
    # 显示图例
    plt.legend(['Penlaty Cost'])
    
    # 保存图形到文件
    plt.savefig("penalty_cost_function_plot.png", dpi=300)  # 保存为PNG文件，高分辨率
    # 显示图形
    # plt.show()
    # 输出最优解
    best_sol.output()

    return best_sol


if __name__ == '__main__':
    main()
    '''
    sol=Sol()
    sol.initialization("method1")
    sol.output()
    '''
    # ff = open("init_sol.bin", "rb")
    # sol = pickle.load(ff)
    # ff.close()
    # sol.penalty()
