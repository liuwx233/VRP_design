from Sol import *


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


def labeling(r, vehicle_type, departure_time):
    """
    TODO: 标签算法（见论文5.5.7），
    :param r: 配送顺序r（r中可能还包含充电站节点，并不一定像论文中都是客户节点）
    :param vehicle_type: r上原本的配送车型
    :param departure_time: r原本的出发时间
    :return: r_ret为返回的路径（决定了充电站、返回配送中心）, vehicle_type_ret为决定的配送车型
    """
    
    R = []
    W = []
    V = []
    T = []
    T_w = []
    d = []
    f = []
    R[0] = 0
    W[0] = 0
    V[0] = 0
    T[0] = departure_time
    T_w[0] = 0
    d[0] = 0
    f[0] = 0
    for i in range(len(r)):
        if i != 0:
            premise1 = (T[i]>=service_time) and (T[i]<=service_time+df_nodes.loc[r[i], "last_receive_tm"])
            premise2 = (W[i]<=df_vehicle.loc[vehicle_type, "max_weight"])
            premise3 = (V[i]<=df_vehicle.loc[vehicle_type, "max_volume"])
            premise4 = R[i] <= df_vehicle.loc[vehicle_type, "driving_range"]
            premise = premise1 and premise2 and premise3 and premise4
            if not premise:
                continue
            
            extension = 1
            #### 拓展1
            if extension == 1:
                T[i+1] = max(T[i]+df_distance.loc[(r[i], r[i+1]), "spend_tm"], df_nodes.loc[r[i+1], "first_receive_tm"])+service_time
                T_w[i+1] = T_w[i] + max(0, df_nodes.loc[r[i+1], "first_receive_tm"]-T[i]-df_nodes.loc[(r[i], r[i+1]), "spend_tm"])
                W[i+1] = W[i] + df_nodes.loc[r[i+1], "pack_total_weight"]
                V[i+1] = V[i] + df_nodes.loc[r[i+1], "pack_total_volume"]   
                R[i+1] = R[i] + df_distance.loc[(r[i], r[i+1]), "distance"]
                d[i+1] = d[i] + df_distance.loc[(r[i], r[i+1]), "distance"]
                time_wait = max(0, df_nodes.loc[r[i+1], "first_receive_tm"]-T[i]-df_nodes.loc[(r[i], r[i+1]), "spend_tm"])
                f[i+1] = f[i]+waiting_cost*time_wait+df_distance.loc[(r[i], r[i+1]), 'distance'] * df_vehicle.loc[vehicle_type, 'unit_trans_cost'] / 1000

            if extension == 2:
                k = find_nearest_recharge_station(r[i], r[i+1])
                T[i+1] = max(T[i]+df_distance.loc[(r[i], k), "spend_tm"]+service_time+df_distance.loc[(k, r[i+1]), "spend_tm"], df_nodes.loc[r[i+1], "first_receive_tm"]) + service_time
                T_w[i+1] = T_w[i] + max(0, df_nodes.loc[r[i+1], "first_receive_tm"]-T[i]-df_nodes.loc[(r[i], k), "spend_tm"]-service_time-df_nodes.loc[(k, r[i+1]), "spend_tm"])
                W[i+1] = W[i] + df_nodes.loc[r[i+1], "pack_total_weight"]
                V[i+1] = V[i] + df_nodes.loc[r[i+1], "pack_total_volume"]
                R[i+1] = df_distance.loc[(k, r[i+1]), "distance"]
                d[i+1] = d[i] + df_distance.loc[(r[i], k), "distance"] + df_distance.loc[(k, r[i+1]), "distance"]
                time_wait = max(0, df_nodes.loc[r[i+1], "first_receive_tm"]-T[i]-df_nodes.loc[(r[i], k), "spend_tm"]-service_time-df_nodes.loc[(k, r[i+1]), "spend_tm"])
                f[i+1] = f[i] + waiting_cost*time_wait + (df_distance.loc[(r[i], k), 'distance']+df_distance.loc[(k, r[i+1]), 'distance']) * df_vehicle.loc[vehicle_type, 'unit_trans_cost']/1000 + charging_cost*service_time

            if extension == 3:
                T[i+1] = max(T[i]+df_distance.loc[(r[i], 0), "spend_tm"]+service_time_depot, df_nodes.loc[r[i+1], "first_receive_tm"])+service_time
                T_w[i+1] = T_w[i] + max(0, df_nodes.loc[r[i+1], "first_receive_tm"]-T[i]-df_nodes.loc[(r[i], 0), "spend_tm"]-service_time_depot-df_nodes.loc[(0, r[i+1]), "spend_tm"])
                W[i+1] = df_nodes.loc[r[i+1], "pack_total_weight"]
                V[i+1] = df_nodes.loc[r[i+1], "pack_total_volume"]
                R[i+1] = df_distance.loc[(0, r[i+1]), "distance"]
                d[i+1] = d[i] + df_distance.loc[(r[i], 0), "distance"] + df_distance.loc[(0, r[i+1]), "distance"]
                time_wait = max(0, df_nodes.loc[r[i+1], "first_receive_tm"]-T[i]-df_nodes.loc[(r[i], 0), "spend_tm"]-service_time_depot-df_nodes.loc[(0, r[i+1]), "spend_tm"]) + service_time_depot
                f[i+1] = f[i] + waiting_cost*time_wait + (df_distance.loc[(r[i], 0), 'distance']+df_distance.loc[(0, r[i+1]), 'distance']) * df_vehicle.loc[vehicle_type, 'unit_trans_cost'] / 1000

    r_ret = r
    vehicle_type_ret = vehicle_type
    return r_ret, vehicle_type_ret


def local_search(sol: Sol, neighbor_structure, lam) -> Sol:
    # 爬山算法
    best_sol = sol
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
    return best_sol


def vns(sol: Sol, lam: float):
    """
    变邻域搜索
    :param sol:
    :return: sol变邻域搜索之后的解, best_sol_changed: 搜索结果和原先解不同
    """
    neighbor_structures = ['2opt*', 'relocate', 'swap', 'insert_remove']
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
    sol = Sol()
    sol.initialization('method1')
    best_sol = sol
    lam = lam0  # 惩罚因子
    continue_infeasible_times = 0
    continue_feasible_times = 0
    for iternum in range(15000):
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
        if iternum > 10000:
            pass
    return best_sol


if __name__ == '__main__':
    main()
