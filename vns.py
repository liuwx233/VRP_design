from Sol import *

def penalty_route(r, vehicle_type=1, depart_time=0):
    """
    包含两部分的内容：违反时间约束的惩罚成本（单位：分钟）+违反电量约束的惩罚成本（单位：百米）
    要加上所有的cost！
    :return: 惩罚成本
    """
    penalty_time = 0
    penalty_elec = 0
    charge = df_vehicle.loc[vehicle_type, 'driving_range']
    res_charge = charge
    path_time = depart_time

    for j in range(len(r) - 1):
        from_point = r[j]
        to_point = r[j+1]

        path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
        if to_point >= 1 and to_point <= 1000:
            penalty_time += max((path_time - df_nodes.loc[to_point, 'last_receive_tm']), 0)  # 晚到惩罚，单位：分钟
        penalty_time += max((path_time - time_horizon), 0)  # 所有点超过time_horizon加一个额外的惩罚
        if to_point != 0:  # 更新时间参数
            path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
        if to_point == 0:
            path_time += 60
        else:
            path_time += service_time

        res_charge = res_charge - df_distance.loc[(from_point, to_point), 'distance']
        if res_charge < 0:
            penalty_elec += -res_charge / 100  # 电量惩罚，单位：百米
        if to_point in index_recharge:  # 更新电量参数
            res_charge = charge
        if to_point == 0:  # 回depot也可以充满的
            res_charge = charge

    total_penalty_cost = penalty_time + penalty_elec

    return total_penalty_cost

def cost_route(r, vehicle_type=1, depart_time=0):
    """
    计算单独一条路的cost。包含
    1. 车辆固定成本
    2. travel cost
    3. waiting cost
    4. charging cost
    :param r: 如[0, 1, 0]
    :param vehicle_type: 1或2,默认为1
    :param depart_time: 出发时间,默认为0
    :return: 这条路的成本
    TODO: 重新实现目标函数，加入惩罚项
    """
    total_fix_cost = 0
    # if vehicle_type == 1:
    #     total_fix_cost = df_vehicle.iloc[0][8]
    # else:
    #     total_fix_cost = df_vehicle.iloc[1][8]
    total_fix_cost = df_vehicle.loc[vehicle_type, 'vehicle_cost']

    total_travel_cost = 0
    for i in range(len(r)-1):
        from_point = r[i]
        to_point = r[i+1]
        total_travel_cost += df_distance.loc[(from_point, to_point), 'distance'] * df_vehicle.loc[vehicle_type, 'unit_trans_cost'] / 1000

    total_waiting_cost = 0
    path_time = depart_time
    path_time_list = [depart_time]  # 注意这个list存储的均为到达时间
    for i in range(len(r)-1):
        from_point = r[i]
        to_point = r[i+1]
        path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
        path_time_list.append(path_time)
        if to_point != 0:
            total_waiting_cost += max((df_nodes.loc[to_point, 'first_receive_tm'] - path_time), 0) * waiting_cost
            path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
        if to_point == 0:
            path_time += 60
        else:
            path_time += service_time
        total_waiting_cost += (r.count(0) - 2) * waiting_cost * 60

    total_charging_cost = 0
    for i in range(len(r)):
        if r[i] in index_recharge:
            total_charging_cost += charging_cost * service_time

    total_cost = total_fix_cost + total_travel_cost + total_waiting_cost + total_charging_cost

    return total_cost

def penalty_cost_route(r, vehicle_type=1, depart_time=0, penalty=False, penalty_lam=0):
    """
    计算单独一条路的cost。包含
    1. 车辆固定成本
    2. travel cost
    3. waiting cost
    4. charging cost
    :param r: 如[0, 1, 0]
    :param vehicle_type: 1或2,默认为1
    :param depart_time: 出发时间,默认为0
    :return: 这条路的成本
    TODO: 重新实现目标函数，加入惩罚项
    """
    total_cost_route = 0
    total_penalty_route = 0
    total_penalty_cost = 0

    total_cost_route = cost_route(r, vehicle_type, depart_time)

    if penalty == True:
        total_penalty_route = penalty_route(r, vehicle_type, depart_time)

    total_penalty_cost += total_cost_route + penalty_lam * total_penalty_route

    return total_penalty_cost

def labeling(r, vehicle_type, departure_time):
    """
    TODO: 标签算法（见论文5.5.7），
    :param r: 配送顺序r（r中可能还包含充电站节点，并不一定像论文中都是客户节点）
    :param vehicle_type: r上原本的配送车型
    :param departure_time: r原本的出发时间
    :return: r_ret为返回的路径（决定了充电站、返回配送中心）, vehicle_type_ret为决定的配送车型
    """
    r_ret = r
    vehicle_type_ret = vehicle_type
    return r_ret, vehicle_type_ret


def local_search(sol: Sol, neighbor_structure, lam) -> Sol:
    # 爬山算法
    best_sol = sol
    while True:
        neighbor_ls = best_sol.neighbor(neighbor_structure)
        find_better = False
        best_sol_value = best_sol.penalty(lam)
        for n in neighbor_ls:
            n_val = n.penalty(lam)
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
    neighbor_structures_local = ['2opt*', 'relocate', 'swap', 'insert_remove']
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
            if x_local.penalty(lam) < x_shaked.penalty(lam):
                x_shaked = x_local
                j = 0
            else:
                j += 1
            if x_local.penalty(lam) < best_sol.penalty(lam):
                best_sol = x_local
                find_local_best = True
                best_sol_changed = True
                break
        if find_local_best:
            i = 0
        else:
            i = i + 1
    return best_sol, best_sol_changed


def main():
    sol = Sol()
    sol.initialization('method1')
    sol.neighbor('2opt*')
    best_sol = sol
    lam = lam0  # 惩罚因子
    for iternum in range(15000):
        # 对best_sol进行vns搜索
        searched_sol, sol_changed = vns(best_sol, lam)
        continue_infeasible_times = 0
        continue_feasible_times = 0
        # 对S进行可行性检验，更新惩罚系数
        feasible = searched_sol.feasible()
        if not feasible:
            # 如果不可行，则更新惩罚系数
            continue_infeasible_times += 1
            continue_feasible_times = 0
            if continue_infeasible_times >= eta_penalty:
                lam *= 10
                continue_infeasible_times = 0
        else:
            # 如果可行，则更新惩罚系数
            continue_infeasible_times = 0
            continue_feasible_times += 1
            if continue_feasible_times > eta_penalty:
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
