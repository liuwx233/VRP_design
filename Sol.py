from input import *
import random
import pickle
from datetime import datetime, timedelta


def vol_constraint_route(r, vehicle_type=1, depart_time=0):
    volume = df_vehicle.loc[vehicle_type, 'max_volume']
    res_volume = volume
    for i in range(len(r)):
        if r[i] == 0:
            res_volume = volume
        elif 1 <= r[i] <= 1000:
            res_volume = res_volume - df_nodes.loc[r[i], 'pack_total_volume']
            if res_volume < 0:
                return False
    return True


def weight_constraint_route(r, vehicle_type=1, depart_time=0):
    weight = df_vehicle.loc[vehicle_type, 'max_weight']
    res_weight = weight
    for i in range(len(r)):
        if r[i] == 0:
            res_weight = weight
        elif 1 <= r[i] <= 1000:
            res_weight = res_weight - df_nodes.loc[r[i], 'pack_total_weight']
            if res_weight < 0:
                return False


def time_constraint_route(r, vehicle_type=1, depart_time=0):
    path_time = depart_time

    for i in range(len(r)-1):
        from_point = r[i]
        to_point = r[i+1]
        path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
        if to_point != 0:
            if path_time > df_nodes.loc[to_point, 'last_receive_tm'] or path_time > time_horizon:
                return False
            # 付出等待时间
            path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
        else:
            if path_time > time_horizon:
                return False
        if to_point == 0:
            # depot服务时间
            path_time += 60
        else:
            # 充电站或客户节点服务时间
            path_time += service_time
    return True


def penalty_route(r, vehicle_type=1, depart_time=0):
    """
    包含两部分的内容：违反时间约束的惩罚成本（单位：分钟）+违反电量约束的惩罚成本（单位：百米）
    要加上所有的cost！
    :return: 惩罚成本
    """
    if len(r) == 2:
        print("warning: route [0,0]")
        return 0
    penalty_time = 0
    penalty_elec = 0
    charge = df_vehicle.loc[vehicle_type, 'driving_range']
    res_charge = charge
    path_time = depart_time

    for j in range(len(r) - 1):
        from_point = r[j]
        to_point = r[j+1]

        path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
        if to_point != 0:
            penalty_time += max((path_time - df_nodes.loc[to_point, 'last_receive_tm']), 0) * Penalty_TW  # 晚到惩罚，单位：分钟
        penalty_time += max((path_time - time_horizon), 0)  # 所有点超过time_horizon加一个额外的惩罚
        if to_point != 0:  # 更新时间参数
            path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
        if to_point == 0:
            path_time += 60
        else:
            path_time += service_time

        res_charge = res_charge - df_distance.loc[(from_point, to_point), 'distance']
        if res_charge < 0:
            penalty_elec += -res_charge * Penalty_range  # 电量惩罚，单位：百米
        if to_point > 1000:  # 更新电量参数
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
    if len(r) == 2:
        print("warning: route [0,0]")
        return 0
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


def penalty_cost_route(r, vehicle_type=1, depart_time=0, penalty=True, penalty_lam=0):
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


class Sol:
    def __init__(self):
        self.routes = []  # 初始解是由depot直达所有客户点，然后直接返回
        self.vehicle_types = []  # 初始车型随机分配
        self.departure_times = []  # 初始发车时间为0
        self.cost_val = 0
        self.penalty_val = 0

    def initialization(self, method, vol_weight_feasible=False):
        """
        TODO: 初始化一个解，解含有的元素有:
        1. 路的集合。一条路由许多个节点id按顺序构成。每条路必须以depot为起点，循环算作一条路。例如从depot->9->6->depot->1->depot这段路的列表为[0, 9, 6, 0, 1, 0]
        2. 每条路所分配的车型。
        3. 每条路的起始时间（注：起始时间给定则路上所有节点的到达、离开时间可以确定）。
        """
        # 计算所有客户点紧迫程度并排序
        if method == 'method1':
            sorted_customer_index = index_customer.copy()
            sorted_customer_index.sort(key=lambda ind: df_nodes.loc[ind, 'last_receive_tm'] - df_distance.loc[(0, ind), 'spend_tm'], reverse=True)  # 越后面Ui越小，紧迫程度越大

            # 每次拿出一个节点, 作为这条路的起始节点:
            while len(sorted_customer_index) > 0:
                new_route = [0]
                now_time = 0  # 默认起始时间为0, 默认车辆类别为1号
                now_vol = df_vehicle.loc[1, 'max_volume']
                now_weight = df_vehicle.loc[1, 'max_weight']
                now_range = df_vehicle.loc[1, 'driving_range']

                now_node = sorted_customer_index.pop()  # pop出紧迫程度最大的点作为起始点
                if_large_vehicle = False
                if df_nodes.loc[now_node, 'pack_total_volume'] > df_vehicle.loc[1, 'max_volume'] or df_nodes.loc[now_node, 'pack_total_weight'] > df_vehicle.loc[1, 'max_weight']:
                    if_large_vehicle = True
                new_route.append(now_node)
                now_time += max(df_distance.loc[(0, now_node), 'spend_tm'], df_nodes.loc[now_node, 'first_receive_tm']) + service_time
                now_vol -= df_nodes.loc[now_node, 'pack_total_volume']
                now_weight -= df_nodes.loc[now_node, 'pack_total_weight']
                now_range -= df_distance.loc[(0, now_node), 'distance']

                while True:
                    # 搜索当前节点的邻域点(暂定为所有点), 将邻域中所有点按照waiting cost + travel cost排序
                    neighbor_nodes = sorted_customer_index.copy()
                    exist_neighbor_can_reach_recharge = False
                    exist_neighbor_but_it_cannot_reach_recharge = False
                    exist_neighbor_cannot_reach_since_elec = False
                    exist_neighbor_can_reach = False

                    neighbor_nodes.sort(key=lambda ind: df_vehicle.loc[1, 'unit_trans_cost'] * df_distance.loc[(now_node, ind), 'distance'] +
                                            waiting_cost * max(0, df_nodes.loc[ind, 'first_receive_tm'] - now_time - df_distance.loc[(now_node, ind), 'spend_tm']), reverse=False)

                    for neighbor_node in neighbor_nodes:
                        # 看weight够不够，vol够不够, 时间够不够, 时间够不够返回depot
                        if_weight = now_weight >= df_nodes.loc[neighbor_node, 'pack_total_weight']
                        if_vol = now_vol >= df_nodes.loc[neighbor_node, 'pack_total_volume']
                        if_time = (now_time + df_distance.loc[(now_node, neighbor_node), 'spend_tm'] <= df_nodes.loc[neighbor_node, 'last_receive_tm'])
                        if_return = (time_horizon >= now_time + df_distance.loc[(now_node, neighbor_node), 'spend_tm'] + service_time + df_distance.loc[(0, neighbor_node), 'spend_tm'])
                        if_elec = (now_range >= df_distance.loc[(now_node, neighbor_node), 'distance'])
                        if if_weight and if_vol and if_time and if_return and if_elec:
                            # 如果几个条件都满足，还要看待插入节点是否能够到达其它的电站或者depot,
                            if_neighbor_can_reach_recharge = False
                            for ch in index_recharge + [0]:
                                if now_range >= df_distance.loc[(now_node, neighbor_node), 'distance'] + df_distance.loc[(neighbor_node, ch), 'distance']:
                                    if_neighbor_can_reach_recharge = True
                                    break
                            if not if_neighbor_can_reach_recharge:
                                # 如果不行，则存在邻居可以到达，但是邻居到不了电站
                                exist_neighbor_but_it_cannot_reach_recharge = True
                                exist_neighbor_can_reach = True
                            else:
                                # 如果上述条件还满足将该节点加入到路中, 将该节点从未选取节点集合中删去
                                new_route.append(neighbor_node)
                                now_weight -= df_nodes.loc[neighbor_node, 'pack_total_weight']
                                now_vol -= df_nodes.loc[neighbor_node, 'pack_total_volume']
                                now_time = max(df_distance.loc[(now_node, neighbor_node), 'spend_tm'] + now_time, df_nodes.loc[neighbor_node, 'first_receive_tm']) + service_time
                                now_range -= df_distance.loc[(now_node, neighbor_node), 'distance']
                                now_node = neighbor_node
                                exist_neighbor_can_reach_recharge = True  # 存在一个邻居能够到达电站
                                exist_neighbor_can_reach = True
                                sorted_customer_index.remove(neighbor_node)
                                break
                        elif if_weight and if_vol and if_time and if_return:
                            exist_neighbor_cannot_reach_since_elec = True  # 至少存在一个邻居，使得冲了电就能够reach
                            exist_neighbor_can_reach = True
                        # else:
                        #     exist_neighbor_can_reach = True  # 任意一个节点或是因为时间，或是因为需求而不能reach

                    if not exist_neighbor_can_reach_recharge:
                        # 如果没有邻居节点可以reach recharge, 意味着要返回depot或者插入一个recharge. 已知Now_node肯定可以到一个recharge.
                        if exist_neighbor_but_it_cannot_reach_recharge or exist_neighbor_cannot_reach_since_elec:
                            # 这两种情况下，货物和时间都够，但是电不够，找最近的充电站直接插入
                            min_charge_dist = float("inf")
                            min_charge_node = -1
                            for insert_charge_node in index_recharge + index_depot:
                                if df_distance.loc[(now_node, insert_charge_node), 'distance'] < min_charge_dist:
                                    min_charge_dist = df_distance.loc[(now_node, insert_charge_node), 'distance']
                                    min_charge_node = insert_charge_node
                            if min_charge_node != 0:
                                new_route.append(min_charge_node)
                                now_time += df_distance.loc[(now_node, min_charge_node), 'spend_tm'] + charging_time
                                now_range = df_vehicle.loc[1, 'driving_range']
                                now_node = min_charge_node
                            else:
                                assert min_charge_node == 0
                                new_route.append(min_charge_node)
                                now_weight = df_vehicle.loc[1, 'max_weight']
                                now_vol = df_vehicle.loc[1, 'max_volume']
                                now_time += df_distance.loc[(now_node, min_charge_node), 'spend_tm'] + service_time_depot
                                now_range = df_vehicle.loc[1, 'driving_range']
                                now_node = min_charge_node
                        else:
                            # 此时因为需求、时间等限制，没有节点可以reach了，则返回depot补货
                            assert not exist_neighbor_can_reach

                            if now_range >= df_distance.loc[(now_node, 0), 'distance']:
                                # 如果可以到depot，直接返回depot, 路程结束
                                new_route.append(0)
                                self.routes.append(new_route)
                                self.vehicle_types.append(2 if if_large_vehicle else 1)
                                self.departure_times.append(0)
                                break
                            else:
                                # 如果电量不足以到depot, 充电后返回
                                min_charge_dist = float("inf")
                                min_charge_node = -1
                                for insert_charge_node in index_recharge:
                                    if now_range >= df_distance.loc[(insert_charge_node, now_node), 'distance']:
                                        if df_distance.loc[(insert_charge_node, now_node), 'distance'] + df_distance.loc[(insert_charge_node, 0), 'distance'] < min_charge_dist:
                                            min_charge_dist = df_distance.loc[(insert_charge_node, now_node), 'distance'] + df_distance.loc[(insert_charge_node, 0), 'distance']
                                            min_charge_node = insert_charge_node
                                new_route.append(min_charge_node)
                                new_route.append(0)
                                self.vehicle_types.append(2 if if_large_vehicle else 1)
                                self.departure_times.append(0)
                                self.routes.append(new_route)
                                break

        elif method == 'method2':
            """
            第二种初始化方式，由石一鸣完成
            """
            # 第二种初始化方式

            # 构造一个包含若干辆空车的空解
            lam0 = 10
            num_vehicles = maximum_vehicle[1] + maximum_vehicle[2]
            self.routes = [[0, 0] for _ in range(num_vehicles)]
            self.vehicle_types = [1] * maximum_vehicle[1] + [2] * maximum_vehicle[2]
            self.departure_times = [0] * num_vehicles

            # 对所有的商户按照最晚服务时间从早到晚排列，得到一个商户列表
            df_customer = df_nodes[df_nodes['type'] == 2]
            # 按照最晚接收时间排序，并获取排序后的ID列表
            sorted_ids = df_customer.sort_values('last_receive_tm').index.tolist()

            while len(sorted_ids) > 0:
                # 从列表的前Z个商户中随机选择一个商户进行插入
                Z = max(int(len(index_customer) * 0.1), 2)
                if Z > len(sorted_ids):
                    Z = len(sorted_ids)
                chosen_customer = random.choice(sorted_ids[:Z])  # 随机选中其中一个
                best_route = None  # 存储插入chosen_customer时目标函数增加最小的路径的索引
                best_position = None  # 存储插入chosen_customer时目标函数增加最小的位置，即插入到best_route的哪个位置
                best_increase = float('inf')

                for i, route in enumerate(self.routes):  # 遍历每一辆车的路径
                    if vol_weight_feasible:
                        route_weight = df_nodes.loc[chosen_customer, 'pack_total_weight']
                        route_volume = df_nodes.loc[chosen_customer, 'pack_total_volume']
                        for j in route:
                            if j==0: continue
                            route_weight += df_nodes.loc[j, 'pack_total_weight']
                            route_volume += df_nodes.loc[j, 'pack_total_volume']

                        if (route_weight > df_vehicle.loc[self.vehicle_types[i], 'max_weight']) or (route_volume > df_vehicle.loc[self.vehicle_types[i], 'max_volume']):
                            continue
                    # 计算原始的目标函数值
                    # 因为我们在某一次特定的循环中只会改变某一辆车的路径，所以只需要计算这辆车带来的变化即可
                    if len(route) <= 2:
                        obj_value = 0
                    else:
                        obj_value = penalty_cost_route(route, self.vehicle_types[i], self.departure_times[i], penalty_lam=lam0)

                    for position in range(len(route)-1):  # 尝试在路径中插入新的顾客
                        # 复制原列表
                        route_new = route[:]
                        # 在序号为position的位置后面插入新元素
                        route_new.insert(position+1, chosen_customer)
                        obj_value_new = penalty_cost_route(route_new, self.vehicle_types[i], self.departure_times[i], penalty_lam=lam0)  # 尝试插入chosen_customer后该route的目标函数值
                        increase = obj_value_new - obj_value  # 插入该顾客带来的成本提高
                        if increase < best_increase:  # 如果提高得很少，则就是我们想要的
                            best_route = i
                            best_position = position
                            best_increase = increase

                self.routes[best_route].insert(best_position + 1, chosen_customer)
                print(f"best route {best_route}, best position {best_position}")
                sorted_ids.remove(chosen_customer)
            
            self.routes = [element for element in self.routes if element != [0, 0]]

        self.cost_val = self.cost()
        self.penalty_val = self.penalty()

    def copy(self):
        """
        复制当前解
        :return:
        """
        ret_sol = Sol()
        ret_sol.routes = copy.deepcopy(self.routes)
        ret_sol.departure_times = copy.deepcopy(self.departure_times)
        ret_sol.vehicle_types = copy.deepcopy(self.vehicle_types)
        ret_sol.cost_val = self.cost_val
        ret_sol.penalty_val = self.penalty_val
        return ret_sol

    def cost(self):
        """
        计算当前解的成本。包含
        1. 车辆固定成本
        2. travel cost
        3. waiting cost
        4. charging cost
        :return: cost
        """
        # total_fix_cost = self.vehicle_types.count(1) * df_vehicle.loc[1, 'vehicle_cost'] + self.vehicle_types.count(2) * df_vehicle.loc[2, 'vehicle_cost']
        # self.fix_cost = total_fix_cost  # 计算固定成本
        #
        # total_travel_cost = 0
        # for i in range(len(self.routes)):
        #     travel_cost = 0
        #     path = self.routes[i]
        #     for j in range(len(path)-1):
        #         from_point = path[j]
        #         to_point = path[j+1]
        #         # TODO: 改
        #         # df_distance.loc[(from_point, to_point), 'distance']
        #         # df_vehicle.loc[vehicle_type, 'unit_trans_cost']
        #         travel_cost += df_distance.loc[(from_point, to_point), 'distance'] * df_vehicle.loc[self.vehicle_types[i], 'unit_trans_cost'] / 1000
        #     total_travel_cost += travel_cost
        # self.travel_cost = total_travel_cost  # 计算行驶成本
        #
        # total_waiting_cost = 0
        # path_time_list = []  # 注意这个list存储的均为到达时间
        # for j in range(len(self.routes)):
        #     path_time = self.departure_times[j]
        #     temp_list = [self.departure_times[j]]
        #     path = self.routes[j]
        #     for i in range(len(path)-1):
        #         from_point = path[i]
        #         to_point = path[i+1]
        #         # TODO
        #         path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
        #         temp_list.append(path_time)
        #         # TODO
        #         if to_point != 0:
        #             total_waiting_cost += max((df_nodes.loc[to_point, 'first_receive_tm'] - path_time), 0) * waiting_cost
        #             path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
        #         if to_point == 0:
        #             path_time += 60
        #         else:
        #             path_time += service_time
        #     path_time_list.append(temp_list)  # 和route同dim的一串list存储相应节点的到达时间
        #     total_waiting_cost += (self.routes[j].count(0) - 2) * waiting_cost * 60  # 除去头尾的0，每经过一次depot存在1h等待成本
        # self.waiting_cost = total_waiting_cost
        #
        # total_charging_cost = 0
        # for path in self.routes:
        #     for i in range(len(path)):
        #         if path[i] in index_recharge:
        #             total_charging_cost += charging_cost * service_time
        # self.charging_cost = total_charging_cost
        #
        # self.total_cost = self.fix_cost + self.travel_cost + self.waiting_cost + self.charging_cost
        #
        # return self.total_cost

        total_cost = 0
        for r_ind, r in enumerate(self.routes):
            total_cost += cost_route(r, self.vehicle_types[r_ind], self.departure_times[r_ind])
        return total_cost

    def penalty(self):
        """
        包含两部分的内容：违反时间约束的惩罚成本（单位：分钟）+违反电量约束的惩罚成本（单位：百米）
        :return: 惩罚成本
        """
        # penalty_time = []
        # penalty_elec = []
        # for i in range(len(self.routes)):
        #     path_time = self.departure_times[i]
        #     charge = df_vehicle.loc[self.vehicle_types[i], 'driving_range']
        #     res_charge = charge
        #     time_penalty = 0
        #     charge_penalty = 0
        #
        #     for j in range(len(self.routes[i])-1):
        #         from_point = self.routes[i][j]
        #         to_point = self.routes[i][j+1]
        #
        #         path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
        #         if to_point >= 1 and to_point <= 1000:
        #             time_penalty += max((path_time - df_nodes.loc[to_point, 'last_receive_tm']), 0) * Penalty_TW  # 晚到惩罚，单位：分钟
        #         time_penalty += max((path_time - time_horizon), 0)  # 所有点超过time_horizon加一个额外的惩罚
        #         if to_point != 0:  # 更新时间参数
        #             path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
        #         if to_point == 0:
        #             path_time += 60
        #         else:
        #             path_time += service_time
        #
        #         res_charge = res_charge - df_distance.loc[(from_point, to_point), 'distance']
        #         if res_charge < 0:
        #             charge_penalty += -res_charge * Penalty_range  # 电量惩罚，单位：百米
        #         if to_point > 1000:  # 更新电量参数
        #             res_charge = charge
        #         if to_point == 0:  # 回depot也可以充满的
        #             res_charge = charge
        #
        #     penalty_time.append(time_penalty)
        #     penalty_elec.append(charge_penalty)
        #
        # total_penalty_cost = sum(penalty_time) + sum(penalty_elec)
        #
        # return total_penalty_cost

        total_penalty_cost = 0
        for r_ind, r in enumerate(self.routes):
            total_penalty_cost += penalty_route(r, self.vehicle_types[r_ind], self.departure_times[r_ind])
        return total_penalty_cost

    def penalty_cost(self, lam):
        """
        TODO: 包含三部分的内容：cost + 违反时间约束的惩罚成本（单位：分钟）+违反电量约束的惩罚成本（单位：百米）
        :return: 惩罚成本
        """
        total_cost = self.cost()
        penalty_cost = self.penalty()
        total_penalty_cost = total_cost + lam * penalty_cost

        return total_penalty_cost

    def elec_constraint(self):
        """
        判定当前解是否符合充电量约束
        :return:
        1. 当前解是否符合充电量约束
        2. 元组(infeasible_route, infeasible_index)列表。元组的第一个元素是不符合约束的route的编号，第二个元素是这个route上第一个违反充电量约束的点。
        例如: [(0,2)]代表第0号route中，到达这条路上第2个节点前电量会耗尽。
        """
        result = []
        for j in range(len(self.routes)):
            # if self.vehicle_types[j] == 1:
            #     charge = df_vehicle.lodc
            # else:
            #     charge = df_vehicle.iloc[1][5]
            charge = df_vehicle.loc[self.vehicle_types[j], 'driving_range']
            res_charge = charge
            for i in range(len(self.routes[j])-1):
                from_point = self.routes[j][i]
                to_point = self.routes[j][i+1]
                # TODO
                res_charge = res_charge - df_distance.loc[(from_point, to_point), 'distance']
                if res_charge < 0:
                    result.append((j, i))  # 第几条路的第几个节点
                if to_point > 1000:
                    res_charge = charge
                if to_point == 0:  # 回depot也可以充满的
                    res_charge = charge
        if len(result) == 0:
            return True, []
        else:
            return False, result

    def vol_constraint(self):
        """
        判定当前解是否符合体积约束
        :return:
        1. 当前解是否符合体积约束
        2. 元组(infeasible_route, infeasible_index)列表. 元组的第一个元素是不符合约束的route的编号，第二个元素是这个route上第一个违反体积约束的点。
        例如：[(0,2)]代表第0号route中，到达这条路上第2个节点时无法满足体积需求。
        """
        result = []
        for j in range(len(self.routes)):
            # if self.vehicle_types[j] == 1:
            #     volume = df_vehicle.iloc[0][2]
            # else:
            #     volume = df_vehicle.iloc[1][2]
            volume = df_vehicle.loc[self.vehicle_types[j], 'max_volume']
            res_volume = volume
            for i in range(len(self.routes[j])):
                if self.routes[j][i] == 0:
                    res_volume = volume
                elif self.routes[j][i] in index_customer:
                    res_volume = res_volume - df_nodes.loc[self.routes[j][i], 'pack_total_volume']
                    if res_volume < 0:
                        result.append((j, i))
        if len(result) == 0:
            return True, []
        else:
            return False, result

    def weight_constraint(self):
        """
        判定当前解是否符合负载重量约束
        :return:
        1. 当前解是否符合负载重量约束
        2. 元组(infeasible_route, infeasible_index)列表. 元组的第一个元素是不符合约束的route的编号，第二个元素是这个route上第一个违反重量约束的点。
        例如：[(0,2)]代表第0号route中，到达这条路上第2个节点时无法满足重量需求。
        """
        result = []
        for j in range(len(self.routes)):
            # if self.vehicle_types[j] == 1:
            #     weight = df_vehicle.iloc[0][3]
            # else:
            #     weight = df_vehicle.iloc[1][3]
            weight = df_vehicle.loc[self.vehicle_types[j], 'max_weight']
            res_weight = weight
            for i in range(len(self.routes[j])):
                if self.routes[j][i] == 0:
                    res_weight = weight
                elif self.routes[j][i] in index_customer:
                    res_weight = res_weight - df_nodes.loc[self.routes[j][i], 'pack_total_weight']
                    if res_weight < 0:
                        result.append((j, i))
        if len(result) == 0:
            return True, []
        else:
            return False, result

    def time_constraint(self):
        """
        判定当前解是否符合最晚到达时间约束
        :return:
        1. 当前解是否符合时间约束
        2. 元组(infeasible_route, infeasible_index)列表. 元组的第一个元素是不符合约束的route的编号，第二个元素是这个route上第一个时间约束的点。
        例如：[(0,2)]代表第0号route中，到达这条路上第2个节点的时间晚于这个节点的最晚到达时间。
        """
        path_time_list = []  # 注意这个list存储的均为到达时间
        result = []  # 存储违反约束的元组的list
        for j in range(len(self.routes)):
            path_time = self.departure_times[j]
            temp_list = []
            if path_time < 0:  # 判断不要在指定发车时间之前发车
                temp_list.append(1)
                result.append((j, 0))
            else:
                temp_list.append(0)
            for i in range(len(self.routes[j])-1):
                from_point = self.routes[j][i]
                to_point = self.routes[j][i+1]
                # TODO
                path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
                if to_point != 0:
                    if path_time <= df_nodes.loc[to_point, 'last_receive_tm'] and path_time <= time_horizon:
                        temp_list.append(0)
                    else:
                        temp_list.append(1)
                        result.append((j, i+1))
                    path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
                else:
                    if path_time <= time_horizon:
                        temp_list.append(0)
                    else:
                        temp_list.append(1)
                        result.append((j, i+1))
                if to_point == 0:
                    path_time += 60
                else:
                    path_time += service_time
            path_time_list.append(temp_list)

        if len(result) == 0:
            return True, []
        else:
            return False, result

    def cover_constraint(self):
        """
        当前解是否cover所有客户
        :return: True or False
        """
        # TODO:循环每次去掉相应的客户，如果去掉两次也会报错
        expected_numbers = set(index_customer)
        for path in self.routes:
            for element in path:
                if element in index_customer:
                    # if element not in expected_numbers:
                    #     return False
                    expected_numbers.remove(element)
        if len(expected_numbers) == 0:
            return True
        else:
            return False

    def feasible(self) -> bool:
        # 判断解是否可行
        return self.elec_constraint()[0] and self.vol_constraint()[0] and self.weight_constraint()[0] \
            and self.time_constraint()[0] and self.cover_constraint()

    def neighbor(self, method='2opt*') -> list:
        """
        输出当前解的邻域
        TODO: 通过在Sol中加入cost项增加计算效率
        :return:
        """
        # neighbor_num = 10  # 暂时设定为随机生成10个解，后面通过商户关联度来减小邻域
        neighbor_ls = []
        for i in range(Gamma_N):
            neighbor_sol = self.copy()
            if method == '2opt*':
                # 随机选择两条路径，随机选择一个位置断开并进行交叉重连

                # 随机选择两个点，第一个点和第二个点的路进行交叉  A-> 1 -> B, C-> 2 -> D  变为 A->2->D, C->1->B, 当然1，2不在一条路上
                # 随机选取一条路上的一个点
                # r1_ind = random.sample(range(len(self.routes)), 1)[0]
                # r1 = self.routes[r1_ind]
                # r1_node_ind = random.sample(range(1, len(r1) - 1), 1)[0]
                # r1_node = r1[r1_node_ind]
                # while r1_node not in index_customer:
                #     r1_node_ind = random.sample(range(1, len(r1) - 1), 1)[0]
                #     r1_node = r1[r1_node_ind]
                # # 再遍历所有的路，看路上节点与该路关联
                # for r2_ind, r2 in enumerate(self.routes):
                #     if r1_ind == r2_ind:
                #         continue
                #     for r2_node_ind, r2_node in enumerate(r2):
                #         if r2_node not in index_customer:
                #             continue
                #         if relation(r1_node, r2_node) > highest_relation_dict[r1_node]:  # 关联度不高的节点不予考虑
                #             continue
                #         # 交换r1_node和r2_node两条路
                #         neighbor_sol = self.copy()
                #         new_r1 = r1[0: r1_node_ind] + r2[r2_node_ind: len(r2)]
                #         new_r2 = r1[r1_node_ind: len(r1)] + r2[0: r2_node_ind]
                #         neighbor_sol.routes[r1_ind] = new_r1
                #         neighbor_sol.routes[r2_ind] = new_r2
                #         neighbor_ls.append(neighbor_sol)
                node1 = random.sample(index_customer, 1)[0]
                # 找到node1对应的路径r1
                r1 = []
                r1_ind = -1
                for r1_ind_, r1_ in enumerate(self.routes):
                    if node1 in r1_:
                        r1 = r1_
                        r1_ind = r1_ind_
                        break

                node1_index = r1.index(node1)
                # 找一个node1的关联度高的点
                node2 = random.sample(highest_relation_dict[node1], 1)[0]
                while node2 in r1:
                    node2 = random.sample(highest_relation_dict[node1], 1)[0]
                # 找node2所属路径
                r2 = []
                r2_ind = -1
                for r2_ind_, r2_ in enumerate(self.routes):
                    if node2 in r2_:
                        r2 = r2_
                        r2_ind = r2_ind_
                        break
                node2_index = r2.index(node2)
                # 更新成本并交换r1, r2
                new_r1 = r1[0: node1_index] + r2[node2_index: len(r2)]
                new_r2 = r2[0: node2_index] + r1[node1_index: len(r1)]

                old_cost = cost_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    cost_route(neighbor_sol.routes[r2_ind],  neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])
                old_penalty = penalty_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    penalty_route(neighbor_sol.routes[r2_ind],  neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])

                neighbor_sol.routes[r1_ind] = new_r1
                neighbor_sol.routes[r2_ind] = new_r2

                new_cost = cost_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    cost_route(neighbor_sol.routes[r2_ind], neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])
                new_penalty = penalty_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    penalty_route(neighbor_sol.routes[r2_ind], neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])

                neighbor_sol.cost_val = neighbor_sol.cost_val + new_cost - old_cost
                neighbor_sol.penalty_val = neighbor_sol.penalty_val + new_penalty - old_penalty

                # 如果有[0,0]路，删掉
                if len(neighbor_sol.routes[r1_ind]) == 2:
                    del neighbor_sol.routes[r1_ind]
                if len(neighbor_sol.routes[r2_ind]) == 2:
                    del neighbor_sol.routes[r2_ind]

            if method == 'relocate':
                # # 随机选择一个客户节点，
                # select_route_num = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                # r1 = neighbor_sol.routes[select_route_num]
                # select_node = -1
                # while select_node not in index_customer:
                #     select_node = random.sample(r1[1:len(r1)-1], 1)[0]
                # # 将其插入到另一个条路的另一个位置
                # select_route_num = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                # r2 = neighbor_sol.routes[select_route_num]
                # pos = random.sample(range(1, len(r2)), 1)[0]  # 插入到pos之前
                # neighbor_sol.routes[select_route_num].insert(select_node, pos)

                # 随机选择一个客户节点，插入到另一个点的前面，两个点的relation必须达到标准
                node1 = random.sample(index_customer, 1)[0]
                node2 = random.sample(highest_relation_dict[node1], 1)[0]
                r1, r1_ind = [], -1
                r2, r2_ind = [], -1
                r1_found, r2_found = False, False
                for r_ind, r in enumerate(self.routes):
                    if node1 in r:
                        r1 = r
                        r1_ind = r_ind
                        r1_found = True
                    if node2 in r:
                        r2 = r
                        r2_ind = r_ind
                        r2_found = True
                    if r1_found and r2_found:
                        break
                node1_index = r1.index(node1)
                node2_index = r2.index(node2)

                old_cost = cost_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    cost_route(neighbor_sol.routes[r2_ind],  neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])
                old_penalty = penalty_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    penalty_route(neighbor_sol.routes[r2_ind],  neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])

                del neighbor_sol.routes[r1_ind][node1_index]
                neighbor_sol.routes[r2_ind].insert(node2_index, node1)

                # 删除r1上的点之后，r1可能出现[0, 0]的情况，此时要删除重复项
                for node1_index, node1 in enumerate(neighbor_sol.routes[r1_ind]):
                    if node1_index == len(neighbor_sol.routes[r1_ind]) - 1:
                        break
                    if node1 == neighbor_sol.routes[r1_ind][node1_index + 1]:
                        del neighbor_sol.routes[r1_ind][node1_index]

                new_cost = cost_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    cost_route(neighbor_sol.routes[r2_ind], neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])
                new_penalty = penalty_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    penalty_route(neighbor_sol.routes[r2_ind], neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])

                neighbor_sol.cost_val = neighbor_sol.cost_val + new_cost - old_cost
                neighbor_sol.penalty_val = neighbor_sol.penalty_val + new_penalty - old_penalty

                if len(neighbor_sol.routes[r1_ind]) == 2:
                    del neighbor_sol.routes[r1_ind]

            if method == 'swap':
                # 随机选两个客户节点点并交换位置
                # select_route_num1 = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                # r1 = neighbor_sol.routes[select_route_num1]
                # select_node_num1 = -1
                # select_node1 = -1
                # while select_node1 not in index_customer:
                #     select_node_num1 = random.sample(list(range(1, len(r1) - 1)), 1)[0]
                #     select_node1 = r1[select_node_num1]
                # select_route_num2 = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                # r2 = neighbor_sol.routes[select_route_num2]
                # select_node_num2 = -1
                # select_node2 = -1
                # while select_node2 not in index_customer:
                #     select_node_num2 = random.sample(list(range(1, len(r2) - 1)), 1)[0]
                #     select_node2 = r2[select_node_num2]
                # neighbor_sol.routes[select_route_num1][select_node_num1] = select_node2
                # neighbor_sol.routes[select_route_num2][select_node_num2] = select_node1

                node1 = random.sample(index_customer, 1)[0]
                node2 = random.sample(highest_relation_dict[node1], 1)[0]
                r1, r1_ind = [], -1
                r2, r2_ind = [], -1
                r1_found, r2_found = False, False
                for r_ind, r in enumerate(self.routes):
                    if node1 in r:
                        r1 = r
                        r1_ind = r_ind
                        r1_found = True
                    if node2 in r:
                        r2 = r
                        r2_ind = r_ind
                        r2_found = True
                    if r1_found and r2_found:
                        break
                node1_index = r1.index(node1)
                node2_index = r2.index(node2)

                # 交换两个点
                old_cost = cost_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    cost_route(neighbor_sol.routes[r2_ind],  neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])
                old_penalty = penalty_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    penalty_route(neighbor_sol.routes[r2_ind],  neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])

                neighbor_sol.routes[r1_ind][node1_index] = node2
                neighbor_sol.routes[r2_ind][node2_index] = node1

                new_cost = cost_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    cost_route(neighbor_sol.routes[r2_ind], neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])
                new_penalty = penalty_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind]) + \
                    penalty_route(neighbor_sol.routes[r2_ind], neighbor_sol.vehicle_types[r2_ind], neighbor_sol.departure_times[r2_ind])

                neighbor_sol.cost_val = neighbor_sol.cost_val + new_cost - old_cost
                neighbor_sol.penalty_val = neighbor_sol.penalty_val + new_penalty - old_penalty

            if method == 'vehicle':
                # 随机改变一个车型
                r1_ind = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]

                old_cost = cost_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind])
                old_penalty = penalty_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind])

                self.departure_times[r1_ind] = 1 if self.departure_times[r1_ind] == 2 else 2

                new_cost = cost_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind])
                new_penalty = penalty_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind])

                neighbor_sol.cost_val = neighbor_sol.cost_val + new_cost - old_cost
                neighbor_sol.penalty_val = neighbor_sol.penalty_val + new_penalty - old_penalty

            if method == 'insert_remove':

                flag = random.sample([0, 1], 1)[0]
                r1_ind = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                old_cost = cost_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind])
                old_penalty = penalty_route(neighbor_sol.routes[r1_ind],  neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind])
                if flag == 0:
                    # 随机插入一个充电站
                    r1 = neighbor_sol.routes[r1_ind]
                    select_node_pos = random.sample(list(range(2, len(r1))), 1)[0]
                    select_recharge = random.sample(index_recharge, 1)[0]
                    neighbor_sol.routes[r1_ind].insert(select_node_pos, select_recharge)
                else:
                    # 随机删除一个充电站
                    deleted = False
                    while not deleted:
                        # 随机选择一条路
                        r1 = neighbor_sol.routes[r1_ind]
                        # 遍历该路，确定该路的充电站
                        recharge_pos = []
                        for i in range(1, len(r1) - 1):
                            if r1[i] in index_recharge:
                                recharge_pos.append(i)
                        if len(recharge_pos) == 0:
                            continue
                        select_del_pos = random.sample(recharge_pos, 1)[0]
                        del neighbor_sol.routes[r1_ind][select_del_pos]
                        deleted = True

                new_cost = cost_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind])
                new_penalty = penalty_route(neighbor_sol.routes[r1_ind], neighbor_sol.vehicle_types[r1_ind], neighbor_sol.departure_times[r1_ind])

                neighbor_sol.cost_val = neighbor_sol.cost_val + new_cost - old_cost
                neighbor_sol.penalty_val = neighbor_sol.penalty_val + new_penalty - old_penalty

            neighbor_ls.append(neighbor_sol)

        return neighbor_ls

    def recycle_opt(self):
        """
        输出将当前解按照recycle原则改善之后的解。
        对于每个路径，找和它的结束时间相差60以上的、最近出发的路径
        :return:
        """
        routes_copy = copy.deepcopy(self.routes)
        vehicle_types_copy = copy.deepcopy(self.vehicle_types)
        departure_times_copy = copy.deepcopy(self.departure_times)

        # 对于所有的路径按照出发时间从小到大的顺序进行排序
        sorted_index = sorted(range(len(departure_times_copy)), key=lambda i: departure_times_copy[i])
        sorted_routes = [routes_copy[i] for i in sorted_index]
        sorted_vehicle_types = [vehicle_types_copy[i] for i in sorted_index]
        sorted_departure_times = [departure_times_copy[i] for i in sorted_index]

        i = 0
        while i < len(sorted_routes):
            path_time = sorted_departure_times[i]
            for k in range(len(sorted_routes)-1):  # 算这条路可以走完所耗的时间
                from_point = sorted_routes[k]
                to_point = sorted_routes[k+1]
                path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
                if to_point == 0:
                    path_time += 60
                else:
                    path_time += max(df_nodes.loc[to_point, 'first_receive_tm'], path_time) + service_time
            indices = [i for i, start_time in enumerate(sorted_departure_times) if (start_time - path_time) >= 0]
            if len(indices) == 0:
                i += 1
            else:
                choice_index = random.choice(indices)
                sorted_routes[i] = sorted_routes[i] + sorted_routes[choice_index][1:]
                sorted_vehicle_types[i] = max(sorted_vehicle_types[i], sorted_vehicle_types[choice_index])
                del sorted_routes[choice_index]
                del sorted_vehicle_types[choice_index]
                del sorted_departure_times[choice_index]

        return sorted_routes, sorted_vehicle_types, sorted_departure_times

    def departure_opt(self):
        """
        输出将当前解按照departure time改善之后的解
        TODO：
        取最大等待时间和最小延迟时间的较小值
        :return:
        """
        arrival_time_list = []  # 注意这个list存储的均为在各个路径的到达时间
        waiting_time_list = []  # 注意这个list存储的均为在各个节点的等待时间
        delay_time_list = []  # 注意这个list存储的均为在各个节点的延迟时间

        for j in range(len(self.routes)):
            path_time = self.departure_times[j]  # 计时器
            temp_wait_time_list = [0]
            temp_delay_time_list = [0]
            for i in range(len(self.routes[j])-1):
                from_point = self.routes[j][i]
                to_point = self.routes[j][i+1]
                path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
                if to_point == 0:
                    temp_wait_time_list.append(0)
                    temp_delay_time_list.append(0)
                    path_time += 60
                else:
                    temp_wait_time_list.append(max((df_nodes.loc[to_point, 'first_receive_tm'] - path_time), 0))
                    temp_delay_time_list.append(max((df_nodes.loc[to_point, 'last_receive_tm'] - path_time), 0))
                    path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time) + service_time
            waiting_time_list.append(temp_wait_time_list)
            delay_time_list.append(temp_delay_time_list)
            arrival_time_list.append(path_time-60)
        for i in range(len(arrival_time_list)):
            if (arrival_time_list[i] - 60) <= time_horizon:  # 要平移不超过时间窗才移动
                self.departure_times[i] += min(max(waiting_time_list[i]), min(delay_time_list[i]))

        return self.departure_times
    
    def output(self, out_str):
        # debug
        print(len(self.routes))
        print(len(self.vehicle_types))
        print(len(self.departure_times))

        # 派车单号
        data = {}
        trans_code = []
        num_car = len(self.vehicle_types)
        for i in range(num_car):
            trans_code.append(f'DP{(i + 1):04d}')
        data['id'] = trans_code

        # 车型
        data['vehicle_type'] = self.vehicle_types

        # 顺序
        data['route'] = self.routes
        df_data = pd.DataFrame(data)
        df_data['route'] = df_data['route'].apply(lambda row: ';'.join(map(str, row)))
        print(self.routes)

        # 首次出发时间
        df_data['departure_time'] = self.departure_times
        # print(df_data)
        # 按照HH:MM输出
        '''new_time = []
        formatted_time = []
        base_time = datetime.strptime('8:00', '%H:%M')
        for i in range(len(self.departure_times)):
            new_time.append(base_time + timedelta(minutes=self.departure_times[i]))
            formatted_time.append(new_time[i].strftime('%H:%M'))
        df_data['departure_time'] = formatted_time'''
        # print(df_data)

        # 最后返回时间
        '''for item in self.routes:
            start_time = []
            for i in range(len(item)):
                start_time += df_distance.loc[(from_point, to_point), 'spend_tm']'''
        path_time_list = []  # 注意这个list存储的均为到达时间
        back_time = []
        for j in range(len(self.routes)):
            path_time = self.departure_times[j]
            temp_list = [self.departure_times[j]]
            path = self.routes[j]
            for i in range(len(path) - 1):
                from_point = path[i]
                to_point = path[i + 1]
                # TODO
                path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
                temp_list.append(path_time)
                # TODO
                if to_point != 0:
                    '''total_waiting_cost += max((df_nodes.loc[to_point, 'first_receive_tm'] - path_time),
                                              0) * waiting_cost'''
                    path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
                if to_point == 0:
                    path_time += 60
                else:
                    path_time += service_time
            path_time_list.append(temp_list)  # 和route同dim的一串list存储相应节点的到达时间
        for j in range(len(self.routes)):
            length = len(path_time_list[j])-1
            last_time = path_time_list[j][length]
            back_time.append(last_time)
        # print(path_time_list)
        # print(back_time)
        df_data['back_time'] = back_time
        '''new_time2 = []
        formatted_time2 = []
        base_time2 = datetime.strptime('8:00', '%H:%M')
        for i in range(len(back_time)):
            new_time2.append(base_time2 + timedelta(minutes=int(back_time[i])))
            formatted_time2.append(new_time2[i].strftime('%H:%M'))
        df_data['back_time'] = formatted_time2'''
        # print(df_data)

        # 总里程
        total_distance = []
        for i in range(len(self.routes)):
            distance = 0
            path = self.routes[i]
            for j in range(len(path)-1):
                from_point = path[j]
                to_point = path[j+1]
                # TODO: 改
                # df_distance.loc[(from_point, to_point), 'distance']
                # df_vehicle.loc[vehicle_type, 'unit_trans_cost']
                distance += df_distance.loc[(from_point, to_point), 'distance']
            total_distance.append(distance)
        # print(total_distance)
        df_data['distance'] = total_distance

        # 运输成本
        total_travel_cost = []
        for i in range(len(self.routes)):
            travel_cost = 0
            path = self.routes[i]
            for j in range(len(path) - 1):
                from_point = path[j]
                to_point = path[j + 1]
                travel_cost += df_distance.loc[(from_point, to_point), 'distance'] * df_vehicle.loc[
                    self.vehicle_types[i], 'unit_trans_cost'] / 1000
            total_travel_cost.append(travel_cost)
        # self.travel_cost = total_travel_cost  # 计算行驶成本
        print(total_travel_cost)
        trans_cost_rounded = [round(num, 2) for num in total_travel_cost]
        df_data['transport_cost'] = trans_cost_rounded

        # 充电成本
        total_charging_cost = []
        for path in self.routes:
            charge_cost = 0
            for i in range(len(path)):
                if path[i] in index_recharge:
                    charge_cost += charging_cost * service_time
            total_charging_cost.append(charge_cost)
        df_data['recharge_cost'] = total_charging_cost

        # 等待成本
        path_time_list = []  # 均为到达时间
        total_waiting_cost = []
        for j in range(len(self.routes)):
            total_wait_cost = 0
            path_time = self.departure_times[j]
            temp_list = [self.departure_times[j]]
            path = self.routes[j]
            for i in range(len(path) - 1):
                from_point = path[i]
                to_point = path[i + 1]
                # TODO
                path_time += df_distance.loc[(from_point, to_point), 'spend_tm']
                temp_list.append(path_time)
                # TODO
                if to_point != 0:
                    total_wait_cost += max((df_nodes.loc[to_point, 'first_receive_tm'] - path_time),
                                              0) * waiting_cost
                    path_time = max(df_nodes.loc[to_point, 'first_receive_tm'], path_time)
                if to_point == 0:
                    path_time += 60
                else:
                    path_time += service_time
            path_time_list.append(temp_list)  # 和route同dim的一串list存储相应节点的到达时间
            total_wait_cost += (self.routes[j].count(0) - 2) * waiting_cost * 60  # 除去头尾的0，每经过一次depot存在1h等待成本
            total_waiting_cost.append(total_wait_cost)
        waiting_cost_rounded = [round(num, 2) for num in total_waiting_cost]
        df_data['wait_cost'] = waiting_cost_rounded

        # 固定成本
        total_fix_cost = []
        fixed_use_cost_int = []
        for j in range(len(self.routes)):
            total_fix_cost_value = 0
            if self.vehicle_types[j] == 1:
                total_fix_cost_value = df_vehicle.loc[1, 'vehicle_cost']
            if self.vehicle_types[j] == 2:
                total_fix_cost_value = df_vehicle.loc[2, 'vehicle_cost']
            total_fix_cost.append(total_fix_cost_value)
        for element in total_fix_cost:
            fixed_use_cost_int.append(int(element))
        df_data['fix_cost'] = fixed_use_cost_int

        # 总成本
        total_cost = [w + x + y + z for w, x, y, z in zip(total_travel_cost, total_charging_cost, total_waiting_cost, total_fix_cost)]
        total_cost_rounded = [round(num, 2) for num in total_cost]
        df_data['total_cost'] = total_cost_rounded

        # 充电次数
        charge_cnt = []
        for path in self.routes:
            charge_count = 0
            for i in range(len(path)):
                if path[i] in index_recharge:
                    charge_count = charge_count + 1
            charge_cnt.append(charge_count)
        df_data['recharge_num'] = charge_cnt
        print(df_data)

        df_data.to_excel('output' + out_str + ".xlsx", index=False)


if __name__ == '__main__':
    # 初始化并存储数据
    sol = Sol()
    method = 'method2'
    vol_feasible = False
    sol.initialization(method, vol_feasible)
    # ff = open("init_sol.bin", "rb")
    # sol = pickle.load(ff)
    # ff.close()
    # sol.vehicle_types = [1 for i in range(len(sol.routes))]
    # sol.departure_times = [0 for i in range(len(sol.routes))]
    fBin = open("init_sol.bin", 'wb')
    pickle.dump(sol, fBin)
    fBin.close()

