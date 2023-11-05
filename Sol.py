from input import *


class Sol:
    def __init(self):
        self.routes = []  # 初始解是由depot直达所有客户点，然后直接返回
        self.vehicle_types = []  # 初始车型随机分配
        self.departure_times = []  # 初始发车时间为0

    def initialization(self, method):
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
            pass

    def cost(self):
        """
        计算当前解的成本。包含
        1. 车辆固定成本
        2. travel cost
        3. waiting cost
        4. charging cost
        :return: cost
        """

    def copy(self):
        """
        复制当前解
        :return:
        """
        ret_sol = Sol()
        ret_sol.routes = copy.deepcopy(self.routes)
        ret_sol.departure_times = copy.deepcopy(self.departure_times)
        ret_sol.vehicle_types = copy.deepcopy(self.vehicle_types)
        return ret_sol

    def penalty(self, lam):
        """
        TODO: 包含三部分的内容：cost + 违反时间约束的惩罚成本（单位：分钟）+违反电量约束的惩罚成本（单位：百米）
        :return: 惩罚成本
        """

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
                if to_point in index_recharge:
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

    def neighbor(self, method='2opt') -> list:
        """
        输出当前解的邻域
        TODO: 减小邻域范围
        :return:
        """
        neighbor_num = 10  # 暂时设定为随机生成10个解，后面通过商户关联度来减小邻域
        neighbor_ls = []
        for i in range(neighbor_num):
            neighbor_sol = self.copy()
            if method == '2opt*':
                # 随机选择两条路径，随机选择一个位置断开并进行交叉重连
                select_routes_num = random.sample(range(0, len(neighbor_sol.routes)), 2)
                r1 = neighbor_sol.routes[select_routes_num[0]]
                r2 = neighbor_sol.routes[select_routes_num[1]]
                select_r1_node = random.sample(r1[1:len(r1) - 1], 1)[0]
                select_r2_node = random.sample(r2[1:len(r2) - 1], 1)[0]
                new_r1 = r1[0: select_r1_node] + r2[select_r2_node: len(r2)]
                new_r2 = r1[select_r1_node: len(r1)] + r2[0: select_r2_node]
                neighbor_sol.routes[select_routes_num[0]] = new_r1
                neighbor_sol.routes[select_routes_num[1]] = new_r2
            if method == 'relocate':
                # 随机选择一个客户节点，
                select_route_num = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                r1 = neighbor_sol.routes[select_route_num]
                select_node = -1
                while select_node not in index_customer:
                    select_node = random.sample(r1[1:len(r1)-1], 1)[0]
                # 将其插入到另一个条路的另一个位置
                select_route_num = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                r2 = neighbor_sol.routes[select_route_num]
                pos = random.sample(range(1, len(r2)), 1)[0]  # 插入到pos之前
                neighbor_sol.routes[select_route_num].insert(select_node, pos)
            if method == 'swap':
                # 随机选两个客户节点点并交换位置
                select_route_num1 = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                r1 = neighbor_sol.routes[select_route_num1]
                select_node_num1 = -1
                select_node1 = -1
                while select_node1 not in index_customer:
                    select_node_num1 = random.sample(list(range(1, len(r1) - 1)), 1)[0]
                    select_node1 = r1[select_node_num1]
                select_route_num2 = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                r2 = neighbor_sol.routes[select_route_num2]
                select_node_num2 = -1
                select_node2 = -1
                while select_node2 not in index_customer:
                    select_node_num2 = random.sample(list(range(1, len(r2) - 1)), 1)[0]
                    select_node2 = r2[select_node_num2]
                neighbor_sol.routes[select_route_num1][select_node_num1] = select_node2
                neighbor_sol.routes[select_route_num2][select_node_num2] = select_node1
            if method == 'insert_remove':
                flag = random.sample([0, 1], 1)[0]
                if flag == 0:
                    # 随机插入一个充电站
                    select_route_num1 = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                    r1 = neighbor_sol.routes[select_route_num1]
                    select_node_pos = random.sample(list(range(2, len(r1))), 1)[0]
                    select_recharge = random.sample(index_recharge, 1)[0]
                    neighbor_sol.routes[select_route_num1].insert(select_recharge, select_node_pos)
                else:
                    # 随机删除一个充电站
                    deleted = False
                    while not deleted:
                        # 随机选择一条路
                        select_route_num1 = random.sample(range(0, len(neighbor_sol.routes)), 1)[0]
                        r1 = neighbor_sol.routes[select_route_num1]
                        # 遍历该路，确定该路的充电站
                        recharge_pos = []
                        for i in range(1, len(r1) - 1):
                            if r1[i] in index_recharge:
                                recharge_pos.append(i)
                        if len(recharge_pos) == 0:
                            continue
                        select_del_pos = random.sample(recharge_pos, 1)[0]
                        del neighbor_sol.routes[select_route_num1][select_del_pos]
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
