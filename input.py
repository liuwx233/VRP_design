"""
version: python 3.10
"""
import random

import numpy as np
import pandas as pd
import sys
import copy
import random

"""
全局变量定义, 可以添加全局变量
"""

index_nodes = []  # 所有点的id集合(包含depot, depot id=0)
index_depot = [0]  # depot id集合
index_recharge = []  # 充电站id集合
index_customer = []  # 顾客id集合
index_vehicle_type = [1, 2]  # 车辆类型id集合，一共两个元素
maximum_vehicle = {1: 80, 2: 80}  # key: 车辆类型id, value: 事先规定的车辆数量上限
df_vehicle = pd.DataFrame()  # 含有车辆信息, 索引位index_vehicle_type中元素
df_nodes = pd.DataFrame()  # 含有客户节点、充电节点的所有信息, 索引为index_customer中元素
df_distance = pd.DataFrame()  # 含有距离信息, 索引为(i,j), i, j分别为index_nodes中的元素。且i≠j
waiting_cost = 24/60  # 等待时间，单位元/分钟
charging_cost = 100/60  # 充电时间，单位元/分钟
time_horizon = 24*60 - 8*60  # 从8点运行到24点
service_time = 30  # 所有客户的服务时间
charging_time = 30
service_time_depot = 60
eta_penalty = 10
lam_min = 0.01
lam_max = 1000
lam0 = 10
gamma_wt = 0.2  # 等待时间惩罚系数，用于计算商户关联度
gamma_tw = 1.0  # 违反时间窗的惩罚系数
highest_relation_dict = {}  # 用于邻域拓展，记载每个客户节点的关联客户节点
Penalty_TW = 1
Penalty_range = 1 / 50
MAX_LOCAL_ITER_NUM = 50
MAX_VNS_ITER_NUM = 500
MAX_MAIN_ITER_NUM = 60
MAX_MAIN_GAP = 0.3
LABEL_IF_MUST = False
INIT_METHOD = 2


# x_o = [[[[]]]]  # 四层依次是第k类车、第l辆、访问路径从i到O再到j，为方便起见x_o定义成与x同dim的
# distance_matrix = [[]]  # 从i到j的距离矩阵,
# time_matrix = [[]]  # 从i到j的时间矩阵
# arrival = []  # 每个点处的到达时间
# start = []  # 每个点处的出发时间


def input_data():
    """
    读入数据给全局变量赋值, 如果添加了全局变量，记得在global关键字之后声明
    :return: None
    """
    global df_vehicle, df_nodes, df_distance, index_nodes, index_recharge, index_customer
    df_vehicle = pd.read_csv('data/input_vehicle_type.txt', sep='\t', header=0, index_col=0)
    df_nodes = pd.read_excel('data/input_node.xlsx', sheet_name='Customer_data', header=0, index_col=0)
    df_distance = pd.read_csv('data/input_distance_time.txt', sep=',', header=0, index_col=(1, 2))
    index_nodes = list(df_nodes.index)
    index_recharge = list(df_nodes[df_nodes['type'] == 3].index)
    index_customer = list(df_nodes[df_nodes['type'] == 2].index)


input_data()
Gamma_N = int(len(index_customer) * 0.4)


def relation(customer1, customer2):
    """
    TODO: 商户关联度(见论文5.5.6)
    :param customer1:
    :param customer2:
    :return:
    """
    tij = df_distance.loc[(customer1, customer2), 'spend_tm']
    return df_distance.loc[(customer1, customer2), 'distance'] +\
        gamma_wt * max(df_nodes.loc[customer2, 'first_receive_tm'] - service_time - tij - df_nodes.loc[customer1, 'last_receive_tm'], 0) +\
        gamma_tw * max(df_nodes.loc[customer1, 'first_receive_tm'] + service_time + tij - df_nodes.loc[customer2, 'last_receive_tm'], 0)


# 构建每个节点的邻节点
highest_relation_dict = np.load('neighbor_dict'+str(Gamma_N)+'.npy', allow_pickle=True).item()



# 储存字典变量
if __name__ == '__main__':

    for i in index_customer:
        highest_relation_dict[i] = sorted(index_customer, key=lambda x: relation(i, x) if x != i else float("inf"))[0: Gamma_N]
    np.save('neighbor_dict'+str(Gamma_N)+'.npy', highest_relation_dict)
    # 读字典变量

