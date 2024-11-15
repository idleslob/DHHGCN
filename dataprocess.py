# -*-codeing =utf-8 -*-
# @Time : 2024/7/15 12:39
# @Author: mo
# @File :Dataset_overlap.py
# @Software:PyCharm
import gzip
import random
from sklearn.utils import shuffle
import os
import time
import dgl
import torch
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix,lil_matrix
import numpy as np
from functools import reduce
import scipy.sparse as sp
from tqdm import tqdm
import torch.nn.functional as F
import pickle


file_path = 'data5/'
path = 'data_origin_5_core/'
price_level_num = 10
dataset='overlap'
datasets_name = ['Clothing_Shoes_and_Jewelry', 'Sports_and_Outdoors']
catename = ['Clothing, Shoes & Jewelry', 'Sports & Outdoors']
sparse_ratio_list = [0.1, 0.3, 0.5]
ratio=0.5
issparse=0

if issparse == 1:
    file_path = os.path.join(file_path,f"sparse_{int(ratio * 100)}/")

def parse(path):
    g = gzip.open(path, 'r')
    for e in g:
        yield eval(e)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def dataprocess(datasets_name,catename):
    print('--------------------' + datasets_name + ' is begin!------------')

    ###  userID-itemID-timestamp
    data_path = path + 'reviews_' + datasets_name + '_5.json.gz'
    # df = getDF('originalData/Tools_and_Home_Improvement.json.gz')
    df = getDF(data_path)
    interaction = df[['reviewerID', 'asin','overall']]
    interaction = interaction[interaction['overall'] > 2]
    ##小数据集不用过滤
    item_inter_num = pd.DataFrame(interaction.groupby(interaction['asin']).count())
    item_inter_num = item_inter_num.reset_index()[['reviewerID', 'asin']]
    item_num = item_inter_num.rename(columns={'reviewerID': 'item_num'})
    interaction = pd.merge(interaction, item_num, how='left', on='asin')
    interaction = interaction[interaction['item_num'] > 5]



    data_path = path + 'meta_' + datasets_name + '.json.gz'
    df_item = getDF(data_path)

    item_property = df_item[['asin', 'price', 'categories']]

    # 判断字符串是否为数字
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    # 去除价格中的美元符号并转换为浮点数
    def reg_price(price):
        # price_num = price.replace('$', '')
        if is_number(price):
            results = float(price)
        else:
            results = ''
        return results

    # 确保类别是非空列表
    def reg_category(cate):
        results = ''
        if isinstance(cate, list):
            if len(cate) == 0:
                results = ''
            else:
                results = cate
        else:
            results = ''
        return results

    # 删除价格和类别为空的商品，转换价格为浮点数，并删除重复的商品记录
    item_property['price_num'] = item_property.price.map(reg_price)
    item_property['categories'] = item_property.categories.map(reg_category)

    # delete items without price
    item_property = item_property.dropna(subset=['price_num'])
    # delete items without category
    item_property = item_property[(item_property['categories'] != '')]
    # delete item without pric
    item_property[['price_num']] = item_property[['price_num']].astype(float)

    item_property.drop_duplicates(subset=['asin'], keep='first', inplace=True)

    def get_cate(cate_list):
        for i in range(len(cate_list)):
            if catename in cate_list[i]:
                cate_list[i].remove(catename)
                return cate_list[i]

    # the last (fine-grained) category of an item is viewed as its category
    item_property['cate'] = item_property.categories.map(get_cate)
    item_all = item_property[['asin', 'price_num', 'cate']]

    # 计算每个类别的商品数量,把每个类别展开
    item_all_expanded = item_all.explode('cate')
    group_cate_num = item_all_expanded.groupby('cate').size().reset_index(name='count')
    # 重置 group_cate_num DataFrame 的索引，并从中提取类别列（cate）和商品数量列（count）
    group_num = group_cate_num.reset_index()[['cate', 'count']]
    # 将最后一列重命名为 count
    group_num = group_num.rename(columns={'count': 'count'})

    # 执行分组统计，计算每个类别的最小价格
    group_cate_min = item_all_expanded.groupby('cate')['price_num'].min().reset_index()
    # 重置 group_cate_min DataFrame 的索引，并提取类别和价格最低值，最后将列名重命名为 min
    group_min = group_cate_min.reset_index()[['cate', 'price_num']].rename(columns={'price_num': 'min'})
    # 使用 groupby 函数按类别分组，并计算每个类别中商品价格的最高值。

    group_cate_max = item_all_expanded.groupby('cate')['price_num'].max().reset_index()
    # 重置 group_cate_max DataFrame 的索引，并提取类别和价格最高值，最后将列名重命名为 max。
    group_max = group_cate_max.reset_index()[['cate', 'price_num']].rename(columns={'price_num': 'max'})

    # 使用 groupby 函数按类别分组，并计算每个类别中商品价格的平均值
    group_cate_mean = item_all_expanded.groupby('cate')['price_num'].mean().reset_index()
    # 重置 group_cate_mean DataFrame 的索引，并提取类别和价格平均值，最后将列名重命名为 mean
    group_mean = group_cate_mean.reset_index()[['cate', 'price_num']].rename(columns={'price_num': 'mean'})

    # 使用 groupby 函数按类别分组，并计算每个类别中商品价格的标准差。
    group_cate_std = item_all_expanded.groupby('cate')['price_num'].std().reset_index()
    # 重置 group_cate_std DataFrame 的索引，并提取类别和价格标准差，最后将列名重命名为 std
    group_std = group_cate_std.reset_index()[['cate', 'price_num']].rename(columns={'price_num': 'std'})

    # 这些行将前面计算得到的统计数据（数量、最低价、最高价、平均价、标准差）与原始商品数据进行合并。
    # 合并后的数据存储在名为 item_data5 的 DataFrame 中
    group_stats = group_num.merge(group_min, on='cate').merge(group_max, on='cate').merge(group_mean, on='cate').merge(
        group_std, on='cate')
    # 合并各类指标并对每个物品所属个各类进行平均
    item_all_exploded = item_all.explode('cate')
    item_all_with_stats = item_all_exploded.merge(group_stats, how='left', on='cate')
    item_aggregated = item_all_with_stats.groupby('asin').agg({
        'count': 'mean',
        'min': 'mean',
        'max': 'mean',
        'mean': 'mean',
        'std': 'mean'
    }).reset_index()
    item_data = item_all.merge(item_aggregated, on='asin', suffixes=('', '_avg'))
    # 筛选出数量大于9且标准差不为0的商品数据
    item_data = item_data[item_data['count'] > 9]
    item_data = item_data[item_data['std'] != 0]
    import math

    def logistic(t, u, s):
        gama = s * (3 ** 0.5) / math.pi
        return 1 / (1 + math.exp((t - u) / gama))

    def get_price_level(price, p_min, p_max, mean, std):
        if std == 0:
            print('only one sample')
            return -1
        fenzi = logistic(price, mean, std) - logistic(p_min, mean, std)
        fenmu = logistic(p_max, mean, std) - logistic(p_min, mean, std)
        if fenmu == 0 or price == 0:
            return -1
        results = fenzi / fenmu * price_level_num
        return int(results)

    item_data['price_level'] = item_data.apply(
        lambda row: get_price_level(row['price_num'], row['min'], row['max'], row['mean'], row['std']), axis=1)
    item_final = item_data[item_data['price_level'] != -1]
    item_final = item_final[['asin', 'price_num', 'cate', 'price_level']]

    user_item1 = pd.merge(interaction , item_final, how='left', on='asin')

    data_all = user_item1.dropna(axis=0)

    # 新加的
    data_all.sort_values(by="reviewerID", inplace=True, ascending=True)

    data = data_all[
        ['reviewerID', 'asin', 'price_num', 'cate', 'price_level']]
    return data


def data_sparse_transfer_dataframe(data, sparse_ratio):
    # 获取用户与商品交互的总数
    num_interactions = len(data)

    # 稀疏化前用户和物品的数量
    original_num_users = data['user'].nunique()  # 统计用户总数
    original_num_items = data['itemID'].nunique()  # 统计物品总数

    print(f"Before sparsification: Total users = {original_num_users}, Total items = {original_num_items}")

    # 计算要移除的交互数
    drop_interaction_num = int(num_interactions * sparse_ratio)

    # 确保每个用户和物品至少保留一条交互
    grouped_by_user = data.groupby('user')
    grouped_by_item = data.groupby('itemID')

    # 找到需要保留的最少交互记录
    user_min_interactions = grouped_by_user.head(1).index  # 保留每个用户的一条交互
    item_min_interactions = grouped_by_item.head(1).index  # 保留每个物品的一条交互

    # 将这两部分合并为必须保留的交互
    must_keep_indices = user_min_interactions.union(item_min_interactions)

    # 确保要删除的交互记录不包括必须保留的交互
    remaining_indices = data.index.difference(must_keep_indices)

    # 确保我们删除的交互数不会超过剩余的可删除交互数
    drop_interaction_num = min(drop_interaction_num, len(remaining_indices))

    # 随机选择要移除的交互
    if drop_interaction_num > 0:
        drop_indices = np.random.choice(remaining_indices, drop_interaction_num, replace=False)
        # 创建稀疏化后的数据（删除选中的交互记录）
        data_sparse = data.drop(drop_indices)
    else:
        data_sparse = data

    # 稀疏化后的用户和物品的数量
    new_num_users = data_sparse['user'].nunique()  # 稀疏化后统计用户总数
    new_num_items = data_sparse['itemID'].nunique()  # 稀疏化后统计物品总数

    print(f"After sparsification: Total users = {new_num_users}, Total items = {new_num_items}")

    # 输出稀疏化前后的稀疏率
    original_sparse_ratio = num_interactions / (original_num_users * original_num_items)
    new_sparse_ratio = len(data_sparse) / (new_num_users * new_num_items)

    print(f"Original sparsity: {original_sparse_ratio:.6f}, Now sparsity: {new_sparse_ratio:.6f}")

    return data_sparse

def datanum(data, datasets_name,intersection):
    reviewerID2userID = {}
    asin2itemID = {}
    category2categoryID = {}
    price2priceID = {}

    userNum = 0
    itemNum = 0
    categoryNum = 0
    priceNum = 0

    data_all = data.rename(
        columns={'reviewerID': 'user', 'asin': 'itemID', 'price_num': 'price', 'price_level': 'priceLevel',
                 'cate': 'categories'})
    data_all = data_all[['user', 'itemID', 'price', 'priceLevel', 'categories']]

    if dataset=='overlap':
        data_all=data_all[data_all['user'].isin(intersection)]

    if issparse == 1:
        print(f"Sparse for {datasets_name} begin")
        data_all = data_sparse_transfer_dataframe(data_all, ratio)

    print('@交互量为：',len(data_all['user']))
    #打乱
    # data_all = data_all.sample(frac=1, random_state=42)

    # 统计不同会话、商品和类别的数量
    for _, row in data_all.iterrows():
        if row['user'] not in reviewerID2userID:
            reviewerID2userID[row['user']] = userNum
            userNum += 1
        if row['itemID'] not in asin2itemID:
            asin2itemID[row['itemID']] = itemNum
            itemNum += 1
        category = frozenset(row['categories'])
        # for category in row['categories']:
        if category not in category2categoryID:
            category2categoryID[category] = categoryNum
            categoryNum += 1
        if row['priceLevel'] not in price2priceID:
            price2priceID[row['priceLevel']] = priceNum
            priceNum += 1

    print('#user: ', userNum)
    print('&item: ', itemNum)
    print('#category: ', categoryNum)
    print('$price: ', priceNum)

    globals()[datasets_name + 'itemNum'] = itemNum
    globals()[datasets_name + 'userNum'] = userNum
    print(datasets_name + '#total user number!!: ', userNum)
    print(datasets_name + '#total item number!!: ', itemNum)

    def reUser(reviewerID):
        if reviewerID in reviewerID2userID:
            return reviewerID2userID[reviewerID]
        else:
            print('user is not recorded')
            return 'none'

    def reItem(asin):
        if asin in asin2itemID:
            return asin2itemID[asin]
        else:
            print('item is not recorded')
            return 'none'

    def reCate(categories):
        category = frozenset(categories)
        if category in category2categoryID:
            return category2categoryID[category]
        else:
            print('cate is not recorded')
            return 'none'


    def priceInt(price):
        return int(price)

    data_all.insert(0, 'userID', '')
    data_all['userID'] = data_all.user.map(reUser)
    data_all['itemID'] = data_all.itemID.map(reItem)
    data_all['priceLevel'] = data_all.priceLevel.map(priceInt)
    data_all['categories'] = data_all.categories.map(reCate)

    train_data_path = file_path + datasets_name

    data = data_all[['user', 'userID', 'itemID', 'priceLevel', 'categories']]

    data = data.sort_values(by='itemID')

    merged = data.groupby(['user', 'userID']).agg({
        'itemID': list,
        'priceLevel': list,
        'categories': list}).reset_index()

    merged_sorted = merged.sort_values(by='userID').reset_index(drop=True)
    merged2 = data.groupby(['itemID']).agg({
        'user': list,
        'userID': list,
        'priceLevel': lambda x: list(set(x)),
        'categories': lambda x: list(set(x))
    }).reset_index()
    merged_sorted.to_csv(train_data_path + '.txt', sep=' ',
                         columns=['user', 'userID', 'itemID', 'priceLevel', 'categories'], index=False)
    merged2.to_csv(train_data_path + '_item.txt', sep=' ',
                   columns=['itemID', 'user', 'userID', 'priceLevel', 'categories'], index=False)

    return data


# ----------------------------构建超图---------------------------------------#
def ConstuctHG(data, datasets_name):
    # dict (sessionID:[itemID,itemID])
    user_all = {}
    # dict (sessionID:[priceLevel, priceLevel])
    price_all = {}
    # dict (sessionID:[cate, cate])
    cate_all = {}

    for _, row in data.iterrows():
        USER = row['user']
        user_id = row['userID']
        item_id = row['itemID']
        price = row['priceLevel']
        cate = row['categories']
        if user_id in user_all:
            user_all[user_id].append(item_id)
            price_all[user_id].append(price)
            cate_all[user_id].append(cate)
        else:
            user_all[user_id] = []
            user_all[user_id].append(item_id)
            price_all[user_id] = []
            price_all[user_id].append(price)
            cate_all[user_id] = []
            cate_all[user_id].append(cate)

    train_data_path = file_path + dataset
    # print("prepare nega_dataset ")
    tra_items,tra_itm,tra_pri,tra_cat,tes_itm,tes_pri,tes_cat,val_itm,val_pri,val_cat,all_item,all_price,all_cate=obtain_neg_tra_test(train_data_path)

    def tomatrix(all_itm, all_pri, all_cate):
        # 每个价格对应的商品列表映射
        price_item_dict = {}
        price_item = []
        # 每个价格对应的类别列表映射
        price_category_dict = {}
        price_category = []
        # 每个类别对应的商品列表映射
        category_item_dict = {}
        category_item = []
        # price-item dict -> {price_id:[1, 3, 4]}

        for it, ps, cs in zip(all_itm, all_pri, all_cate):
            for i_temp, p_temp, c_temp in zip(it, ps, cs):
                if p_temp not in price_item_dict: # print('price_new: ',p_temp)
                    price_item_dict[p_temp] = []
                if p_temp not in price_category_dict:
                    price_category_dict[p_temp] = []
                if c_temp not in category_item_dict: #  print('category_new: ',c_temp)
                        category_item_dict[c_temp] = []
                price_item_dict[p_temp].append(i_temp)
                price_category_dict[p_temp].append(c_temp)
                category_item_dict[c_temp].append(i_temp)
        # 键值对按照键进行排序
        price_item_dict = dict(sorted(price_item_dict.items()))
        price_category_dict = dict(sorted(price_category_dict.items()))
        category_item_dict = dict(sorted(category_item_dict.items()))#有的cate估计没有item对应
        print("#price", len(price_item_dict))
        print("#去掉0的category", len(category_item_dict))

        price_item = list(price_item_dict.values())
        price_category = list(price_category_dict.values())
        category_item = list(category_item_dict.values())
        return price_item, price_category, category_item

    # 将输入的会话数据 all_sessions 转换为稀疏矩阵的三元组形式，以便后续处理或分析
    def HG_matrix(all_data):
        #     coo_matrix
        # indptr：用于存储稀疏矩阵的指针数组，记录每行数据在 indices 和 data 中的起始位置
        # indices：存储非零元素的列索引。
        # 存储非零元素的值
        indptr, indices, data = [], [], []
        indptr.append(0)
        for j in range(len(all_data)):
            eachnum = np.unique(all_data[j])
            length = len(eachnum)
            s = indptr[-1]
            indptr.append((s + length))
            for i in range(length):
                indices.append(eachnum[i])
                data.append(1)
        results = (data, indices, indptr)
        return results

    tra_pi, tra_pc, tra_ci = tomatrix(all_item, all_price,all_cate)

    tra = (tra_itm, tra_pri, HG_matrix(tra_itm), HG_matrix(tra_pri), HG_matrix(tra_cat), HG_matrix(tra_pi),HG_matrix(tra_pc),HG_matrix(tra_ci))


    # ----------保存异构超图------------#

    path_data_train = train_data_path + "_HGtrain.txt"
    pickle.dump(tra, open(path_data_train, 'wb'))

    print("dataset: ", datasets_name)
    return tra_items

single_top_k_ratio = 0.5


def reconstruct_tra_items(H_sparse):
    # 获取稀疏矩阵的非零元素索引 (行和列)
    user_indices, item_indices = H_sparse.nonzero()  # nonzero() 返回行和列的两个数组

    # 初始化 tra_items
    tra_items = {}

    # 遍历非零元素，并构建 tra_items
    for user, item in zip(user_indices, item_indices):
        if user not in tra_items:
            tra_items[user] = []
        tra_items[user].append(item)
    return tra_items

def UerHG(tra_items, datasets_name):
    # ----------------创建用户超图-----------------
    def cal_single_domain_matrix(single_H: csc_matrix, k: int, out_file_u):
        # 计算用户和物品之间的相似度矩

        single_Hu = [single_H]
        H_for_user = single_H.transpose().__matmul__(single_H)

        Hu_k = H_for_user
        print(f"Calculate Hu_k succeed")
        # 将数组Hu_k中的所有元素都设置为1
        Hu_k.data[:] = 1
        print(f"Transfer to 1 succeed")

        # 到时候尝试一下不用（0，1）的结果
        for i in tqdm(range(1, k)):

            Hu_add = single_H.dot(Hu_k.minimum(1))
            Hud_full = sp.csr_matrix(np.full(Hu_add.shape, single_H.sum(axis=1).reshape((-1, 1))))

            Hu_column = Hu_add.multiply(Hud_full)
            Hu_degree_sum=Hu_column.sum(axis=0).A1
            Hu_column_sum = sorted(Hu_degree_sum.copy())
            Hu_flag = Hu_column_sum[int(len(Hu_column_sum) * 0.5)]

            for j in tqdm(range(len(Hu_column_sum))):
                if Hu_column_sum[j] < Hu_flag:
                    Hu_add[:, j] = 0

            single_Hu.append(Hu_add)
            print(f"Append single H succeed")
            Hu_k = Hu_k.dot(H_for_user)

            single_Hu.append(Hu_add)
            print(f"Append single H succeed")
            Hu_k = Hu_k.dot(H_for_user)
        # single_Hu稀疏矩阵
        single_Hu = sp.hstack(single_Hu)
        print(f"Concatenate succeed")
        Duv, Due = calculate_D_matrix(single_Hu)
        print(f"Calculate D matrix succeed")
        # 在求-0.5次方的时候对角线元素必须都有值，否则求出的结果所有元素会都为nan
        Duv = my_power_D(Duv, -0.5)
        Due = my_power_D(Due, -1)

        print(f"Calculate power succeed")
        conv_u = reduce(csc_matrix.__matmul__, [Duv, single_Hu, Due, single_Hu.T, Duv])
        # print(conv_u.toarray())
        print(f"matmul process succeed")
        # 转换为稠密矩阵进行保存
        conv_u_dense = conv_u.toarray()
        np.savez(out_file_u, conv_u_dense)
        # sp.save_npz(out_file_u, conv_u)
        # sp.save_npz(out_file_i, conv_i)
        print(f"save succeed")

    def my_power_D(H, pow):
        diag_elements = H.diagonal()
        diag_elements = np.where(diag_elements != 0, np.power(diag_elements, pow), 0)
        return sp.diags(diag_elements)

    item_to_index = {}
    index = 0
    for items in tra_items.values():
        for item in items:
            if item not in item_to_index:
                item_to_index[item] = index
                index += 1
    # 准备稀疏矩阵的数据
    rows = []
    cols = []
    data = []

    for user, items in tra_items.items():
        for item in items:
            rows.append(user)
            cols.append(item_to_index[item])
            data.append(1)

    # 创建稀疏矩阵,用全局变量
    num_users = globals()[datasets_name + 'userNum']
    num_items = globals()[datasets_name + 'itemNum']
    HG = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    k=2
    print(f"Cal matrixs begin for k = {k}......")
    back = f"_{k}_top05.npz"
    print(f"back = {back}")
    cal_single_domain_matrix(HG, k, os.path.join(file_path, "conv_" + datasets_name + back))
    print(f"Cal matrixs for k = {k} succeed......")

def load_data(train_data_path):
    print('loading data...{}')
    item2peo = pd.read_csv(train_data_path+ '_item.txt', sep=' ')
    peo2item = pd.read_csv(train_data_path + '.txt', sep=' ')
    return item2peo, peo2item


def obtain_neg_tra_test(train_data_path):

    item2peo, peo2item= load_data(train_data_path)
    num_item = len(item2peo)
    nega = {}
    train_items = []
    train_price = []
    train_cate = []
    data_test = []
    data_vali = []

    test_items = []
    test_price = []
    test_cate = []

    val_items = []
    val_price = []
    val_cate = []

    all_item=[]
    all_price=[]
    all_cate=[]
    tra_items = dict()
    for i in range(len(peo2item['userID'])):
        user = peo2item['user'][i]
        userid=peo2item['userID'][i]
        Items = eval(peo2item['itemID'][i])
        Price = eval(peo2item['priceLevel'][i])
        Category = eval(peo2item['categories'][i])
        all_item.append(Items.copy())
        all_price.append(Price.copy())
        all_cate.append(Category.copy())
        if len(Items) > 2:
            items_choices = random.sample(range(len(Items)), 2)

            item_test = eval(peo2item['itemID'][i])[items_choices[0]]
            price_test = eval(peo2item['priceLevel'][i])[items_choices[0]]
            cate_test = eval(peo2item['categories'][i])[items_choices[0]]

            item_vali = eval(peo2item['itemID'][i])[items_choices[1]]
            price_vali=eval(peo2item['priceLevel'][i])[items_choices[1]]
            cate_vali=eval(peo2item['categories'][i])[items_choices[1]]

            #去除
            Items.remove(item_test)
            Items.remove(item_vali)
            Price.remove(price_test)
            Price.remove(price_vali)
            Category.remove(cate_test)
            Category.remove(cate_vali)
        elif len(Items)<2:
            item_test = eval(peo2item['itemID'][i])[0]
            price_test = eval(peo2item['priceLevel'][i])[0]
            cate_test = eval(peo2item['categories'][i])[0]

            item_vali = eval(peo2item['itemID'][i])[0]
            price_vali=eval(peo2item['priceLevel'][i])[0]
            cate_vali=eval(peo2item['categories'][i])[0]
        else:
            item_test = eval(peo2item['itemID'][i])[0]
            price_test = eval(peo2item['priceLevel'][i])[0]
            cate_test = eval(peo2item['categories'][i])[0]

            item_vali = eval(peo2item['itemID'][i])[1]
            price_vali=eval(peo2item['priceLevel'][i])[1]
            cate_vali=eval(peo2item['categories'][i])[1]

        data_vali.append([user, userid, item_vali, price_vali, cate_vali])
        data_test.append([user, userid, item_test, price_test, cate_test])
            ##测试集
        test_items.append(item_test)
        test_price.append(price_test)
        test_cate.append(cate_test)
        ###验证集
        val_items.append(item_vali)
        val_price.append(price_vali)
        val_cate.append(cate_vali)

        train_items.append(Items)
        train_price.append(Price)
        train_cate.append(Category)
        tra_items[peo2item['userID'][i]]=Items

        all_nega = list((set(list(range(num_item))) - set(Items)))
        nega[i] = shuffle(all_nega, random_state=2020)[:99]

    df_vali = pd.DataFrame(data_vali, columns=['user','userID', 'itemID', 'price', 'category'])
    df_vali.to_csv(train_data_path + "_vali.txt", sep=' ', index=False)

    df_test = pd.DataFrame(data_test, columns=['user','userID', 'itemID', 'price', 'category'])
    df_test.to_csv(train_data_path + "_test.txt", sep=' ', index=False)

    nega = [(key, value) for key, value in nega.items()]
    nega = pd.DataFrame.from_dict(nega, orient='columns')
    nega.to_csv(train_data_path + '_nega.txt', sep=' ', index=False)

    result = {
        'tra_items': tra_items,
        'train_items': train_items,
        'train_price': train_price,
        'train_cate': train_cate,
        'test_items': test_items,
        'test_price': test_price,
        'test_cate': test_cate,
        'val_items': val_items,
        'val_price': val_price,
        'val_cate': val_cate,
        'all_item': all_item,
        'all_price': all_price,
        'all_cate': all_cate
    }
    tra_items=result['tra_items']
    train_items=result['train_items']
    train_price=result['train_price']
    train_cate=result['train_cate']
    test_items=result['test_items']
    test_price=result['test_price']
    test_cate=result['test_cate']
    val_items=result['val_items']
    val_price=result['val_price']
    val_cate=result['val_cate']
    all_item=result['all_item']
    all_price=result['all_price']
    all_cate=result['all_cate']
    # return result
    return tra_items,train_items,train_price,train_cate,test_items,test_price,test_cate,val_items,val_price,val_cate,all_item,all_price,all_cate

def create_train_graph(datasets_name, tra_items):
    # 用于存储图的边的起始节点和目标节点
    src = []
    dst = []
    u_i_pairs = set()  # 存储已经添加过的用户-物品对，以避免重复添加。
    USER_DICT = tra_items
    max_iid = -1  # 动态计算物品ID的最大值
    # for uid in TRAIN_RECORDS[task]:
    for uid in USER_DICT.keys():
        iids = USER_DICT[uid]
        for iid in iids:
            if (uid, iid) not in u_i_pairs:
                src.append(int(uid))
                dst.append(int(iid))
                u_i_pairs.add((uid, iid))
                if int(iid) > max_iid:
                    max_iid = int(iid)
    u_num = globals()[datasets_name + 'userNum']
    i_num =  globals()[datasets_name + 'itemNum']
    src_ids = torch.tensor(src)
    dst_ids = torch.tensor(dst) + u_num
    # 所有物品ID加上用户数量 u_num 以确保物品节点的ID不与用户节点的ID冲突
    # 根据起始节点和目标节点创建一个有向图 g，节点总数为用户数量和物品数量之和。
    g = dgl.graph((src_ids, dst_ids), num_nodes=u_num + i_num)
    g = dgl.to_bidirected(g)  # 将有向图转换为双向图，即每条边都添加一个反向边，增加图的连接性。
    return g


def create_overlap_dict(target, source, USER_OVERLAP_DICT):
    intersection_AB = pd.read_csv('intersection_AB_' + target + '.txt', sep=' ')
    intersection_BA = pd.read_csv('intersection_AB_' + source + '.txt', sep=' ')

    # intersection_AB=pd.read_csv('dataset-Amazon/jz_to_text_jpg/reviews_'+target+'_5_overlapuser_item.txt', sep=' ')
    # intersection_BA=pd.read_csv('dataset-Amazon/jz_to_text_jpg/reviews_'+source+'_5_overlapuser_item.txt', sep=' ')

    if (target, source) not in USER_OVERLAP_DICT:
        USER_OVERLAP_DICT[(target, source)] = [[], []]
    USER_OVERLAP_DICT[(target, source)][0] = intersection_AB['userID']
    USER_OVERLAP_DICT[(target, source)][1] = intersection_BA['userID']
    return USER_OVERLAP_DICT


def batch(x, bs):
    x = list(range(x))
    return [x[i:i + bs] for i in range(0, len(x), bs)]


# 计算两个图之间的节点对齐
def cal_align(target, sorce, train_graphs):
    u_rw_len = 4  # 用户随机游走的长度
    i_rw_len = 3  # 物品随机游走的长度
    rw_times = 500  # 随机游走次数
    node_sample_num_max = 10000  # 定义节点采样的最大数量
    batch_size = 4196
    USER_OVERLAP_DICT = dict()
    USER_OVERLAP_DICT = create_overlap_dict(target, sorce, USER_OVERLAP_DICT)

    g1 = train_graphs[target]
    ppr_g1 = torch.zeros((g1.num_nodes(), g1.num_nodes()))
    d_g1 = []
    for nodes in batch(g1.num_nodes(), batch_size):  # 对第一个图的节点进行批处理
        # trace_g1_u 和 trace_g1_i 是两次随机游走得到的节点轨迹（路径）
        trace_g1_u, _ = dgl.sampling.random_walk(g1, torch.tensor(nodes).repeat(rw_times), length=u_rw_len)
        trace_g1_i, _ = dgl.sampling.random_walk(g1, torch.tensor(nodes).repeat(rw_times), length=i_rw_len)
        # trace_g1_u[:, -1] 和 trace_g1_i[:, -1] 分别表示随机游走轨迹的最后一个节点
        # 如果轨迹的最后一个节点是 -1（表示无效节点），则返回轨迹的第一个节点 trace_g1_u[:, 0] 或 trace_g1_i[:, 0]；
        # 否则返回轨迹的最后一个节点。
        tr_g1_u_end = torch.where(trace_g1_u[:, -1] == -1, trace_g1_u[:, 0], trace_g1_u[:, -1])
        tr_g1_i_end = torch.where(trace_g1_i[:, -1] == -1, trace_g1_i[:, 0], trace_g1_i[:, -1])
        # 将轨迹的最后一个节点重新排列
        dst_g1_u = tr_g1_u_end.reshape(rw_times, len(nodes)).T
        dst_g1_i = tr_g1_i_end.reshape(rw_times, len(nodes)).T
        dst_g1 = torch.cat([dst_g1_u, dst_g1_i], dim=1)
        d_g1.append(dst_g1)
    dst_g1 = torch.cat(d_g1)  # 轨迹

    g2 = train_graphs[sorce]
    indices = torch.tensor(USER_OVERLAP_DICT[(target, sorce)])
    sampling_size = min(node_sample_num_max, indices[0].shape[0])
    ppr_g2 = torch.zeros((g2.num_nodes(), g2.num_nodes()))
    sampled_src = indices[:, random.sample(range(indices[0].shape[0]), sampling_size)]
    d_g2 = []
    for nodes in batch(g2.num_nodes(), batch_size):
        trace_g2_u, _ = dgl.sampling.random_walk(g2, torch.tensor(nodes).repeat(rw_times), length=u_rw_len)
        trace_g2_i, _ = dgl.sampling.random_walk(g2, torch.tensor(nodes).repeat(rw_times), length=i_rw_len)

        tr_g2_u_end = torch.where(trace_g2_u[:, -1] == -1, trace_g2_u[:, 0], trace_g2_u[:, -1])
        tr_g2_i_end = torch.where(trace_g2_i[:, -1] == -1, trace_g2_i[:, 0], trace_g2_i[:, -1])

        dst_g2_u = tr_g2_u_end.reshape(rw_times, len(nodes)).T
        dst_g2_i = tr_g2_i_end.reshape(rw_times, len(nodes)).T
        dst_g2 = torch.cat([dst_g2_u, dst_g2_i], dim=1)
        d_g2.append(dst_g2)
    dst_g2 = torch.cat(d_g2)
    for m in range(g1.num_nodes()):
        ppr_g1[m] = torch.bincount(dst_g1[m], minlength=g1.num_nodes())
    ppr_g1_norm = F.normalize(ppr_g1[:, sampled_src[0]], p=2)
    for m in range(g2.num_nodes()):
        ppr_g2[m] = torch.bincount(dst_g2[m], minlength=g2.num_nodes())
    ppr_g2_norm = F.normalize(ppr_g2[:, sampled_src[1]], p=2)

    sims = torch.matmul(ppr_g1_norm, ppr_g2_norm.T)
    return sims


def fun_load_original_data(datasets_name, data_music, data_instrument):
    # 计算两两域之间的用户交集

    intersection_AB = set(data_music['user']) & set(data_instrument['user'])
    print('domain间两两的用户交集数目：AB= ', len(intersection_AB))

    records1 = data_music[data_music['user'].isin(intersection_AB)]
    records1.to_csv('intersection_AB_' + datasets_name[0] + '.txt', sep=' ', index=False)

    records2 = data_instrument[data_instrument['user'].isin(intersection_AB)]
    records2.to_csv('intersection_AB_' + datasets_name[1] + '.txt', sep=' ', index=False)


def dual_domain_gcn_prepare(H, out_file_dict, out_file_name):
    H[(H > 0.2) & (H < 0.4)] = 1
    H[(H >= 0.4) & (H < 1)] = 2
    H[(H <= 0.2)] = 0
    # csr_matrix 函数构建一个稀疏矩阵，然后将其转换为稠密矩
    H_numpy = H.numpy()
    # 使用 csr_matrix 函数构建一个稀疏矩阵，然后将其转换为稠密矩阵
    H_result = csr_matrix(H_numpy).toarray()
    np.savez(out_file_name, H_result)
    # 构造结果文件的路径
    out_file_name_with_k = os.path.join(out_file_dict, out_file_name + ".npz")
    print(f"Out file path : {out_file_name_with_k}")
    # 保存结果
    np.savez(out_file_name_with_k, H_result)
    print("Save succeed")


def calculate_D_matrix(H):  # 使用了 Scipy 库来创建稀疏对角矩阵
    # v_degree = H.sum(axis=1)
    # e_degree = H.sum(axis=0)
    v_degree = np.array(H.sum(axis=1).reshape(1, H.sum(axis=1).shape[0])).squeeze()
    e_degree = np.array(H.sum(axis=0)).squeeze()
    # return torch.diag(v_degree), torch.diag(e_degree)
    return sp.diags(v_degree), sp.diags(e_degree)
def calculate_D_matrix_np(H):#使用了 Numpy 库来
    #计算每个节点的出度
    v_degree = H.sum(axis=1)
    #计算每个节点的入度
    e_degree = H.sum(axis=0)
    # return torch.diag(v_degree), torch.diag(e_degree)
    #返回了顶点度和边的度构成的对角矩阵
    return np.diag(v_degree).astype(np.float32), np.diag(e_degree).astype(np.float32)

def cal_dual_GCN_params(H, out_file_dict, out_file_name):
    out_file_name = os.path.join(out_file_dict, out_file_name + f".npz")  # 构造结果文件的路径
    print(f"Out file path : {out_file_name}")
    H = csc_matrix(H)
    Dv, De = calculate_D_matrix(H)
    Dv21 = csc_matrix.power(Dv, -0.5)
    De21 = csc_matrix.power(De, -0.5)
    # reduce函数:按照np.matul的原则进行计算，对参数序列中元素进行累积，返回值是一个数值。
    # np.matmul计算两个矩阵的乘积
    result_matrix = reduce(csc_matrix.__matmul__, [Dv21, H, De21])
    result = result_matrix.toarray()
    np.savez(out_file_name, result)
    print("Save succeed")


def cal_all_dual_GCN_params(target, source):
    print(f"Cal topk dual domain matrix for GCN, k = {100}")
    print(f"Cal Ha begin")
    Ha_fuse_u = np.load(os.path.join(file_path, "HG_u_" + target + "_fuse.npz"))["arr_0"]
    cal_dual_GCN_params(Ha_fuse_u, os.path.join(file_path), "Ha_GCN_u")

    Ha_fuse_i = np.load(os.path.join(file_path, "HG_i_" + target + "_fuse.npz"))["arr_0"]
    cal_dual_GCN_params(Ha_fuse_i, os.path.join(file_path), "Ha_GCN_i")

    print(f"Cal Hb begin")
    Hb_fuse_u = np.load(os.path.join(file_path, "HG_u_" + source + "_fuse.npz"))["arr_0"]
    cal_dual_GCN_params(Hb_fuse_u, os.path.join(file_path), "Hb_GCN_u")

    Hb_fuse_i = np.load(os.path.join(file_path, "HG_i_" + source + "_fuse.npz"))["arr_0"]
    cal_dual_GCN_params(Hb_fuse_i, os.path.join(file_path), "Hb_GCN_i")

    print("===================================================================")

if __name__ == '__main__':


    data_music = dataprocess(datasets_name[0],catename[0])
    data_instrument = dataprocess(datasets_name[1],catename[1])

    intersection = set(data_music['reviewerID']) & set(data_instrument['reviewerID'])


    data_music=datanum(data_music,datasets_name[0],intersection)
    data_instrument=datanum(data_instrument,datasets_name[1],intersection)
    #
    #
    #
    itemdata_music=pd.read_csv(file_path+datasets_name[0]+'_item.txt', sep=' ')
    itemdata_isntrument = pd.read_csv(file_path + datasets_name[1] + '_item.txt', sep=' ')



    music = pd.read_csv(file_path+datasets_name[0]+'.txt', sep=' ')
    instrument = pd.read_csv(file_path+datasets_name[1]+'.txt', sep=' ')
    intersection_AB = fun_load_original_data(datasets_name, music, instrument)

    graph_pairs = dict()
    for dataset in datasets_name:
        train_data_path = file_path + dataset
        # 用户-物品交互图(train版)，基于共现关系
        if dataset == datasets_name[0]:
            data = data_music
        else:
            data = data_instrument

        tra_items = ConstuctHG(data, dataset)
        print("prepare single-domain userHG:")
        UerHG(tra_items, dataset)

        print("prepare dual-domain HG")
        graph_pairs[dataset] = create_train_graph(dataset, tra_items)
    # 域间：

    HG = {}
    for target in datasets_name:
        for j in range(len(datasets_name)):
            if datasets_name[j] != target:
                align_dict = cal_align(target, datasets_name[j], graph_pairs)
                au_num = globals()[target + 'userNum']
                bu_num = globals()[datasets_name[j] + 'userNum']
                # 将 'Ha_i_' 与 target 连接起来作为字典的键
                u_name = 'HG_u_' + target
                i_name = 'HG_i_' + target
                # 使用字典来存储变量
                HG[u_name] = align_dict[:au_num, :bu_num]
                HG[i_name] = align_dict[au_num:, bu_num:]
        # -------------------user---------------
        print(f"Cal HG_u_" + target + " begin")
        dual_domain_gcn_prepare(HG[u_name], os.path.join(file_path), "HG_u_" + target + "_fuse")
        print(f"Cal HG_i_" + target + " begin")

        dual_domain_gcn_prepare(HG[i_name], os.path.join(file_path), "HG_i_" + target + "_fuse")
        print("===================================================================")

    cal_all_dual_GCN_params(datasets_name[0], datasets_name[1])
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("done")