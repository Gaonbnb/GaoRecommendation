#!D:/Program/Anaconda/anaconda/envs/pych/python.exe
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import json
import pickle

def read_data(path):
    """
    读取数据并进行命名的更改
    """

    sample_data = pd.read_csv(path)

    # history中包括shop_id, cate click/buy floor time，将原始的文件格式生成多个序列的格式
    sample_data.columns = ["userid", "history", "age", "sex", "level_id"]
    def Split(data):
        dataListResult = []
        dataList = data.split("|")
        for i in [0, 2, 3, 4, 5]:
            dataList_ = [data.split("#")[i]  for data in dataList]
            dataList_ = "#".join(dataList_)
            dataListResult.append(dataList_)
        return dataListResult
    for i, state in enumerate(["shop_id", "cate", "event_type", "floor", "time"]):
        sample_data[state] = sample_data.history.apply(lambda x: Split(x)[i])

    sample_data = sample_data.drop("history", axis=1, inplace=False)
    return sample_data


def generate_dict(colname):
    """
    生成映射字典
    """
    shop_id_set = set()
    for shop_id_data in sample_data[colname].tolist():
        shop_id_data = shop_id_data.split("#")
        for data in shop_id_data:
            shop_id_set.add(data)
    
    unq, idx = np.unique(list(shop_id_set), return_inverse=True)
    shop_id_dict = dict(zip(unq, idx))
    return shop_id_dict



def replace_value(x, col):
    """
    替换原有的id变成index，为神经网络中的embedding作准备
    """
    res = []
    for mask in x.split("#"):
        if col == "shop_id":    
            res.append(shop_id_dict[mask])
        elif col == "cate":
            res.append(cate_dict[mask])
        else:
            res.append(floor_dict[mask])
    return res


def generate_neg_dict_df(sample_data):
    """
    生成时间比较长,30分钟
    """
    neg_dict_df = pd.DataFrame(columns=("userid", "last_click", "last_time", "label"))

    for userid in sample_data["userid"]:
        
        last_time = sample_data[sample_data["userid"] == userid]["time"].tolist()[0][-1]
        
        random_num = np.random.randint(0, 145, 2)
        for single_num in random_num:
            if single_num not in user_shop_id_dict[userid]:
                
                neg_dict_df = neg_dict_df.append(pd.DataFrame([[userid, single_num, last_time, 0]], columns = ["userid", "last_click", "last_time", "label"]))
        

    # 生成所有的负采样的结果
    neg_dict_df = sample_data.merge(neg_dict_df, on=["userid"], how="right")



    # 直接写入csv list格式变成了str，需要按行写入。。。以后存储一定要用#。。。
    ### neg_dict_df.to_csv("neg_sample_df.csv")
    neg_dict_df.shop_id = neg_dict_df.shop_id.apply(lambda x: "#".join(list(map(str, x))))
    neg_dict_df.cate = neg_dict_df.cate.apply(lambda x: "#".join(list(map(str, x))))
    neg_dict_df.floor = neg_dict_df.floor.apply(lambda x: "#".join(list(map(str, x))))
    neg_dict_df.to_csv("neg_sample_df.csv")

    neg_dict_df = pd.read_csv("neg_sample_df.csv", index_col=0)
    neg_dict_df.shop_id = neg_dict_df.shop_id.apply(lambda x: list(map(int, x.split("#"))))
    neg_dict_df.cate = neg_dict_df.cate.apply(lambda x: list(map(int, x.split('#'))))
    neg_dict_df.floor = neg_dict_df.floor.apply(lambda x: list(map(int, x.split("#"))))

    neg_dict_df = neg_dict_df.sample(n=10, replace=False, random_state=1, axis=0)
    neg_dict_df = neg_dict_df.reset_index(drop=True)
    return neg_dict_df

def read_store_data(path):
    """
    读取物品侧数据
    """
    store_cate = pd.read_csv(path, header=None)
    store_cate = store_cate.rename(columns={0:"last_click", 3:"cate_id"})
    store_cate = store_cate[["last_click", "cate_id"]]
    store_cate.last_click = store_cate.last_click.map(lambda x: shop_id_dict[str(x)])
    store_cate.cate_id = store_cate.cate_id.map(lambda x: cate_dict[str(x)])
    return store_cate

def get_seq_sample_data(sample_data):
    """
    下面是将所有序列的正样本都抽取出来
    """
    # 下面是将所有序列的正样本都抽取出来
    sample_data["last_click"] = sample_data.shop_id.apply(lambda x : x[-1])
    sample_data["last_time"] = sample_data.time.apply(lambda x: x[-1])
    sample_data["shop_id"] = sample_data.shop_id.apply(lambda x: x[:-1])
    sample_data["cate"] = sample_data.cate.apply(lambda x: x[:-1])
    sample_data["floor"] = sample_data.floor.apply(lambda x: x[:-1])
    sample_data["time"] = sample_data.time.apply(lambda x: x[:-1])
    cur_sample_data = sample_data

    while not cur_sample_data.event_type.empty:
        wantList = []
        for index, list_ in enumerate(cur_sample_data.shop_id):
            if len(list_) > 1:
                wantList.append(index)

        # 找到现在的索引
        cur_sample_data = cur_sample_data.iloc[wantList, :]
        cur_sample_data.reset_index(drop=True, inplace=True)
        
        cur_sample_data.last_click = cur_sample_data.shop_id.apply(lambda x : x[-1])
        cur_sample_data.last_time = cur_sample_data.time.apply(lambda x: x[-1])
        cur_sample_data["shop_id"] = cur_sample_data.shop_id.map(lambda x: x[:-1])
        cur_sample_data.cate = cur_sample_data.cate.map(lambda x: x[:-1])
        cur_sample_data.floor = cur_sample_data.floor.map(lambda x: x[:-1])
        cur_sample_data.time = cur_sample_data.time.map(lambda x: x[:-1])

        # 纵向堆叠
        sample_data = pd.concat([sample_data, cur_sample_data], axis = 0)
    return sample_data

def save_sample_data(sample_data):
    sample_data.to_csv("./use_data/finished_sample_data.csv", index = False)

def save_json(shop_id_dict):
    """将映射字典存储为json格式"""
    if isinstance(shop_id_dict, str):
        dict = eval(shop_id_dict)
    with open('./use_data/shop_id_dict.txt', 'w', encoding='utf-8') as f:
        # f.write(str(dict))  # 直接这样存储的时候，读取时会报错JSONDecodeError，因为json读取需要双引号{"aa":"BB"},python使用的是单引号{'aa':'bb'}
        str_ = json.dumps(shop_id_dict, ensure_ascii=False) # TODO：dumps 使用单引号''的dict ——> 单引号''变双引号"" + dict变str
        #print(type(str_), str_)
        f.write(str_)

def load_json():
    """将json格式的映射字典读取出来"""
    with open('./use_data/shop_id_dict.txt', 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        #print(type(data), data)
        dict = json.loads(data)
        #print(type(dict),dict)
        return dict

def save_dict(obj, name):
    """将字典映射为pickle类型"""
    with open('./use_data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    """将pickle读出"""
    with open('./use_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # 读取数据
    sample_data = read_data("rec_raw_data_hot_clickbuy_1.0.csv")
    # 生成映射字典
    shop_id_dict = generate_dict("shop_id")
    cate_dict = generate_dict("cate")
    floor_dict = generate_dict("floor")
    # 利用映射字典生成对应的index
    sample_data.shop_id = sample_data.shop_id.map(lambda x : replace_value(x, "shop_id"))
    sample_data.cate = sample_data.cate.map(lambda x: replace_value(x, "cate"))
    sample_data.floor = sample_data.floor.map(lambda x: replace_value(x, "floor"))
    sample_data.time = sample_data.time.map(lambda x: x.split("#"))
    # 现有的负样本读取
    # neg_dict_df = generate_neg_dict_df(sample_data)
    neg_dict_df = pd.read_csv("neg_sample_df.csv", index_col=0)
    # 控制负采样数量
    neg_dict_df = neg_dict_df.sample(n=10, replace=False, random_state=1, axis=0)
    neg_dict_df = neg_dict_df.reset_index(drop=True)
    # 用户对应历史shop字典
    user_shop_id_dict = {userid: set(shop_id) for userid,  shop_id in zip(sample_data["userid"], sample_data["shop_id"])}
    # DSSM物品侧
    store_cate = read_store_data("store_profile_hot.csv")
    sample_data = get_seq_sample_data(sample_data)
    # 生成标签
    sample_data["label"] = 1
    # 正负样本堆叠
    sample_data = pd.concat([sample_data, neg_dict_df], axis=0)
    # 将用户侧和物品侧拼接起来
    sample_data = pd.merge(sample_data, store_cate, on=["last_click"], how="left")
    # 保存sample_data
    save_sample_data(sample_data)
    # 保存字典
    save_dict(shop_id_dict, "shop_id_dict")
    
    