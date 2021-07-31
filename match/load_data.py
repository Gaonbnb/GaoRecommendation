#!D:/Program/Anaconda/anaconda/envs/pych/python.exe
# -*- coding: UTF-8 -*-
"""
将之前数据预处理得到的数据载入模型的数据载入模块
"""
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader

class PrepareData:
    """
        将原始数据通过这个数据准备的类放入数据载入模块
    """
    def __init__(self, sample_data):
        """读入数据"""
        self.sample_data = sample_data
        
    def train_sample_data(self):
        """候选物品点击时间在2018-04-11之前的数据"""
        train_sample_data = self.sample_data[self.sample_data.last_time < "2018-04-11"]
        return train_sample_data
        
    def valid_sample_data(self):
        """候选物品点击时间在2018-04-11之后的数据"""
        valid_sample_data = self.sample_data[self.sample_data.last_time >= "2018-04-11"]
        return valid_sample_data

    def train_softmax_loss_sample_data(self):
        """候选物品点击时间在2018-04-11之前的数据，并且标签为1"""
        train_softmax_loss_sample_data = self.sample_data[(self.sample_data.last_time < "2018_04-11") & (self.sample_data["label"] == 1)]
        return train_softmax_loss_sample_data

    def valid_softmax_loss_sample_data(self):
        """候选物品点击时间在2018-04-11之后的数据，并且标签为1"""
        valid_softmax_loss_sample_data = self.sample_data[(self.sample_data.last_time >= "2018-04-11") & (self.sample_data.label == 1)]
        return valid_softmax_loss_sample_data

    def user_embedding_sample_data(self):
        """生成userembedding的数据"""
        user_embedding_sample_data = self.sample_data.groupby(by="userid").head(1).reset_index(drop=True)
        return user_embedding_sample_data

    def item_embedding_sample_data(self):
        """生成itemembedding的数据"""
        item_embedding_sample_data = self.sample_data.groupby(by="last_click").head(1).reset_index(drop=True)
        return item_embedding_sample_data


# 
class JdataDataset(Dataset):
    """dataset中尽量存储np格式的数据,但本类返回的都是list中的tensor数据，初始设计的问题"""

    def __init__(self, data, sampled_softmax_loss=False):
        """
        sampled_softmax_loss:控制是否进行负采样，假如进行负采样则最后的label是负采样出来的shop_id
        """
        self.sampled_softmax_loss = sampled_softmax_loss
        self.data = data[["sex", "level_id", "shop_id", "cate", "floor", "last_click", "label", "cate_id"]]
        self.data = self.data.values

    def __getitem__(self, idx):
        """对每个读入的数据返回他的idx情况"""
            
        # 下面这句话直接把初始化的self.data给改了。。。。。。
        # self.data = self.data[idx]
        cur_data = self.data[idx]
        
        # 转tensor的几种方式，Tensor()默认是浮点数
        self.data_sex = torch.from_numpy(np.array(cur_data[0]))
        self.data_level_id = torch.from_numpy(np.array(cur_data[1]))
        # list：str转int
        # if isinstance(cur_data[2], int):
        #     cur_data[2] = list(map(int, cur_data[2].strip("[]").strip(","))) if "," in cur_data[2] else list(map(int, cur_data[2].strip("[]")))
        # else:
        #     cur_data[2] = list(map(int, cur_data[2].strip("[]").strip(","))) if "," in cur_data[2] else list(map(int, cur_data[2].strip("[]")))

        # if isinstance(cur_data[3], int):
        #     cur_data[3] = list(map(int, cur_data[3].strip("[]").strip(","))) if "," in cur_data[3] else list(map(int, cur_data[3].strip("[]")))
            
        # else:
        #     cur_data[2] = list(map(int, cur_data[2].strip("[]").strip(","))) if "," in cur_data[2] else list(map(int, cur_data[2].strip("[]")))
        # if isinstance(cur_data[4], int):
        #     cur_data[4] = list(map(int, cur_data[4].strip("[]").strip(","))) if "," in cur_data[4] else list(map(int, cur_data[4].strip("[]")))
        # else:
        #     cur_data[2] = list(map(int, cur_data[2].strip("[]").strip(","))) if "," in cur_data[2] else list(map(int, cur_data[2].strip("[]")))
            
        self.data_shop_id = torch.LongTensor(cur_data[2])
        self.data_cate = torch.LongTensor(cur_data[3])
        self.data_floor = torch.LongTensor(cur_data[4])
        self.data_last_click_id = torch.from_numpy(np.array(cur_data[5]))
        self.data_last_click_id_cate = torch.from_numpy(np.array(cur_data[7]))
        
        if not self.sampled_softmax_loss:
            self.label = torch.from_numpy(np.array(cur_data[6]))
            return [self.data_sex, self.data_level_id, self.data_shop_id, self.data_cate, self.data_floor, self.data_last_click_id, self.data_last_click_id_cate], self.label
        else:
            with open('./use_data/' + "shop_id_dict" + '.pkl', 'rb') as f:
                shop_id_dict =  pickle.load(f)
            neg_label = np.random.randint(0, len(shop_id_dict)-1, 10)
            if self.data_shop_id in neg_label:
                neg_label_list = neg_label
                neg_label = np.delete(neg_label, neg_label_list.index(self.data_shop_id))
            return [self.data_sex, self.data_level_id, self.data_shop_id, self.data_cate, self.data_floor, self.data_last_click_id, self.data_last_click_id_cate], neg_label
            
    def __len__(self):
        """所有读入数据的总体长度定义"""
        return len(self.data)


class JdataDataset_yield(Dataset):
    """
        将数据集变成生成器，生成数据
        """
    def __init__(self, data):
        
        self.data = data[["sex", "level_id", "shop_id", "cate", "floor", "last_click", "label", "cate_id"]]
        self.data = self.data.values
        
    def __getitem__(self, idx):
        return next(self.get_data(idx))

    def get_data(self, idx):
        cur_data = self.data[idx]
        # 转tensor的几种方式，Tensor()默认是浮点数
        self.data_sex = torch.from_numpy(np.array(cur_data[0]))
        self.data_level_id = torch.from_numpy(np.array(cur_data[1]))
        self.data_shop_id = torch.LongTensor(cur_data[2])
        self.data_cate = torch.LongTensor(cur_data[3])
        self.data_floor = torch.LongTensor(cur_data[4])
        self.data_last_click_id = torch.from_numpy(np.array(cur_data[5]))
        self.data_last_click_id_cate = torch.from_numpy(np.array(cur_data[7]))
        # label
        self.label = torch.from_numpy(np.array(cur_data[6]))
        # 生成
        yield [self.data_sex, self.data_level_id, self.data_shop_id, self.data_cate, self.data_floor, self.data_last_click_id, self.data_last_click_id_cate], self.label

    def __len__(self):
        return len(self.data)

# 每个batch要有相同的大小,因为这里是embedding，所以就直接把不相等的拿进去了 
def collate_fn(batch_data):
    """
    自定义batch内各个数据条目的组织方式
    param data:
    return
    """
    
    batch_sex = [item[0][0] for item in batch_data]
    batch_level_id = [item[0][1] for item in batch_data]
       
    batch_shop_id = [item[0][2] for item in batch_data]
    batch_cate = [item[0][3] for item in batch_data]
    batch_floor = [item[0][4] for item in batch_data]
    batch_candidate_shop_id = [item[0][5] for item in batch_data]
    batch_candidate_cate = [item[0][6] for item in batch_data]
    batch_label = [item[1] for item in batch_data]
    
    return [batch_sex, batch_level_id, batch_shop_id, batch_cate, batch_floor, batch_candidate_shop_id, batch_candidate_cate], batch_label
    
    # batch_data.sort(key=lambda x: len(x[3]), reverse=True)
    # data_length = [len(x[3]) for x in batch_data]

if __name__ == "__main__":
    sample_data = pd.read_csv("./use_data/finished_sample_data.csv")
    
    preparedata = PrepareData(sample_data)

    train_data = JdataDataset(preparedata.train_sample_data())
    # 这里数据是正确的
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn, drop_last=True)

    valid_data = JdataDataset(preparedata.valid_sample_data())

    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)

    user_embedding_data = JdataDataset(preparedata.user_embedding_sample_data())

    user_embedding_loader = DataLoader(user_embedding_data, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)

    item_embedding_data = JdataDataset(preparedata.item_embedding_sample_data())
    item_embedding_loader = DataLoader(item_embedding_data, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)