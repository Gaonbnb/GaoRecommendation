from load_data import PrepareData, JdataDataset, collate_fn
from criteria import recall_embedding, hit_rate
from model import DSSM, YouTubeDNN, MIND, GRU4REC
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from process_data import load_dict
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["NCCL_DEBUG"] = "INFO"

if __name__ == "__main__":
    # 首先执行
    # python process_data.py
    sample_data = pd.read_csv("./use_data/finished_sample_data.csv")
    sample_data = sample_data.dropna(axis=0, how="any")


    # 准备数据
    preparedata = PrepareData(sample_data)
    # 训练数据
    train_data = JdataDataset(preparedata.user_embedding_sample_data())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn, drop_last=True)
    # 验证数据
    
    valid_data = JdataDataset(preparedata.valid_softmax_loss_sample_data())
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    # user线上embedding数据
    user_embedding_data = JdataDataset(preparedata.user_embedding_sample_data())
    user_embedding_loader = DataLoader(user_embedding_data, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    # item线上embedding数据from torch.utils.data import Dataset, DataLoaderfrom torch.utils.data import Dataset, DataLoader
    item_embedding_data = JdataDataset(preparedata.item_embedding_sample_data())
    item_embedding_loader = DataLoader(item_embedding_data, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)

    shop_id_dict, cate_dict, floor_dict = load_dict("shop_id_dict"), load_dict("cate_dict"), load_dict("floor_dict")
    
    model = MIND([5, 1, len(shop_id_dict), len(cate_dict), len(floor_dict)], softmax_dims=147)
    model.fit(train_loader, valid_loader, train_epoch=25)
    userembedding = model.predict(user_embedding_loader, get_user_embedding=True)
    itemembedding = model.predict(item_embedding_loader, get_item_embedding=True)
    
    match_result = recall_embedding(itemembedding, userembedding)
    real_result = sample_data.groupby(by="userid").head(1).last_click
    
    hit_rate_score = hit_rate(match_result, real_result)
    print(hit_rate_score)
    
    
