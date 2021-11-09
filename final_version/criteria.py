import faiss
import numpy as np

def recall_embedding(xb, xq):
    xb, xq = np.array(xb).astype("float32"), np.array(xq).astype("float32")
    index = faiss.IndexFlatL2(8)
    print(index.is_trained) # true
    print(xb.shape)
    index.add(xb) # item_length * dim
    # print(index.ntotal) # item_length
    k = 5
    distance, itemindex = index.search(xq, k) # return user_length * k_length
    # print(distance) #  每个user对应的最近距离
    # print(itemindex)
    return itemindex

def hit_rate(match_result, real_result):
    all_number = len(real_result)
    hit_number = 0
    for i, item in enumerate(real_result):     
        if item in match_result[i]:
            hit_number += 1
    return hit_number / all_number

# 检测召回结果函数，越靠近0就说明召回的结果中结果越差，越靠近len(cur_recall_result)说明结果比较好
def ARHR_evaluation(recall_result, real_result):
    """
    recall_result: id_list
    real_result: last_click

    """
    
    average_reciprocal_hit_rank_denominator = average_reciprocal_hit_rank_molecule = 0
    for cur_recall_result, cur_real_result in zip(recall_result, real_result):

        average_reciprocal_hit_rank_denominator += 1
        if cur_real_result not in cur_recall_result: continue
        
        average_reciprocal_hit_rank_molecule += len(cur_recall_result) - cur_recall_result.index(cur_real_result)
        
    return average_reciprocal_hit_rank_molecule / average_reciprocal_hit_rank_denominator

if __name__ == "__main__":
    pass