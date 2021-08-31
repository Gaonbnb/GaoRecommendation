import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
import os
import random
from sklearn.metrics import log_loss, roc_auc_score
import log


class BasicModel(nn.Module):
    """基础模型"""
    def __init__(self):
        super(BasicModel, self).__init__()

    def fit(self):
        pass
    
    def evaluate(self):
        pass

    def predict(self):
        pass

    def get_parameter_number(self, model):
        """打印参数数量"""
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def print_state_dict(self):
        """打印模型的 state_dict"""
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())
            
    def print_opt_state_dict(self, optimizer):
        """打印优化器的参数"""
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

    def write_log(self, w):
        """定义log日志"""
        file_name = './log_data/' + datetime.date.today().strftime('%m%d')+"_{}.log".format("deepfm")
        t0 = datetime.datetime.now().strftime('%H:%M:%S')
        info = "{} : {}".format(t0, w)
        print(info)
        with open(file_name, 'a') as f: 
            f.write(info + '\n') 
            
    def __save_model(self):
        """保存模型,私有方法,示例用"""
        # 保存模型, state_dict是序列化的字典，需要先反序列化
        torch.save(self.state_dict(), "DSSM_model.pkl")
        self.load_state_dict(torch.load("DSSM_model.pkl"))
        self.eval() # 固定dropout和归一化层

        # 保存模型
        torch.save(self, "dssm_model.pt")
        model = torch.load("dssm_model.pt")
        model.eval()
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

#######DSSM模型最后的loss永远是31.33，不过是初始化，归一化，截断数据各种操作的结果都是一样的，不知道为什么,加减层数，改变神经元的个数都是不变的
class DSSM(BasicModel):
    """DSSM模型"""
    def __init__(self, feature_size, embedding_dim = 4, hidden_dims = [4, 4, 4], dropout = 0.0):
        """
        注意参数可以自行修改，输出的向量都是64维度
        
        """
        super(DSSM, self).__init__()
        
        # # 初始化参数
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dtype = torch.long
        # self.bias = torch.nn.Parameter(torch.randn(1))
        
        # 首先进行用户侧embedding
        self.embed_sex = nn.Embedding(self.feature_size[0], self.embedding_dim)
        self.embed_level_id = nn.Embedding(self.feature_size[1], self.embedding_dim)
        self.embed_shop_id = nn.Embedding(self.feature_size[2] + 1, self.embedding_dim, padding_idx=0)
        self.embed_cate = nn.Embedding(self.feature_size[3] + 2, self.embedding_dim, padding_idx=0)
        self.embed_floor = nn.Embedding(self.feature_size[4] + 1, self.embedding_dim, padding_idx=0)

        # 进行现在候选物品的embedding
        # self.candidate_shop_id = nn.Embedding(self.feature_size[2], self.embedding_dim)
        # self.candidate_cate = nn.Embedding(self.feature_size[3], self.embedding_dim)

        # 然后输入全连接网络
        self.fc1_user = nn.Linear(self.embedding_dim * 5, self.hidden_dims[0])
        self.fc1_item = nn.Linear(self.embedding_dim * 2, self.hidden_dims[0])
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.fc3 = nn.Linear(self.hidden_dims[1], self.hidden_dims[2])
        self.dropout = nn.Dropout(dropout)

        self.bn = torch.nn.BatchNorm1d(4)
        # self.output = nn.Linear(self.hidden_dims[2], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate):
        """前向传播"""    
        
        output_user = self.user_tower(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor)
        output_item = self.item_tower(candidate_shop_id, candidate_cate)
        # 计算物品和用户之间的相似度
        
        cosine = torch.cosine_similarity(output_user, output_item, dim = 1, eps = 1e-8)
        #cosine = self.sigmoid(cosine)
        # print(cosine) # 加了sigmoid之后模型最后只有0.7311和0.2689两个数字
        
        
        return cosine 

    def predict(self, embedding_loader, get_user_embedding=False, get_item_embedding=False):
        """
        提取得到embedding，user_embedding和item_embedding不可以同时取到，只能一个为True，并且注意输入的loader分别应该为不同的
        """
        if (get_user_embedding and get_item_embedding) or (not get_user_embedding and not get_item_embedding):
            raise Exception("不可以同时得到user_embedding item_embedding, 或者同时不得到user_embedding item_embedding!")

        with torch.no_grad():
            all_user_embedding, all_item_embedding = [], []

            for step, (x, label) in enumerate(embedding_loader):
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x

                if get_user_embedding:
                    cat_fea_sex = torch.tensor([item.item() for item in cat_fea_sex])
                    cat_fea_level_id = torch.tensor([item.item() for item in cat_fea_level_id])
                    cat_fea_sex = cat_fea_sex + 1
                    
                    #cat_fea_sex = cat_fea_sex.cuda()
                    #cat_fea_level_id = cat_fea_level_id.cuda()
                    user_embedding = self.user_tower(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor)
                    user_embedding = user_embedding.numpy().tolist()
                    all_user_embedding.extend(user_embedding)
                else:
                    candidate_shop_id = torch.tensor([item.item() for item in candidate_shop_id])
                    candidate_cate = torch.tensor([item.item() for item in candidate_cate])
                    # candidate_shop_id = candidate_shop_id.cuda()
                    # candidate_cate = candidate_cate.cuda()
                    item_embedding = self.item_tower(candidate_shop_id, candidate_cate)
                    item_embedding = item_embedding.numpy().tolist()
                    all_item_embedding.extend(item_embedding)
            return all_user_embedding if get_user_embedding else all_item_embedding
                
    def user_tower(self, cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor):
        """用户侧"""
        
        # 对非序列数据进行处理
        # cat_fea_sex = cat_fea_sex.cuda()
        # cat_fea_level_id = cat_fea_level_id.cuda()
        # iter_fea_shop_id = iter_fea_shop_id.cuda()
        # iter_fea_cate = iter_fea_cate.cuda()
        # iter_fea_floor = iter_fea_floor.cuda()
        
        embedding_sex = self.embed_sex(cat_fea_sex) # batch_size * embedding_dim
        
        embedding_level_id = self.embed_level_id(cat_fea_level_id) # batch_size * embedding_dim
        # # 对序列数据进行处理
        
        embedding_shop_id = self.embed_shop_id(iter_fea_shop_id).sum(1)
        embedding_cate = self.embed_cate(iter_fea_cate).sum(1)
        embedding_floor = self.embed_floor(iter_fea_floor).sum(1)
        
        # embedding_shop_id = []
        # for shop_id in iter_fea_shop_id:
            
        #     shop_id = shop_id.cuda() 
            
        #     cur_embedding_shop_id = self.embed_shop_id(shop_id) # 序列长度 * embedding_dim
        #     cur_embedding_shop_id = cur_embedding_shop_id.sum(0) # embedding_dim_sum_pooling
        #     embedding_shop_id.append(cur_embedding_shop_id)  # batch_size *  embedding_dim
        
        
        # embedding_shop_id = torch.LongTensor([item.cpu().detach().numpy() for item in embedding_shop_id]).cuda() # batch_size * embedding_dim
        # #embedding_shop_id = torch.LongTensor([item.detach().numpy() for item in embedding_shop_id])
        
        # embedding_cate = []
        # for cate in iter_fea_cate:
        #     cate = cate.cuda() 
            
        #     cur_embedding_cate = self.embed_cate(cate)
        #     cur_embedding_cate = cur_embedding_cate.sum(0)
        #     embedding_cate.append(cur_embedding_cate)
        # embedding_cate = torch.LongTensor([item.cpu().detach().numpy() for item in embedding_cate]).cuda()
        # #embedding_cate = torch.LongTensor([item.detach().numpy() for item in embedding_cate])
        
        # embedding_floor = []
        # for floor in iter_fea_floor:
        #     floor = floor.cuda() 
            
        #     cur_embedding_floor = self.embed_floor(floor)
        #     cur_embedding_floor = cur_embedding_floor.sum(0)
        #     embedding_floor.append(cur_embedding_floor)
            
        # embedding_floor = torch.LongTensor([item.cpu().detach().numpy() for item in embedding_floor]).cuda()
        # #embedding_floor = torch.LongTensor([item.detach().numpy() for item in embedding_floor])
        # # 全都是batch_size * embedding_dim
        
        embedding_cat = torch.cat([embedding_sex, embedding_level_id, embedding_shop_id, embedding_cate, embedding_floor], dim = 1)
        # print(embedding_cat.size()) # batch_size * (embedding_dim * 5)
        # 将concat的部分输入全连接网络
        embedding_cat = self.dropout(torch.tanh(self.fc1_user(embedding_cat)))
        embedding_cat = self.bn(embedding_cat)
        embedding_cat = self.dropout(torch.tanh(self.fc2(embedding_cat)))
        embedding_cat = self.bn(embedding_cat)
        embedding_cat = self.dropout(torch.tanh(self.fc3(embedding_cat)))

        self.output_user = embedding_cat
        # print(embedding_cat) 数量级下降
        return self.output_user


    def item_tower(self, candidate_shop_id, candidate_cate):
        """物品侧"""
        # candidate_shop_id = candidate_shop_id.cuda()
        # candidate_cate = candidate_cate.cuda()
        
        embedding_candidate_shop_id = self.embed_shop_id(candidate_shop_id)
        # print(embedding_candidate_shop_id) 取出来的数据非常贴近0
        
        
        embedding_candidate_cate = self.embed_cate(candidate_cate)
        #  print(embedding_candidate_cate) 到了一个阶段之后就非常的趋近0
        embedding_cat_item = torch.cat([embedding_candidate_shop_id, embedding_candidate_cate], dim = 1)
        
        embedding_cat_item = self.dropout(torch.tanh(self.fc1_item(embedding_cat_item)))
        embedding_cat_item = self.dropout(torch.tanh(self.fc2(embedding_cat_item)))
        embedding_cat_item = self.dropout(torch.tanh(self.fc3(embedding_cat_item)))
        # print(embedding_cat_item) 下降的数量级没有embeddding部分严重，但也是下降了
        self.output_item = embedding_cat_item
        return self.output_item
    
    def evaluate(self, model, valid_loader):
        """模型的验证，在模型的fit时自动被调用"""
        model.eval()
        loss_fct = nn.BCELoss()
        with torch.no_grad():
            valid_labels, valid_preds = [], []
            # all_user_embedding, all_item_embedding = [], []

            for step, (x, label) in enumerate(valid_loader):
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x
                
                
                pred = model(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                pred = pred.view(-1) # batch_size
                label = torch.FloatTensor([item.item() for item in label])
                
                loss = loss_fct(pred, label)

                valid_labels.extend(label.cpu().numpy().tolist())
                valid_preds.extend(pred)

                # print("valid_data: step {:04d} | loss {:.4f}".format(step+1, loss))
                # print("********************************evaluation_metric*****************************")
                
            try:
                cur_auc = roc_auc_score(valid_labels, valid_preds)    
            except UnboundLocalError:
                print("只有一种分类的标签，要进行负采样并且shuffle才行")           
            except:
                print("发生了未知的错误")
            else:
                print(cur_auc)
                return cur_auc
            finally:
                print("this epoch's valid process is finished, if then there are errors, look here")
                
    def fit(self, train_loader, valid_loader, train_epoch = 50, learning_rate = 0.1, weight_decay = 0.000):
        """模型进行训练"""
        best_auc = 0.0
        # 选择设备
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.cuda()
        # self = nn.DataParallel(self) #会把tensor都给拆开了，但是非tensor是不会动的
        self.apply(self.weights_init)
        #loss_fct = nn.BCELoss() # 二分类之前需要加sigmoid，crossentropy是自动softmax多分类
        loss_fct = nn.BCEWithLogitsLoss() # 代替了sigmoid + becloss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # print(self.get_parameter_number(self))
        
        for epoch in range(train_epoch):
            
            model = self.train()
            train_loss_sum = 0.0
            start_time = time.time()
            for step, (x, label) in enumerate(train_loader):
                
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x
                
                pred = model(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                
                ## batch_size * 1
                pred = pred.view(-1) # batch_size
                label = torch.FloatTensor([item.item() for item in label])
                # print(pred.size()) 数量级一样
                # print(label.size())
                # label = label.cuda()
                
                loss = loss_fct(pred, label)
                
                optimizer.zero_grad()
                loss.backward()
                # 查看时候梯度回传
                # for item in self.named_parameters():
                #     if item[0] == 'embed_sex.weight':
                #         h = item[1].register_hook(lambda grad: print(grad))
                # for name, parms in self.named_parameters():
                    
                #     print("-->name:", name, "-->grad_requirs:", parms.requires_grad, "-->weight", torch.mean(parms.data), "-->grad_value:", torch.mean(parms.grad))
                optimizer.step()
                train_loss_sum += loss.cpu().item()
                if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                        print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                            epoch+1, step+1, len(train_loader), train_loss_sum/(step+1), time.time() - start_time))
            
            # cur_auc = self.evaluate(model, valid_loader)
            # if cur_auc > best_auc:
            #     best_auc = cur_auc
            #     os.makedirs("./save_model", exist_ok=True)
            #     torch.save(model.state_dict(), "./save_model/DSSM.pkl")
            #     torch.save(model, "./save_model/DSSM_model.pkl")
            # print(best_auc)
######模型倒是可以loss下降，但是最后的业务指标非常的差，不知道原因
class YouTubeDNN(BasicModel):
    """YouTubeDNN模型"""
    def __init__(self, feature_size, softmax_dims, embedding_dim = 4, hidden_dims = [8, 4, 4], dropout = 0.0):
        """
        输出softmax的结果
        """
        super(YouTubeDNN, self).__init__()
        self.softmax_dims = softmax_dims
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dtype = torch.long
        #self.bias = torch.nn.Parameter(torch.randn(1))

        # 用户信息embedding
        self.embed_sex = nn.Embedding(self.feature_size[0], self.embedding_dim)
        self.embed_level_id = nn.Embedding(self.feature_size[1], self.embedding_dim)
        
        # 进行过往经历的embedding
        self.embed_shop_id = nn.Embedding(self.feature_size[2]+1, self.embedding_dim)
        self.embed_cate = nn.Embedding(self.feature_size[3]+2, self.embedding_dim)
        self.embed_floor = nn.Embedding(self.feature_size[4]+1, self.embedding_dim)

        # 进行现在候选物品的embedding
        # self.candidate_shop_id = nn.Embedding(self.feature_size[2], self.embedding_dim)
        # self.candidate_cate = nn.Embedding(self.feature_size[3], self.embedding_dim)

        self.relu_layer = nn.Sequential(
            nn.Linear(self.embedding_dim * 7, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[2]),
            nn.ReLU()      
        )
        self.output = nn.Linear(hidden_dims[2], self.softmax_dims)
        
    def forward(self, cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate):
        """前向传播"""
        # 用户数据的embedding
        cat_fea_sex = cat_fea_sex.cuda()
        cat_fea_level_id = cat_fea_level_id.cuda()
        candidate_cate = candidate_cate.cuda()
        candidate_shop_id = candidate_shop_id.cuda()
        iter_fea_shop_id = iter_fea_shop_id.cuda()
        iter_fea_cate = iter_fea_cate.cuda()
        iter_fea_floor = iter_fea_floor.cuda()
        embedding_sex = self.embed_sex(cat_fea_sex) # batch_size * embedding_dim
        embedding_level_id = self.embed_level_id(cat_fea_level_id) # batch_size * embedding_dim

        # # 过往经历的embedding
        embedding_shop_id = self.embed_shop_id(iter_fea_shop_id).sum(1)
        embedding_cate = self.embed_cate(iter_fea_cate).sum(1)
        embedding_floor = self.embed_floor(iter_fea_floor).sum(1)
        # embedding_shop_id = []
        # for shop_id in iter_fea_shop_id:
        #     # shop_id = shop_id.cuda()           
        #     cur_embedding_shop_id = self.embed_shop_id(shop_id) # 序列长度 * embedding_dim
        #     cur_embedding_shop_id = cur_embedding_shop_id.mean(0) # embedding_dim_sum_pooling
        #     embedding_shop_id.append(cur_embedding_shop_id)  # batch_size *  embedding_dim
        # # embedding_shop_id = torch.LongTensor([item.cpu().detach().numpy() for item in embedding_shop_id]).cuda() # batch_size * embedding_dim
        # embedding_shop_id = torch.LongTensor([item.detach().numpy() for item in embedding_shop_id])
        # embedding_cate = []
        # for cate in iter_fea_cate:
        #     #cate = cate.cuda()
        #     cur_embedding_cate = self.embed_cate(cate)
        #     cur_embedding_cate = cur_embedding_cate.mean(0)
        #     embedding_cate.append(cur_embedding_cate)
        # # embedding_cate = torch.LongTensor([item.cpu().detach().numpy() for item in embedding_cate]).cuda()
        # embedding_cate = torch.LongTensor([item.detach().numpy() for item in embedding_cate])
        # embedding_floor = []
        # for floor in iter_fea_floor:
        #     #floor = floor.cuda()    
        #     cur_embedding_floor = self.embed_floor(floor)
        #     cur_embedding_floor = cur_embedding_floor.mean(0)
        #     embedding_floor.append(cur_embedding_floor)
        # #embedding_floor = torch.LongTensor([item.cpu().detach().numpy() for item in embedding_floor]).cuda()
        # embedding_floor = torch.LongTensor([item.detach().numpy() for item in embedding_floor])

        # 候选物品的embedding
        embedding_candidate_shop_id = self.embed_shop_id(candidate_shop_id)
        embedding_candidate_cate = self.embed_cate(candidate_cate)

        embedding_cat = torch.cat([embedding_sex, embedding_level_id, embedding_shop_id, embedding_cate, embedding_floor, embedding_candidate_shop_id, embedding_candidate_cate], dim = 1) 
        
        x = self.relu_layer(embedding_cat)
        # 将之前的多分类问题利用负采样变成二分类问题
        
        score = self.output(x)
        # 外层已经有crossentropy代替logsoftmax + nilloss
        #score = F.softmax(x, dim=1)
                
        return score, x


    def fit(self, train_loader, valid_loader, train_epoch = 1, learning_rate = 0.1, weight_decay = 0.00):
        """Youtube dnn的训练，这里是softmax训练"""
        # youtube dnn配套
        # 选择设备
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self = nn.DataParallel(self)
        self.cuda()
        loss_entroy = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.005, weight_decay=0.001)
        #self.print_opt_state_dict(optimizer)
    
        # print(self.get_parameter_number(self))

        for epoch in range(train_epoch):
            model = self.train()
            # model.cuda()
            train_loss_sum = 0.0
            start_time = time.time()
            for step, (x, label) in enumerate(train_loader):
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x
            
                #cat_fea_sex = cat_fea_sex.cuda()
                #cat_fea_level_id = cat_fea_level_id.cuda()
                # candidate_shop_id = candidate_shop_id.cuda()
                # candidate_cate = candidate_cate.cuda()

                pred,x = model(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                
                ## batch_size * 1
                label = torch.FloatTensor([item.item() for item in label])
                candidate_shop_id = candidate_shop_id.cuda()
                # print(candidate_shop_id.size())
                # print(pred.size())
                loss = loss_entroy(pred, candidate_shop_id)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.cpu().item()
                if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                        print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                            epoch+1, step+1, len(train_loader), train_loss_sum/(step+1), time.time() - start_time))
            
            # self.evaluate(model, valid_loader)


    def evaluate(self, model, valid_loader):
        model.eval()
        loss_entroy = nn.CrossEntropyLoss()
        valid_loss_sum = 0.0
        with torch.no_grad():
            valid_labels, valid_preds = [], []

            for step, (x, label) in enumerate(valid_loader):
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x
                
                
                pred = model(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                
                label = torch.LongTensor([item.item() for item in label])
                candidate_shop_id = candidate_shop_id.cuda()
                loss = loss_entroy(pred, candidate_shop_id)

                valid_labels.extend(label.cpu().numpy().tolist())
                valid_preds.extend(pred)
                valid_loss_sum += loss.cpu().item()

                if (step + 1) % 50 == 0 or (step + 1) == len(valid_loader):
                        print("valid_data: Step {:04d} / {} | Loss {:.4f}".format(
                            step+1, len(valid_loader), valid_loss_sum/(step+1)))
      
    def predict(self, embedding_loader, get_user_embedding=False, get_item_embedding=False):
        """提取embedding"""

        if (get_user_embedding and get_item_embedding) or (not get_user_embedding and not get_item_embedding):
            raise Exception("不可以同时得到user_embedding item_embedding, 或者同时不得到user_embedding item_embedding!")
        if get_item_embedding:
            # output表
            return self.output.weight.data.cpu()
        else:
            with torch.no_grad():
                all_user_embedding = []

                for step, (x, label) in enumerate(embedding_loader):
                    cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x   
                    
                    #cat_fea_sex = cat_fea_sex.cuda()
                    #cat_fea_level_id = cat_fea_level_id.cuda()
                    # candidate_shop_id = candidate_shop_id.cuda()
                    # candidate_cate = candidate_cate.cuda()
                    pred, x = self(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                    all_user_embedding.extend(x.cpu().numpy().tolist())
                return torch.Tensor(all_user_embedding)

class Routing(nn.Module):
    """Routing层，胶囊网络的实现"""
    def __init__(self, max_len, input_units, output_units, iteration=1, max_k=3):
        """
        max_k：输出向量的个数
        input_units：输入向量的维度
        output_units:输出向量的维度
        iteration:胶囊网络的循环次数
        max_len：将所有的序列进行截断处理
        S_matrix:双线性映射矩阵
        """
        super(Routing, self).__init__()
        self.iteration = iteration
        self.max_k = max_k
        self.max_len = max_len
        self.input_units = input_units
        self.output_units = output_units
        
        # 双先行映射矩阵
        self.S_matrix = nn.init.normal_(torch.empty(self.input_units, self.output_units), mean=0, std=1)
        
    def forward(self, low_capsule, seq_len):
        """前向传播，B_matrix是公式中间的b矩阵，但是和序列有关系，所以没有写到初始化中"""
        # 最后输出的高维胶囊
        global high_capsule
        B, _, embed_size = low_capsule.size() # batch_size * seq_len * embed_size
        self.max_len = torch.max(seq_len).item()
        self.B_matrix = nn.init.normal_(torch.empty(1, self.max_k, self.max_len), mean=0, std=1)
        self.B_matrix.requires_grad = False
        
        assert torch.max(seq_len).item() <= self.max_len # 补齐之后的序列长度要小于max_len，要不就最长的补了之后还是不够长
        seq_len_tile = seq_len.repeat(1, self.max_k) # 对应维度乘倍数，然后多了的用0补前面，且repeat参数的个数不能少于张亮的维度的个数，这个就是在原来的维度重复max-k， 那可能就是batch_size * (self._max_k)
        
        for i in range(self.iteration): # 迭代次数r
            #mask = sequence_mask(seq_len_tile, self.max_len) # 直接找最长的就可以了
            mask = self.sequence_mask(seq_len_tile)
            
            mask = torch.reshape(mask, [B, self.max_k, self.max_len])
            ## mask: B * max_k * max_len
            ## W: B * max_k * max_len
            ## low_capsule_new: B * max_len * hidden_units
            pad = torch.ones_like(mask, dtype=torch.float32) * (-2 ** 16 + 1)
            B_tile = self.B_matrix.repeat(B, 1, 1) # batch_size * max_k * max_len
            
            B_mask = torch.where(mask, B_tile, pad) # mask 是规则，true为B_tile, false为pad, batchsize * max_k * max_len, 矩阵表示为batch ， bj bi，后面的用超大负数代替
            
            #B_mask = B_mask.transpose(1, 2) 网上说原文写错了
            # wij = softmax(bij)
            W = nn.functional.softmax(B_mask, dim=-1) # 最后一个维度做softmax
            
            # Sei
            # zj = sum(wijSei)
            # uj = squash(zj)
            # print("wsize", W.size())
            # print("low_capsule",low_capsule.size())
            # print("smatrix", self.S_matrix.size())
            #low_capsule_new = torch.einsum("ijk, lo->ilk", (low_capsule, self.S_matrix)) # 爱因斯坦求
            low_capsule_new = torch.matmul(low_capsule, self.S_matrix)
            # print("wsize", W.size())
            # print("low_bapsule_new",low_capsule_new.size())
            high_capsule_tmp = torch.bmm(W, low_capsule_new)   # batchsize * klen * outputunit
            high_capsule = self.squash(high_capsule_tmp) # batch_size * klen * outputunit
            
            # bij + ujSei
            # batch * outunit * seqlen
            # 1 * ken * seqlen
            B_delta = torch.sum(
                torch.matmul(high_capsule, torch.transpose(low_capsule_new, dim0=1, dim1=2)),
                dim=0, keepdim=True
            )
            # print(B_delta.size()) 
            # 1 * k_len * seqlen
            self.B_matrix += B_delta
        
        # batch_size * max_k * output_units 输出k个兴趣胶囊，每个胶囊都有output_units维度
        return high_capsule

    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        """将序列数据中存在的变为true，不存在的变成false"""
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1)
        matrix = torch.unsqueeze(lengths, dim=-1) # batch_size * max_k * 1
        mask = row_vector < matrix # 看如何截断,有一种广播的效果，就是最后的1维广播成maxlen的大小，并且看存的值和row_vector进行比较的bool值
        mask.type(dtype)
        return mask

    def squash(self, inputs):
        """计算公式中的squash值"""
        vec_squared_norm = torch.sum(torch.square(inputs), dim = 1, keepdim=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-9)
        vec_squashed = scalar_factor * inputs # element-wise
        return vec_squashed

class MIND(BasicModel):
    """MIND模型主网络，中间包括了胶囊网络"""
    def __init__(self, feature_size, embedding_dim=4, max_len=4, input_units=4, output_units=8, iteration=1, max_k=3, p=2):
        """
        feature_size:各个输入特征的分类数
        embedding_dim：开始的embedding层的维度
        p：power的大小，通过这个调节不同的attention权重是平滑还是尖锐
        其余的参数看胶囊网络的部分，在这里进行调节
        """
        super(MIND, self).__init__()
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.max_k = max_k
        self.p = p

        # 用户信息embedding
        self.embed_sex = nn.Embedding(self.feature_size[0], self.embedding_dim)
        self.embed_level_id = nn.Embedding(self.feature_size[1], self.embedding_dim)
        
        # 进行过往经历的embedding
        self.embed_shop_id = nn.Embedding(self.feature_size[2]+1, self.embedding_dim, padding_idx=0)
        self.embed_cate = nn.Embedding(self.feature_size[3]+1, self.embedding_dim, padding_idx=0)
        self.embed_floor = nn.Embedding(self.feature_size[4]+1, self.embedding_dim,padding_idx=0)

        self.routing = Routing(max_len, input_units, output_units, iteration, max_k)
        self.label_linear = nn.Linear(self.embedding_dim, output_units)
        self.user_linear = nn.Linear(self.embedding_dim, output_units)
        self.capsule_linuear = nn.Linear(self.embedding_dim, output_units)
        # 两个relu层
        self.relu_linear_1 = nn.Linear(output_units, 128)
        self.relu_linear_2 = nn.Linear(128, output_units)
        self.output_units = output_units
        self.input_units = input_units

    def forward(self, cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate):
        # for cur_shop_id, cur_cate, cur_floor in zip(iter_fea_shop_id, iter_fea_cate, iter_fea_floor):
            # print(self.embed_shop_id(cur_shop_id).size()) # sep_len * embed_dim
            # print(self.embed_cate(cur_cate).size())
            # print(self.embed_floor(cur_floor).size())
            # print(sum([self.embed_shop_id(cur_shop_id), self.embed_cate(cur_cate), self.embed_floor(cur_floor)])) # saq_len * embed_dim
            # 只能利用原生sum进行对应元素相加
        # batch_size * seqlen * embed_dim
        # seq_embed = [sum([self.embed_shop_id(cur_shop_id), self.embed_cate(cur_cate), self.embed_floor(cur_floor)]) for cur_shop_id, cur_cate, cur_floor in zip(iter_fea_shop_id, iter_fea_cate, iter_fea_floor)]
        seq_lens = []
        for cur_shop_id in iter_fea_shop_id:
            seq_lens.append(cur_shop_id.shape[0])
        seq_lens = torch.tensor(seq_lens).reshape(-1, 1)
      
        # pad_iter_fea_shop_id = torch.nn.utils.rnn.pad_sequence(iter_fea_shop_id, padding_value=self.feature_size[2]).t() # batch_size * maxseq
        # pad_iter_fea_cate = torch.nn.utils.rnn.pad_sequence(iter_fea_cate, padding_value=self.feature_size[3]).t()
        # pad_iter_fea_floor = torch.nn.utils.rnn.pad_sequence(iter_fea_floor, padding_value=self.feature_size[4]).t()
        seq_embed_pad_iter_fea_shop_id = self.embed_shop_id(iter_fea_shop_id)
        seq_embed_pad_iter_fea_cate = self.embed_cate(iter_fea_cate)
        seq_embed_pad_iter_fea_floor = self.embed_floor(iter_fea_floor)

        B = seq_embed_pad_iter_fea_shop_id.shape[0] # batchsize
        # 序列的embed
        seq_embed = sum([seq_embed_pad_iter_fea_shop_id, seq_embed_pad_iter_fea_floor, seq_embed_pad_iter_fea_cate]) # batch_size * max_seqlen * embed_dim
        # 不存在的最后的embedding就是0了
        
        # other feature
        embedding_sex = self.embed_sex(cat_fea_sex) # batch_size * embedding_dim
        embedding_level_id = self.embed_level_id(cat_fea_level_id) # batch_size * embedding_dim
        # user_embedc
        user_other_feature = torch.cat([torch.unsqueeze(embedding_sex, 1), torch.unsqueeze(embedding_level_id, 1)], dim=1) # 原始embed并列concat
        user_other_feature = torch.sum(user_other_feature, dim=1,keepdim=True)

        # candidate feature
        # 候选物品的embedding
        embedding_candidate_shop_id = self.embed_shop_id(candidate_shop_id)
        embedding_candidate_cate = self.embed_cate(candidate_cate)
        # label embed
        candidate_feature = sum([embedding_candidate_cate, embedding_candidate_shop_id]).unsqueeze(1)

        user_ids_embedding = self.user_linear(user_other_feature) # batchsize * 1 * output_units
        self.label_embedding = self.label_linear(candidate_feature) # batchsize * 1 * output_units

        capsule_output = self.routing(seq_embed, seq_lens) # seq_lens没有定义
        # capsule_output = self.capsule_linuear(capsule_output) # batch_size * max_k * output_units
        
        # 两个相加相同纬度的只要小的是1维就可以通过广播来进行相加
        capsule_output_user_added = capsule_output + user_ids_embedding
        capsule_output_user_added = self.relu_linear_1(capsule_output_user_added)
        capsule_output_user_added = F.relu_(capsule_output_user_added) # batchsize * max_k * output_units
        capsule_output_user_added = self.relu_linear_2(capsule_output_user_added)
        self.capsule_output_user_added = F.relu_(capsule_output_user_added)

        
        pos_label_embedding = self.label_embedding.reshape(B, -1, self.output_units)
        # print(pos_label_embedding.size())
        attention_weight = torch.mul(capsule_output_user_added, pos_label_embedding.repeat(1, self.max_k, 1)) # batchsize * k_len * outputunit
        # print(attention_weight.size())
        attention_weight = torch.pow(attention_weight, self.p)# batchsize * k_len * outputunit
        # print(attention_weight.size())
        attention_weight = torch.sum(attention_weight, dim=-1, keepdim=False) # batchsize * k_len
        # print(attention_weight.size())
        attention_weight = nn.functional.softmax(attention_weight, dim=1) # batch_size * k_len
        attention_output = capsule_output_user_added * attention_weight.unsqueeze(dim=-1)  # batch_size * k_len * output_nuit

        attention_output = attention_output.view(attention_weight.size()[0], -1) # batch_size * (len_k * output_unit)

        # 生成负样本
        neg_sample_id = self.neg_sample(candidate_shop_id)
        
        # 因为不定长度所以在这个创建
        self.final_linear = nn.Linear(attention_output.size()[1], 146)
        
        attention_output = self.final_linear(attention_output)
        # print(attention_output.size()) # batchsize * 146
        return attention_output

    def neg_sample(self, candidate_shop_id):
        neg_sample_id = list()
        for item in candidate_shop_id.numpy().tolist():
            cur_sample_id = list()
            # 对应15个负样本
            while len(cur_sample_id) < 15:
                non_match = [item]
                neg_sample_data = random.randint(0, len(shop_id_dict))
                if neg_sample_data != item:
                    cur_sample_id.append(neg_sample_data)
            neg_sample_id.append(cur_sample_id)
        return neg_sample_id
    def fit(self, train_loader, valid_loader):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        optimizer = optim.Adam(self.parameters(), lr=0.005, weight_decay=0.001)
        # self.print_opt_state_dict(optimizer)
    
        print(self.get_parameter_number(self))
        for epoch in range(1):
            model = self.train()
            # model.cuda()
            train_loss_sum = 0.0
            start_time = time.time()
            for step, (x, label) in enumerate(train_loader):
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x
                
                    
                cat_fea_sex = torch.tensor([item.item() for item in cat_fea_sex])
                
                cat_fea_level_id = torch.tensor([item.item() for item in cat_fea_level_id])
                
                cat_fea_sex = cat_fea_sex + 1

                candidate_shop_id = torch.tensor([item.item() for item in candidate_shop_id])

                candidate_cate = torch.tensor([item.item() for item in candidate_cate])


                #cat_fea_sex = cat_fea_sex.cuda()
                #cat_fea_level_id = cat_fea_level_id.cuda()
                # candidate_shop_id = candidate_shop_id.cuda()
                # candidate_cate = candidate_cate.cuda()

                pred = model(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                         
                ## batch_size * 1
       
                # pred = pred.squeeze(dim=-1).float()
                # label = label.float()
                # loss = F.binary_cross_entropy_with_logits(pred,label)
                
                loss = nn.CrossEntropyLoss()(pred, candidate_shop_id.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.cpu().item()
                if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                        print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                            epoch+1, step+1, len(train_loader), train_loss_sum/(step+1), time.time() - start_time))
            
            #self.evaluate(model, valid_loader)


    def evaluate(self, model, valid_loader):

        model.eval()
        
        valid_loss_sum = 0.0
        with torch.no_grad():
            valid_labels, valid_preds = [], []

            for step, (x, label) in enumerate(valid_loader):
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x
                
                cat_fea_sex = torch.tensor([item.item() for item in cat_fea_sex])
                
                cat_fea_level_id = torch.tensor([item.item() for item in cat_fea_level_id])
                
                cat_fea_sex = cat_fea_sex + 1

                candidate_shop_id = torch.tensor([item.item() for item in candidate_shop_id])

                candidate_cate = torch.tensor([item.item() for item in candidate_cate])

                #cat_fea_sex = cat_fea_sex.cuda()
                #cat_fea_level_id = cat_fea_level_id.cuda()
                
                pred = model(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                
                label = torch.LongTensor([item.item() for item in label])
                pred = pred.squeeze(dim=-1).float()
                label = label.float()
                loss = F.binary_cross_entropy_with_logits(pred,label)

                valid_labels.extend(label.cpu().numpy().tolist())
                valid_preds.extend(pred)
                valid_loss_sum += loss.item()
                if (step + 1) % 50 == 0 or (step + 1) == len(valid_loader):
                        print("valid_data: Step {:04d} / {} | Loss {:.4f}".format(
                            step+1, len(valid_loader), valid_loss_sum/(step+1)))

    def predict(self, embedding_loader, get_user_embedding=False, get_item_embedding=False):

        if (get_user_embedding and get_item_embedding) or (not get_user_embedding and not get_item_embedding):
            raise Exception("不可以同时得到user_embedding item_embedding, 或者同时不得到user_embedding item_embedding!")
        if get_item_embedding:
            # output表
            with torch.no_grad():
                all_item_embedding = []
                for step, (x, label) in enumerate(embedding_loader):
                    cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x   
                    cat_fea_sex = torch.tensor([item.item() for item in cat_fea_sex])                
                    cat_fea_level_id = torch.tensor([item.item() for item in cat_fea_level_id])
                    cat_fea_sex = cat_fea_sex + 1
                    candidate_shop_id = torch.tensor([item.item() for item in candidate_shop_id])
                    candidate_cate = torch.tensor([item.item() for item in candidate_cate])
                    pred = self(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                    all_item_embedding.extend(self.label_embedding.numpy().tolist())
                return torch.Tensor(all_item_embedding)

        else:
            with torch.no_grad():
                all_user_embedding = []

                for step, (x, label) in enumerate(embedding_loader):
                    cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x   
                    cat_fea_sex = torch.tensor([item.item() for item in cat_fea_sex])                
                    cat_fea_level_id = torch.tensor([item.item() for item in cat_fea_level_id])
                    cat_fea_sex = cat_fea_sex + 1
                    candidate_shop_id = torch.tensor([item.item() for item in candidate_shop_id])
                    candidate_cate = torch.tensor([item.item() for item in candidate_cate])
                    #cat_fea_sex = cat_fea_sex.cuda()
                    #cat_fea_level_id = cat_fea_level_id.cuda()
                    # candidate_shop_id = candidate_shop_id.cuda()
                    # candidate_cate = candidate_cate.cuda()
                    pred = self(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                    all_user_embedding.extend(self.capsule_output_user_added.numpy().tolist())
                return torch.Tensor(all_user_embedding)                  
 
class GRU4REC(BasicModel):
    """简单的gru模型提取序列信息"""
    def __init__(self, feature_size, embedding_dim=4, hidden_size=8, num_layers=2, dropout_hidden=.5):
        super(GRU4REC, self).__init__()
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        # embedding
        self.embed_sex = nn.Embedding(self.feature_size[0], self.embedding_dim)
        self.embed_level_id = nn.Embedding(self.feature_size[1], self.embedding_dim)
        self.embed_shop_id = nn.Embedding(self.feature_size[2]+1, self.embedding_dim, padding_idx=self.feature_size[2])
        self.embed_cate = nn.Embedding(self.feature_size[3]+1, self.embedding_dim, padding_idx=self.feature_size[3])
        self.embed_floor = nn.Embedding(self.feature_size[4]+1, self.embedding_dim, padding_idx=self.feature_size[4])
        # 多层gru
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
        self.linear = nn.Linear(13, 146)

    def forward(self, cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate):
        """这里用了padding，专门用于rnn相关模型"""
        # 首先进行padding
        iter_fea_cate, iter_fea_cate_len = self.pad_sequence(iter_fea_cate, self.feature_size[3]) # batchsize * seq
        iter_fea_floor, iter_fea_floor_len = self.pad_sequence(iter_fea_floor, self.feature_size[4])
        iter_fea_shop_id, iter_fea_shop_id_len = self.pad_sequence(iter_fea_shop_id, self.feature_size[2])
        
        # 后面没有的用embedding为length补0 embedding
        iter_fea_shop_id = self.embed_shop_id(iter_fea_shop_id) # batchsize * seq * embedding_dim
        iter_fea_cate = self.embed_cate(iter_fea_cate)
        iter_fea_floor = self.embed_floor(iter_fea_floor)
        

        # pack将之前的填充的部分给压缩,都是纵向的一列列堆叠数量和排列方式
        iter_fea_shop_id = nn.utils.rnn.pack_padded_sequence(iter_fea_shop_id, iter_fea_shop_id_len, batch_first=True)
        iter_fea_cate = nn.utils.rnn.pack_padded_sequence(iter_fea_cate, iter_fea_cate_len, batch_first=True)
        iter_fea_floor = nn.utils.rnn.pack_padded_sequence(iter_fea_floor, iter_fea_floor_len, batch_first=True)
        
        output, hidden = self.gru(iter_fea_shop_id)
        
        # 还是没有的就进行补0 embedding操作,这句话就是将每个上面的拿出来的
        output, output_len = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # 两个hidden做sum pooling
        sum_hidden = torch.sum(hidden, dim=0)
        
        cat_fea_sex_onehot = torch.nn.functional.one_hot(cat_fea_sex, num_classes=4)
        cat_fea_level_id_onehot = torch.nn.functional.one_hot(cat_fea_level_id, num_classes=1)
        concat_output = torch.cat([sum_hidden, cat_fea_sex_onehot, cat_fea_level_id_onehot], dim=1)
        
        # print(concat_output.size()) # batchsize * concat_result
        output  = self.linear(concat_output)
        return output

    def pad_sequence(self, iter_feature, padding_value):
        iter_feature.sort(key=lambda data: len(data.numpy().tolist()), reverse=True)
        data_length = [len(data.numpy().tolist()) for data in iter_feature]
        iter_feature = nn.utils.rnn.pad_sequence(iter_feature, batch_first=True, padding_value=padding_value)
        return iter_feature, data_length
    def fit(self, train_loader, valid_loader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        loss_entroy = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.005)
        self.print_opt_state_dict(optimizer)
    
        print(self.get_parameter_number(self))
        for epoch in range(10):
            model = self.train()
            # model.cuda()
            start_time = time.time()
            train_loss_sum = 0.0
            for step, (x, label) in enumerate(train_loader):
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x
                    
                cat_fea_sex = torch.tensor([item.item() for item in cat_fea_sex])
                
                cat_fea_level_id = torch.tensor([item.item() for item in cat_fea_level_id])
                
                cat_fea_sex = cat_fea_sex + 1

                candidate_shop_id = torch.tensor([item.item() for item in candidate_shop_id])

                candidate_cate = torch.tensor([item.item() for item in candidate_cate])
                
                #cat_fea_sex = cat_fea_sex.cuda()
                #cat_fea_level_id = cat_fea_level_id.cuda()
                # candidate_shop_id = candidate_shop_id.cuda()
                # candidate_cate = candidate_cate.cuda()

                pred = model(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                label = torch.FloatTensor([item.item() for item in label])
                
                loss = loss_entroy(pred, candidate_shop_id)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.cpu().item()
                if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                        print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                            epoch+1, step+1, len(train_loader), train_loss_sum/(step+1), time.time() - start_time))
            self.evaluate(model, valid_loader)

    def evaluate(self, model, valid_loader):
        model.eval()
        loss_entroy = nn.CrossEntropyLoss()
        valid_loss_sum = 0.0
        with torch.no_grad():
            valid_labels, valid_preds = [], []

            for step, (x, label) in enumerate(valid_loader):
                cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate = x
                
                cat_fea_sex = torch.tensor([item.item() for item in cat_fea_sex])
                
                cat_fea_level_id = torch.tensor([item.item() for item in cat_fea_level_id])
                
                cat_fea_sex = cat_fea_sex + 1

                candidate_shop_id = torch.tensor([item.item() for item in candidate_shop_id])

                candidate_cate = torch.tensor([item.item() for item in candidate_cate])

                #cat_fea_sex = cat_fea_sex.cuda()
                #cat_fea_level_id = cat_fea_level_id.cuda()
                
                pred = model(cat_fea_sex, cat_fea_level_id, iter_fea_shop_id, iter_fea_cate, iter_fea_floor, candidate_shop_id, candidate_cate)
                
                label = torch.LongTensor([item.item() for item in label])
                
                loss = loss_entroy(pred, label)

                valid_labels.extend(label.cpu().numpy().tolist())
                valid_preds.extend(pred)
                valid_loss_sum += loss.cpu().item()
                if (step + 1) % 50 == 0 or (step + 1) == len(valid_loader):
                        print("valid_data: Step {:04d} / {} | Loss {:.4f}".format(
                            step+1, len(valid_loader), valid_loss_sum/(step+1)))
            


if __name__ == "__main__":
    # 前两行是模型的训练
    model = DSSM([4, 1, len(shop_id_dict), len(cate_dict), len(floor_dict)])
    model.fit(train_epoch=2)
    # 以下是从存储的最优模型中读取的模型
    model = DSSM([4, 1, len(shop_id_dict), len(cate_dict), len(floor_dict)])
    # load_state_dict没有返回值，要直接接收
    model.load_state_dict(torch.load("./save_model/DSSM.pkl"))
    #model = torch.load("./save_model/DSSM.pkl")
    print(model)

    user_embedding = torch.Tensor(model.predict(valid_loader, get_user_embedding=True))
    item_embedding = torch.Tensor(model.predict(valid_loader, get_item_embedding=True))

    # youtube dnn
    model = YouTubeDNN([4, 1, len(shop_id_dict), len(cate_dict), len(floor_dict)])
    model.fit()
    userembedding = model.predict(valid_loader, get_user_embedding=True)
    itemembedding = model.predict(valid_loader, get_item_embedding=True)
