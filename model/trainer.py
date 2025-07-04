import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.UniCDR import UniCDR
from torch.nn import TransformerDecoder
from utils import torch_utils
from Deno.A import Model, vgae_encoder, vgae_decoder, DenoisingNet, vgae
import numpy as np
import pdb
import math
import torch as t

from utils.Utils import innerProduct, pairPredict, calcRegLoss


class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch=None):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class CrossTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if self.opt["model"] == "UniCDR":
            self.model = UniCDR(opt).to(opt["device"])
            # self.source_GNN = VBGE(opt)
            # self.target_GNN = VBGE(opt)
            self.model_x = Model(opt, self.opt["source_user_num"], self.opt["source_item_num"])
            self.model_y = Model(opt, self.opt["target_user_num"], self.opt["target_item_num"])
        else :
            print("please input right model name!")
            exit(0)

        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'], opt["weight_decay"])
        self.epoch_rec_loss = []

        self.source_sport_pos_embs = []
        self.source_sport_neg_embs = []
        self.target_cloth_pos_embs = []
        self.target_cloth_neg_embs = []
        self.source_ib_embs = []
        self.target_ib_embs = []

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            item = inputs[1]
            context_item = inputs[2]
            context_score = inputs[3]
            global_item = inputs[4]
            global_score = inputs[5]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            item = inputs[1]
            context_item = inputs[2]
            context_score = inputs[3]
            global_item = inputs[4]
            global_score = inputs[5]
        return user, item, context_item, context_score, global_item, global_score

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            pos_item = inputs[1]
            neg_item = inputs[2]
            context_item = inputs[3]
            context_score = inputs[4]
            global_item = inputs[5]
            global_score = inputs[6]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            pos_item = inputs[1]
            neg_item = inputs[2]
            context_item = inputs[3]
            context_score = inputs[4]
            global_item = inputs[5]
            global_score = inputs[6]
        return user, pos_item, neg_item, context_item, context_score, global_item, global_score

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1).to(memory.device)  # 确保 index 和 memory 在同一设备
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def generator_generate(self, generator, adj):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj_1 = deepcopy(adj)
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(adj, idxs, adj_1)

        return view

    # def combined_w(self,domain_id , batch):
    #     user, pos_item, n_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)
    #     user_feature, pos_dis, neg_dis = self.model.forward_user(domain_id, user, context_item, context_score,
    #                                                              global_item, global_score)
    #
    #     epsilon = 1e-6
    #     min_pos_distances = pos_dis.min(dim=1)[0]
    #     pos_weights = 1 / (min_pos_distances + epsilon)
    #
    #     min_neg_distances = neg_dis.min(dim=1)[0]
    #     neg_weights = 1 / (min_neg_distances + epsilon)
    #     pos_weights = pos_weights / pos_weights.sum()
    #     neg_weights = neg_weights / neg_weights.sum()
    #     alpha = 0.7  # 正样本权重占比
    #     beta = 0.3  # 负样本权重占比
    #     combined_weights = (alpha * pos_weights + beta * neg_weights)
    #     combined_weights_1d = torch.cat([combined_weights, combined_weights], dim=0)  # (1024,)
    #     combined_weights_1d[combined_weights.shape[0]:] = 0.002
    #     return combined_weights_1d

    def reconstruction_loss_x(self, opt,domain_id,batch ,  users, item, n_item,adj, temperature,pos_weight, neg_weight ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        s_user, pos_item, neg_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)
        self.model_x = self.model_x.to(self.device)
        user = s_user.to(self.device)
        users = users.to(self.device)
        item = item.to(self.device)
        n_item = n_item.to(self.device)
        encoder_x = vgae_encoder(self.opt, self.opt["source_user_num"], self.opt["source_item_num"], self.device)
        decoder_x = vgae_decoder(self.opt, self.opt["source_user_num"], self.opt["source_item_num"])
        self.encoder_x = encoder_x.to(self.device)
        self.decoder_x = decoder_x.to(self.device)
        # 初始化生成器并移动到设
        self.generator_1 = vgae(self.encoder_x, self.decoder_x).to(self.device)
        self.x_generator_2 = DenoisingNet(self.model_x.getGCN(), self.model_x.getEmbeds(), self.opt,
                                          self.opt["source_user_num"], self.opt["source_item_num"]).to(self.device)
        self.x_generator_2.set_fea_adj(self.opt["source_user_num"] + self.opt["source_item_num"],
                                       deepcopy(adj).to(self.device))
        data1 = self.generator_generate(self.generator_1, adj.to(self.device))
        out_x = self.model_x.forward_graphcl(data1)
        x_out = self.model_x.forward_graphcl_(self.x_generator_2)
        assert out_x.requires_grad, "out_x does not require gradients"
        assert x_out.requires_grad, "x_out does not require gradients"
        # 损失计算 ib

        loss_x = (self.model_x.loss_graphcl(out_x, x_out, user, item))
        if loss_x.shape[0] > pos_weight.shape[0]:
            pad_size = loss_x.shape[0] - pos_weight.shape[0]
            pad_values = torch.full((pad_size,), pos_weight.mean().item()* self.opt["ssl_reg"], device=pos_weight.device)  # 补充值
            pos_weight = torch.cat([pos_weight* self.opt["ssl_reg"], pad_values])
        if loss_x.shape[0] > neg_weight.shape[0]:
            pad_size = loss_x.shape[0] - neg_weight.shape[0]
            pad_values = torch.full((pad_size,), neg_weight.mean().item()* self.opt["ssl_reg"], device=neg_weight.device)  # 补充值
            neg_weight = torch.cat([neg_weight * self.opt["ssl_reg"] , pad_values])
            # print(neg_weight)
        # loss_x = ((loss_x * pos_weight) - (loss_x * neg_weight)).mean() * self.opt["ssl_reg"]
        # print("loss_x: ", loss_x)
        loss_x =  (loss_x * neg_weight).mean()* self.opt["ssl_reg_game"]
        # loss_x = (loss_x ).mean() * self.opt["ssl_reg"]
        x_out1 = self.model_x.forward_graphcl(data1)
        x_out2 = self.model_x.forward_graphcl_(self.x_generator_2)
        x_loss_ib = (self.model_x.loss_graphcl(out_x, out_x.detach(), users, item)
                     + self.model_x.loss_graphcl(x_out, x_out.detach(), users, item))
        # loss_x += x_loss_ib.mean()
        loss_x += x_loss_ib.mean() * self.opt["ib_reg"]
        #
        x_usrEmbeds, x_itmEmbeds = self.model_x.forward_gcn(deepcopy(adj).to(self.device))
        x_ancEmbeds = x_usrEmbeds[users]
        x_posEmbeds = x_itmEmbeds[item]
        x_negEmbeds = x_itmEmbeds[n_item]
        x_scoreDiff = pairPredict(x_ancEmbeds, x_posEmbeds, x_negEmbeds)
        x_bprLoss = -(x_scoreDiff).sigmoid().log().sum() / self.opt["data_batch_size"]
        x_regLoss = calcRegLoss(self.model_x) * self.opt["reg"]


        loss_x += x_bprLoss + x_regLoss
        x_loss_1 = self.generator_1(deepcopy(adj).to(self.device), users, item, n_item)
        x_loss_2 = self.x_generator_2(users, item, n_item, temperature)
        loss_x += x_loss_1 + x_loss_2

        return loss_x

    def reconstruction_loss_y(self, opt, domain_id, batch,users,item, n_item, adj, temperature,pos_weight, neg_weight ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        s_user, pos_item, neg_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)
        self.model_y = self.model_y.to(self.device)
        # S_user, S_pos_item, S_n_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)
        user = s_user.to(self.device)
        users = users.to(self.device)
        item = item.to(self.device)
        n_item = n_item.to(self.device)
        encoder_y = vgae_encoder(self.opt, self.opt["target_user_num"], self.opt["target_item_num"], self.device)
        decoder_y = vgae_decoder(self.opt, self.opt["target_user_num"], self.opt["target_item_num"])
        self.encoder_y = encoder_y.to(self.device)
        self.decoder_y = decoder_y.to(self.device)
        self.generator_2 = vgae(self.encoder_y, self.decoder_y).to(self.device)
        self.y_generator_2 = DenoisingNet(self.model_y.getGCN(), self.model_y.getEmbeds(), self.opt,
                                          self.opt["target_user_num"], self.opt["target_item_num"]).to(self.device)
        self.y_generator_2.set_fea_adj(self.opt["target_user_num"] + self.opt["target_item_num"],
                                       deepcopy(adj).to(self.device))
        data2 = self.generator_generate(self.generator_2, adj.to(self.device))
        out_y = self.model_y.forward_graphcl(data2)
        y_out = self.model_y.forward_graphcl_(self.y_generator_2)
        assert out_y.requires_grad, "out_y does not require gradients"
        assert y_out.requires_grad, "y_out does not require gradients"
        loss_y = (self.model_y.loss_graphcl(out_y, y_out, user, item) )

        if loss_y.shape[0] > pos_weight.shape[0]:
            pad_size = loss_y.shape[0] - pos_weight.shape[0]
            pad_values = torch.full((pad_size,), pos_weight.mean().item()* self.opt["ssl_reg"], device=pos_weight.device)  # 补充值
            pos_weight = torch.cat([pos_weight* self.opt["ssl_reg"], pad_values])
        if loss_y.shape[0] > neg_weight.shape[0]:
            pad_size = loss_y.shape[0] - neg_weight.shape[0]
            pad_values = torch.full((pad_size,), neg_weight.mean().item() * self.opt["ssl_reg"], device=neg_weight.device)  # 补充值
            neg_weight = torch.cat([neg_weight * self.opt["ssl_reg"], pad_values])
            # print(neg_weight)
        # loss_y = ((loss_y * pos_weight) - (loss_y * neg_weight)).mean() * self.opt["ssl_reg"]
        #
        loss_y =  (loss_y * neg_weight).mean() * self.opt["ssl_reg_game"]
        # loss_y = (loss_y ).mean() * self.opt["ssl_reg"]
        y_out1 = self.model_y.forward_graphcl(data2)

        y_out2 = self.model_y.forward_graphcl_(self.y_generator_2)
        y_loss_ib = self.model_y.loss_graphcl(y_out1, out_y.detach(), users,
                                              item) + self.model_y.loss_graphcl(y_out2, y_out.detach(),
                                                                                      users, item)
        # loss_y += y_loss_ib.mean()
        loss_y += y_loss_ib.mean()*self.opt["ib_reg"]
        #
        y_usrEmbeds, y_itmEmbeds = self.model_y.forward_gcn(deepcopy(adj).to(self.device))
        y_ancEmbeds = y_usrEmbeds[users]
        y_posEmbeds = y_itmEmbeds[item]
        y_negEmbeds = y_itmEmbeds[n_item]
        y_scoreDiff = pairPredict(y_ancEmbeds, y_posEmbeds, y_negEmbeds)
        y_bprLoss = -(y_scoreDiff).sigmoid().log().sum() / self.opt["data_batch_size"]
        y_regLoss = calcRegLoss(self.model_y) * self.opt["reg"]
        loss_y += y_bprLoss + y_regLoss
        y_loss_1 = self.generator_2(deepcopy(adj).to(self.device), users, item, n_item)
        y_loss_2 = self.y_generator_2(users,item,n_item, temperature)
        loss_y += y_loss_1 + y_loss_2
        return loss_y

    def reconstruct_graph(self, opt,domain_id, batch):
        user, pos_item, neg_item, context_item, context_score, global_item, global_score = self.unpack_batch(batch)

        user_feature,pos_dis, neg_dis = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
        # # 确保张量在 CPU 上，并分离计算图（如果需要）
        # pos_dis = pos_dis.cpu().detach()
        # neg_dis = neg_dis.cpu().detach()

        # # 方法 1：保存为 .npy 文件（推荐）
        # # 转换为 NumPy 数组
        # pos_dis_np = pos_dis.numpy()
        # neg_dis_np = neg_dis.numpy()
        #
        # # 保存到 .npy 文件
        # np.save('pos_dis.npy', pos_dis_np)
        # np.save('neg_dis.npy', neg_dis_np)
        # print(
        #     "pos_dis 和 neg_dis 已保存到文件：pos_dis.npy, neg_dis.npy, pos_dis.pt, neg_dis.pt, pos_dis.csv, neg_dis.csv")
        epsilon = 1e-6
        # scale = 1000.0  # 缩放因子，可以调整（例如 100、1000、10000）
        min_pos_distances = pos_dis.min(dim=1)[0]
        pos_weights = 1 / (min_pos_distances + epsilon)

        min_neg_distances = neg_dis.min(dim=1)[0]
        neg_weights = -1 / (min_neg_distances + epsilon)
        pos_weights = pos_weights / pos_weights.sum()
        neg_weights = neg_weights / neg_weights.sum()
        alpha = self.opt["alpha"]  # 正样本权重占比
        beta = self.opt["beta1"]# 负样本权重占比
        combined_weights = (alpha* pos_weights + beta *neg_weights)


        pos_item_feature = self.model.forward_item(domain_id, pos_item)
        neg_item_feature = self.model.forward_item(domain_id, neg_item)

        pos_score = self.model.predict_dot(user_feature, pos_item_feature)
        neg_score = self.model.predict_dot(user_feature, neg_item_feature)

        pos_labels, neg_labels = torch.ones(pos_score.size()), torch.zeros(neg_score.size())

        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()
        loss = self.opt["lambda_loss"] * (self.criterion(pos_score, pos_labels) + self.criterion(neg_score, neg_labels)) + (1 - self.opt["lambda_loss"]) * self.model.critic_loss
        # print(type(loss))
        # print(loss)

        if self.opt["aggregator"] == "Transformer":
            prop_loss = self.model.n_prop_loss * self.opt["lambda_pp"]
            # prop_loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            return loss, prop_loss, pos_weights, combined_weights

        return loss, None

    # def train_IB(self,opt, source_UV, source_VU, target_UV, target_VU):
    #     # print("执行；了+++++++++++++++++++++++++++++++")
    #     self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
    #     self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
    #     self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
    #     self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])
    #
    #     # self.shared_user = torch.arange(0, self.opt["shared_user"], 1)
    #     self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
    #     self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
    #     self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
    #     self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)
    #     self.discri = nn.Sequential(
    #         nn.Linear(opt["feature_dim"] * 2 * opt["GNN"], opt["feature_dim"]),
    #         nn.ReLU(),
    #         nn.Linear(opt["feature_dim"], 100),
    #         nn.ReLU(),
    #         nn.Linear(100, 1),
    #     )
    #
    #     self.source_GNN = VBGE(opt)
    #     self.target_GNN = VBGE(opt)
    #     # 获取源领域用户嵌入
    #     source_user = self.source_user_embedding(self.source_user_index)
    #     # 获取目标领域用户嵌入
    #     target_user = self.target_user_embedding(self.target_user_index)
    #     # 获取源领域物品嵌入
    #     source_item = self.source_item_embedding(self.source_item_index)
    #     # 获取目标领域物品嵌入
    #     target_item = self.target_item_embedding(self.target_item_index)
    #
    #     # 通过源领域的GNN更新用户和物品表示
    #     source_learn_user, source_learn_item = self.source_GNN(source_user, source_item, source_UV, source_VU)
    #     # 通过目标领域的GNN更新用户和物品表示
    #     target_learn_user, target_learn_item = self.target_GNN(target_user, target_item, target_UV, target_VU)
    #
    #
    #     return self.source_GNN.encoder[-1].kld_loss, self.target_GNN.encoder[-1].kld_loss
    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def predict(self, domain_id, eval_dataloader):
        MRR = 0.0
        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0
        valid_entity = 0

        for test_batch in eval_dataloader:
            user, item, context_item, context_score, global_item, global_score = self.unpack_batch_predict(test_batch)

            user_feature = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
            item_feature = self.model.forward_item(domain_id, item)

            scores = self.model.predict_dot(user_feature, item_feature)

            scores = scores.data.detach().cpu().numpy()

            for pred in scores:

                rank = (-pred).argsort().argsort()[0].item()

                valid_entity += 1
                MRR += 1 / (rank + 1)
                if rank < 1:
                    NDCG_1 += 1 / np.log2(rank + 2)
                    HT_1 += 1
                if rank < 5:
                    NDCG_5 += 1 / np.log2(rank + 2)
                    HT_5 += 1
                if rank < 10:
                    NDCG_10 += 1 / np.log2(rank + 2)
                    HT_10 += 1
                if valid_entity % 100 == 0:
                    print('+', end='')

        print("")
        metrics = {}
        # metrics["MRR"] = MRR / valid_entity
        # metrics["NDCG_5"] = NDCG_5 / valid_entity
        metrics["NDCG_10"] = NDCG_10 / valid_entity
        # metrics["HT_1"] = HT_1 / valid_entity
        # metrics["HT_5"] = HT_5 / valid_entity
        metrics["HT_10"] = HT_10 / valid_entity

        return metrics


    def predict_full_rank(self, domain_id, eval_dataloader, train_map, eval_map):

        def nDCG(ranked_list, ground_truth_length):
            dcg = 0
            idcg = IDCG(ground_truth_length)
            for i in range(len(ranked_list)):
                if ranked_list[i]:
                    rank = i + 1
                    dcg += 1 / math.log(rank + 1, 2)
            return dcg / idcg

        def IDCG(n):
            idcg = 0
            for i in range(n):
                idcg += 1 / math.log(i + 2, 2)
            return idcg

        def precision_and_recall(ranked_list, ground_number):
            hits = sum(ranked_list)
            pre = hits / (1.0 * len(ranked_list))
            rec = hits / (1.0 * ground_number)
            return pre, rec

        ndcg_list = []
        pre_list = []
        rec_list = []

        NDCG_10 = 0.0
        HT_10 = 0

        # pdb.set_trace()
        for test_batch in eval_dataloader:
            user, item, context_item, context_score, global_item, global_score = self.unpack_batch_predict(test_batch)

            user_feature = self.model.forward_user(domain_id, user, context_item, context_score, global_item, global_score)
            item_feature = self.model.forward_item(domain_id, item)

            scores = self.model.predict_dot(user_feature, item_feature)

            scores = scores.data.detach().cpu().numpy()
            user = user.data.detach().cpu().numpy()
            # pdb.set_trace()
            for idx, pred in enumerate(scores):
                rank = (-pred).argsort()
                score_list = []

                hr=0
                for i in rank:
                    i = i + 1
                    if (i in train_map[user[idx]]) and (i not in eval_map[user[idx]]):
                        continue
                    else:
                        if i in eval_map[user[idx]]:
                            hr = 1
                            # nd += 1 / np.log2(len(score_list) + 2)
                            score_list.append(1)
                        else:
                            score_list.append(0)
                        if len(score_list) == 10:
                            break

                HT_10 += hr

                pre, rec = precision_and_recall(score_list, len(eval_map[user[idx]]))
                pre_list.append(pre)
                rec_list.append(rec)
                ndcg_list.append(nDCG(score_list, len(eval_map[user[idx]])))

                if len(ndcg_list) % 100 == 0:
                    print('+', end='')
        print("")

        metrics = {}
        metrics["HT_10"] = HT_10 / len(ndcg_list)
        metrics["NDCG_10"] = sum(ndcg_list) / len(ndcg_list)

        return metrics
