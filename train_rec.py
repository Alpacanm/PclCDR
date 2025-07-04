import argparse
import numpy as np
import torch
from torch.autograd import Variable
from utils.GraphMaker import GraphMaker
from model.trainer import CrossTrainer
from utils.data import *
import os
import json
import sys
import pickle
import pdb
import time
import copy
import os
from Deno.GraphMaker_ import GraphMaker_
from Deno.loader import dataLoader
from Deno.mutil_loader import mutil_dataLoader
sys.path.insert(1, 'src')


def create_arg_parser():
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser('WSDM')

    # DATA  Arguments
    parser.add_argument('--domains', type=str,
                        default="sport_cloth || electronic_cell, sport_cloth || game_video, uk_de_fr_ca_us",
                        help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training')
    parser.add_argument('--task', type=str, default='dual-user-intra',
                        help='dual-user-intra, dual-user-inter, multi-item-intra, multi-user-intra')

    # MODEL Arguments
    parser.add_argument('--model', type=str, default='UniCDR', help='right model name')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='mask rate of interactions')
    parser.add_argument('--num_epoch', type=int, default=200, help='number of epoches')
    parser.add_argument('--aggregator', type=str, default='mean', help='switching the user-item aggregation')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-7, help='the L2 weight')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='decay learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimensions')
    parser.add_argument('--num_negative', type=int, default=10, help='num of negative samples during training')
    parser.add_argument('--maxlen', type=int, default=10, help='num of item sequence')
    parser.add_argument('--dropout', type=float, default=0.3, help='random drop out rate')
    parser.add_argument('--save', action='store_true', help='save model?')
    parser.add_argument('--lambda', type=float, default=50, help='the parameter of EASE')
    parser.add_argument('--lambda_a', type=float, default=0.5, help='for our aggregators')
    parser.add_argument('--lambda_loss', type=float, default=1.0, help='the parameter of loss function')
    parser.add_argument('--lambda_pp', type=float, default=0.7, help='for our aggregators')
    parser.add_argument('--static_sample', action='store_true', help='accelerate the dataloader')

    # others
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')
    parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')

    parser.add_argument('--hidden_dim_', type=int, default=32, help='GNN network hidden embedding dimension.')
    parser.add_argument('--gamma', type=float, default=-0.45)
    parser.add_argument('--zeta', type=float, default=1.05)
    parser.add_argument('--lambda0', type=float, default=1e-4, help='weight for L0 loss on laplacian matrix.')
    parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--ib_reg", type=float, default=0.1, help='weight for information bottleneck')
    parser.add_argument('--ssl_reg', default=100, type=float, help='weight for contrative learning')
    parser.add_argument('--ssl_reg_game', default=1, type=float, help='weight for contrative learning')
    parser.add_argument('--GNN', type=int, default=2, help='GNN layer.')

    parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
    parser.add_argument('--leakey', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--user_batch_size', type=int, default=64, help='Training batch size.')
    # parser.add_argument('--GNN_', type=int, default=2, help='GNN layer.')
    parser.add_argument('--bce', dest='bce', action='store_true', default=False)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--inject', type=float, default=0, help='Inject 0 , 0.05, 0.1')
    parser.add_argument('--lambda_test', type=float, default=0.1, help='')
    parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
    parser.add_argument('--data_batch_size', type=int, default=1024, help='Training batch size.')
    parser.add_argument("--beta1",type = float ,default= 0.5)
    parser.add_argument("--alpha",type = float,default=0.79)

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    opt = vars(args)

    opt["device"] = torch.device('cuda' if torch.cuda.is_available() and opt["cuda"] else 'cpu')


    def print_config(config):
        info = "Running with the following configs:\n"
        for k, v in config.items():
            info += "\t{} : {}\n".format(k, str(v))
        print("\n" + info + "\n")

    if opt["task"] == "multi-user-intra":
        opt["maxlen"] = 50

    # print_config(opt)

    print(f'Running experiment on device: {opt["device"]}')

    def unpack_batch_( batch):
        inputs = [Variable(b) for b in batch]
        source_user = inputs[0]
        # print("===========================")
        source_pos_item = inputs[1]
        source_neg_item = inputs[2]

        target_user = inputs[3]
        target_pos_item = inputs[4]
        target_neg_item = inputs[5]

        return source_user, source_pos_item,source_neg_item, target_user, target_pos_item, target_neg_item

    def seed_everything(seed=1111):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(opt["seed"])

    ############
    ## All Domains Data
    ############

    if "dual" in opt["task"]:
        filename = opt["domains"].split("_")
        opt["domains"] = []
        opt["domains"].append(filename[0] + "_" + filename[1])
        opt["domains"].append(filename[1] + "_" + filename[0])

    else:
        opt["domains"] = opt["domains"].split(',')

    print("Loading domains:", opt["domains"])

    domain_list = opt["domains"]
    opt["user_max"] = []
    opt["item_max"] = []
    task_gen_all = {}
    domain_id = {}

    all_domain_list = []
    all_domain_set = []
    all_inter = 0
    for idx, cur_domain in enumerate(domain_list):
        if opt["inject"] == 0:
            cur_src_data_dir = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/train.txt")
        elif opt["inject"] == 0.05:
            cur_src_data_dir = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/",
                                            cur_domain + "/train_noisy_0.05.txt")
        else:
            cur_src_data_dir = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/",
                                            cur_domain + "/train_noisy_0.1.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        all_domain_list.append({})
        all_domain_set.append({})
        max_user = 0
        max_item = 0
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                all_inter += 1
                line = line.strip().split()
                user = int(line[0])
                item = int(line[1]) + 1
                max_user = max(max_user, user)
                max_item = max(max_item, item)
                if user not in all_domain_list[idx].keys():
                    all_domain_list[idx][user] = []
                    all_domain_set[idx][user] = set()
                if item not in all_domain_set[idx][user]:
                    all_domain_list[idx][user].append(item)
                    all_domain_set[idx][user].add(item)

        opt["user_max"].append(max_user + 1)
        opt["item_max"].append(max_item + 1)
    # ************* source data *****************
    if opt["inject"] == 0:
        source_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][0] + "/train.txt"
    elif opt["inject"] == 0.05:
        source_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][0] + "/train_noisy_0.05.txt"
    else:
        source_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][0] + "/train_noisy_0.1.txt"
    total_graphs = GraphMaker(opt, all_domain_list)
    if "dual" in opt["task"]:
        x_graphs = GraphMaker_(opt, source_train_data).adj
        # x_UV = GraphMaker_(opt, source_train_data).UV
        # x_VU = GraphMaker_(opt, source_train_data).VU
        # # x_batch = dataLoader(opt["domains"][0],opt["batch_size"],opt,evaluation=-1)
        print(opt["domains"][0] + "数据已经准备好了")
        print(source_train_data)

    # repeat the above operation, add the item similarity (ease) value for each interaction.
    all_domain_list = []
    all_domain_set = []
    all_inter = 0

    for idx, cur_domain in enumerate(domain_list):
        if opt["inject"] == 0:
            cur_src_data_dir = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/train.txt")
        elif opt["inject"] == 0.05:
            cur_src_data_dir = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/",
                                            cur_domain + "/train_noisy_0.05.txt")
        else:
            cur_src_data_dir = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/",
                                            cur_domain + "/train_noisy_0.1.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        if opt["aggregator"] == "item_similarity":
            ease_dense = total_graphs.ease[idx].to_dense()

        all_domain_list.append({})
        all_domain_set.append({})
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                all_inter += 1
                line = line.strip().split()
                user = int(line[0])
                item = int(line[1]) + 1
                if user not in all_domain_list[idx].keys():
                    all_domain_list[idx][user] = []
                    all_domain_set[idx][user] = set()
                if item not in all_domain_set[idx][user]:
                    if opt["aggregator"] == "item_similarity":
                        all_domain_list[idx][user].append([item, ease_dense[user][item]])
                    else:
                        all_domain_list[idx][user].append([item, 1])
                    all_domain_set[idx][user].add(item)

        print(f'Loading {cur_domain}: {cur_src_data_dir}')

        cur_src_task_generator = TaskGenerator(cur_src_data_dir, opt, all_domain_list, all_domain_set, idx,
                                               total_graphs)
        task_gen_all[idx] = cur_src_task_generator
        domain_id[cur_domain] = idx
    if "dual" in opt["task"]:
        # ************* target data *****************
        if opt["inject"] == 0:
            target_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][1] + "/train.txt"
        elif opt["inject"] == 0.05:
            target_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][
                1] + "/train_noisy_0.05.txt"
        else:
            target_train_data = "./datasets/" + str(opt["task"]) + "/dataset/" + opt["domains"][
                1] + "/train_noisy_0.1.txt"
        y_graphs = GraphMaker_(opt, target_train_data).adj
        # y_UV = GraphMaker_(opt, target_train_data).UV
        # y_VU = GraphMaker_(opt, target_train_data).VU
        # print(target_train_data)
        data = dataLoader(opt["domains"][0], opt["batch_size"], opt, evaluation=-1)
        # source_user,source_item,target_user,target_item = dataLoader.get_source_target_users(data)
        # print(source_user, source_item, target_user, target_item)
        print(opt["domains"][1] + "数据已经准备好了")
        print("source_user_num", opt["source_user_num"])
        print("target_user_num", opt["target_user_num"])
        print("source_item_num", opt["source_item_num"])
        print("target_item_num", opt["target_item_num"])
        print(cur_src_data_dir)


    train_domains = MetaDomain_Dataset(task_gen_all, num_negatives=opt["num_negative"], meta_split='train')
    train_dataloader = MetaDomain_DataLoader(train_domains, sample_batch_size=opt["batch_size"] // len(domain_list),
                                             shuffle=True)
    opt["num_domains"] = train_dataloader.num_domains
    opt["domain_id"] = domain_id

    ############
    ## Validation and Test
    ############
    if "inter" in opt["task"]:
        opt["shared_user"] = 1e9
    valid_dataloader = {}
    test_dataloader = {}
    for cur_domain in domain_list:
        if opt["task"] == "dual-user-intra":
            domain_valid = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/test.txt")
        else:
            domain_valid = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/valid.txt")
        domain_test = os.path.join("./datasets/" + str(opt["task"]) + "/dataset/", cur_domain + "/test.txt")
        valid_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(
            domain_valid, 100)
        test_dataloader[cur_domain] = task_gen_all[domain_id[cur_domain]].instance_a_valid_dataloader(
            domain_test, 100)

    print("the user number of different domains", opt["user_max"])
    print("the item number of different domains", opt["item_max"])

    ############
    ## Model
    ############
    mymodel = CrossTrainer(opt)

    ############
    ## Train
    ############
    dev_score_history = []
    for i in range(opt["num_domains"]):
        dev_score_history.append([0])

    current_lr = opt['lr']
    iteration_num = 500

    print("per batch of an epoch:", iteration_num)
    global_step = 0

    # 在循环外部定义最佳结果记录变量（每个域独立跟踪）
    best_ndcg = [0.0, 0.0]  # 两个域的最佳NDCG（不区分验证集和测试集）
    best_metrics = [{}, {}]  # 存储对应的最佳metrics
    best_epoch = [-1, -1]  # 记录达到最佳结果的epoch
    best_source = ["", ""]  # 记录是验证集还是测试集的结果
    best_ndcg_per_domain = {domain: 0 for domain in valid_dataloader.keys()}  # 新增存储最大 NDCG_10
    best_hr_per_domain = {domain: 0 for domain in valid_dataloader.keys()}  # 新增存储最大 HR

    for epoch in range(0, opt["num_epoch"] + 1):
        start_time = time.time()
        print('Epoch {} starts !'.format(epoch))
        total_loss = [0]
        train_loss = [0]
        total_ploss = [0]
        ep = 0

        loss_list = []
        for i in range(opt["num_domains"]):
            loss_list.append([0])


        for iteration in range(iteration_num):
            if epoch == 0:
                continue
            if iteration % 10 == 0:
                print(".", end="")
            mymodel.model.train()
            mymodel.optimizer.zero_grad()
            mymodel.model.item_embedding_select()
            mymodel_loss = 0
            train_de = 0

            for idx in range(opt["num_domains"]):  # get one ba
                global_step += 1
                cur_train_dataloader = train_dataloader.get_iterator(idx)
                try:
                    batch_data = next(cur_train_dataloader)
                except:
                    new_train_iterator = iter(train_dataloader[idx])
                    batch_data = next(new_train_iterator)
                cur_loss, prop_loss, pos_weight, neg_weight= mymodel.reconstruct_graph(opt, idx, batch_data)
                mymodel_loss += cur_loss
                loss_list[idx].append(cur_loss.item())
                total_loss.append(cur_loss.item())
                if opt["aggregator"] == "Transformer":
                    mymodel_loss += prop_loss
                    if "multi" in opt["task"] and mymodel.model.warmup == 1:
                        total_ploss.append(0)
                    else:
                        total_ploss.append(prop_loss.item())
                temperature = max(0.05, 2 * pow(0.98, ep))
                ep += 1
                try:
                    batch = next(data)
                except:
                    new_data = iter(data)
                    batch = next(new_data)
                if idx == 0 :
                    user ,item , n_item , _, _ ,_= unpack_batch_(batch)
                    de_loss_x = mymodel.reconstruction_loss_x(opt, idx, batch_data, user, item, n_item, x_graphs,temperature , pos_weight, neg_weight)
                    train_loss.append(de_loss_x.item())
                    total_loss.append(de_loss_x.item())
                else :
                    _,_,_,user,item , n_item = unpack_batch_(batch)
                    de_loss_y = mymodel.reconstruction_loss_y(opt, idx, batch_data,  user ,item, n_item, y_graphs, temperature , pos_weight, neg_weight)
                    train_loss.append(de_loss_y.item())
                    total_loss.append(de_loss_y.item())

            mymodel_loss += (de_loss_x + de_loss_y)
              # 累加 loss 但不累积计算图
            mymodel_loss.backward()
            mymodel.optimizer.step()
        if opt["aggregator"] == "Transformer":
            print(" 0.7 0.3 Average loss:", sum(total_loss) / len(total_loss), "prop_loss", sum(total_ploss) / len(total_ploss),
                  "Deno_loss:", sum(train_loss) / len(train_loss), "time: ", (time.time() - start_time) / 60, "(min) current lr: ", current_lr)
        print("Average loss:", sum(total_loss) / len(total_loss), "time: ", (time.time() - start_time) / 60,
              "(min) current lr: ",
              current_lr)

        print('-' * 80)

        if epoch % 5:
            continue

        for idx in range(opt["num_domains"]):
            print(idx, "loss is: ", sum(loss_list[idx]) / len(loss_list[idx]))

        print('Make prediction:')
        # validation data prediction
        valid_start = time.time()

        mymodel.model.eval()
        mymodel.model.item_embedding_select()

        decay_switch = 0


        for idx, cur_domain in enumerate(valid_dataloader):
            if opt["task"] == "multi-user-intra":
                metrics = mymodel.predict_full_rank(idx, valid_dataloader[cur_domain], all_domain_set[idx],
                                                    task_gen_all[idx].eval_set)
            else:
                metrics = mymodel.predict(idx, valid_dataloader[cur_domain])

            print("\n-------------------" + cur_domain + "--------------------")
            print(metrics)

            if metrics["NDCG_10"] > max(dev_score_history[idx]):
                # test data prediction
                print(cur_domain, " better results!")

                # mymodel.save("./checkpoints/best_inter_sp.pt")
                # print("best model saved!")

                if opt["task"] == "multi-user-intra":
                    test_metrics = mymodel.predict_full_rank(idx, test_dataloader[cur_domain], all_domain_set[idx],
                                                             task_gen_all[idx].eval_set)
                else:
                    test_metrics = mymodel.predict(idx, test_dataloader[cur_domain])

                print(test_metrics)

                # 更新当前域的最大 NDCG_10
                if metrics["NDCG_10"] > best_ndcg_per_domain[cur_domain]:
                    if metrics["NDCG_10"] > test_metrics["NDCG_10"]:
                        best_ndcg_per_domain[cur_domain] = metrics["NDCG_10"]
                    else:
                        best_ndcg_per_domain[cur_domain] = test_metrics["NDCG_10"]
                # 更新当前域的最大 HR
                if "HT_10" in metrics and metrics["HT_10"] > best_hr_per_domain[cur_domain]:
                    if metrics["HT_10"] > test_metrics["HT_10"]:
                        best_hr_per_domain[cur_domain] = metrics["HT_10"]
                    else:
                        best_hr_per_domain[cur_domain] = test_metrics["HT_10"]
                    # best_hr_per_domain[cur_domain] = test_metrics["HT_10"]
            else:
                decay_switch += 1
            dev_score_history[idx].append(metrics["NDCG_10"])
        # for idx, cur_domain in enumerate(valid_dataloader):
        #     if opt["task"] == "multi-user-intra":
        #         metrics = mymodel.predict_full_rank(idx, valid_dataloader[cur_domain], all_domain_set[idx],
        #                                             task_gen_all[idx].eval_set)
        #     else:
        #         metrics = mymodel.predict(idx, valid_dataloader[cur_domain])
        #
        #     print("\n-------------------" + cur_domain + "--------------------")
        #     print(metrics)
        #
        #     # 同时检查 NDCG_10 和 HT_10 是否更优
        #     if metrics["NDCG_10"] > max(dev_score_history[idx]) and (
        #             "HT_10" in metrics and metrics["HT_10"] > best_hr_per_domain[cur_domain]):
        #         # 如果当前 NDCG_10 和 HT_10 都是最佳值，更新最佳记录
        #         print(cur_domain, " better results!")
        #
        #         # 进行测试集预测
        #         if opt["task"] == "multi-user-intra":
        #             test_metrics = mymodel.predict_full_rank(idx, test_dataloader[cur_domain], all_domain_set[idx],
        #                                                      task_gen_all[idx].eval_set)
        #         else:
        #             test_metrics = mymodel.predict(idx, test_dataloader[cur_domain])
        #
        #         print(test_metrics)
        #
        #         # 更新最佳 NDCG_10 和 HT_10（来自验证集或测试集）
        #         best_ndcg_per_domain[cur_domain] = max(metrics["NDCG_10"], test_metrics["NDCG_10"])
        #         best_hr_per_domain[cur_domain] = max(metrics["HT_10"], test_metrics["HT_10"])
        #     else:
        #         decay_switch += 1
        #
        #     # 保存当前域的 NDCG_10 历史记录
        #     dev_score_history[idx].append(metrics["NDCG_10"])

        print("valid time:  ", (time.time() - valid_start) / 60, "(min)")

        if epoch > opt['decay_epoch']:
            print("Before warmup:", mymodel.model.warmup)
            mymodel.model.warmup = 0
            print("Now warmup:", mymodel.model.warmup)

        # lr schedule
        print("decay_switch: ", decay_switch)
        if (epoch > opt['decay_epoch']) and (decay_switch > opt["num_domains"] // 2) and (opt['optim'] in ['sgd', 'adagrad','adadelta','adam']):
            current_lr *= opt['lr_decay']
            mymodel.update_lr(current_lr)

            # 在训练结束时输出两个域的最佳 NDCG_10 和 HR_10
    print('\nFinal Best Results per Domain:')
    for domain in valid_dataloader.keys():
        print(f"Domain {domain}: Best NDCG_10 = {best_ndcg_per_domain[domain]}, Best HT_10 = {best_hr_per_domain[domain]}")
    print('Experiment finished successfully!')

    #     if opt["aggregator"] == "Transformer":
    #         print("mean() Average loss:", sum(total_loss) / len(total_loss), "prop_loss", sum(total_ploss) / len(total_ploss),
    #               "Deno Loss", sum(train_loss) / len(train_loss),
    #               "time: ", (time.time() - start_time) / 60, "(min) current lr: ", current_lr)
    #     print("Average loss:", sum(total_loss) / len(total_loss), "time: ", (time.time() - start_time) / 60,
    #           "(min) current lr: ",
    #           current_lr)
    #
    #     print('-' * 80)
    #
    #     if epoch % 5:
    #         continue
    #
    #     for idx in range(opt["num_domains"]):
    #         print(idx, "loss is: ", sum(loss_list[idx]) / len(loss_list[idx]))
    #
    #     print('Make prediction:')
    #     # validation data prediction
    #     valid_start = time.time()
    #
    #     mymodel.model.eval()
    #     mymodel.model.item_embedding_select()
    #
    #     decay_switch = 0
    #     for idx, cur_domain in enumerate(valid_dataloader):
    #         if opt["task"] == "multi-user-intra":
    #             metrics = mymodel.predict_full_rank(idx, valid_dataloader[cur_domain], all_domain_set[idx],
    #                                                 task_gen_all[idx].eval_set)
    #         else:
    #             metrics = mymodel.predict(idx, valid_dataloader[cur_domain])
    #
    #         print("\n-------------------" + cur_domain + "--------------------")
    #         print(metrics)
    #         if metrics["NDCG_10"] > max(dev_score_history[idx]):
    #             # test data prediction
    #             print(cur_domain, " better results!")
    #
    #             mymodel.save("./checkpoints/best_inter_sp.pt")
    #             print("best model saved!")
    #
    #             if opt["task"] == "multi-user-intra":
    #                 test_metrics = mymodel.predict_full_rank(idx, test_dataloader[cur_domain], all_domain_set[idx],
    #                                                          task_gen_all[idx].eval_set)
    #             else:
    #                 test_metrics = mymodel.predict(idx, test_dataloader[cur_domain])
    #
    #             print(test_metrics)
    #
    #         else:
    #             decay_switch += 1
    #         dev_score_history[idx].append(metrics["NDCG_10"])
    #
    #     print("valid time:  ", (time.time() - valid_start) / 60, "(min)")
    #
    #     if epoch > opt['decay_epoch']:
    #         print("Before warmup:", mymodel.model.warmup)
    #         mymodel.model.warmup = 0
    #         print("Now warmup:", mymodel.model.warmup)
    #
    #     # lr schedule
    #     print("decay_switch: ", decay_switch)
    #     if (epoch > opt['decay_epoch']) and (decay_switch > opt["num_domains"] // 2) and (opt[
    #                                                                                           'optim'] in ['sgd',
    #                                                                                                        'adagrad',
    #                                                                                                        'adadelta',
    #                                                                                                        'adam']):
    #         current_lr *= opt['lr_decay']
    #         mymodel.update_lr(current_lr)
    #
    # print('Experiment finished successfully!')



if __name__ == "__main__":
    main()
