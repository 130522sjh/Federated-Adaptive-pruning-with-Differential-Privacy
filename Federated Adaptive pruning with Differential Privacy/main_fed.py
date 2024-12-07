#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import random
import shutil
import time

import torchvision
from torchvision.datasets import VOCDetection

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import cifar_iid, cifar_noniid, svhn_iid, svhn_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNSVHN
from models.Fed import FedAvg, FedProx, FedMA
from models.test import test_img
import os
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def evaluate_and_save_results(Lepochs, net_glob, dataset_train, dataset_test, args, tpath='./save_test_acc'):
    """
    Evaluate the global model on training and testing datasets, then save the results.

    Parameters:
    - net_glob: The global neural network model.
    - dataset_train: The training dataset.
    - dataset_test: The testing dataset.
    - args: Arguments containing hyperparameters and settings.
    - tpath: The directory path to save the results.
    """
    # 设置模型为评估模式
    net_glob.eval()

    # 评估训练集和测试集的准确率和损失
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    # 打印训练和测试准确率
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    # 创建目录（如果不存在）
    if not os.path.exists(tpath):
        os.makedirs(tpath)

    # 生成保存结果的文件路径
    file_path = os.path.join(tpath,
                             'fed_{}_{}_{}_C{}_iid{}_local_ep{}_lr{}_dp_epsilon{}_dp_mechanism{}.FedMANOPDirichlet.txt'.format(
                                 args.dataset, args.model, Lepochs, args.frac, args.iid,
                                 args.local_ep, args.lr, args.dp_epsilon, args.dp_mechanism))

    # 将训练和测试准确率写入文件
    with open(file_path, 'w') as test_ac:
        test_ac.write(f"{acc_train}\n")
        test_ac.write(f"{acc_test}\n")

    print(f'结果已保存到 {file_path}')




if __name__ == '__main__':

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    trans_voc = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a standard size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # Normalization values typically used for pre-trained models
    ])


    if args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'svhn':
        trans_svhn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.SVHN('../data/svhn', split='train', download=True, transform=trans_svhn)
        dataset_test = datasets.SVHN('../data/svhn', split='test', download=True, transform=trans_svhn)
        if args.iid:
            dict_users = svhn_iid(dataset_train, args.num_users)
        else:
            dict_users = svhn_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'svhn':
        net_glob = CNNSVHN(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden1=200, dim_hidden2=100, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)
    net_glob.train()

    # copy weights  获取全局模型 net_glob 的参数字典
    w_glob = net_glob.state_dict()

    # 训练过程中的准确率列表
    acc_train_list = []
    loss_train = []
    epochs_sub = []
    jisuancishu_list = []
    sub = 0
    jisuancishu = 0
    if args.all_clients:
        print("Aggregation over all clients")
        #这样做是为了在所有客户端上使用相同的全局模型进行训练，而不是每个客户端都使用各自训练后的局部模型进行聚合。
        w_locals = [w_glob for i in range(args.num_users)]

    # 迭代训练过程
    for iter in range(args.epochs):
        loss_locals = []
        accuracy_locals = []

        if not args.all_clients:
            w_locals = []
        #默认取m个frac是比例
        m = max(int(args.frac * args.num_users), 1)
        #包含m个客户端的索引
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            b_f = len(dict_users[idx])
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # 训练局部模型
            w, loss, accuracy, idxs, count = local.train(net=copy.deepcopy(net_glob).to(args.device))
            jisuancishu += count
            dict_users[idx] = idxs

            later = len(dict_users[idx])
            sub += (b_f - later)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            accuracy_locals.append(copy.deepcopy(accuracy))
        jisuancishu_list.append(jisuancishu)
        epochs_sub.append(copy.deepcopy(sub))
        # 更新全局权重
        w_glob = FedAvg(w_locals)
        # 更新全局模型
        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_avg = sum(accuracy_locals) / len(accuracy_locals)

        # 打印和记录损失和准确率
        print('Round {:3d}, Average loss {:.3f}, Average train accuracy {:.2f}'.format(iter, loss_avg, acc_avg))
        loss_train.append(loss_avg)
        acc_train_list.append(acc_avg)
        if iter + 1 in [10,20,30,40,50,60,70,80,90,100]:
            evaluate_and_save_results(iter+1, net_glob, dataset_train, dataset_test, args)
            print('计算次数:{}'.format(jisuancishu))



#训练准确率日志
    rootpath = './log_self'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(
        rootpath + '/accfile_fed_{}_{}_{}_iid{}_dp_{}_local_ep{}_dp_epsilon{}_dp_mechanism{}.FedMANOPDirichlet.dat'.format(args.dataset, args.model, args.epochs, args.iid, args.dp_mechanism, args.local_ep, args.dp_epsilon, args.dp_mechanism), "w")

    for ac in acc_train_list:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()




    plt.figure()
    plt.plot(range(len(acc_train_list)), acc_train_list)
    plt.ylabel('train_accuracy')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_acc_local_ep{}_lr{}_dp_epsilon{}_dp_mechanism{}.FedMANOPDirichlet.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.lr, args.dp_epsilon, args.dp_mechanism))




    # 将每轮删减的数据个数的累加记录下来。
    subpath = './sub_log'
    if not os.path.exists(subpath):
        os.makedirs(subpath)
    sub_write = open(
        subpath + '/epochs_sub_file_{}_{}_{}_iid{}_dp_{}_local_ep{}_dp_epsilon{}_dp_mechanism{}.FedMANOPDirichlet.dat'.format(
            args.dataset, args.model, args.epochs, args.iid, args.dp_mechanism, args.local_ep, args.dp_epsilon,
            args.dp_mechanism), "w")  # 定义文件名为 'epochs_sub_file.dat'

    for epoch in epochs_sub:
        epoch_str = str(epoch)  # 将数据转换为字符串形式
        sub_write.write(epoch_str)  # 将字符串写入文件
        sub_write.write('\n')  # 写入换行符，以便下一行数据写入新的一行

    sub_write.close()  # 关闭文件

    # 保存计算次数日志
    jisuanpath = './jisuan_log'
    if not os.path.exists(jisuanpath):
        os.makedirs(jisuanpath)

    # 将每轮的计算次数存储到文件中
    jisuan_write = open(
        jisuanpath + '/jisuancishu_file_{}_{}_{}_iid{}_dp_{}_local_ep{}_dp_epsilon{}_dp_mechanism{}.FedMANOPDirichlet.dat'.format(
            args.dataset, args.model, args.epochs, args.iid, args.dp_mechanism, args.local_ep, args.dp_epsilon,
            args.dp_mechanism),
        "w")

    # 写入 jisuancishu_list 中的每个计算次数
    for count in jisuancishu_list:
        jisuan_write.write(str(count))
        jisuan_write.write('\n')  # 写入换行符

    jisuan_write.close()