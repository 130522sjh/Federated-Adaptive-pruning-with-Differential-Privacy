#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from mdit_py_plugins.myst_blocks.index import target
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.sampling import  cifar_iid, cifar_noniid
import torch.nn.functional as F
import random
import time



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        idx = self.idxs[item]
        return image, label, idx

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.idxs = idxs
        #选择idxs个图片作为一个训练子集然后每次加载的批次大小为batch_size
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)
        self.times = self.args.epochs * self.args.frac
        self.max_norm = 1



    def train(self, net):
        net.train()
        # 训练和更新
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        epoch_accuracy = []
        #记录初始每个客户端数据集大小
        # target = 500\
        target=500
        count = 0

        # # 在训练前记录时间
        # start_time_training = time.time()
        # 每个客户端训练的轮数local_ep
        for iter in range(self.args.local_ep):

            batch_loss = []
            correct = 0
            total = 0
            count += len(self.idxs)

            # 循环batch_idx次，返回一个元组，labels是个张量，是每个批次各个图片对应的label
            for batch_idx, (images, labels, idxs1) in enumerate(self.ldr_train):
                images, labels, idxs1 = images.to(self.args.device), labels.to(self.args.device), idxs1.to(
                    self.args.device)
                indices_to_remove = []
                # 获得每个批次中各个图片在原来数据集中的索引列表
                indices_in_idxs1 = [idxs1[i.item()] for i in torch.arange(images.size(0))]
                net.zero_grad()
                # 将图像输入到神经网络模型中进行前向传播，得到预测结果
                log_probs = net(images)
                probs = F.softmax(log_probs, dim=1)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_norm)

                # 计算噪声标准差
                sensitivity = self.compute_sensitivity(net.parameters())  # 计算梯度的敏感度
                noise_std = sensitivity / self.args.dp_epsilon

                for param in net.parameters():
                    # 添加拉普拉斯噪声
                    noise = np.random.laplace(0, noise_std*(len(self.idxs)/target), param.grad.shape)
                    param.grad += torch.tensor(noise, device=self.args.device)

                # noise_std = 1.0  # 你可以根据需要调整这个值
                #
                # for param in net.parameters():
                #     # 添加高斯噪声
                #     noise = np.random.normal(0, noise_std * (len(self.idxs) / target), param.grad.shape)
                #     param.grad += torch.tensor(noise, device=self.args.device)

                optimizer.step()

                # 计算准确率
                premaxnum, predicted = torch.max(log_probs, 1)
                max_probs, max_indices = torch.max(probs, dim=1)
                #计算每张图的fisher
                # fisher_prob = []
                # for i in range(len(probs)):
                #     A  = self.compute_fisher_information(probs[i])
                #     fisher_prob.append(A)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                batch_loss.append(loss.item())


                #删除图片
                # if (len(self.idxs) > target * 0.1):
                #      for idx, prob in enumerate(max_probs):
                #          if prob > self.args.ML and fisher_prob[idx] > self.args.FN:
                #              indices_to_remove.append(idx)
                #      removed_images = set()
                #      for idx in indices_to_remove:
                #          # 检查索引是否在范围内，以防止越界访问
                #          if idx < len(indices_in_idxs1):
                #          # 根据索引从 indices_in_idxs1 中获取元素，并添加到 removed_images 列表中
                #              removed_images.add(indices_in_idxs1[idx].item())
                #          # 从索引列表中移除对应要删除的图片的索引
                #      self.idxs = [idx for idx in self.idxs if idx not in removed_images]

            #计算epoch准确率
            accuracy = 100 * correct / total
            epoch_accuracy.append(accuracy)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # # 计算训练时间
        # training_time = time.time() - start_time_training
        # print(f"训练时间: {training_time:.4f} 秒")
        average_loss = sum(epoch_loss) / len(epoch_loss)
        average_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)

        return net.state_dict(), average_loss, average_accuracy, self.idxs, count


    def compute_sensitivity(self, parameters):
        max_norm = max(p.grad.norm(2).item() for p in parameters)
        return max_norm

    # def compute_fisher_information(self, probs):
    #
    #     # 将概率预测张量从GPU移动到CPU
    #     probs_cpu = probs.cpu().detach().numpy()
    #
    #     # 计算似然函数的一阶导数（Score function）
    #     score_function = np.gradient(np.log(probs_cpu))
    #
    #     # 计算Score function 的方差，即Fisher信息
    #     fisher = np.var(score_function)
    #
    #     return fisher






