#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
from collections import defaultdict

import numpy as np
from torchvision import datasets, transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA






#函数返回一个字典，键是客户端的索引，值是对应客户端所拥有的图像索引的集合
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


#分层采样百分之30独立同分布
# def cifar_iid(dataset, num_users):
#     """
#     从CIFAR-10数据集中按类别分层采样30%后生成独立同分布的客户端数据
#     :param dataset: CIFAR-10数据集
#     :param num_users: 客户端数量
#     :return: dict of image index
#     """
#     num_classes = 10
#     num_samples_per_class = int(len(dataset) / num_classes * 0.3)  # 每个类别保留的样本数量
#
#     # 初始化字典，用于存储每个类别的索引列表
#     class_indices = {i: [] for i in range(num_classes)}
#
#     # 按类别对数据集进行索引
#     for idx, label in enumerate(dataset.targets):
#         class_indices[int(label)].append(idx)
#
#     # 对每个类别进行分层子采样
#     sampled_indices = []
#     for label, indices in class_indices.items():
#         sampled_indices.extend(random.sample(indices, min(num_samples_per_class, len(indices))))
#
#     # 打乱采样的索引
#     np.random.shuffle(sampled_indices)
#
#     # 按照用户数量分配数据
#     num_items = int(len(sampled_indices) / num_users)
#     dict_users = {i: set() for i in range(num_users)}
#     all_idxs = np.array(sampled_indices)
#
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = np.setdiff1d(all_idxs, list(dict_users[i]))
#
#     return dict_users




#普通的cifar非独立同分布
# def cifar_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D. client data from CIFAR10 dataset
#     :param dataset: CIFAR-10 dataset
#     :param num_users: number of clients
#     :return: dict of image index
#     """
#     num_shards, num_imgs = 200, 250
#     idx_shard = [i for i in range(num_shards)]
#     #创建了个字典，键是客户端编号，值是该客户端对应的索引，是个numpy数组
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = dataset.targets
#
#     # sort labels确保每个客户端数据按照标签数据分配
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#
#     return dict_users






##Dirichlet 来生成CIFAR-10的非独立同分布数据
def cifar_noniid(dataset, num_users ):
    """
    使用 Dirichlet 分布生成 CIFAR-10 数据集的非独立同分布（non-IID）数据划分。

    参数：
    - dataset: CIFAR-10 数据集。
    - num_users: 用户（客户端）数量。
    - alpha: Dirichlet 分布的参数（控制非IID程度）。

    返回：
    - dict_users: 一个字典，将用户编号映射到他们的数据样本。
    """
    num_classes = 10
    alpha = 0.5
    num_items = len(dataset) // num_users  # 每个用户的数据样本数（假设数据均匀分配）
    labels = np.array([target for _, target in dataset])  # 提取标签
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}  # 每个类别的样本索引
    dict_users = defaultdict(list)  # 用来存储每个用户的数据样本索引

    # 对每个用户进行样本分配
    for user in range(num_users):
        dirichlet_probs = np.random.dirichlet(np.repeat(alpha, num_classes))  # 从Dirichlet分布获取比例
        class_sizes = (dirichlet_probs * num_items).astype(int)  # 计算每个类别分配的样本数

        # 处理样本数不足或过多的情况，确保每个用户分配的样本总数为 num_items
        diff = num_items - class_sizes.sum()  # 计算总数与目标数量的差值
        if diff != 0:
            adjustment = np.sign(diff)  # 决定增加或减少
            for i in range(num_classes):
                if diff == 0:
                    break
                # 根据差值调整
                class_sizes[i] += adjustment
                diff -= adjustment

        # 为每个类别采样，确保采样数量不超过该类别的剩余样本数量
        for class_id, size in enumerate(class_sizes):
            size = min(size, len(class_indices[class_id]))  # 确保采样大小不超过该类别剩余的样本数
            selected_indices = np.random.choice(class_indices[class_id], size, replace=False)
            dict_users[user].extend(selected_indices)
            class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected_indices)  # 更新剩余样本
    for i in range(num_users):
        num_samples = int(len(dict_users[i]) * 0.5)
        dict_users[i] = set(np.random.choice(dict_users[i], num_samples, replace=False))

    return dict_users







#分层采样非独立同分布
# def cifar_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D. client data from CIFAR10 dataset with stratified subsampling of 30%
#     :param dataset: CIFAR-10 dataset
#     :param num_users: number of clients
#     :return: dict of image index
#     """
#     num_classes = 10
#     num_samples_per_class = int(len(dataset) / num_classes * 0.3)  # 每个类别保留的样本数量
#
#
#
#     # 初始化字典，用于存储每个类别的索引列表
#     class_indices = {i: [] for i in range(num_classes)}
#
#     # 按类别对数据集进行索引
#     for idx, label in enumerate(dataset.targets):
#         class_indices[label].append(idx)
#
#     # 对每个类别进行分层子采样
#     sampled_indices = []
#     for label, indices in class_indices.items():
#         sampled_indices.extend(random.sample(indices, min(num_samples_per_class, len(indices))))
#
#     # 创建索引列表，并排序
#     sampled_indices = np.array(sampled_indices)
#     labels = np.array([dataset.targets[idx] for idx in sampled_indices])
#     idxs_labels = np.vstack((sampled_indices, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     sampled_indices = idxs_labels[0, :].astype(int)
#
#     # 创建客户端数据字典
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#
#     # 随机分配分片到每个客户端
#     num_shards = num_users * 2
#     num_imgs = len(sampled_indices) // num_shards
#     idx_shard = [i for i in range(num_shards)]
#
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], sampled_indices[rand * num_imgs:(rand + 1) * num_imgs]),
#                                            axis=0)
#
#     return dict_users







def svhn_iid(dataset, num_users):
    """
    Sample I.I.D. client data from SVHN dataset
    :param dataset: SVHN dataset
    :param num_users: Number of clients
    :return: dict of image indices for each client
    """
    num_items = len(dataset) // num_users
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    np.random.shuffle(idxs)
    for i in range(num_users):
        start_idx = i * num_items
        end_idx = (i + 1) * num_items
        dict_users[i] = idxs[start_idx:end_idx]
    return dict_users


#普通的svhn的非独立同分布
# def svhn_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D. client data from SVHN dataset
#     :param dataset: SVHN dataset
#     :param num_users: Number of clients
#     :return: dict of image indices for each client
#     """
#     total_images = len(dataset)
#     num_shards = 200
#     num_imgs_per_shard = total_images // num_shards
#     dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
#     idxs = np.arange(total_images)
#     labels = dataset.labels
#
#     # Sort labels to ensure each client gets non-IID data
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :].astype(int)
#
#     # Divide and assign
#     for i in range(num_users):
#         rand_set = np.random.choice(num_shards, 2, replace=False)
#         for rand in rand_set:
#             start_idx = rand * num_imgs_per_shard
#             end_idx = (rand + 1) * num_imgs_per_shard
#             dict_users[i] = np.concatenate((dict_users[i], idxs[start_idx:end_idx]), axis=0)
#
#     return dict_users




##Dirichlet 来生成SVHN的非独立同分布数据
def svhn_noniid(dataset, num_users ):
    """
    使用 Dirichlet 分布生成 SVHN 数据集的非独立同分布（non-IID）数据划分。

    参数：
    - dataset: SVHN 数据集。
    - num_users: 用户（客户端）数量。
    - alpha: Dirichlet 分布的参数（控制非IID程度）。

    返回：
    - dict_users: 一个字典，将用户编号映射到他们的数据样本。
    """
    num_classes = 10
    alpha = 0.5
    num_items = len(dataset) // num_users  # 每个用户的数据样本数（假设数据均匀分配）
    labels = np.array([target for _, target in dataset])  # 提取标签
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}  # 每个类别的样本索引
    dict_users = defaultdict(list)  # 用来存储每个用户的数据样本索引

    # 对每个用户进行样本分配
    for user in range(num_users):
        dirichlet_probs = np.random.dirichlet(np.repeat(alpha, num_classes))  # 从Dirichlet分布获取比例
        class_sizes = (dirichlet_probs * num_items).astype(int)  # 计算每个类别分配的样本数

        # 处理样本数不足或过多的情况，确保每个用户分配的样本总数为 num_items
        diff = num_items - class_sizes.sum()  # 计算总数与目标数量的差值
        if diff != 0:
            adjustment = np.sign(diff)  # 决定增加或减少
            for i in range(num_classes):
                if diff == 0:
                    break
                # 根据差值调整
                class_sizes[i] += adjustment
                diff -= adjustment

        # 为每个类别采样，确保采样数量不超过该类别的剩余样本数量
        for class_id, size in enumerate(class_sizes):
            size = min(size, len(class_indices[class_id]))  # 确保采样大小不超过该类别剩余的样本数
            selected_indices = np.random.choice(class_indices[class_id], size, replace=False)
            dict_users[user].extend(selected_indices)
            class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected_indices)  # 更新剩余样本


    return dict_users













#分层采样百分之30 iid
# def svhn_iid(dataset, num_users):
#
#     num_classes = 10
#     num_samples_per_class = int(len(dataset) / num_classes * 0.3)  # 每个类别保留的样本数量
#
#     # 初始化字典，用于存储每个类别的索引列表
#     class_indices = {i: [] for i in range(num_classes)}
#
#     # 按类别对数据集进行索引
#     for idx, label in enumerate(dataset.labels):
#         class_indices[int(label)].append(idx)
#
#     # 对每个类别进行分层子采样
#     sampled_indices = []
#     for label, indices in class_indices.items():
#         sampled_indices.extend(random.sample(indices, min(num_samples_per_class, len(indices))))
#
#     # 打乱采样的索引
#     np.random.shuffle(sampled_indices)
#
#     # 按照用户数量分配数据
#     num_items = int(len(sampled_indices) / num_users)
#     dict_users = {i: set() for i in range(num_users)}
#     all_idxs = np.array(sampled_indices)
#
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = np.setdiff1d(all_idxs, list(dict_users[i]))
#
#     return dict_users



#分层采样百分之30 non_iid
# def svhn_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D. client data from SVHN dataset with stratified subsampling
#     :param dataset: SVHN dataset
#     :param num_users: Number of clients
#     :return: dict of image indices for each client
#     """
#     num_classes = 10
#     num_samples_per_class = int(len(dataset) / num_classes * 0.3)  # Number of samples to keep per class
#
#     # Initialize dictionary to store indices for each class
#     class_indices = {i: [] for i in range(num_classes)}
#
#     # Index the dataset by class
#     for idx, label in enumerate(dataset.labels):
#         class_indices[label].append(idx)
#
#     # Perform stratified subsampling for each class
#     sampled_indices = []
#     for label, indices in class_indices.items():
#         sampled_indices.extend(random.sample(indices, min(num_samples_per_class, len(indices))))
#
#     # Create list of indices and sort them
#     sampled_indices = np.array(sampled_indices)
#     labels = np.array([dataset.labels[idx] for idx in sampled_indices])
#     idxs_labels = np.vstack((sampled_indices, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     sampled_indices = idxs_labels[0, :].astype(int)
#
#     # Create dictionary for client data
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#
#     # Randomly allocate shards to each client
#     num_shards = num_users * 2
#     num_imgs = len(sampled_indices) // num_shards
#     idx_shard = [i for i in range(num_shards)]
#
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], sampled_indices[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#
#     return dict_users





if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)


