import torch
import os
import pickle
import glob
import h5py as h5
import numpy as np
from torch.nn import functional as F
from models.protonet import ProtoNet
from utils.cuda_util import cast_cuda
from utils.checkpoint_util import load_model_checkpoint
from dataloaders.loader import sample_pointcloud

# 模型预测
def pc_vis(args):
    model = ProtoNet(args)
    model.cuda()
    model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')

    save_path = os.path.join('../vis/scannet/S%d' % args.cvfold, 'episode5')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    [support_x, support_y, query_x, query_y] = creat_episode(save_path)
    model.eval()
    with torch.no_grad():
        logits, loss = model(support_x, support_y, query_x, query_y)
        pred = F.softmax(logits, dim=1).argmax(dim=1)  # (n_queries, num_points)

        # 生成用于可视化的预测数据
        query_pred_data = torch.cat([query_x.transpose(1, 2), pred.unsqueeze(2)], dim=2)  # (n_queries, num_points, in_channels+1)
        pt0 = query_pred_data[0].cpu().numpy()
        # pt1 = query_pred_data[1].cpu().numpy()
        np.savetxt(os.path.join(save_path, 'pred.txt'), pt0)
        # np.savetxt(os.path.join(save_path, 'pred1.txt'), pt1)

        correct = torch.eq(pred, query_y).sum().item()
        accuracy = correct / (query_y.shape[0] * query_y.shape[1])

    print(accuracy)

# 从静态的h5文件中读取每个episode的数据
def read_episode(save_path, epi_num):
    file_name = '../datasets/ScanNet/blocks_bs1_s1/S_0_N_2_K_1_test_episodes_100_pts_2048_vis/%d.h5' %epi_num
    data_file = h5.File(file_name, 'r')
    support_ptclouds = data_file['support_ptclouds'][:]
    support_masks = data_file['support_masks'][:]
    query_ptclouds = data_file['query_ptclouds'][:]  # (n_queries, num_points, in_channels)
    query_labels = data_file['query_labels'][:]  # (n_queries, num_points)
    sampled_classes = data_file['sampled_classes'][:]

    # 生成用于可视化的原始点云数据
    labels = np.expand_dims(query_labels, 2) # (n_queries, num_points, 1)
    query_data = np.concatenate((query_ptclouds, labels), 2) # (n_queries, num_points, in_channels+1)
    pt1 = query_data[0]  # (num_points, in_channels+1)
    pt2 = query_data[1]  # (num_points, in_channels+1)
    data_path = os.path.join(save_path, 'raw')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    np.savetxt(os.path.join(data_path, 'query1.txt'), pt1)
    np.savetxt(os.path.join(data_path, 'query2.txt'), pt2)

    data = [torch.from_numpy(support_ptclouds).transpose(2, 3), torch.from_numpy(support_masks),
            torch.from_numpy(query_ptclouds).transpose(1, 2),
            torch.from_numpy(query_labels.astype(np.int64))]
    cast_cuda(data)

    return data, sampled_classes

# 自己构建一个完整场景数据的episode
def creat_episode(save_path):
    scene_name = ['scene0027_00', 'scene0013_00', 'scene0000_01']
    data_path = '../datasets/ScanNet/scenes'
    num_point = 10000
    PC_AUGMENT_CONFIG = {'scale': 0,
                         'rot': 1,
                         'mirror_prob': 0,
                         'jitter': 1
                         }
    pc_attribs = 'xyzrgbXYZ'
    pc_augm = True
    sampled_classes = [8, 6]

    # 构建支持集
    support_ptclouds_0, support_labels_0 = sample_pointcloud(data_path, num_point, pc_attribs, pc_augm,
                                                             PC_AUGMENT_CONFIG,
                                                             scene_name[0], sampled_classes, sampled_classes[0],
                                                             support=True)
    support_data_0 = np.concatenate((support_ptclouds_0, np.expand_dims(support_labels_0, 1)), 1)
    # print(support_data_0.shape)
    np.savetxt(os.path.join(save_path, 'support0.txt'), support_data_0)
    support_ptclouds_1, support_labels_1 = sample_pointcloud(data_path, num_point, pc_attribs, pc_augm,
                                                             PC_AUGMENT_CONFIG,
                                                             scene_name[1], sampled_classes, sampled_classes[1],
                                                             support=True)
    support_data_1 = np.concatenate((support_ptclouds_1, np.expand_dims(support_labels_1, 1)), 1)
    # print(support_data_1.shape)
    np.savetxt(os.path.join(save_path, 'support1.txt'), support_data_1)
    support_ptclouds = np.stack([support_ptclouds_0, support_ptclouds_1], 0)
    support_ptclouds = np.expand_dims(support_ptclouds, 1).astype(np.float32)
    support_labels = np.stack([support_labels_0, support_labels_1], 0)
    support_masks = np.expand_dims(support_labels, 1).astype(np.int32)
    # print(support_ptclouds.shape)
    # print(support_masks.shape)

    # 构建查询集
    query_ptclouds_0, query_labels_0 = sample_pointcloud(data_path, num_point, pc_attribs, pc_augm,
                                                         PC_AUGMENT_CONFIG,
                                                         scene_name[2], sampled_classes, sampled_classes[0],
                                                         random_sample=True)
    # query_ptclouds_1, query_labels_1 = sample_pointcloud(data_path, num_point, pc_attribs, pc_augm,
    #                                                      PC_AUGMENT_CONFIG,
    #                                                      scene_name[2], sampled_classes, sampled_classes[1],
    #                                                      random_sample=True)
    query_ptclouds_1 = np.copy(query_ptclouds_0)
    query_labels_1 = np.copy(query_labels_0)
    query_ptclouds = np.stack([query_ptclouds_0, query_ptclouds_1], 0).astype(np.float32)
    query_labels = np.stack([query_labels_0, query_labels_1], 0).astype(np.int64)
    # print(query_ptclouds.shape)
    # print(query_labels.shape)

    # 生成用于可视化的原始点云数据
    labels = np.expand_dims(query_labels, 2)  # (n_queries, num_points, 1)
    query_data = np.concatenate((query_ptclouds, labels), 2)  # (n_queries, num_points, in_channels+1)
    pt = query_data[0]  # (num_points, in_channels+1)
    np.savetxt(os.path.join(save_path, '{}.txt'.format(scene_name[2])), pt)

    data = [torch.from_numpy(support_ptclouds).transpose(2, 3), torch.from_numpy(support_masks),
            torch.from_numpy(query_ptclouds).transpose(1, 2),
            torch.from_numpy(query_labels.astype(np.int64))]
    cast_cuda(data)

    return data

def npy2txt():
    scene_paths = glob.glob('../datasets/ScanNet/scenes/data/*.npy')
    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)[:-4]
        data = np.load(scene_path)
        print(data.shape)
        np.savetxt('../vis/scannet/GT/{}.txt'.format(scene_name), data)

if __name__ == '__main__':
    npy2txt()