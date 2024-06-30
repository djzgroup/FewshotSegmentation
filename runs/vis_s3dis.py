import torch
import os
import pickle
import h5py as h5
import numpy as np
from torch.nn import functional as F
from models.protonet import ProtoNet
from utils.cuda_util import cast_cuda
from utils.checkpoint_util import load_model_checkpoint

# 模型预测
def pc_vis(args):
    model = ProtoNet(args)
    model.cuda()
    model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')

    epi_num = 1418
    save_path = os.path.join('../vis/s3dis/S%d' % args.cvfold, 'episode%d' % epi_num)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    [support_x, support_y, query_x, query_y], _ = read_episode(save_path, epi_num)
    model.eval()
    with torch.no_grad():
        logits, loss = model(support_x, support_y, query_x, query_y)
        pred = F.softmax(logits, dim=1).argmax(dim=1)  # (n_queries, num_points)

        # 生成用于可视化的预测数据
        query_pred_data = torch.cat([query_x.transpose(1, 2), pred.unsqueeze(2)], dim=2)  # (n_queries, num_points, in_channels+1)
        pt1 = query_pred_data[0].cpu().numpy()
        pt2 = query_pred_data[1].cpu().numpy()
        pred_path = os.path.join(save_path, 'pred')
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
        np.savetxt(os.path.join(pred_path, 'pred1.txt'), pt1)
        np.savetxt(os.path.join(pred_path, 'pred2.txt'), pt2)

        correct = torch.eq(pred, query_y).sum().item()
        accuracy = correct / (query_y.shape[0] * query_y.shape[1])

    print(accuracy)

# 从静态的h5文件中读取每个episode的数据
def read_episode(save_path, epi_num):
    file_name = '../datasets/S3DIS/blocks_bs1_s1/S_1_N_2_K_1_test_episodes_100_pts_2048_vis/%d.h5' %epi_num
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

def npy2txt():
    block_name = 'Area_4_office_13_block_15'
    file_name = '../datasets/S3DIS/blocks_bs1_s1/data/{}.npy'.format(block_name)
    data = np.load(file_name)
    print(data.shape)
    np.savetxt('../vis/s3dis/S1/vis.txt', data)

if __name__ == '__main__':
    npy2txt()