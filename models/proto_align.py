""" Prototypical Network

Author: Zhao Na, 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.attention import SelfAttention


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x


class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = args.dist_method
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        # self.bn = nn.BatchNorm1d(32)
        # self.fc = nn.Conv1d(5, 1, 1)
        # self.fc2 = nn.Sequential(nn.Conv1d(2, 32, 1),
        #                          self.bn,
        #                          nn.ReLU())
        # self.fc3 = nn.Linear(32, 1)
        # self.q = nn.Sequential(nn.Linear(32, 32),
        #                        nn.ReLU())
        # self.k = nn.Sequential(nn.Linear(32, 32),
        #                        nn.ReLU())
        # self.v = nn.Sequential(nn.Linear(32, 32),
        #                        nn.ReLU())

        # self.q = nn.Linear(192, 192)
        # self.k = nn.Linear(192, 192)

        self.conv1 = nn.Conv1d(self.n_way+1, 1, 1)
        self.bn = nn.BatchNorm1d(32)
        self.conv2 = nn.Sequential(nn.Conv1d(2, 32, 1),
                                   self.bn,
                                   nn.ReLU())
        self.conv3 = nn.Conv1d(32, 1, 1)
        self.temperature = 32 ** 0.5
        self.q = nn.Linear(32, 32)
        self.k = nn.Linear(32, 32)
        self.v = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.1)


    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)#(n_support, feat_dim, num_points)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)

        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        support_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)

        # prototype learning
        # fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, support_bg_feat)
        # fg_prototypes, bg_prototype = self.getSCIPrototype(support_fg_feat, support_bg_feat)
        fg_prototypes, bg_prototype = self.getSCIPlusPrototype(support_fg_feat, support_bg_feat)

        prototypes = [bg_prototype] + fg_prototypes

        # non-parametric metric learning
        similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]

        query_pred = torch.stack(similarity, dim=1) #(n_queries, n_way+1, num_points)
        loss = self.computeCrossEntropyLoss(query_pred, query_y)
        return query_pred, loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            return torch.cat((feat_level1, att_feat, feat_level3), dim=1)
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    # def getCompensateMaskedFeature(self, feat, mask):
    #     """
    #     Extract foreground and background features via masked average pooling
    #
    #     Args:
    #         feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
    #         mask: binary mask, shape: (n_way, k_shot, num_points)
    #     Return:
    #         masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
    #     """
    #     feat = feat.permute(0, 1, 3, 2).contiguous().view(self.n_way*self.k_shot, self.n_points, -1)
    #     mask = mask.view(self.n_way*self.k_shot, -1)
    #     masked_feat = []
    #     for i in range(self.n_way*self.k_shot):
    #         one_shot_feat = feat[i]  # (num_points, dim)
    #         one_shot_mask = mask[i]  # (num_points,)
    #         ground_mask = one_shot_mask.nonzero().view(-1)  # (ground_num,)
    #         ground_feat = torch.index_select(one_shot_feat, 0, ground_mask)  # (ground_num, dim)
    #         ground_feat = ground_feat.unqueeze(0)  # (1, ground_num, dim)
    #         raw_mask_feat = ground_feat.mean(1)  # (1, dim)
    #         supplement_feat = self.fc(ground_feat)  # (1, 1, 1dim)
    #         p2s = torch.cat([raw_mask_feat.unsqueeze(1), supplement_feat], dim=1)  # (1, 2, feat_dim)
    #         fusion_feat = self.fc2(p2s).transpose(1, 2)  # (1, feat_dim, 32)
    #         q = self.q(fusion_feat)  # (1, feat_dim, 32)
    #         k = self.k(fusion_feat)  # (1, feat_dim, 32)
    #         v = self.v(fusion_feat)  # (1, feat_dim, 32)
    #         Rs = torch.matmul(q, k.transpose(1, 2))  # (n_way, feat_dim, feat_dim)
    #         Rs = F.softmax(Rs, 2)
    #         v_hat = torch.matmul(Rs, v)  # (1, feat_dim, 32)
    #         refine_feat = self.fc3(v_hat).squeeze()  # (1, feat_dim)
    #         upt_mask_feat = raw_mask_feat + refine_feat
    #         masked_feat.append(upt_mask_feat)
    #     masked_feat = torch.stack(masked_feat, dim=0).view(self.n_way, self.k_shot, -1)  # (n_way, k_shot, dim)
    #     return masked_feat

    # def getMaskedMPMFeatures(self, feat, mask):
    #     """
    #     Extract foreground and background features via masked average pooling
    #
    #     Args:
    #         feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
    #         mask: binary mask, shape: (n_way, k_shot, num_points)
    #     Return:
    #         masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
    #     """
    #     feat = feat.permute(0, 1, 3, 2).contiguous().view(self.n_way*self.k_shot, self.n_points, -1)
    #     mask = mask.view(self.n_way*self.k_shot, -1)
    #     masked_feat = []
    #     for i in range(self.n_way*self.k_shot):
    #         one_shot_feat = feat[i]  # (num_points, dim)
    #         one_shot_mask = mask[i]  # (num_points,)
    #         ground_mask = one_shot_mask.nonzero().view(-1)  # (ground_num,)
    #         ground_feat = torch.index_select(one_shot_feat, 0, ground_mask)  # (ground_num, dim)
    #         ground_feat = ground_feat[0:(len(ground_mask)-len(ground_mask)%16), :]
    #         ground_feat = ground_feat.unsqueeze(0).permute(0, 2, 1)  # (1, dim, ground_num)
    #         batch, feat_dim, n_points = ground_feat.shape
    #         bin_feat = []
    #         for bin in self.bin_num:
    #             z = ground_feat.view(batch, feat_dim, bin, -1)
    #             z_max, _ = z.max(3)
    #             z = z.mean(3) + z_max
    #             bin_feat.append(z)
    #         bin_feat = torch.cat(bin_feat, 2).contiguous()  # (batchsize, 192, 63)
    #         bin_feat = self.transform(bin_feat).squeeze()
    #         masked_feat.append(bin_feat)
    #     masked_feat = torch.stack(masked_feat, dim=0).view(self.n_way, self.k_shot, -1)  # (n_way*k_shot, dim)
    #
    #     return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype =  bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def getSCIPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)

        fg_prototypes = torch.stack(fg_prototypes, dim=0)  # (n_way, feat_dim)
        bg_prototype = bg_prototype.view(1, -1)  # (1, feat_dim)
        prototypes = torch.cat([fg_prototypes, bg_prototype], dim=0)  # (n_way+1, feat_dim)

        feature = prototypes.unsqueeze(1)  # (n_way+1, 1, feat_dim)
        q = self.q(prototypes)  # (n_way+1, feat_dim)
        k = self.k(prototypes)  # (n_way+1, feat_dim)
        R = torch.matmul(q.unsqueeze(-1), k.unsqueeze(1))  # (n_way+1, feat_dim, feat_dim)
        R_hat = F.softmax(R, 1)  # (n_way+1, feat_dim, feat_dim)
        v = torch.matmul(feature, R_hat)  # (n_way+1, 1, feat_dim)
        upd_feat = v.squeeze() + prototypes  # (n_way+1, feat_dim)

        fg_prototypes = [upd_feat[i, ...] for i in range(self.n_way)]
        bg_prototype = upd_feat[-1]

        return fg_prototypes, bg_prototype

    def getSCIPlusPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)

        fg_prototypes = torch.stack(fg_prototypes, dim=0)  # (n_way, feat_dim)
        bg_prototype = bg_prototype.view(1, -1)  # (1, feat_dim)
        prototypes = torch.cat([fg_prototypes, bg_prototype], dim=0)  # (n_way+1, feat_dim)

        task_aware_feat = self.conv1(prototypes.unsqueeze(0)).repeat(self.n_way+1, 1, 1)  # (n_way+1, 1, feat_dim)
        fusion_feat = torch.cat([prototypes.unsqueeze(1), task_aware_feat], dim=1)  # (n_way+1, 2, feat_dim)
        fusion_feat = self.conv2(fusion_feat).transpose(1, 2)  # (n_way+1, feat_dim, 32)
        q = self.q(fusion_feat)  # (n_way+1, feat_dim, 32)
        k = self.k(fusion_feat)  # (n_way+1, feat_dim, 32)
        v = self.v(fusion_feat)  # (n_way+1, feat_dim, 32)
        Rs = torch.matmul(q / self.temperature, k.transpose(1, 2))  # (n_way+1, feat_dim, feat_dim)
        Rs = self.dropout(F.softmax(Rs, 2))
        v_hat = torch.matmul(Rs, v)  # (n_way+1, feat_dim, 32)
        refine_feat = self.conv3(v_hat.transpose(1, 2)).squeeze()  # (n_way+1, feat_dim)
        upt_proto = prototypes + refine_feat  # (n_way+1, feat_dim)
        fg_prototypes = [upt_proto[i, ...] for i in range(self.n_way)]
        bg_prototype =  upt_proto[-1]

        return fg_prototypes, bg_prototype

    # def getCompensateProto(self, fg_feat, bg_feat):
    #     """
    #     Average the features to obtain the prototype
    #
    #     Args:
    #         fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
    #         bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
    #     Returns:
    #         fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
    #         bg_prototype: background prototype, a vector with shape (feat_dim,)
    #     """
    #     fg_prototypes = fg_feat.mean(1)  # (n_way, feat_dim)
    #     supplement_fg_feat = self.fc(fg_feat)  # (n_way, 1, feat_dim)
    #     # bg_feat = bg_feat.view(1, self.n_way * self.k_shot, -1)
    #     # bg_prototype = bg_feat.mean(1)  # (1, feat_dim)
    #     # supplement_bg_feat = self.fc4(bg_feat)  # (1, 1, feat_dim)
    #     #
    #     # prototypes = torch.cat([fg_prototypes, bg_prototype], dim=0)  # (n_way+1, feat_dim)
    #     # supplement_feat = torch.cat([supplement_fg_feat, supplement_bg_feat], dim=0)  # (n_way+1, 1, feat_dim)
    #
    #     p2s = torch.cat([fg_prototypes.unsqueeze(1), supplement_fg_feat], dim=1)  # (n_way+1, 2, feat_dim)
    #     fusion_feat = self.fc2(p2s).transpose(1, 2)  # (n_way+1, feat_dim, 32)
    #     q = self.q(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     k = self.k(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     v = self.v(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     Rs = torch.matmul(q, k.transpose(1, 2))  # (n_way+1, feat_dim, feat_dim)
    #     Rs = F.softmax(Rs, 2)
    #     v_hat = torch.matmul(Rs, v)  # (n_way+1, feat_dim, 32)
    #     refine_feat = self.fc3(v_hat).squeeze()  # (n_way+1, feat_dim)
    #     upt_proto = fg_prototypes + refine_feat  # (n_way+1, feat_dim)
    #
    #     fg_prototypes = [upt_proto[i, ...] for i in range(self.n_way)]
    #     bg_prototype =  bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
    #
    #     return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            # similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
            similarity = - F.pairwise_distance(feat.transpose(1, 2), prototype[None, None, ...], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)
