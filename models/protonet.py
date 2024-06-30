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
        self.weight = args.weight

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        # self.criterion_generator = GMMNLoss(sigma=[2, 5, 10, 20, 40, 80], cuda=True).build_loss()

        # self.proto_weight = nn.Sequential(nn.Linear(2, 32),
        #                                   nn.ReLU(),
        #                                   nn.Linear(32, 2),
        #                                   nn.Softmax(-1))

        # self.bn = nn.BatchNorm1d(32)
        # self.fc = nn.Conv1d(self.k_shot, 1, 1)
        # self.fc2 = nn.Sequential(nn.Conv1d(2, 32, 1),
        #                          self.bn,
        #                          nn.ReLU())
        # self.fc3 = nn.Linear(32, 1)
        # self.fc4 = nn.Conv1d(self.k_shot, 1, 1)
        #
        # self.q = nn.Sequential(nn.Linear(32, 32),
        #                        nn.ReLU())
        # self.k = nn.Sequential(nn.Linear(32, 32),
        #                        nn.ReLU())
        # self.v = nn.Sequential(nn.Linear(32, 32),
        #                        nn.ReLU())
        #

        # self.q = nn.Linear(192, 192)
        # self.k = nn.Linear(192, 192)

        # self.conv1 = nn.Conv1d(self.n_way+1, 1, 1)
        # self.bn = nn.BatchNorm1d(32)
        # self.conv2 = nn.Sequential(nn.Conv1d(2, 32, 1),
        #                            self.bn,
        #                            nn.ReLU())
        # self.conv3 = nn.Conv1d(32, 1, 1)
        # self.temperature = 32 ** 0.5
        # self.q = nn.Linear(32, 32)
        # self.k = nn.Linear(32, 32)
        # self.v = nn.Linear(32, 32)
        # self.dropout = nn.Dropout(0.1)

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
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, support_bg_feat)
        prototypes = [bg_prototype] + fg_prototypes

        # self-construction
        regulize_loss = self.sup_regulize_Loss(prototypes, support_feat, fg_mask, bg_mask)

        # non-parametric metric learning
        similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]
        query_pred = torch.stack(similarity, dim=1) #(n_queries, n_way+1, num_points)

        # prototypes iteration
        upt_proto = self.getFusionPrototype(query_feat, query_pred, torch.stack(prototypes, dim=0))
        prototypes = [upt_proto[i] for i in range(self.n_way + 1)]
        similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]
        query_pred = torch.stack(similarity, dim=1)  # (n_queries, n_way+1, num_points)

        loss = self.computeCrossEntropyLoss(query_pred, query_y)
        align_loss = self.alignLoss_trans(query_feat, query_pred, support_feat, fg_mask, bg_mask)
        # euc_loss = self.euclideanLoss(query_feat, query_pred, support_fg_feat, bg_prototype)
        # que_regulize_loss = self.que_regulize_Loss(query_pred, query_feat, query_y)


        return query_pred, loss + regulize_loss + align_loss

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

    def getFusionPrototype(self, query_feat, pred, prototypes, isfirst=True):
        """
        Args:
            query_feat: embedding features for query images
                        shape: n_queries x C x num_points
            pred: predicted segmentation score
                        shape: n_queries x (1 + way) x num_points
            prototypes: prototypes from support feature
                        shape: (1 + way) x C
        """
        n_way, k_shot = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # (n_queries, 1, num_points)
        binary_masks = [pred_mask == i for i in range(1 + n_way)]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(query_feat.unsqueeze(1) * pred_mask, dim=(0, 3)) / \
                         (pred_mask.sum(dim=(0, 3)) + 1e-5)  # (n_way+1, feat_dim)

        # updatad by channel weight
        cat_proto = torch.cat([qry_prototypes.unsqueeze(1), prototypes.unsqueeze(1)], dim=1).permute(0, 2, 1)  # (n_way+1, feat_dim, 2)
        if isfirst:
            proto_weight = self.proto_weight(cat_proto)
        else:
            proto_weight = self.proto_weight2(cat_proto)
        upper_proto = torch.sum(cat_proto * proto_weight, -1)

        # updated by instance weight
        # sim = cos_sim(prototypes, qry_prototypes)  # (n_way + 1, n_way + 1)
        # p2q_sim = F.softmax(sim, dim=-1)
        # lower_proto = torch.matmul(p2q_sim, qry_prototypes)  # (n_way + 1, feat_dim)

        upt_prototypes = upper_proto
        return upt_prototypes

    # def getSCIPrototype(self, fg_feat, bg_feat):
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
    #     fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
    #     bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)
    #
    #     fg_prototypes = torch.stack(fg_prototypes, dim=0)  # (n_way, feat_dim)
    #     # bg_prototype = bg_prototype.view(1, -1)  # (1, feat_dim)
    #     # prototypes = torch.cat([fg_prototypes, bg_prototype], dim=0)  # (n_way+1, feat_dim)
    #     prototypes = fg_prototypes
    #
    #     feature = prototypes.unsqueeze(1)  # (n_way+1, 1, feat_dim)
    #     q = self.q(prototypes)  # (n_way+1, feat_dim)
    #     k = self.k(prototypes)  # (n_way+1, feat_dim)
    #     R = torch.matmul(q.unsqueeze(-1), k.unsqueeze(1))  # (n_way+1, feat_dim, feat_dim)
    #     R_hat = F.softmax(R, 1)  # (n_way+1, feat_dim, feat_dim)
    #     v = torch.matmul(feature, R_hat)  # (n_way+1, 1, feat_dim)
    #     upd_feat = v.squeeze() + prototypes  # (n_way+1, feat_dim)
    #
    #     fg_prototypes = [upd_feat[i, ...] for i in range(self.n_way)]
    #     # bg_prototype = upd_feat[-1]
    #
    #     return fg_prototypes, bg_prototype

    # def getSCIPlusPrototype(self, fg_feat, bg_feat):
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
    #     fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
    #     bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)
    #
    #     fg_prototypes = torch.stack(fg_prototypes, dim=0)  # (n_way, feat_dim)
    #     bg_prototype = bg_prototype.view(1, -1)  # (1, feat_dim)
    #     prototypes = torch.cat([fg_prototypes, bg_prototype], dim=0)  # (n_way+1, feat_dim)
    #
    #     task_aware_feat = self.conv1(prototypes.unsqueeze(0)).repeat(self.n_way+1, 1, 1)  # (n_way+1, 1, feat_dim)
    #     fusion_feat = torch.cat([prototypes.unsqueeze(1), task_aware_feat], dim=1)  # (n_way+1, 2, feat_dim)
    #     fusion_feat = self.conv2(fusion_feat).transpose(1, 2)  # (n_way+1, feat_dim, 32)
    #     q = self.q(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     k = self.k(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     v = self.v(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     Rs = torch.matmul(q / self.temperature, k.transpose(1, 2))  # (n_way+1, feat_dim, feat_dim)
    #     Rs = self.dropout(F.softmax(Rs, 2))
    #     v_hat = torch.matmul(Rs, v)  # (n_way+1, feat_dim, 32)
    #     refine_feat = self.conv3(v_hat.transpose(1, 2)).squeeze()  # (n_way+1, feat_dim)
    #     upt_proto = prototypes + refine_feat  # (n_way+1, feat_dim)
    #     fg_prototypes = [upt_proto[i, ...] for i in range(self.n_way)]
    #     bg_prototype =  upt_proto[-1]
    #
    #     return fg_prototypes, bg_prototype
    #
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
    #     supplement_bg_feat = self.fc4(bg_feat).mean(0, keepdim=True)  # (1, 1, feat_dim)
    #     bg_feat = bg_feat.view(1, self.n_way * self.k_shot, -1)
    #     bg_prototype = bg_feat.mean(1)  # (1, feat_dim)
    #
    #     prototypes = torch.cat([fg_prototypes, bg_prototype], dim=0)  # (n_way+1, feat_dim)
    #     supplement_feat = torch.cat([supplement_fg_feat, supplement_bg_feat], dim=0)  # (n_way+1, 1, feat_dim)
    #
    #     p2s = torch.cat([prototypes.unsqueeze(1), supplement_feat], dim=1)  # (n_way+1, 2, feat_dim)
    #     fusion_feat = self.fc2(p2s).transpose(1, 2)  # (n_way+1, feat_dim, 32)
    #     q = self.q(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     k = self.k(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     v = self.v(fusion_feat)  # (n_way+1, feat_dim, 32)
    #     Rs = torch.matmul(q, k.transpose(1, 2))  # (n_way+1, feat_dim, feat_dim)
    #     Rs = F.softmax(Rs, 2)
    #     v_hat = torch.matmul(Rs, v)  # (n_way+1, feat_dim, 32)
    #     refine_feat = self.fc3(v_hat).squeeze()  # (n_way+1, feat_dim)
    #     upt_proto =   prototypes + refine_feat  # (n_way+1, feat_dim)
    #
    #     fg_prototypes = [upt_proto[i, ...] for i in range(self.n_way)]
    #     bg_prototype = upt_proto[-1]
    #     # bg_prototype =  bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)
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
        """
        Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)

    def alignLoss_trans(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)
                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss

    def sup_regulize_Loss(self, prototype_supp, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype suppoort self alignment branch

        Args:
            prototypes: embedding features for query images
                expect shape: N x C x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            prototypes = [prototype_supp[0], prototype_supp[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss

    # def que_regulize_Loss(self, query_feat, pred, query_y):
    #     """
    #     Compute the loss for the query prototypes self alignment branch
    #
    #     Args:
    #         query_feat: embedding features for query images
    #                     shape: n_queries x C x num_points
    #         pred: predicted segmentation score
    #                     shape: n_queries x (1 + way) x num_points
    #         query_y: the ground truth of query
    #                     shape: n_queries x num_points
    #
    #     """
    #     n_way, k_shot = self.n_way, self.k_shot
    #
    #     # Mask and get query prototype
    #     pred_mask = pred.argmax(dim=1, keepdim=True)  # (n_queries, 1, num_points)
    #     binary_masks = [pred_mask == i for i in range(1 + n_way)]
    #     pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
    #     qry_prototypes = torch.sum(query_feat.unsqueeze(1) * pred_mask, dim=(0, 3)) / \
    #                      (pred_mask.sum(dim=(0, 3)) + 1e-5)  # (n_way+1, feat_dim)
    #
    #     prototypes = [qry_prototypes[i] for i in range(self.n_way + 1)]
    #     similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]
    #     query_pred = torch.stack(similarity, dim=1)  # (n_queries, n_way+1, num_points)
    #     loss = self.computeCrossEntropyLoss(query_pred, query_y)
    #     return loss


    def euclideanLoss(self, query_feat, pred, support_fg_feat, bg_prototype, weight=0.1):
        """
        Calculate the euclidean Loss for support prototypes and query prototypes

        Args:
            query_feat: embedding features for query images
                        shape: n_queries x C x num_points
            pred: predicted segmentation score
                        shape: n_queries x (1 + way) x num_points
            support_fg_feat:
                        shape: n_way x k_shot x feat_dim
        """
        n_way, k_shot = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # (n_queries, 1, num_points)
        binary_masks = [pred_mask == i for i in range(1 + n_way)]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(query_feat.unsqueeze(1) * pred_mask, dim=(0, 3)) / \
                         (pred_mask.sum(dim=(0, 3)) + 1e-5)  # (n_way+1, feat_dim)

        # Compute the prototype loss of each support sample
        loss = 0
        for way in range(n_way):
            # query_bg_proto = qry_prototypes[0].unsqueeze(0)
            query_fg_proto = qry_prototypes[way+1].unsqueeze(0)
            for shot in range(k_shot):
                # support_bg_proto = support_bg_feat[way, shot].unsqueeze(0)
                support_fg_proto = support_fg_feat[way, shot].unsqueeze(0)
                # bg_loss = F.pairwise_distance(query_bg_proto, support_bg_proto)
                # fg_loss = F.pairwise_distance(query_fg_proto, support_fg_proto)
                fg_loss = F.pairwise_distance(query_fg_proto, support_fg_proto)
                loss = loss + fg_loss
        query_bg_proto = qry_prototypes[0].unsqueeze(0)
        bg_loss = F.pairwise_distance(bg_prototype.unsqueeze(0), query_bg_proto)
        if(k_shot == 1):
            return loss / k_shot + weight * bg_loss
        else:
            return loss / k_shot

    # def euclideanLoss(self, query_feat, pred, prototypes, weight=0.1):
    #     """
    #     Calculate the euclidean Loss for support prototypes and query prototypes
    #
    #     Args:
    #         query_feat: embedding features for query images
    #                     shape: n_queries x C x num_points
    #         pred: predicted segmentation score
    #                     shape: n_queries x (1 + way) x num_points
    #         prototypes: prototypes from support feature
    #                     shape: (1 + way) x C
    #     """
    #     n_ways, n_shots = self.n_way, self.k_shot
    #
    #     # Mask and get query prototype
    #     pred_mask = pred.argmax(dim=1, keepdim=True)  # (n_queries, 1, num_points)
    #     binary_masks = [pred_mask == i for i in range(1 + n_ways)]
    #     pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
    #     qry_prototypes = torch.sum(query_feat.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
    #
    #     # Compute the prototypes loss
    #     bg_prototype = prototypes[:1, :]
    #     fg_prototypes = prototypes[1:, :]
    #     bg_qry_prototype = qry_prototypes[:1, :]
    #     fg_qry_prototypes = qry_prototypes[1:, :]
    #
    #     bg_loss = F.pairwise_distance(bg_prototype, bg_qry_prototype)
    #     fg_loss = torch.sum(F.pairwise_distance(fg_prototypes, fg_qry_prototypes))
    #     loss = fg_loss + weight * bg_loss
    #
    #     return loss
    #
    # def cosineLoss(self, query_feat, pred, prototypes, weight=0.1):
    #     """
    #     Calculate the cosine Loss for support prototypes and query prototypes
    #
    #     Args:
    #         query_feat: embedding features for query images
    #                     shape: n_queries x C x num_points
    #         pred: predicted segmentation score
    #                     shape: n_queries x (1 + way) x num_points
    #         prototypes: prototypes from support feature
    #                     shape: (1 + way) x C
    #
    #     """
    #     # Mask and get query prototype
    #     pred_mask = pred.argmax(dim=1, keepdim=True)  # (n_queries, 1, num_points)
    #     binary_masks = [pred_mask == i for i in range(1 + self.n_way)]
    #     pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
    #     qry_prototypes = torch.sum(query_feat.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
    #
    #     # Compute the prototypes loss
    #     bg_prototype = prototypes[:1, :]
    #     fg_prototypes = prototypes[1:, :]
    #     bg_qry_prototype = qry_prototypes[:1, :]
    #     fg_qry_prototypes = qry_prototypes[1:, :]
    #     bg_loss = 1 - F.cosine_similarity(bg_prototype, bg_qry_prototype)
    #     fg_loss = self.n_way * 1 - torch.sum(F.cosine_similarity(fg_prototypes, fg_qry_prototypes))
    #     loss = fg_loss + weight * bg_loss
    #
    #     return loss
    #
    # def gmmnLoss(self, query_feat, pred, prototypes, weight=0.1):
    #     """
    #     Calculate the euclidean Loss for support prototypes and query prototypes
    #
    #     Args:
    #         query_feat: embedding features for query images
    #                     shape: n_queries x C x num_points
    #         pred: predicted segmentation score
    #                     shape: n_queries x (1 + way) x num_points
    #         prototypes: prototypes from support feature
    #                     shape: (1 + way) x C
    #
    #     """
    #     n_ways, n_shots = self.n_way, self.k_shot
    #
    #     # Mask and get query prototype
    #     pred_mask = pred.argmax(dim=1, keepdim=True)  # (n_queries, 1, num_points)
    #     binary_masks = [pred_mask == i for i in range(1 + n_ways)]
    #     pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
    #     qry_prototypes = torch.sum(query_feat.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
    #
    #     # Compute the prototypes loss
    #     bg_prototype = prototypes[:1, :]
    #     fg_prototypes = prototypes[1:, :]
    #     bg_qry_prototype = qry_prototypes[:1, :]
    #     fg_qry_prototypes = qry_prototypes[1:, :]
    #
    #     bg_loss = self.criterion_generator(bg_qry_prototype, bg_prototype)
    #     fg_loss = self.criterion_generator(fg_qry_prototypes, fg_prototypes)
    #     loss = fg_loss + weight * bg_loss
    #
    #     return loss