# --------------------------------------------------------
# IFCVQA（Image Feature Cropping VQA）
# Licensed under The MIT License [see LICENSE for details]
# Written by Ziteng Xu
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted, att_map = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted, att_map  # 返回注意力映射

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value), att_map  # 返回注意力映射



# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        # 获取注意力映射
        att_output, att_map = self.mhatt(x, x, x, x_mask)
        
        x = self.norm1(x + self.dropout1(att_output))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x, att_map  # 返回注意力映射



# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):

        att_output, _ = self.mhatt1(x, x, x, x_mask)
        x = self.norm1(x + self.dropout1(att_output))

        att_output, att_map = self.mhatt2(y, y, x, y_mask)
        x = self.norm2(x + self.dropout2(att_output))

        """
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))
        """

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x, att_map


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        for idx, enc in enumerate(self.enc_list):
            x, att_map = enc(x, x_mask)

        for idx, dec in enumerate(self.dec_list):
            y, att_map = dec(y, x, y_mask, x_mask)

            if idx == 3:
                contribution_score = self.compute_score(att_map, y)  # Score culculating
                topk_indices = self.compute_threshold_att_inplace(contribution_score, 0.6, 3)  # Mask Generating
                y = self.select_topk_features(y, topk_indices, 0.1) # Feature cropping
                #input()

        return x, y

    def compute_score(self, att_map, features):
        
        att_map_avg = att_map.mean(dim=1)
        
        contribution = torch.sum(att_map_avg, dim=2)

        return contribution

    def compute_threshold_att_inplace(self, att_map, threshold_ratio = 0.6, k = 3):
        
        topk_values = att_map.topk(k=k, dim=1, largest=True, sorted=False).values

        avg_topk_values = topk_values.mean(dim=1, keepdim=True)

        threshold = avg_topk_values * threshold_ratio

        att_map[:] = (att_map >= threshold).float()

        return att_map

    def select_topk_features(self, x, topk_indices, scale_factor = 0.1):
        if x.shape[:2] != topk_indices.shape:
            raise ValueError("Index tensor shape must match the first two dimensions of target tensor.")

        mask = topk_indices.unsqueeze(-1)  

        masked_target_tensor = torch.where(mask == 1, x, x * scale_factor)
        
        return masked_target_tensor