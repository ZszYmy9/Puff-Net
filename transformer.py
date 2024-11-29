import os

import numpy as np
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.new_ps = nn.Conv2d(512, 512, (1, 1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style, mask, content, pos_embed_c, pos_embed_s):

        # content-aware positional embedding
        content_pool = self.averagepooling(content)
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=style.shape[-2:])

        '''
        # position embedding (sin-cos)
        pe = torch.zeros(1024, 512)
        position = torch.arange(0, 1024, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 512, 2).float() * (-math.log(10000.0) / 512))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        pos_embed_c = pe.cuda()
        '''

        # flatten NxCxHxW to HWxNxC
        style = style.flatten(2).permute(2, 0, 1)
        if pos_embed_s is not None:
            pos_embed_s = pos_embed_s.flatten(2).permute(2, 0, 1)

        content = content.flatten(2).permute(2, 0, 1)
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1)

        output = content.clone()
        '''
        output = style.clone()
        output = torch.rand(content.shape)
        output = torch.zeros(content.shape)
        '''
        output = output.to(device)

        hs = self.encoder(content, style, output,
                          out_key_padding_mask=mask,
                          pos_c=pos_embed_c,
                          pos_s=pos_embed_s)

        # HWxNxC to NxCxHxW to
        N, B, C = hs.shape
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0)
        hs = hs.view(B, C, -1, H)

        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                content, style, output,
                mask: Optional[Tensor] = None,
                out_key_padding_mask: Optional[Tensor] = None,
                pos_c: Optional[Tensor] = None,
                pos_s: Optional[Tensor] = None,):

        for layer in self.layers:
            output = layer(content, style, output,
                           out_mask=mask,
                           out_key_padding_mask=out_key_padding_mask,
                           pos_c=pos_c, pos_s=pos_s)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     content, style, output,
                     out_mask: Optional[Tensor] = None,
                     out_key_padding_mask: Optional[Tensor] = None,
                     pos_c: Optional[Tensor] = None,
                     pos_s: Optional[Tensor] = None):
        q = self.with_pos_embed(content, pos_c)
        k = self.with_pos_embed(style, pos_s)
        # q = k = src
        # print(q.size(),k.size(),src.size())
        output_2 = self.self_attn(q, k, value=style, attn_mask=out_mask,
                              key_padding_mask = None)[0]
        output = output_2 + self.dropout1(output)
        output = self.norm1(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        return output

    def forward_pre(self,
                    content, style, output,
                    out_mask: Optional[Tensor] = None,
                    out_key_padding_mask: Optional[Tensor] = None,
                    pos_c: Optional[Tensor] = None,
                    pos_s: Optional[Tensor] = None):
        output = self.norm1(output)
        q = self.with_pos_embed(content, pos_c)
        k = self.with_pos_embed(style, pos_s)
        # q = k = src
        # print(q.size(),k.size(),src.size())
        output_2 = self.self_attn(q, k, value=style, attn_mask=out_mask,
                                  key_padding_mask=out_key_padding_mask)[0]
        output = output_2 + self.dropout1(output)
        output = self.norm2(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        return output

    def forward(self,
                content, style, output,
                out_mask: Optional[Tensor] = None,
                out_key_padding_mask: Optional[Tensor] = None,
                pos_c: Optional[Tensor] = None,
                pos_s: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(content, style, output, out_key_padding_mask, pos_c, pos_s)
        return self.forward_post(content, style, output, out_key_padding_mask, pos_c, pos_s)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")