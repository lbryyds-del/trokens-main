"""Attention modules"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from trokens.models.common import DropPath, Mlp

class Attention(nn.Module):
    """Attention module"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                attn_drop=0., proj_drop=0., with_qkv=True, use_mask=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.use_mask = use_mask

    def forward(self, x, mask=None, key_bias=None):
        """Forward pass"""
        return self.aggregate(*self.prepare(x, mask, key_bias))

    def prepare_logits(self, x, mask=None):
        """Project once; retain PRE-softmax scores, unmodified V and key mask.

        Different label priors can reuse these temporal scores without sharing
        the final attention probabilities. Spatial inputs remain serial.
        """
        batch_size, seq_len, feat_dim = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads,
                                    feat_dim // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(batch_size, seq_len, self.num_heads,
                            feat_dim // self.num_heads).permute(0, 2, 1, 3)
            q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        key_mask = None
        if self.use_mask:
            mask = mask.to(device=attn.device).bool()
            if len(mask.shape) == 2:
                # extra_dim for heads and dim on which attention is not applied
                key_mask = mask[:, None, None, :]
            else:
                # extra_dim for heads
                # mask -> bs x num_spacetime_tokens x num_spacetime_tokens
                key_mask = mask[:, None, :, :]
        return attn, v, key_mask

    def _weights(self, attn, key_mask, key_bias):
        if key_bias is not None:
            if key_bias.shape != (attn.shape[0], attn.shape[-1]):
                raise ValueError("Attention key_bias must be [batch, source/key tokens].")
            # Broadcast across heads and query positions, NOT the key axis.
            attn = attn + key_bias[:, None, None, :].to(attn.dtype)
        valid_rows = None
        if key_mask is not None:
            valid_rows = key_mask.any(dim=-1, keepdim=True)
            attn = attn.masked_fill(~key_mask, float('-inf'))
            # softmax([-inf, ...]) has NaN forward AND backward. Evaluate
            # empty rows on finite constants, then give them exactly zero mass.
            attn = torch.where(valid_rows, attn, torch.zeros_like(attn))

        attn = attn.softmax(dim=-1)
        if key_mask is not None:
            attn = attn.masked_fill(~key_mask, 0.0)
        attn = self.attn_drop(attn)
        return attn, valid_rows

    def prepare(self, x, mask=None, key_bias=None):
        logits, v, key_mask = self.prepare_logits(x, mask)
        attn, valid_rows = self._weights(logits, key_mask, key_bias)
        return attn, v, valid_rows

    def aggregate_logits(self, logits, v, key_mask=None, key_bias=None):
        attn, valid_rows = self._weights(logits, key_mask, key_bias)
        return self.aggregate(attn, v, valid_rows)

    def aggregate(self, attn, v, valid_rows=None):
        batch_size, heads, seq_len, head_dim = v.shape
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, heads * head_dim)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        if valid_rows is not None:
            # Also remove projection bias when there was no valid key. This
            # does not invent a visible point or alter any nonempty row.
            x = torch.where(valid_rows.squeeze(1), x, torch.zeros_like(x))
        return x


class TrajectoryAttentionBlock(nn.Module):
    """Trajectory attention block"""
    def __init__(
            self, cfg=None, dim=768, num_heads=12,
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            pt_attention='divided_space_time', use_pt_visibility=False,
            num_mlp_layers=1,
        ):
        super().__init__()
        self.cfg = cfg
        self.pt_attention = pt_attention
        self.norm1 = norm_layer(dim)
        if pt_attention in 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            temp_mask_flag = True if pt_attention == 'temp_patch' else False
            self.temporal_attn = Attention(dim, num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop, use_mask=(use_pt_visibility or temp_mask_flag))
            self.temporal_fc = nn.Linear(dim, dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop,
                                  use_mask=use_pt_visibility)
        else:
            raise NotImplementedError(f"Unsupported attention type {pt_attention}")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = []
        for _ in range(num_mlp_layers):
            mlp_hidden_dim = int(dim * mlp_ratio)
            mlp_ratio = max(mlp_ratio // 2, 1)
            self.mlp.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                                act_layer=act_layer, drop=drop))
        self.mlp = nn.Sequential(*self.mlp)

        self.use_pt_visibility = use_pt_visibility
        self.checkpoint_mlp = False

    def _mlp(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, thw, pt_mask, with_cls_token=True,
                temporal_state=None, temporal_bias=None, spatial_bias=None):
        """Forward pass"""
        if self.pt_attention == 'divided_space_time':
            if self.cfg.MODEL.USE_CLS_TOKEN:
                xt = x[:,1:,:]
                temporal_mask = pt_mask[:,1:].clone()
            else:
                xt = x
                temporal_mask = pt_mask.clone()
            num_frames, height, width = thw
            batch_size = x.shape[0]
            spatial_mask = None

            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',
                            b=batch_size, h=height, w=width, t=num_frames)
            if self.use_pt_visibility:
                temporal_mask = rearrange(temporal_mask, 'b (h w t) -> (b h w) t',
                                            b=batch_size, h=height, w=width, t=num_frames)
            if temporal_state is None:
                temporal_output = self.temporal_attn(
                    self.temporal_norm1(xt), temporal_mask, key_bias=temporal_bias)
            else:
                temporal_output = self.temporal_attn.aggregate_logits(
                    *temporal_state, key_bias=temporal_bias)
            res_temporal = self.drop_path(temporal_output)
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',
                                    b=batch_size, h=height, w=width, t=num_frames)
            res_temporal = self.temporal_fc(res_temporal)
                # res_temporal = res_temporal * temporal_mask.unsqueeze(-1)


            if self.cfg.MODEL.USE_CLS_TOKEN:
                xt = x[:,1:,:] + res_temporal
                init_cls_token = x[:,0,:].unsqueeze(1)
                cls_token = init_cls_token.repeat(1, num_frames, 1)
                if self.use_pt_visibility:
                    intit_cls_pt_mask = pt_mask[:,0].unsqueeze(1)
                    spatial_mask = pt_mask[:,1:].clone()
                    intit_cls_pt_mask = intit_cls_pt_mask.repeat(1, num_frames)

                cls_token = rearrange(cls_token, 'b t m -> (b t) 1 m',
                                            b=batch_size, t=num_frames)
            else:
                xt = x + res_temporal
                if self.use_pt_visibility:
                    spatial_mask = pt_mask.clone()
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',
                                b=batch_size, h=height, w=width, t=num_frames)
            if self.cfg.MODEL.USE_CLS_TOKEN:
                xs = torch.cat((cls_token, xs), 1)
                if self.use_pt_visibility:
                    intit_cls_pt_mask = rearrange(intit_cls_pt_mask, 'b t -> (b t) 1',
                                                b=batch_size,t=num_frames)
                    spatial_mask = rearrange(spatial_mask, 'b (h w t) -> (b t) (h w)',
                                        b=batch_size,h=height,w=width,t=num_frames)
                    spatial_mask = torch.cat((intit_cls_pt_mask, spatial_mask), 1)
            else:
                if self.use_pt_visibility:
                    spatial_mask = rearrange(spatial_mask, 'b (h w t) -> (b t) (h w)',
                                            b=batch_size,h=height,w=width,t=num_frames)


            if spatial_bias is not None and self.cfg.MODEL.USE_CLS_TOKEN:
                # CLS is a neutral key (g=1); patch priors also affect its query.
                spatial_bias = torch.cat((spatial_bias.new_zeros(batch_size * num_frames, 1),
                                          spatial_bias), dim=-1)
            res_spatial = self.drop_path(self.attn(
                self.norm1(xs), spatial_mask, key_bias=spatial_bias))
            if self.cfg.MODEL.USE_CLS_TOKEN:
                ### Taking care of CLS token
                cls_token = res_spatial[:,0,:]
                cls_token = rearrange(cls_token, '(b t) m -> b t m',b=batch_size,t=num_frames)
                cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
                res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',
                                    b=batch_size,h=height,w=width,t=num_frames)
            res = res_spatial
            x = xt

            ## Mlp
            if self.cfg.MODEL.USE_CLS_TOKEN:
                x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            else:
                x = x + res

        elif self.pt_attention == 'trajectory':
            x  = x + self.drop_path(
                self.attn(
                    self.norm1(x),
                    thw,
                    with_cls_token=with_cls_token,
                )[0]
            )
        else:
            raise NotImplementedError(f"Unsupported attention type {self.pt_attention}")


        mlp = (checkpoint(self._mlp, x, use_reentrant=False)
               if self.checkpoint_mlp and self.training and torch.is_grad_enabled()
               else self._mlp(x))
        x = x + self.drop_path(mlp)
        return x, thw
