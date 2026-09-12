"""Pointformer model."""
import os
import sys
from functools import partial
from collections import OrderedDict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from sympy import divisors
import numpy as np
from torch.nn.init import trunc_normal_

from trokens.models.attention import TrajectoryAttentionBlock
from trokens.models.cat_spatial_aggregation import CATSpatialCostAggregator
from trokens.models.temporal_similarity_refiner import (
    TrajectoryTemporalSimilarityRefiner,
)
from trokens.models.branches.motion_blocks import (
    CrossMotionModule,
    HODMotionModule
)
from trokens.datasets.hod import get_orientation_hist
from .build import MODEL_REGISTRY

# pylint: disable=unused-argument,redefined-builtin

@MODEL_REGISTRY.register()
class Pointformer(nn.Module):
    """ Main model for point tracking based transformer model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        # self.patch_size = cfg.MF.PATCH_SIZE
        self.feat_extractor_type = cfg.MODEL.FEAT_EXTRACTOR
        if self.feat_extractor_type == "dino":
            dino_config  = cfg.MODEL.DINO_CONFIG
            vit_mode = dino_config.split("_")[1]
            if 'vits' in vit_mode:
                vit_type = 'vits'
                self.embed_dim = self.dino_feat_size = 384
            elif 'vitb' in vit_mode:
                vit_type = 'vitb'
                self.embed_dim = self.dino_feat_size = 768
            elif 'vitl' in vit_mode:
                vit_type = 'vitl'
                self.embed_dim = self.dino_feat_size = 1024

            else:
                raise NotImplementedError("Only supports ViT-B and ViT-S for DINO")
            self.patch_size = int(vit_mode.replace(vit_type, ""))
        elif self.feat_extractor_type == "clip_vit_b16":
            self.embed_dim = 768
            self.patch_size = 16
        elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
            self.embed_dim = 1024
            self.patch_size = 14
        else:
            raise NotImplementedError('Feature extractor not implemented')


        self.in_chans = cfg.MF.CHANNELS
        if cfg.TRAIN.DATASET == "epickitchens" and cfg.TASK == 'classification':
            self.num_classes = [97, 300]
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES

        self.depth = cfg.MF.DEPTH
        self.num_heads = cfg.MF.NUM_HEADS
        self.mlp_ratio = cfg.MF.MLP_RATIO
        self.qkv_bias = cfg.MF.QKV_BIAS
        self.drop_rate = cfg.MF.DROP
        self.drop_path_rate = cfg.MF.DROP_PATH
        self.head_dropout = cfg.MF.HEAD_DROPOUT
        self.video_input = cfg.MF.VIDEO_INPUT
        self.temporal_resolution = cfg.DATA.NUM_FRAMES
        self.use_mlp = cfg.MF.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.MF.ATTN_DROPOUT
        self.head_act = cfg.MF.HEAD_ACT
        self.cfg = cfg
        self.pot_route_cfg = cfg.FEW_SHOT.POT_ROUTE
        self.text_align_cfg = cfg.FEW_SHOT.TEXT_ALIGN
        self.cost_agg_cfg = getattr(cfg.FEW_SHOT, "COST_AGG", None)
        self.support_text_fusion_cfg = getattr(
            cfg.FEW_SHOT,
            "SUPPORT_TEXT_FUSION",
            None,
        )
        self.query_null_cfg = getattr(
            cfg.FEW_SHOT,
            "QUERY_NULL_ROUTE",
            None,
        )
        self.is_multilabel_few_shot = (
            cfg.TASK == 'few_shot'
            and cfg.DATA.MULTI_LABEL
            and not cfg.MODEL.APPEARANCE_MODULE_DISABLE
        )
        self.use_frame_softmax_route = (
            self.is_multilabel_few_shot
            and self.feat_extractor_type == "dinotxt_vitl14_reg4"
            and self.pot_route_cfg.ENABLE
        )
        support_text_fusion_requested = bool(
            getattr(self.support_text_fusion_cfg, "ENABLE", False)
        )
        self.use_support_text_fusion = (
            support_text_fusion_requested and self.use_frame_softmax_route
        )
        cost_agg_requested = bool(
            getattr(self.cost_agg_cfg, "ENABLE", False)
        )
        self.use_cat_cost_aggregation = (
            cost_agg_requested and self.use_frame_softmax_route
        )
        self.use_query_partial_q2s = (
            self.use_frame_softmax_route
            and bool(getattr(self.pot_route_cfg, "QUERY_PARTIAL_ENABLE", False))
        )
        query_null_requested = bool(
            getattr(self.query_null_cfg, "ENABLE", False)
        )
        self.use_query_null_route = (
            query_null_requested and self.use_query_partial_q2s
        )
        self.use_text_alignment = (
            self.is_multilabel_few_shot
            and self.feat_extractor_type == "dinotxt_vitl14_reg4"
            and self.text_align_cfg.ENABLE
        )
        if (
            self.is_multilabel_few_shot
            and self.pot_route_cfg.ENABLE
            and self.feat_extractor_type != "dinotxt_vitl14_reg4"
        ):
            raise NotImplementedError(
                "POT_ROUTE text routing currently requires the dinotxt_vitl14_reg4 backbone."
            )
        if (
            self.is_multilabel_few_shot
            and bool(getattr(self.pot_route_cfg, "QUERY_PARTIAL_ENABLE", False))
            and not self.pot_route_cfg.ENABLE
        ):
            raise NotImplementedError(
                "QUERY_PARTIAL_ENABLE currently requires POT_ROUTE.ENABLE."
            )
        if query_null_requested and not self.use_query_partial_q2s:
            raise NotImplementedError(
                "QUERY_NULL_ROUTE requires POT_ROUTE.ENABLE and "
                "POT_ROUTE.QUERY_PARTIAL_ENABLE."
            )
        if (
            self.is_multilabel_few_shot
            and self.text_align_cfg.ENABLE
            and self.feat_extractor_type != "dinotxt_vitl14_reg4"
        ):
            raise NotImplementedError(
                "TEXT_ALIGN currently requires the dinotxt_vitl14_reg4 backbone."
            )
        if cost_agg_requested and not self.use_frame_softmax_route:
            raise NotImplementedError(
                "COST_AGG currently requires the enabled frame_softmax text route."
            )
        if support_text_fusion_requested and not self.use_frame_softmax_route:
            raise NotImplementedError(
                "SUPPORT_TEXT_FUSION currently requires the enabled "
                "frame_softmax text route."
            )
        if self.use_support_text_fusion and self.use_cat_cost_aggregation:
            raise NotImplementedError(
                "SUPPORT_TEXT_FUSION and COST_AGG cannot be enabled together: "
                "CAT supplies precomputed query costs and would bypass the fused "
                "query features."
            )
        if self.use_cat_cost_aggregation and not cfg.POINT_INFO.ENABLE:
            raise NotImplementedError(
                "COST_AGG currently requires POINT_INFO.ENABLE so dense cost "
                "corrections can be sampled at pred_tracks."
            )
        self.use_label_text_features = (
            self.use_frame_softmax_route
            or self.use_text_alignment
        )
        if self.use_query_null_route:
            score_min = float(self.query_null_cfg.SCORE_MIN)
            score_max = float(self.query_null_cfg.SCORE_MAX)
            score_init = float(self.query_null_cfg.SCORE_INIT)
            token_init_std = float(self.query_null_cfg.TOKEN_INIT_STD)
            value_scale = float(self.query_null_cfg.VALUE_SCALE)
            ortho_weight = float(self.query_null_cfg.ORTHO_WEIGHT)
            if not all(
                np.isfinite(value)
                for value in (score_min, score_max, score_init)
            ) or not score_min < score_init < score_max:
                raise ValueError(
                    "QUERY_NULL_ROUTE requires finite SCORE_MIN < SCORE_INIT "
                    "< SCORE_MAX."
                )
            if not np.isfinite(token_init_std) or token_init_std <= 0.0:
                raise ValueError(
                    "QUERY_NULL_ROUTE.TOKEN_INIT_STD must be finite and positive."
                )
            if not np.isfinite(value_scale) or value_scale <= 0.0:
                raise ValueError(
                    "QUERY_NULL_ROUTE.VALUE_SCALE must be finite and positive."
                )
            if not np.isfinite(ortho_weight) or ortho_weight < 0.0:
                raise ValueError(
                    "QUERY_NULL_ROUTE.ORTHO_WEIGHT must be finite and non-negative."
                )

            score_ratio = (score_init - score_min) / (score_max - score_min)
            raw_score = np.log(score_ratio / (1.0 - score_ratio))
            self.query_null_token = nn.Parameter(
                torch.empty(1, self.embed_dim)
            )
            self.query_null_score_raw = nn.Parameter(
                torch.tensor(raw_score, dtype=torch.float32)
            )
        self.num_patches = (224 // self.patch_size) ** 2
        if cfg.POINT_INFO.ENABLE:
            self.point_grid_size = self.get_point_grid_size()

        else:
            self.point_grid_size = int(self.num_patches ** 0.5)

        # CLS token
        if cfg.MODEL.USE_CLS_TOKEN:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = nn.Identity()

        # # Positional embedding

        self.pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)

        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        ##
        blocks = []
        for i in range(self.depth):
            # pt_attention is introduced, for now its just space-time attention
            _block = TrajectoryAttentionBlock(
                cfg = cfg,
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                pt_attention=cfg.MF.PT_ATTENTION,
                use_pt_visibility=cfg.MF.USE_PT_VISIBILITY or cfg.POINT_INFO.USE_PT_QUERY_MASK,
                num_mlp_layers=cfg.MF.NUM_MLP_LAYERS,
            )

            blocks.append(_block)
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(self.embed_dim)
        if self.cfg.MODEL.APPEARANCE_MODULE_DISABLE:
            assert (
            self.cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE or
            self.cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE), "One motion module must be enabled"

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh()
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, hidden_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()
            self.agg_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.agg_cls_token, std=.02)


        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, f'head{a}', nn.Linear(
                                        self.embed_dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(self.embed_dim, self.num_classes)
                if self.num_classes > 0 else nn.Identity())
        self.patch_num_side = 224 // self.patch_size
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.embed_dim, self.patch_num_side,
                                                        self.patch_num_side))

        trunc_normal_(self.spatial_pos_embed, std=.02)
        if cfg.MODEL.FEAT_EXTRACTOR == 'resnet':
            #TODO(pulkit): Remove hard coding
            self.space_pos_embed = nn.Parameter(torch.zeros(1,49, self.embed_dim))
        else:
            self.space_pos_embed = nn.Parameter(
                                torch.zeros(1,self.num_patches, self.embed_dim))

        self.time_pos_embed = nn.Parameter(
                        torch.zeros(1,self.cfg.DATA.NUM_FRAMES, self.embed_dim))
        trunc_normal_(self.space_pos_embed, std=.02)
        trunc_normal_(self.time_pos_embed, std=.02)
        self.space_pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        self.time_pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)



        self.spatial_pos_embed_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        self.layer_to_use = None
        if self.use_label_text_features:
            if self.feat_extractor_type == "clip_vit_b16":
                self.text_feature_dim = 512
            elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
                self.text_feature_dim = self.embed_dim
            else:
                raise NotImplementedError(
                    f"Text support branch is not supported for {self.feat_extractor_type}."
                )
            if self.text_feature_dim == self.embed_dim:
                self.text_to_model_proj = nn.Identity()
            else:
                self.text_to_model_proj = nn.Linear(self.text_feature_dim, self.embed_dim)
            if self.use_frame_softmax_route or self.use_text_alignment:
                self.atomic_label_names = self._load_atomic_label_names()

        if self.use_cat_cost_aggregation:
            cost_dim = int(getattr(self.cost_agg_cfg, "COST_DIM", 32))
            guidance_dim = int(getattr(self.cost_agg_cfg, "GUIDANCE_DIM", 32))
            num_heads = int(getattr(self.cost_agg_cfg, "NUM_HEADS", 4))
            window_size = int(getattr(self.cost_agg_cfg, "WINDOW_SIZE", 4))
            num_layers = int(getattr(self.cost_agg_cfg, "NUM_LAYERS", 1))
            mlp_ratio = float(getattr(self.cost_agg_cfg, "MLP_RATIO", 4.0))
            class_agg_cfg = getattr(self.cost_agg_cfg, "CLASS_AGG", None)
            self.cat_spatial_cost_aggregator = CATSpatialCostAggregator(
                appearance_dim=self.embed_dim,
                cost_dim=cost_dim,
                guidance_dim=guidance_dim,
                num_heads=num_heads,
                window_size=window_size,
                num_layers=num_layers,
                input_resolution=(self.patch_num_side, self.patch_num_side),
                mlp_ratio=mlp_ratio,
                attn_dropout=float(
                    getattr(self.cost_agg_cfg, "ATTN_DROPOUT", 0.0)
                ),
                proj_dropout=float(
                    getattr(self.cost_agg_cfg, "PROJ_DROPOUT", 0.0)
                ),
                class_attention_enabled=bool(
                    getattr(class_agg_cfg, "ENABLE", False)
                ),
                class_guidance_dim=int(
                    getattr(class_agg_cfg, "GUIDANCE_DIM", 32)
                ),
                class_num_heads=int(
                    getattr(class_agg_cfg, "NUM_HEADS", num_heads)
                ),
                class_attention_type=str(
                    getattr(class_agg_cfg, "ATTENTION_TYPE", "full")
                ),
                class_pooling_size=getattr(
                    class_agg_cfg,
                    "POOLING_SIZE",
                    1,
                ),
                class_pad_len=int(
                    getattr(class_agg_cfg, "PAD_LEN", 0)
                ),
                class_mlp_ratio=float(
                    getattr(class_agg_cfg, "MLP_RATIO", 4.0)
                ),
                class_gate_init=float(
                    getattr(class_agg_cfg, "GATE_INIT", 0.0)
                ),
            )

        # Initialize weights
        self.init_weights()
        self.apply(self._init_weights)
        if self.use_query_null_route:
            trunc_normal_(
                self.query_null_token,
                std=float(self.query_null_cfg.TOKEN_INIT_STD),
            )
        if self.feat_extractor_type == "dino":
            dino_config  = cfg.MODEL.DINO_CONFIG
            torch_home = os.environ.get("TORCH_HOME", os.path.join(os.getcwd(), ".torch-cache"))
            os.environ.setdefault("TORCH_HOME", torch_home)
            local_path = os.path.join(torch_home, 'hub')
            if 'v2' in dino_config:
                local_path = os.path.join(local_path , 'facebookresearch_dinov2_main')
                self.dino = torch.hub.load(local_path, dino_config, source='local')
            else:
                local_path = os.path.join(local_path , 'facebookresearch_dino_main')
                self.dino = torch.hub.load(local_path, dino_config, source='local')
                self.dino.num_register_tokens = 0

            self.feat_dict = dict()
            #output of last norm to be taken.
            layer = self.dino.norm
            self.hook = layer.register_forward_hook(self.hook_fn(self.feat_dict, 'dino'))

            self.dino.cuda()
            # Set all DINO parameters to not require gradients
            for param in self.dino.parameters():
                param.requires_grad = False
        elif self.feat_extractor_type == "clip_vit_b16":
            import clip

            clip_model, _ = clip.load("ViT-B/16", device="cuda", jit=False)
            self.clip_model = clip_model
            self.clip_visual = clip_model.visual
            self.clip_model.cuda()
            for param in self.clip_model.parameters():
                param.requires_grad = False
        elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
            self.dinotxt_visual_model = self._load_dinotxt_visual_model()
            if self.use_label_text_features:
                self.dinotxt_text_model = self._load_dinotxt_text_model()
                self.dinotxt_tokenizer = self._load_dinotxt_tokenizer()
        else:
            raise NotImplementedError('Feature extractor not implemented')

        if cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE:
            self.cross_motion_module = CrossMotionModule(
                out_feature_dim=self.embed_dim,
                num_patches=self.num_patches,
                in_fea_dim_crossmotion=2*self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE,
            )

        if cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE:
            if cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID:
                #TODO(pulkit): make this dynamic
                assert cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID_LEVELS == 3, 'hard coded for now'
                feat_dim = cfg.POINT_INFO.HOD.NUM_BINS * 7
            else:
                feat_dim = cfg.POINT_INFO.HOD.NUM_BINS
            self.hod_motion_module = HODMotionModule(
                in_feature_dim=feat_dim,
                out_feature_dim=self.embed_dim,
                num_patches=self.num_patches,
            )

        # Build last so the optional branch cannot change initialization of
        # existing parameters or be overwritten by the global weight init.
        self._init_temporal_similarity_refinement()

    def _init_temporal_similarity_refinement(self):
        temporal_cfg = getattr(self.pot_route_cfg, "TEMPORAL_REFINEMENT", None)
        self.use_temporal_similarity_refinement = bool(
            getattr(temporal_cfg, "ENABLE", False)
        )
        self.temporal_similarity_refiner = None
        if not self.use_temporal_similarity_refinement:
            return
        if not self.use_frame_softmax_route or not self.use_query_partial_q2s:
            raise ValueError(
                "TEMPORAL_REFINEMENT requires the frame-softmax Query route."
            )
        match_cfg = getattr(self.cfg.FEW_SHOT, "QUERY_CLASS_MATCHABILITY", None)
        if (
            self.use_cat_cost_aggregation
            or self.use_query_null_route
            or (
                bool(getattr(match_cfg, "ENABLE", False))
                and bool(getattr(match_cfg, "LOCAL_REFINEMENT_ENABLE", False))
            )
        ):
            raise ValueError(
                "TEMPORAL_REFINEMENT cannot be combined with COST_AGG, "
                "QUERY_NULL_ROUTE or local confuser refinement: these bypass "
                "the shared text-similarity route."
            )
        if not self.cfg.POINT_INFO.ENABLE:
            raise ValueError("TEMPORAL_REFINEMENT requires tracked point inputs.")
        # Preserve the baseline RNG stream for subsequent episodic sampling.
        with torch.random.fork_rng(devices=[]):
            self.temporal_similarity_refiner = TrajectoryTemporalSimilarityRefiner(
                hidden_dim=getattr(temporal_cfg, "HIDDEN_DIM", 16),
                kernel_size=getattr(temporal_cfg, "KERNEL_SIZE", 3),
                long_dilation=getattr(temporal_cfg, "LONG_DILATION", 2),
                max_logit_delta=getattr(temporal_cfg, "MAX_LOGIT_DELTA", 0.5),
                use_visibility=getattr(temporal_cfg, "USE_VISIBILITY", True),
            )

    def hook_fn(self, feat_dict, layer_name):
        """Hook function to extract features of specific layers"""
        def hook(module, input, output):
            # Store the extracted features as an attribute of the model
            feat_dict[layer_name] = output
        return hook

    def backward_hook_fn(self, feat_dict, layer_name):
        """Backward hook function for extracting gradients"""
        def hook(module, grad_inputs, grad_outputs):
            # Store the extracted features as an attribute of the model
            grad_out_norm = np.mean([torch.norm(grad_output).item()
                                        for grad_output in grad_outputs])
            grad_in_norm = np.mean([torch.norm(grad_input).item()
                                        for grad_input in grad_inputs])
            feat_dict[layer_name] = {
                'grad_out': np.around(grad_out_norm, 3),
                'grad_in': np.around(grad_in_norm, 3)
            }
        return hook


    def init_weights(self):
        """Initialize weights"""
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """No weight decay params"""
        if self.cfg.MF.POS_EMBED == "joint":
            skip = {'pos_embed', 'cls_token', 'st_embed'}
        else:
            skip = {'pos_embed', 'cls_token', 'temp_embed'}
        if getattr(self, "use_query_null_route", False):
            skip.update({"query_null_token", "query_null_score_raw"})
        return skip

    def get_classifier(self):
        """Get classifier"""
        return self.head


    def get_point_grid_size(self):
        """Get point grid size"""
        all_divisors = divisors(self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE)
        return all_divisors[len(all_divisors) // 2]

    def _resolve_sav_label_map_path(self):
        """Resolve the SAV atomic label map used for DinoTxt label text."""
        candidate_paths = []
        if self.cfg.DATA.PATH_TO_DATA_DIR:
            data_root = self.cfg.DATA.PATH_TO_DATA_DIR
            if os.path.isabs(data_root):
                candidate_paths.append(
                    os.path.join(data_root, "education_first_label.pbtxt")
                )
            else:
                candidate_paths.append(
                    os.path.join(
                        os.getcwd(),
                        data_root,
                        "education_first_label.pbtxt",
                    )
                )
        candidate_paths.append(
            os.path.join(os.getcwd(), "data", "sav", "education_first_label.pbtxt")
        )
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                return candidate_path
        raise FileNotFoundError(
            "Could not locate SAV atomic label map education_first_label.pbtxt."
        )

    def _load_atomic_label_names(self):
        """Load the 0-based SAV atomic action names for POT routing."""
        dataset_names = {
            str(getattr(self.cfg.TRAIN, "DATASET", "")).lower(),
            str(getattr(self.cfg.TEST, "DATASET", "")).lower(),
        }
        if "sav" not in dataset_names:
            raise NotImplementedError(
                "POT support routing currently expects SAV atomic label names."
            )

        label_names = [str(class_id) for class_id in range(self.num_classes)]
        label_map_path = self._resolve_sav_label_map_path()
        current_name = None
        current_id = None
        with open(label_map_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line.startswith("name:"):
                    current_name = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("id:"):
                    current_id = int(line.split(":", 1)[1].strip()) - 1

                if current_name is not None and current_id is not None:
                    if 0 <= current_id < len(label_names):
                        label_names[current_id] = current_name
                    current_name = None
                    current_id = None
        return label_names

    def _get_sav_label_prompts(self, label_name):
        """Return visually grounded prompt variants for every SAV atomic label."""
        base_prompt = label_name.replace("_", " ")
        prompt_bank = {
            "sit": [
                "sit",
                "a student sits on a chair",
                "the person's hips rest on a seat",
                "the upper body remains in a seated posture",
                "a student is seated at a classroom desk",
            ],
            "stand": [
                "stand",
                "a student stands upright",
                "the person's body is supported by both feet",
                "the torso rises above the chair",
                "an upright person stands beside a classroom desk",
            ],
            "look_forward": [
                "look forward",
                "a student looks toward the front of the classroom",
                "the head and face point straight ahead",
                "eyes are directed toward the teacher or board",
                "the head remains centered without turning sideways",
            ],
            "look_sideways": [
                "look sideways",
                "a student looks to the left or right",
                "the head turns toward one side",
                "the face is oriented away from the front",
                "the eyes and head point toward something nearby",
            ],
            "read": [
                "read",
                "a student reads a book or paper",
                "eyes looking down at reading material",
                "holding or looking at a book on a desk",
                "reading pages without writing",
            ],
            "flip_books": [
                "flip books",
                "a student turns pages of a book",
                "hand flipping book pages on a desk",
                "fingers moving along page edges",
                "a book page changes position over time",
            ],
            "touch_sth": [
                "touch something",
                "a student touches a nearby object",
                "a hand reaches toward an object on the desk",
                "fingers make contact with an item",
                "the arm extends and the hand contacts something",
            ],
            "raise_hand": [
                "raise hand",
                "a student raises one hand in class",
                "an arm moves upward above shoulder level",
                "a hand is held up to get the teacher's attention",
                "the raised arm extends upward away from the desk",
            ],
            "hands_down": [
                "hands down",
                "a student lowers one or both hands",
                "hands move downward from a raised position",
                "the arms descend toward the desk or the body",
                "both hands finish below shoulder level",
            ],
            "take_notes": [
                "take notes",
                "a student writes notes on paper",
                "hand holding a pen and writing",
                "small repetitive hand motion on notebook",
                "writing on a desk with paper",
            ],
            "applaud": [
                "applaud",
                "a student claps their hands",
                "both hands repeatedly move together and apart",
                "the palms meet in front of the body",
                "repeated hand clapping in the classroom",
            ],
            "bend": [
                "bend",
                "a student bends the upper body",
                "the torso leans forward or downward",
                "the head and shoulders move toward the desk",
                "the body changes from upright to a bent posture",
            ],
            "turn_around": [
                "turn around",
                "a person turns the head or body around",
                "torso orientation changes over time",
                "head rotates from front to side or back",
                "shoulder direction changes across frames",
            ],
            "talk_with_others": [
                "talk with others",
                "a student talks with another person",
                "face turned toward a nearby person while speaking",
                "mouth movement during conversation",
                "two people interacting or conversing",
            ],
            "answer_questions": [
                "answer questions",
                "a student answers a question in class",
                "a student speaks while looking toward the teacher",
                "a student responds in the classroom",
                "head and mouth movement while answering",
            ],
        }
        return prompt_bank.get(label_name, [base_prompt])

    def _get_pot_label_text_features(self, global_class_indices, dtype):
        """Encode episode label prompts with DinoTxt text encoder for POT routing."""
        if self.feat_extractor_type != "dinotxt_vitl14_reg4":
            raise NotImplementedError(
                "POT label text features are only implemented for dinotxt_vitl14_reg4."
            )

        prompt_groups = [
            self._get_sav_label_prompts(self.atomic_label_names[int(class_id)])
            for class_id in global_class_indices.detach().cpu().tolist()
        ]
        flat_prompts = [prompt for group in prompt_groups for prompt in group]
        group_sizes = [len(group) for group in prompt_groups]
        text_device = next(self.dinotxt_text_model.parameters()).device
        self.dinotxt_text_model.eval()
        with torch.no_grad():
            tokenized = self.dinotxt_tokenizer.tokenize(flat_prompts).to(
                text_device,
                non_blocking=True,
            )
            flat_text_features = self.dinotxt_text_model(tokenized).float()
            flat_text_features = flat_text_features[
                :,
                flat_text_features.shape[-1] // 2 :,
            ]
        flat_text_features = self.text_to_model_proj(flat_text_features)
        flat_text_features = F.normalize(flat_text_features, dim=-1)

        label_text_features = []
        start_idx = 0
        for group_size in group_sizes:
            group_features = flat_text_features[start_idx : start_idx + group_size]
            label_feature = group_features.mean(dim=0)
            label_text_features.append(F.normalize(label_feature, dim=0))
            start_idx += group_size

        text_features = torch.stack(label_text_features, dim=0)
        return text_features.to(device=global_class_indices.device, dtype=dtype)

    def _torchhub_dirs(self):
        """Return candidate torch hub dirs for cached DINOv2 assets."""
        candidates = [
            Path(torch.hub.get_dir()),
            Path(os.getcwd()) / ".torch-cache" / "hub",
            Path.home() / ".cache" / "torch" / "hub",
        ]
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _find_torchhub_checkpoint(self, filename):
        """Find a non-empty checkpoint in known torch hub cache dirs."""
        for hub_dir in self._torchhub_dirs():
            checkpoint_path = hub_dir / "checkpoints" / filename
            if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
                return checkpoint_path
        return None

    def _find_dinov2_hub_repo(self):
        """Find or populate the cached DINOv2 hub repository."""
        for hub_dir in self._torchhub_dirs():
            hub_repo = hub_dir / "facebookresearch_dinov2_main"
            if hub_repo.exists():
                return hub_repo

        torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            trust_repo=True,
            skip_validation=True,
        )
        for hub_dir in self._torchhub_dirs():
            hub_repo = hub_dir / "facebookresearch_dinov2_main"
            if hub_repo.exists():
                return hub_repo
        raise FileNotFoundError("Could not locate the cached DINOv2 hub repository.")

    def _get_dinov2_hub_modules(self):
        """Make DINOv2 hub modules importable and return the hub repo path."""
        hub_repo = self._find_dinov2_hub_repo()
        hub_repo_str = str(hub_repo)
        if hub_repo_str not in sys.path:
            sys.path.insert(0, hub_repo_str)
        return hub_repo

    def _download_torchhub_checkpoint(self, filename, url):
        """Download a checkpoint-like asset into the active torch hub cache."""
        target_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(target_path), progress=True)
        return target_path

    def _load_dinotxt_visual_model(self):
        """Load the DinoTxt visual tower without loading the text encoder."""
        self._get_dinov2_hub_modules()

        from dinov2.hub.backbones import dinov2_vitl14_reg
        from dinov2.hub.text.dinov2_wrapper import DINOv2Wrapper
        from dinov2.hub.text.vision_tower import VisionTower
        from dinov2.hub.utils import _DINOV2_BASE_URL

        backbone_checkpoint = self._find_torchhub_checkpoint(
            "dinov2_vitl14_reg4_pretrain.pth"
        )
        if backbone_checkpoint is not None:
            backbone = dinov2_vitl14_reg(weights=str(backbone_checkpoint))
        else:
            backbone = dinov2_vitl14_reg()

        visual_model = VisionTower(
            backbone=DINOv2Wrapper(backbone),
            freeze_backbone=True,
            embed_dim=2048,
            num_head_blocks=2,
            head_blocks_block_drop_path=0.3,
            use_class_token=True,
            use_patch_tokens=True,
            patch_token_layer=1,
            patch_tokens_pooler_type="mean",
            use_linear_projection=False,
        )

        vision_head_checkpoint = self._find_torchhub_checkpoint(
            "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth"
        )
        if vision_head_checkpoint is not None:
            vision_head_state_dict = torch.load(
                vision_head_checkpoint,
                map_location="cpu",
            )
        else:
            vision_head_state_dict = torch.hub.load_state_dict_from_url(
                _DINOV2_BASE_URL
                + "/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
                map_location="cpu",
            )

        visual_model.head.load_state_dict(vision_head_state_dict, strict=True)
        visual_model.cuda()
        visual_model.eval()
        for param in visual_model.parameters():
            param.requires_grad = False
        return visual_model

    def _load_dinotxt_text_model(self):
        """Load the DinoTxt text tower for prompt embeddings."""
        self._get_dinov2_hub_modules()

        from dinov2.hub.text.text_tower import TextTower
        from dinov2.hub.text.text_transformer import TextTransformer
        from dinov2.hub.utils import _DINOV2_BASE_URL

        text_backbone = TextTransformer(
            context_length=77,
            vocab_size=49408,
            dim=1280,
            num_heads=20,
            num_layers=24,
            ffn_ratio=4,
            is_causal=True,
            ls_init_value=None,
            dropout_prob=0.0,
        )
        text_model = TextTower(
            backbone=text_backbone,
            freeze_backbone=False,
            embed_dim=2048,
            num_head_blocks=0,
            head_blocks_is_causal=False,
            head_blocks_block_drop_prob=0.0,
            tokens_pooler_type="argmax",
            use_linear_projection=True,
        )

        text_checkpoint = self._find_torchhub_checkpoint(
            "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth"
        )
        if text_checkpoint is not None:
            text_state_dict = torch.load(text_checkpoint, map_location="cpu")
        else:
            text_state_dict = torch.hub.load_state_dict_from_url(
                _DINOV2_BASE_URL
                + "/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
                map_location="cpu",
            )

        text_model.load_state_dict(text_state_dict, strict=True)
        text_model.cuda()
        text_model.eval()
        for param in text_model.parameters():
            param.requires_grad = False
        return text_model

    def _load_dinotxt_tokenizer(self):
        """Load the DinoTxt tokenizer, caching the BPE vocabulary if needed."""
        self._get_dinov2_hub_modules()

        from dinov2.hub.text.tokenizer import Tokenizer
        from dinov2.hub.utils import _DINOV2_BASE_URL

        vocab_filename = "bpe_simple_vocab_16e6.txt.gz"
        vocab_path = self._find_torchhub_checkpoint(vocab_filename)
        if vocab_path is None:
            vocab_path = self._download_torchhub_checkpoint(
                vocab_filename,
                _DINOV2_BASE_URL + f"/thirdparty/{vocab_filename}",
            )
        return Tokenizer(vocab_path=str(vocab_path))

    def _sample_point_features(self, feat_to_use, pred_tracks, add_pt_pos_embed=False):
        """Sample point features from a dense patch feature map."""
        bs, num_frames = feat_to_use.shape[:2]
        feat_to_use = rearrange(feat_to_use, 'b t p q d -> (b t) p q d')
        feat_to_use = rearrange(feat_to_use, 'b p q d -> b d p q')
        num_x, num_y = feat_to_use.shape[-2:]
        assert self.num_patches == num_x * num_y, "Number of patches mismatch"
        pred_tracks = pred_tracks.view(bs * num_frames, -1, 1, 2)
        sampled_feat = F.grid_sample(
            feat_to_use,
            pred_tracks,
            align_corners=True,
            mode=self.cfg.MODEL.FEAT_EXTRACT_MODE,
        )
        if add_pt_pos_embed:
            spatial_pos_embed = self.spatial_pos_embed.repeat(bs * num_frames, 1, 1, 1)
            sample_pos_embedding = F.grid_sample(
                spatial_pos_embed,
                pred_tracks,
                align_corners=True,
                mode='bilinear',
            )
            sampled_feat = sampled_feat + sample_pos_embedding
        sampled_feat = rearrange(sampled_feat, 'b d p q -> b p q d')
        sampled_feat = sampled_feat.squeeze(-2)
        return rearrange(sampled_feat, '(b t) p d -> b t p d', t=num_frames)

    def _masked_space_time_mean(self, feat, point_mask):
        """Average support tokens over valid space-time positions only."""
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if point_mask is None:
            point_mask = torch.ones(
                feat.shape[:3],
                device=feat.device,
                dtype=torch.bool,
            )
        else:
            point_mask = point_mask.to(device=feat.device).bool()

        weights = point_mask[..., None].to(feat.dtype)
        denom = weights.sum(dim=(1, 2)).clamp_min(1e-6)
        pooled = (feat * weights).sum(dim=(1, 2)) / denom
        return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    def _masked_softmax_1d(self, scores, mask, dim=-1, tau=1.0):
        """Masked softmax that returns zeros for fully invalid rows."""
        scores = torch.nan_to_num(
            scores.float(),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool, device=scores.device)
        else:
            mask = mask.to(device=scores.device).bool()
            if mask.shape != scores.shape:
                mask = mask.expand_as(scores)

        if scores.numel() == 0:
            return torch.zeros_like(scores)

        tau = max(float(tau), 1e-6)
        masked_scores = (scores / tau).masked_fill(~mask, -1e4)
        probs = torch.softmax(masked_scores, dim=dim)
        probs = probs * mask.to(probs.dtype)
        denom = probs.sum(dim=dim, keepdim=True)
        probs = torch.where(
            denom > 0.0,
            probs / denom.clamp_min(1e-12),
            torch.zeros_like(probs),
        )
        return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_support_text_alignment(self, patch_tokens, metadata):
        """Align support global visual features with episode label text features."""
        support_mask = metadata['support_mask'].bool()
        episode_positive_labels = metadata['episode_positive_labels'].float()
        base_pt_mask = (
            metadata['pred_query_mask']
            if self.cfg.POINT_INFO.USE_PT_QUERY_MASK
            else metadata['pred_visibility']
        )
        episode_class_ids = metadata['episode_class_ids'].long()
        episode_class_ids = (
            episode_class_ids[0]
            if episode_class_ids.ndim == 2
            else episode_class_ids
        )
        if support_mask.sum().item() == 0:
            return None

        support_patch_tokens = patch_tokens[support_mask]
        support_point_mask = base_pt_mask[support_mask]
        support_targets = episode_positive_labels[support_mask].to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        valid_support = support_targets.sum(dim=-1) > 0
        if not valid_support.any():
            return None

        support_patch_tokens = support_patch_tokens[valid_support]
        support_point_mask = support_point_mask[valid_support]
        support_targets = support_targets[valid_support]

        support_global = self._masked_space_time_mean(
            support_patch_tokens,
            support_point_mask,
        )
        episode_label_text = self._get_pot_label_text_features(
            episode_class_ids,
            patch_tokens.dtype,
        )
        support_global = F.normalize(support_global.float(), dim=-1)
        episode_label_text = F.normalize(episode_label_text.float(), dim=-1)
        logits = float(self.text_align_cfg.LOGIT_SCALE) * torch.matmul(
            support_global,
            episode_label_text.transpose(0, 1),
        )

        target = support_targets / support_targets.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        pred = torch.softmax(logits, dim=-1)
        align_loss = ((pred - target.float()) ** 2).sum(dim=-1).mean()
        return {
            'text_align_loss': torch.nan_to_num(
                align_loss.to(dtype=patch_tokens.dtype),
                nan=0.0,
                posinf=1e4,
                neginf=0.0,
            ),
        }

    def _aggregate_weighted_st_support_tokens(self, feat, point_mask, st_weights):
        """Aggregate spatio-temporal token weights into per-frame support prototypes."""
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        point_mask = point_mask.to(device=feat.device).bool()
        st_weights = torch.nan_to_num(
            st_weights.to(device=feat.device, dtype=feat.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        temporal_dim, num_points = point_mask.shape
        st_weights = st_weights.view(temporal_dim, num_points)

        frame_weights = point_mask.to(feat.dtype) * st_weights
        weighted_denom = frame_weights.sum(dim=1, keepdim=True)
        weighted_proto = (
            feat * frame_weights.unsqueeze(-1)
        ).sum(dim=1) / weighted_denom.clamp_min(1e-6)

        valid_weights = point_mask.to(feat.dtype)
        valid_proto = (
            feat * valid_weights.unsqueeze(-1)
        ).sum(dim=1) / valid_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        proto = torch.where(
            weighted_denom > 0.0,
            weighted_proto,
            valid_proto,
        )
        return torch.nan_to_num(proto, nan=0.0, posinf=0.0, neginf=0.0)

    def _safe_l2_normalize(self, value, dim=-1):
        """Normalize features without producing NaNs for zero vectors."""
        value = torch.nan_to_num(
            value.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return value / value.norm(dim=dim, keepdim=True).clamp_min(1e-12)

    def _compute_batched_point_text_similarity(
        self,
        patch_tokens,
        label_text_features,
    ):
        """Compute post-Pointformer cosine as ``[B,K,T,N]``."""
        if patch_tokens.ndim != 4:
            raise ValueError(
                "patch_tokens must have shape [B,T,N,D]; got "
                f"{tuple(patch_tokens.shape)}."
            )
        if label_text_features.ndim != 2:
            raise ValueError(
                "label_text_features must have shape [K,D]; got "
                f"{tuple(label_text_features.shape)}."
            )
        if patch_tokens.shape[-1] != label_text_features.shape[-1]:
            raise ValueError(
                "patch/text feature dimensions differ: "
                f"{patch_tokens.shape[-1]} versus "
                f"{label_text_features.shape[-1]}."
            )
        token_norm = self._safe_l2_normalize(patch_tokens, dim=-1)
        text_norm = self._safe_l2_normalize(
            label_text_features.to(device=patch_tokens.device),
            dim=-1,
        )
        similarity = torch.einsum("kc,btnc->bktn", text_norm, token_norm)
        return torch.nan_to_num(
            similarity,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

    def _get_temporal_similarity_mask(self, point_mask, metadata):
        """Keep temporal observation validity separate from spatial routing."""
        if not bool(getattr(self, "use_temporal_similarity_refinement", False)):
            return None
        visibility = metadata.get("pred_visibility")
        if visibility is None or tuple(visibility.shape) != tuple(point_mask.shape):
            raise ValueError(
                "TEMPORAL_REFINEMENT requires pred_visibility matching point_mask."
            )
        return point_mask.bool() & visibility.to(device=point_mask.device).bool()

    def _refine_trajectory_similarity(self, similarity, point_mask, softmax_tau=None):
        if not bool(getattr(self, "use_temporal_similarity_refinement", False)):
            return similarity
        refiner = getattr(self, "temporal_similarity_refiner", None)
        if refiner is None:
            return similarity
        if softmax_tau is None:
            softmax_tau = getattr(self.pot_route_cfg, "FRAME_SOFTMAX_TAU", 0.07)
        tau = max(float(softmax_tau), 1e-6)
        # The refiner returns a bounded logit residual; convert it to the
        # similarity units consumed by the canonical spatial-softmax builder.
        delta_logits = refiner(similarity, point_mask)
        return similarity.float() + tau * delta_logits.float()

    def _compute_frame_softmax_text_prototypes(
        self,
        patch_tokens,
        point_mask,
        label_text_features,
        temporal_point_mask=None,
    ):
        """Build per-text frame prototypes with patch-wise masked softmax."""
        label_text_features = torch.nan_to_num(
            label_text_features.to(device=patch_tokens.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        token_norm = self._safe_l2_normalize(patch_tokens, dim=-1)
        text_norm = self._safe_l2_normalize(label_text_features, dim=-1)
        similarity = torch.einsum("kc,tnc->ktn", text_norm, token_norm)
        similarity = torch.nan_to_num(
            similarity,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        tau = max(float(getattr(self.pot_route_cfg, "FRAME_SOFTMAX_TAU", 0.07)), 1e-6)
        similarity = self._refine_trajectory_similarity(
            similarity,
            point_mask if temporal_point_mask is None else temporal_point_mask,
            softmax_tau=tau,
        )
        return self._compute_frame_softmax_prototypes_from_similarity(
            patch_tokens,
            point_mask,
            similarity,
            softmax_tau=tau,
        )

    def _compute_batched_frame_softmax_text_prototypes(
        self,
        patch_tokens,
        point_mask,
        label_text_features,
        temporal_point_mask=None,
    ):
        """Batched equivalent of the canonical text-routed prototype path."""
        similarity = self._compute_batched_point_text_similarity(
            patch_tokens,
            label_text_features,
        )
        tau = max(
            float(getattr(self.pot_route_cfg, "FRAME_SOFTMAX_TAU", 0.07)),
            1e-6,
        )
        similarity = self._refine_trajectory_similarity(
            similarity,
            point_mask if temporal_point_mask is None else temporal_point_mask,
            softmax_tau=tau,
        )
        return self._compute_batched_frame_softmax_prototypes_from_similarity(
            patch_tokens,
            point_mask,
            similarity,
            softmax_tau=tau,
        )

    def _compute_batched_frame_softmax_prototypes_from_similarity(
        self,
        patch_tokens,
        point_mask,
        similarity,
        softmax_tau=None,
    ):
        """Build ``[B,K,T,D]`` prototypes without per-sample/class loops."""
        if patch_tokens.ndim != 4 or similarity.ndim != 4:
            raise ValueError(
                "Batched frame-softmax expects [B,T,N,D] tokens and "
                "[B,K,T,N] similarity."
            )
        batch, temporal_dim, num_points, _ = patch_tokens.shape
        if tuple(point_mask.shape) != (batch, temporal_dim, num_points):
            raise ValueError("point_mask must match patch tokens [B,T,N].")
        if (
            similarity.shape[0] != batch
            or tuple(similarity.shape[2:]) != (temporal_dim, num_points)
        ):
            raise ValueError(
                "Batched similarity must match patch tokens [B,K,T,N]."
            )

        patch_tokens = torch.nan_to_num(
            patch_tokens, nan=0.0, posinf=0.0, neginf=0.0,
        )
        point_mask = point_mask.to(device=patch_tokens.device).bool()
        similarity = torch.nan_to_num(
            similarity.to(device=patch_tokens.device).float(),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        if softmax_tau is None:
            softmax_tau = getattr(
                self.pot_route_cfg,
                "FRAME_SOFTMAX_TAU",
                0.07,
            )
        patch_weights = self._masked_softmax_1d(
            similarity,
            point_mask.unsqueeze(1),
            dim=-1,
            tau=max(float(softmax_tau), 1e-6),
        ).to(dtype=patch_tokens.dtype)

        frame_weights = patch_weights * point_mask.unsqueeze(1).to(
            patch_weights.dtype,
        )
        weighted_denom = frame_weights.sum(dim=-1, keepdim=True)
        weighted_proto = torch.einsum(
            "bktn,btnd->bktd",
            frame_weights,
            patch_tokens,
        ) / weighted_denom.clamp_min(1e-6)

        valid_weights = point_mask.to(patch_tokens.dtype)
        valid_proto = torch.einsum(
            "btn,btnd->btd",
            valid_weights,
            patch_tokens,
        ) / valid_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        prototypes = torch.where(
            weighted_denom > 0.0,
            weighted_proto,
            valid_proto.unsqueeze(1),
        )
        return (
            torch.nan_to_num(
                prototypes, nan=0.0, posinf=0.0, neginf=0.0,
            ),
            torch.nan_to_num(
                patch_weights, nan=0.0, posinf=0.0, neginf=0.0,
            ),
        )

    def _compute_frame_softmax_prototypes_from_similarity(
        self,
        patch_tokens,
        point_mask,
        similarity,
        softmax_tau=None,
    ):
        """Build per-frame prototypes from a precomputed [K,T,N] cost.

        ``softmax_tau`` is optional for backwards compatibility.  The local
        confuser-aware route uses it to convert a bounded logit residual back
        into the same similarity units while reusing this canonical masked
        softmax/aggregation implementation.
        """
        patch_tokens = torch.nan_to_num(
            patch_tokens,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        point_mask = point_mask.to(device=patch_tokens.device).bool()
        similarity = similarity.to(device=patch_tokens.device).float()

        num_labels = similarity.shape[0]
        temporal_dim, num_points, feat_dim = patch_tokens.shape
        expected_shape = (num_labels, temporal_dim, num_points)
        if tuple(similarity.shape) != expected_shape:
            raise ValueError(
                "Frame-softmax similarity must have shape [K,T,N] matching "
                f"the patch tokens; got {tuple(similarity.shape)}, expected "
                f"{expected_shape}."
            )
        if num_labels == 0:
            return (
                patch_tokens.new_zeros(0, temporal_dim, feat_dim),
                patch_tokens.new_zeros(0, temporal_dim, num_points),
            )

        similarity = torch.nan_to_num(
            similarity,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        if softmax_tau is None:
            softmax_tau = getattr(
                self.pot_route_cfg,
                "FRAME_SOFTMAX_TAU",
                0.07,
            )
        tau = max(float(softmax_tau), 1e-6)
        patch_weights = self._masked_softmax_1d(
            similarity,
            point_mask.unsqueeze(0),
            dim=-1,
            tau=tau,
        ).to(dtype=patch_tokens.dtype)

        prototypes = torch.stack(
            [
                self._aggregate_weighted_st_support_tokens(
                    patch_tokens,
                    point_mask,
                    patch_weights[label_idx],
                )
                for label_idx in range(num_labels)
            ],
            dim=0,
        )
        prototypes = torch.nan_to_num(
            prototypes,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return prototypes, patch_weights

    def _get_query_null_score(self):
        """Map the Query Null score parameter into its configured cosine range."""
        score_min = float(self.query_null_cfg.SCORE_MIN)
        score_max = float(self.query_null_cfg.SCORE_MAX)
        return score_min + (score_max - score_min) * torch.sigmoid(
            self.query_null_score_raw.float()
        )

    def _compute_frame_softmax_query_prototypes_with_null(
        self,
        patch_tokens,
        point_mask,
        label_text_features,
    ):
        """Build Query frame prototypes with one shared Null alternative."""
        label_text_features = torch.nan_to_num(
            label_text_features.to(device=patch_tokens.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        token_norm = self._safe_l2_normalize(patch_tokens, dim=-1)
        text_norm = self._safe_l2_normalize(label_text_features, dim=-1)
        similarity = torch.einsum("kc,tnc->ktn", text_norm, token_norm)
        similarity = torch.nan_to_num(
            similarity,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        return (
            self
            ._compute_frame_softmax_query_prototypes_with_null_from_similarity(
                patch_tokens,
                point_mask,
                similarity,
            )
        )

    def _compute_frame_softmax_query_prototypes_with_null_from_similarity(
        self,
        patch_tokens,
        point_mask,
        similarity,
    ):
        """Route Query patch and Null mass from a precomputed ``[K,T,N]`` score."""
        output_dtype = patch_tokens.dtype
        patch_tokens_fp32 = torch.nan_to_num(
            patch_tokens.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        point_mask = point_mask.to(device=patch_tokens.device).bool()
        similarity = similarity.to(device=patch_tokens.device).float()

        temporal_dim, num_points, feat_dim = patch_tokens.shape
        if tuple(point_mask.shape) != (temporal_dim, num_points):
            raise ValueError(
                "Query point_mask must have shape [T,N] matching patch tokens; "
                f"got {tuple(point_mask.shape)}, expected "
                f"{(temporal_dim, num_points)}."
            )
        if similarity.ndim != 3:
            raise ValueError(
                "Query Null similarity must have shape [K,T,N]; got "
                f"{tuple(similarity.shape)}."
            )
        num_labels = similarity.shape[0]
        expected_shape = (num_labels, temporal_dim, num_points)
        if tuple(similarity.shape) != expected_shape:
            raise ValueError(
                "Query Null similarity must have shape [K,T,N] matching the "
                f"patch tokens; got {tuple(similarity.shape)}, expected "
                f"{expected_shape}."
            )
        if num_labels == 0:
            return (
                patch_tokens.new_zeros(0, temporal_dim, feat_dim),
                patch_tokens.new_zeros(0, temporal_dim, num_points),
                patch_tokens.new_zeros(0, temporal_dim),
            )

        similarity = torch.nan_to_num(
            similarity,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        tau = max(
            float(getattr(self.pot_route_cfg, "FRAME_SOFTMAX_TAU", 0.07)),
            1e-6,
        )
        expanded_mask = point_mask.unsqueeze(0).expand_as(similarity)
        patch_logits = (similarity / tau).masked_fill(~expanded_mask, -1e4)

        null_score = self._get_query_null_score().to(device=patch_tokens.device)
        null_logits = (null_score / tau).view(1, 1, 1).expand(
            num_labels,
            temporal_dim,
            1,
        )
        if bool(getattr(
            self.query_null_cfg,
            "CARDINALITY_CORRECTION",
            True,
        )):
            valid_count = point_mask.sum(dim=-1).clamp_min(1).float()
            null_logits = (
                null_logits
                + valid_count.log().view(1, temporal_dim, 1)
            )

        joint_weights = torch.softmax(
            torch.cat((patch_logits, null_logits), dim=-1),
            dim=-1,
        )
        patch_weights = (
            joint_weights[..., :-1]
            * expanded_mask.to(dtype=joint_weights.dtype)
        )
        null_weights = joint_weights[..., -1]

        mask_fp32 = point_mask.float()
        token_norms = patch_tokens_fp32.norm(dim=-1)
        valid_count_fp32 = mask_fp32.sum(dim=-1)
        frame_scale = (
            token_norms * mask_fp32
        ).sum(dim=-1) / valid_count_fp32.clamp_min(1.0)
        frame_scale = torch.where(
            valid_count_fp32 > 0.0,
            frame_scale,
            torch.ones_like(frame_scale),
        )
        if bool(getattr(
            self.query_null_cfg,
            "DETACH_FRAME_SCALE",
            True,
        )):
            frame_scale = frame_scale.detach()

        null_unit = self._safe_l2_normalize(
            self.query_null_token.to(device=patch_tokens.device),
            dim=-1,
        ).squeeze(0)
        null_values = (
            float(getattr(self.query_null_cfg, "VALUE_SCALE", 1.0))
            * frame_scale.unsqueeze(-1)
            * null_unit.unsqueeze(0)
        )
        patch_component = torch.einsum(
            "ktn,tnd->ktd",
            patch_weights.float(),
            patch_tokens_fp32,
        )
        null_component = (
            null_weights.unsqueeze(-1)
            * null_values.unsqueeze(0)
        )
        query_prototypes = torch.nan_to_num(
            patch_component + null_component,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return (
            query_prototypes.to(dtype=output_dtype),
            patch_weights.to(dtype=output_dtype),
            null_weights,
        )

    def _compute_query_null_support_cosines(
        self,
        support_prototypes,
        detach_support,
    ):
        """Return Null-to-Support frame cosines and their validity mask."""
        if support_prototypes.ndim != 3:
            raise ValueError(
                "support_prototypes must have shape [K,T,D]; got "
                f"{tuple(support_prototypes.shape)}."
            )
        support = torch.nan_to_num(
            support_prototypes.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if detach_support:
            support = support.detach()
        valid = support.norm(dim=-1) > 1e-12
        support_unit = self._safe_l2_normalize(support, dim=-1)
        null_unit = self._safe_l2_normalize(
            self.query_null_token.to(device=support.device),
            dim=-1,
        ).squeeze(0)
        cosine = torch.einsum("d,ktd->kt", null_unit, support_unit)
        cosine = torch.nan_to_num(
            cosine,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        return cosine, valid

    def _compute_query_null_orthogonal_loss(self, support_prototypes):
        """Penalize squared cosine between the Null token and valid Support frames."""
        cosine, valid = self._compute_query_null_support_cosines(
            support_prototypes,
            detach_support=bool(getattr(
                self.query_null_cfg,
                "ORTHO_DETACH_SUPPORT",
                True,
            )),
        )
        if not valid.any():
            return cosine.new_zeros(())
        return cosine[valid].square().mean()

    def _compute_query_null_support_cosine_stats(self, support_prototypes):
        """Return detached mean/max absolute Null-to-Support cosine diagnostics."""
        cosine, valid = self._compute_query_null_support_cosines(
            support_prototypes,
            detach_support=True,
        )
        if not valid.any():
            zero = cosine.new_zeros(())
            return zero, zero
        absolute = cosine[valid].abs()
        return absolute.mean().detach(), absolute.max().detach()

    def _build_frame_softmax_support_prototypes(
        self,
        value_tokens,
        point_mask,
        support_mask,
        episode_positive_labels,
        episode_label_text,
        precomputed_similarity=None,
        temporal_point_mask=None,
    ):
        """Average true-label text-routed support prototypes by episode class."""
        support_mask = support_mask.to(device=value_tokens.device).bool()
        point_mask = point_mask.to(device=value_tokens.device).bool()
        episode_positive_labels = episode_positive_labels.to(
            device=value_tokens.device,
        ).bool()
        num_labels = episode_label_text.shape[0]
        temporal_dim = value_tokens.shape[1]
        feat_dim = value_tokens.shape[-1]
        support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
        if support_indices.numel() == 0:
            return value_tokens.new_zeros(num_labels, temporal_dim, feat_dim)

        support_tokens = value_tokens.index_select(0, support_indices)
        support_points = point_mask.index_select(0, support_indices)
        if precomputed_similarity is None:
            support_temporal_points = (
                None
                if temporal_point_mask is None
                else temporal_point_mask.index_select(0, support_indices)
            )
            all_support_prototypes, _ = (
                self._compute_batched_frame_softmax_text_prototypes(
                    support_tokens,
                    support_points,
                    episode_label_text,
                    temporal_point_mask=support_temporal_points,
                )
            )
        else:
            all_support_prototypes, _ = (
                self._compute_batched_frame_softmax_prototypes_from_similarity(
                    support_tokens,
                    support_points,
                    precomputed_similarity.index_select(0, support_indices),
                )
            )

        per_class_prototypes = [[] for _ in range(num_labels)]
        for support_local_idx, sample_idx in enumerate(support_indices.tolist()):
            positive_indices = torch.nonzero(
                episode_positive_labels[sample_idx],
                as_tuple=False,
            ).flatten()
            if positive_indices.numel() == 0:
                continue
            sample_prototypes = all_support_prototypes[
                support_local_idx
            ].index_select(0, positive_indices)
            for local_idx, class_idx in enumerate(positive_indices.tolist()):
                per_class_prototypes[class_idx].append(sample_prototypes[local_idx])

        support_prototypes = value_tokens.new_zeros(
            num_labels,
            temporal_dim,
            feat_dim,
        )
        for class_idx, class_prototypes in enumerate(per_class_prototypes):
            if class_prototypes:
                support_prototypes[class_idx] = torch.stack(
                    class_prototypes,
                    dim=0,
                ).mean(dim=0)
        return torch.nan_to_num(
            support_prototypes,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _fuse_episode_text_with_support_visual(
        self,
        episode_label_text,
        support_prototypes,
    ):
        """Fuse episode text with masked temporal support prototypes for Query."""
        if episode_label_text.ndim != 2:
            raise ValueError(
                "episode_label_text must have shape [K,D]; got "
                f"{tuple(episode_label_text.shape)}."
            )
        if support_prototypes.ndim != 3:
            raise ValueError(
                "support_prototypes must have shape [K,T,D]; got "
                f"{tuple(support_prototypes.shape)}."
            )
        expected_shape = (
            episode_label_text.shape[0],
            episode_label_text.shape[1],
        )
        if (
            support_prototypes.shape[0] != expected_shape[0]
            or support_prototypes.shape[2] != expected_shape[1]
        ):
            raise ValueError(
                "Support/text class or feature dimensions differ: text is "
                f"{tuple(episode_label_text.shape)}, support is "
                f"{tuple(support_prototypes.shape)}."
            )

        output_dtype = episode_label_text.dtype
        output_device = episode_label_text.device
        support_frames = torch.nan_to_num(
            support_prototypes.to(device=output_device).float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid_frames = support_frames.norm(dim=-1) > 1e-12
        valid_classes = valid_frames.any(dim=1)
        frame_count = valid_frames.sum(dim=1, keepdim=True).to(
            dtype=support_frames.dtype,
        )
        support_visual = (
            support_frames * valid_frames.unsqueeze(-1).to(support_frames.dtype)
        ).sum(dim=1) / frame_count.clamp_min(1.0)

        fusion_cfg = getattr(self, "support_text_fusion_cfg", None)
        if bool(getattr(fusion_cfg, "VISUAL_DETACH", True)):
            support_visual = support_visual.detach()
        text_weight = float(getattr(fusion_cfg, "TEXT_WEIGHT", 1.0))
        visual_weight = float(getattr(fusion_cfg, "VISUAL_WEIGHT", 1.0))
        if (
            not np.isfinite(text_weight)
            or not np.isfinite(visual_weight)
            or text_weight < 0.0
            or visual_weight < 0.0
            or text_weight + visual_weight <= 0.0
        ):
            raise ValueError(
                "SUPPORT_TEXT_FUSION weights must be finite, non-negative, "
                "and not both zero."
            )

        text_norm = self._safe_l2_normalize(
            episode_label_text.to(device=output_device),
            dim=-1,
        )
        visual_norm = self._safe_l2_normalize(support_visual, dim=-1)
        combined = text_weight * text_norm + visual_weight * visual_norm
        combined_valid = combined.norm(dim=-1) > 1e-12
        fused = self._safe_l2_normalize(combined, dim=-1)
        use_fused = valid_classes & combined_valid
        fused = torch.where(use_fused.unsqueeze(-1), fused, text_norm)

        return (
            fused.to(device=output_device, dtype=output_dtype),
            visual_norm.to(device=output_device, dtype=output_dtype),
            valid_classes.to(device=output_device),
        )

    def _sample_dense_cost_at_tracks(self, dense_cost, pred_tracks):
        """Sample [B,T,K,H,W] dense costs at [B,T,N,2] track coordinates."""
        if dense_cost.ndim != 5:
            raise ValueError(
                "Dense cost must have shape [B,T,K,H,W]; got "
                f"{tuple(dense_cost.shape)}."
            )
        if pred_tracks.ndim != 4 or pred_tracks.shape[-1] != 2:
            raise ValueError(
                "pred_tracks must have shape [B,T,N,2]; got "
                f"{tuple(pred_tracks.shape)}."
            )
        batch_size, temporal_dim, num_labels, _, _ = dense_cost.shape
        if tuple(pred_tracks.shape[:2]) != (batch_size, temporal_dim):
            raise ValueError(
                "Dense cost and pred_tracks must share B,T; got "
                f"{tuple(dense_cost.shape[:2])} and "
                f"{tuple(pred_tracks.shape[:2])}."
            )

        model_cfg = getattr(self.cfg, "MODEL", None)
        sample_mode = str(
            getattr(model_cfg, "FEAT_EXTRACT_MODE", "nearest")
        ).lower()
        # Preserve the exact FP32 pred_tracks coordinates used by the original
        # dense-feature sampler. Casting the grid to an autocast FP16 cost dtype
        # can move boundary coordinates into a different nearest patch cell.
        with torch.autocast(device_type=dense_cost.device.type, enabled=False):
            dense_cost_bt = rearrange(
                dense_cost,
                "b t k h w -> (b t) k h w",
            ).float()
            track_grid = rearrange(
                pred_tracks,
                "b t n c -> (b t) n 1 c",
            ).to(device=dense_cost.device, dtype=torch.float32)
            track_grid = torch.nan_to_num(
                track_grid,
                nan=2.0,
                posinf=2.0,
                neginf=-2.0,
            )
            sampled_cost = F.grid_sample(
                dense_cost_bt,
                track_grid,
                mode=sample_mode,
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(-1)
        sampled_cost = rearrange(
            sampled_cost,
            "(b t) k n -> b k t n",
            b=batch_size,
            t=temporal_dim,
            k=num_labels,
        )
        return torch.nan_to_num(
            sampled_cost,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _rasterize_point_cost_and_guidance(
        self,
        patch_tokens,
        point_similarity,
        pred_tracks,
        point_mask,
        resolution,
    ):
        """Nearest-scatter trajectory costs/features onto their current grid.

        Colliding trajectories are averaged.  Cells receiving no valid
        trajectory remain exactly zero and are marked false in the returned
        occupancy mask.
        """
        if patch_tokens.ndim != 4:
            raise ValueError(
                "patch_tokens must have shape [B,T,N,D]; got "
                f"{tuple(patch_tokens.shape)}."
            )
        if point_similarity.ndim != 4:
            raise ValueError(
                "point_similarity must have shape [B,K,T,N]; got "
                f"{tuple(point_similarity.shape)}."
            )
        if pred_tracks.ndim != 4 or pred_tracks.shape[-1] != 2:
            raise ValueError(
                "pred_tracks must have shape [B,T,N,2]; got "
                f"{tuple(pred_tracks.shape)}."
            )
        if point_mask.ndim != 3:
            raise ValueError(
                "point_mask must have shape [B,T,N]; got "
                f"{tuple(point_mask.shape)}."
            )
        height, width = (int(resolution[0]), int(resolution[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}.")

        batch, temporal, num_points, channels = patch_tokens.shape
        num_labels = point_similarity.shape[1]
        if tuple(point_similarity.shape) != (
            batch,
            num_labels,
            temporal,
            num_points,
        ):
            raise ValueError("point_similarity does not match patch token B/T/N.")
        if tuple(pred_tracks.shape[:3]) != (batch, temporal, num_points):
            raise ValueError("pred_tracks does not match patch token B/T/N.")
        if tuple(point_mask.shape) != (batch, temporal, num_points):
            raise ValueError("point_mask does not match patch token B/T/N.")

        device = patch_tokens.device
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = torch.nan_to_num(
                patch_tokens.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            similarity = torch.nan_to_num(
                point_similarity.float(),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).clamp(-1.0, 1.0)
            tracks = pred_tracks.to(device=device, dtype=torch.float32)
            finite_tracks = torch.isfinite(tracks).all(dim=-1)
            in_bounds = (
                tracks.ge(-1.0).all(dim=-1)
                & tracks.le(1.0).all(dim=-1)
            )
            valid = (
                point_mask.to(device=device).bool()
                & finite_tracks
                & in_bounds
            )
            safe_tracks = torch.nan_to_num(
                tracks,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).clamp(-1.0, 1.0)
            x_index = torch.round(
                (safe_tracks[..., 0] + 1.0) * 0.5 * max(width - 1, 0)
            ).long()
            y_index = torch.round(
                (safe_tracks[..., 1] + 1.0) * 0.5 * max(height - 1, 0)
            ).long()
            flat_index = y_index * width + x_index
            cells = height * width
            valid_float = valid.float()

            counts = torch.zeros(
                batch,
                temporal,
                cells,
                device=device,
                dtype=torch.float32,
            ).scatter_add(2, flat_index, valid_float)

            guidance_index = flat_index.unsqueeze(-1).expand(
                -1, -1, -1, channels
            )
            guidance_sum = torch.zeros(
                batch,
                temporal,
                cells,
                channels,
                device=device,
                dtype=torch.float32,
            ).scatter_add(
                2,
                guidance_index,
                tokens * valid_float.unsqueeze(-1),
            )

            cost_source = similarity.permute(0, 2, 1, 3)
            cost_index = flat_index.unsqueeze(2).expand(
                -1, -1, num_labels, -1
            )
            cost_sum = torch.zeros(
                batch,
                temporal,
                num_labels,
                cells,
                device=device,
                dtype=torch.float32,
            ).scatter_add(
                3,
                cost_index,
                cost_source * valid_float.unsqueeze(2),
            )

            denominator = counts.clamp_min(1.0)
            dense_guidance = guidance_sum / denominator.unsqueeze(-1)
            dense_cost = cost_sum / denominator.unsqueeze(2)
            occupancy = counts > 0.0

            dense_guidance = dense_guidance.reshape(
                batch, temporal, height, width, channels
            )
            dense_cost = dense_cost.reshape(
                batch, temporal, num_labels, height, width
            )
            occupancy = occupancy.reshape(batch, temporal, height, width)
        return dense_cost, dense_guidance, occupancy

    def _compute_cat_refined_point_similarity(
        self,
        patch_tokens,
        point_mask,
        pred_tracks,
        label_text_features,
    ):
        """Replace point cosine with masked CAT cost built from the same tokens."""
        if patch_tokens is None or pred_tracks is None:
            raise ValueError(
                "COST_AGG requires post-Pointformer patch_tokens and pred_tracks."
            )
        point_similarity = self._compute_batched_point_text_similarity(
            patch_tokens,
            label_text_features,
        )
        resolution = self.cat_spatial_cost_aggregator.input_resolution
        dense_cost, dense_guidance, occupancy = (
            self._rasterize_point_cost_and_guidance(
                patch_tokens,
                point_similarity,
                pred_tracks,
                point_mask,
                resolution,
            )
        )
        refined_dense_cost = (
            self.cat_spatial_cost_aggregator.forward_precomputed(
                dense_cost,
                dense_guidance,
                occupancy,
                text_features=label_text_features,
            )
        )
        refined_point_cost = self._sample_dense_cost_at_tracks(
            refined_dense_cost,
            pred_tracks,
        )
        return torch.nan_to_num(
            refined_point_cost,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

    def _compute_split_cat_refined_point_similarity(
        self,
        patch_tokens,
        point_mask,
        pred_tracks,
        support_mask,
        episode_positive_labels,
        episode_class_ids,
        episode_label_text,
        raw_positive_labels=None,
    ):
        """Apply CAT with separate support/query label information.

        Query samples are refined jointly against every candidate on the
        episode label axis.  A support sample is instead refined against only
        its known *global* ground-truth labels.  The refined support costs that
        overlap the episode axis are then mapped back to their episode slots.

        Importantly, query rows from ``raw_positive_labels`` are never read.
        They are evaluation targets, not model inputs.  If global labels are
        unavailable, support samples fall back to their known positive labels
        on the episode axis.
        """
        device = patch_tokens.device
        batch = patch_tokens.shape[0]
        support_mask = support_mask.to(device=device).bool().flatten()
        if support_mask.numel() != batch:
            raise ValueError(
                "support_mask must have one entry per sample; got "
                f"{tuple(support_mask.shape)} for batch size {batch}."
            )

        episode_class_ids = episode_class_ids.to(
            device=device,
            dtype=torch.long,
        ).flatten()
        episode_positive_labels = episode_positive_labels.to(
            device=device,
        ).bool()
        expected_positive_shape = (batch, episode_class_ids.numel())
        if tuple(episode_positive_labels.shape) != expected_positive_shape:
            raise ValueError(
                "episode_positive_labels must have shape [B,K]; got "
                f"{tuple(episode_positive_labels.shape)}, expected "
                f"{expected_positive_shape}."
            )
        if tuple(episode_label_text.shape[:1]) != (
            episode_class_ids.numel(),
        ):
            raise ValueError(
                "episode label ids/text length mismatch: got "
                f"{episode_class_ids.numel()} ids and "
                f"{episode_label_text.shape[0]} text features."
            )

        support_raw_labels = None
        if raw_positive_labels is not None:
            raw_positive_labels = raw_positive_labels.to(device=device).bool()
            raw_positive_labels = raw_positive_labels.reshape(batch, -1)
            # Slice only support rows.  Query ground truth must not influence
            # either its label axis or any class-attention computation.
            support_indices = torch.nonzero(
                support_mask,
                as_tuple=False,
            ).flatten()
            support_raw_labels = raw_positive_labels.index_select(
                0,
                support_indices,
            )
        else:
            support_indices = torch.nonzero(
                support_mask,
                as_tuple=False,
            ).flatten()

        # Non-positive support/episode slots remain zero and are never consumed
        # downstream.  In particular, support samples do not even compute a
        # cost against non-ground-truth episode labels.
        refined_similarity = torch.zeros(
            batch,
            episode_class_ids.numel(),
            patch_tokens.shape[1],
            patch_tokens.shape[2],
            device=device,
            dtype=torch.float32,
        )

        query_indices = torch.nonzero(
            ~support_mask,
            as_tuple=False,
        ).flatten()
        if query_indices.numel() > 0:
            query_similarity = self._compute_cat_refined_point_similarity(
                patch_tokens.index_select(0, query_indices),
                point_mask.index_select(0, query_indices),
                pred_tracks.index_select(0, query_indices),
                episode_label_text,
            )
            refined_similarity = refined_similarity.index_copy(
                0,
                query_indices,
                query_similarity,
            )

        for local_support_idx, sample_idx_tensor in enumerate(support_indices):
            sample_idx = int(sample_idx_tensor.item())
            positive_episode_indices = torch.nonzero(
                episode_positive_labels[sample_idx],
                as_tuple=False,
            ).flatten()
            fallback_global_labels = episode_class_ids.index_select(
                0,
                positive_episode_indices,
            )

            if support_raw_labels is None:
                true_global_labels = fallback_global_labels
            else:
                true_global_labels = torch.nonzero(
                    support_raw_labels[local_support_idx],
                    as_tuple=False,
                ).flatten().to(dtype=torch.long)
                # Metadata should agree, but keep all known episode positives
                # if a partially populated raw-label vector is supplied.
                if fallback_global_labels.numel() > 0:
                    true_global_labels = torch.unique(
                        torch.cat(
                            (true_global_labels, fallback_global_labels),
                            dim=0,
                        ),
                        sorted=True,
                    )

            if true_global_labels.numel() == 0:
                continue

            true_label_text = self._get_pot_label_text_features(
                true_global_labels,
                patch_tokens.dtype,
            )
            true_label_similarity = self._compute_cat_refined_point_similarity(
                patch_tokens[sample_idx : sample_idx + 1],
                point_mask[sample_idx : sample_idx + 1],
                pred_tracks[sample_idx : sample_idx + 1],
                true_label_text,
            )[0]

            true_axis_indices, episode_axis_indices = torch.nonzero(
                true_global_labels[:, None] == episode_class_ids[None, :],
                as_tuple=True,
            )
            if episode_axis_indices.numel() == 0:
                continue
            sample_similarity = refined_similarity[sample_idx].index_copy(
                0,
                episode_axis_indices,
                true_label_similarity.index_select(0, true_axis_indices),
            )
            refined_similarity = refined_similarity.index_copy(
                0,
                sample_idx_tensor.reshape(1),
                sample_similarity.unsqueeze(0),
            )

        return torch.nan_to_num(
            refined_similarity,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

    def _compute_bidirectional_frame_similarity(
        self,
        query_prototypes_frame,
        support_prototypes_frame,
    ):
        """Return diagonal per-class bidirectional frame cosine similarity."""
        # query_prototypes_frame: [Q, N, Tq, C]; support_prototypes_frame: [N, Ts, C]
        query_norm = self._safe_l2_normalize(query_prototypes_frame, dim=-1)
        support_norm = self._safe_l2_normalize(support_prototypes_frame, dim=-1)
        sim = torch.einsum("qntc,nsc->qnts", query_norm, support_norm)
        sim = torch.nan_to_num(
            sim,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        bidir_sim = 0.5 * (
            sim.max(dim=3)[0].mean(dim=2) + sim.max(dim=2)[0].mean(dim=2)
        )
        return torch.nan_to_num(
            bidir_sim,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

    def _build_frame_softmax_q2s_aux(
        self,
        value_tokens,
        metadata,
        pred_tracks=None,
        matchability_evidence_tokens=None,
    ):
        """Build direct text-routed frame prototypes and q2s logits."""
        support_mask = metadata["support_mask"].to(device=value_tokens.device).bool()
        query_mask = ~support_mask
        if not query_mask.any():
            return None

        point_mask = (
            metadata["pred_query_mask"]
            if self.cfg.POINT_INFO.USE_PT_QUERY_MASK
            else metadata["pred_visibility"]
        ).to(device=value_tokens.device).bool()
        temporal_point_mask = self._get_temporal_similarity_mask(point_mask, metadata)
        episode_positive_labels = metadata["episode_positive_labels"].to(
            device=value_tokens.device,
        ).bool()
        episode_class_ids = metadata["episode_class_ids"].to(
            device=value_tokens.device,
        ).long()
        episode_class_ids = (
            episode_class_ids[0]
            if episode_class_ids.ndim == 2
            else episode_class_ids
        ).flatten()
        if episode_class_ids.numel() == 0:
            return None

        episode_label_text = self._get_pot_label_text_features(
            episode_class_ids,
            value_tokens.dtype,
        )
        refined_similarity = None
        if bool(getattr(self, "use_cat_cost_aggregation", False)):
            refined_similarity = self._compute_split_cat_refined_point_similarity(
                value_tokens,
                point_mask,
                pred_tracks,
                support_mask,
                episode_positive_labels,
                episode_class_ids,
                episode_label_text,
                raw_positive_labels=metadata.get("raw_positive_labels"),
            )
        support_prototypes = self._build_frame_softmax_support_prototypes(
            value_tokens,
            point_mask,
            support_mask,
            episode_positive_labels,
            episode_label_text,
            precomputed_similarity=refined_similarity,
            temporal_point_mask=temporal_point_mask,
        )
        query_label_features = episode_label_text
        support_visual = None
        support_visual_valid = None
        if bool(getattr(self, "use_support_text_fusion", False)):
            if refined_similarity is not None:
                raise RuntimeError(
                    "SUPPORT_TEXT_FUSION cannot consume CAT precomputed query costs."
                )
            (
                query_label_features,
                support_visual,
                support_visual_valid,
            ) = self._fuse_episode_text_with_support_visual(
                episode_label_text,
                support_prototypes,
            )

        query_indices = torch.nonzero(query_mask, as_tuple=False).flatten()
        query_null_weights = []
        if bool(getattr(self, "use_query_null_route", False)):
            query_prototypes = []
            for sample_idx in query_indices.tolist():
                if refined_similarity is None:
                    (
                        sample_prototypes,
                        _,
                        sample_null_weights,
                    ) = self._compute_frame_softmax_query_prototypes_with_null(
                        value_tokens[sample_idx],
                        point_mask[sample_idx],
                        query_label_features,
                    )
                else:
                    (
                        sample_prototypes,
                        _,
                        sample_null_weights,
                    ) = (
                        self
                        ._compute_frame_softmax_query_prototypes_with_null_from_similarity(
                            value_tokens[sample_idx],
                            point_mask[sample_idx],
                            refined_similarity[sample_idx],
                        )
                    )
                query_null_weights.append(sample_null_weights.unsqueeze(0))
                query_prototypes.append(sample_prototypes.unsqueeze(0))
            if not query_prototypes:
                return None
            query_prototypes = torch.cat(query_prototypes, dim=0)
        else:
            query_tokens = value_tokens.index_select(0, query_indices)
            query_points = point_mask.index_select(0, query_indices)
            if refined_similarity is None:
                query_temporal_points = (
                    None
                    if temporal_point_mask is None
                    else temporal_point_mask.index_select(0, query_indices)
                )
                query_prototypes, _ = (
                    self._compute_batched_frame_softmax_text_prototypes(
                        query_tokens,
                        query_points,
                        query_label_features,
                        temporal_point_mask=query_temporal_points,
                    )
                )
            else:
                query_prototypes, _ = (
                    self._compute_batched_frame_softmax_prototypes_from_similarity(
                        query_tokens,
                        query_points,
                        refined_similarity.index_select(0, query_indices),
                    )
                )

        diag_similarity = self._compute_bidirectional_frame_similarity(
            query_prototypes,
            support_prototypes,
        )
        alpha = float(getattr(self.pot_route_cfg, "QUERY_PARTIAL_LOGIT_ALPHA", 10.0))
        bias = float(getattr(self.pot_route_cfg, "QUERY_PARTIAL_LOGIT_BIAS", -2.0))
        q2s_logits = alpha * diag_similarity + bias
        q2s_logits = torch.nan_to_num(
            q2s_logits,
            nan=bias,
            posinf=1e4,
            neginf=-1e4,
        )
        target_label_indices = torch.arange(
            episode_class_ids.numel(),
            device=value_tokens.device,
            dtype=torch.long,
        )
        result = {
            "query_partial_q2s_logits": q2s_logits.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_partial_query_prototypes": query_prototypes.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_partial_support_prototypes": support_prototypes.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_partial_diag_similarity": diag_similarity.to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_partial_alpha_sim_term": (alpha * diag_similarity).to(
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_partial_bias_term": torch.full_like(
                q2s_logits,
                fill_value=bias,
                device=value_tokens.device,
                dtype=value_tokens.dtype,
            ),
            "query_partial_query_sample_indices": query_indices.to(
                device=value_tokens.device,
                dtype=torch.long,
            ),
            "query_partial_label_axis_global_labels": episode_class_ids.to(
                device=value_tokens.device,
                dtype=torch.long,
            ),
            "query_partial_target_label_indices": target_label_indices,
        }
        if bool(getattr(self, "use_query_null_route", False)):
            query_null_weights = torch.cat(query_null_weights, dim=0)
            null_ortho_loss = self._compute_query_null_orthogonal_loss(
                support_prototypes,
            )
            (
                support_mean_abs_cosine,
                support_max_abs_cosine,
            ) = self._compute_query_null_support_cosine_stats(
                support_prototypes,
            )
            result.update({
                "query_null_weights": query_null_weights.to(
                    device=value_tokens.device,
                    dtype=torch.float32,
                ),
                "query_null_score": self._get_query_null_score().to(
                    device=value_tokens.device,
                ),
                "query_null_orthogonal_loss": null_ortho_loss,
                "query_null_support_mean_abs_cosine": (
                    support_mean_abs_cosine
                ),
                "query_null_support_max_abs_cosine": (
                    support_max_abs_cosine
                ),
            })
        if support_visual is not None:
            result.update({
                "support_text_fusion_query_features": query_label_features.to(
                    device=value_tokens.device,
                    dtype=value_tokens.dtype,
                ),
                "support_text_fusion_visual_prototypes": support_visual.to(
                    device=value_tokens.device,
                    dtype=value_tokens.dtype,
                ),
                "support_text_fusion_valid_classes": support_visual_valid.to(
                    device=value_tokens.device,
                    dtype=torch.bool,
                ),
            })
        return result

    def get_dino_features(self, x):
        """ Get DINO features
        Args:
            x (torch.Tensor): Input features of shape [BS, T, C, H, W]

        Returns:
            torch.Tensor: DINO features of shape [BS, T, P, Q, D]
        """
        self.dino.eval()
        batch_size, num_frames, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        if self.cfg.MODEL.TRAIN_BACKBONE:
            _ = self.dino(x)
        else:
            with torch.no_grad():
                _ = self.dino(x)
        #using hooks to get the patch tokens
        feat = self.feat_dict['dino'][:, self.dino.num_register_tokens + 1 :]
        feat_size = feat.shape[-1]
        #dino patch side is fine
        feat = feat.view(batch_size, num_frames, self.patch_num_side,
                         self.patch_num_side, feat_size)

        return feat

    def get_clip_features(self, x, return_projected=False):
        """Get CLIP ViT-B/16 patch features."""
        self.clip_visual.eval()
        batch_size, num_frames, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        x = x.type(self.clip_visual.conv1.weight.dtype)
        if self.cfg.MODEL.TRAIN_BACKBONE:
            feat = self._forward_clip_visual(x, return_projected=return_projected)
        else:
            with torch.no_grad():
                feat = self._forward_clip_visual(x, return_projected=return_projected)
        if return_projected:
            raw_feat, projected_feat = feat
            raw_feat = raw_feat.float().view(
                batch_size,
                num_frames,
                self.patch_num_side,
                self.patch_num_side,
                raw_feat.shape[-1],
            )
            projected_feat = projected_feat.float().view(
                batch_size,
                num_frames,
                self.patch_num_side,
                self.patch_num_side,
                projected_feat.shape[-1],
            )
            return raw_feat, projected_feat
        feat = feat.float()
        feat = feat.view(
            batch_size,
            num_frames,
            self.patch_num_side,
            self.patch_num_side,
            feat.shape[-1],
        )
        return feat

    def _forward_clip_visual(self, x, return_projected=False):
        """Return raw CLIP patch tokens before ln_post/proj."""
        visual = self.clip_visual
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        raw_patch = x[:, 1:, :]
        if not return_projected:
            return raw_patch
        projected_patch = (visual.ln_post(x) @ visual.proj)[:, 1:, :]
        return raw_patch, projected_patch

    def get_dinotxt_features(self, x):
        """Get DinoTxt ViT-L/14 visual patch features."""
        self.dinotxt_visual_model.eval()
        batch_size, num_frames, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        if self.cfg.MODEL.TRAIN_BACKBONE:
            _, patch_tokens = self.dinotxt_visual_model.get_class_and_patch_tokens(x)
        else:
            with torch.no_grad():
                _, patch_tokens = self.dinotxt_visual_model.get_class_and_patch_tokens(x)
        patch_tokens = patch_tokens.float()
        return patch_tokens.view(
            batch_size,
            num_frames,
            self.patch_num_side,
            self.patch_num_side,
            patch_tokens.shape[-1],
        )


    def pt_forward(self, x, metadata):
        """ Forward pass for point tracking based transformer model.

        Args:
            x (torch.Tensor): Input features of shape [BS, T, N, D]
            metadata (dict): Metadata containing prediction masks

        Returns:
            torch.Tensor: Class token features
            torch.Tensor: Patch token features
        """
        if self.cfg.POINT_INFO.USE_PT_QUERY_MASK:
            pt_mask = metadata['pred_query_mask'] # [BS, T, N]
        else:
            pt_mask = metadata['pred_visibility'] # [BS, T, N]

        bs, temporal_dim, num_points, _ = x.shape
        # reshaping the input according to the attention block
        x = rearrange(x, 'b t n d -> b n t d')
        pt_mask = rearrange(pt_mask, 'b t n -> b n t')
        x = rearrange(x, 'b n t d -> b (n t) d')
        pt_mask = rearrange(pt_mask, 'b n t -> b (n t)')
        if self.cfg.MODEL.USE_CLS_TOKEN:
            cls_tokens = self.cls_token.expand(bs, -1, -1) # [BS, 1, dim]
            x = torch.cat((cls_tokens, x), dim=1) # [BS, N, dim]
            cls_token_mask = torch.ones(bs, 1).bool().to(x.device)
            pt_mask = torch.cat((cls_token_mask, pt_mask), dim=1) # [BS, N, dim]
        # Apply positional dropout
        x = self.pos_drop(x) # [BS, N, dim]
        # Encoding using transformer layers
        thw = [self.temporal_resolution, self.point_grid_size,
            int(num_points / self.point_grid_size)]
        for _, blk in enumerate(self.blocks):
            x, _ = blk(
                x,
                thw,
                pt_mask
            )
        if self.cfg.MODEL.ADAPOOLING.ENABLE:
            if self.cfg.MODEL.ADAPOOLING.TYPE == 'temporal_spatial':
                extra_cls_token =  self.agg_cls_token.expand(bs * num_points, -1, -1)
                cls_x, patch_x = self.adaptive_pooling(x, extra_cls_token)
                patch_x = rearrange(patch_x, 'b n d -> b 1 n d')

            elif self.cfg.MODEL.ADAPOOLING.TYPE == 'spatial_temporal':
                extra_cls_token = self.agg_cls_token.expand(bs * temporal_dim, -1, -1)
                cls_x, patch_x = self.adaptive_pooling(x, extra_cls_token)
                patch_x = rearrange(patch_x, 'b t d -> b t 1 d')

        else:
            x = self.norm(x)
            if self.cfg.MODEL.USE_CLS_TOKEN:
                cls_x, patch_x = x[:, 0], x[:, 1:]
                if self.cfg.MODEL.USE_PATCH_AS_CLS:
                    cls_x = patch_x.mean(dim=1)
            else:
                # If cls token is not ued, for now using global average pooling
                cls_x = x.mean(dim=1)
                patch_x = x

            cls_x = self.pre_logits(cls_x)
            # Taking the patch tokens back to the input shape
            patch_x = rearrange(patch_x, 'b (n t) d -> b t n d', t=temporal_dim)
        if not torch.isfinite(x).all():
            print("WARNING: nan in features out")
        return cls_x, patch_x

    def add_st_pos_embeddings(self, x):
        """ Add spatial and temporal positional embeddings to the input features.

        Args:
            x (torch.Tensor): Input features of shape [BS, T, P, Q, D]

        Returns:
            torch.Tensor: Output features of shape [BS, T, P, Q, D]
        """
        _, _, sp_dim_1, sp_dim_2, _ = x.shape
        # reshaping the input according to the attention block
        x = rearrange(x, 'b t p q d -> b t (p q) d')
        x = x + self.space_pos_embed.unsqueeze(0)
        x = self.space_pos_drop(x)
        x = rearrange(x, 'b t p d -> b p t d')
        x = x + self.time_pos_embed.unsqueeze(0)
        x = self.time_pos_drop(x)
        x = rearrange(x, 'b (p q) t d -> b t p q d', p=sp_dim_1, q=sp_dim_2)
        return x

    def forward(self, input_to_use):
        """Forward pass of the model"""
        x = input_to_use['video']
        metadata = input_to_use['metadata']
        few_shot_aux = {}
        matchability_evidence_tokens = None

        if 'skip_feat_extractor' in input_to_use:
            skip_feat_extractor = input_to_use['skip_feat_extractor']
        else:
            skip_feat_extractor = False
        if not self.cfg.MODEL.APPEARANCE_MODULE_DISABLE:
            if skip_feat_extractor:
                embed_dim = self.embed_dim
                batch_size, num_frames = x.shape[:2]
                feat_to_use = torch.randn(batch_size, num_frames,
                                            int(self.num_patches**0.5),
                                            int(self.num_patches**0.5),
                                            embed_dim).to(x.device)
            else:
                if self.feat_extractor_type == "dino":
                    feat_to_use = self.get_dino_features(x)
                    if self.cfg.POINT_INFO.USE_CORRELATION:
                        # for ablation study without point tracking module
                        new_metadata = get_points_using_correlation(self.cfg, feat_to_use)
                        metadata.update(new_metadata)
                elif self.feat_extractor_type == "clip_vit_b16":
                    feat_to_use = self.get_clip_features(x)
                    if self.cfg.POINT_INFO.USE_CORRELATION:
                        new_metadata = get_points_using_correlation(self.cfg, feat_to_use)
                        metadata.update(new_metadata)
                elif self.feat_extractor_type == "dinotxt_vitl14_reg4":
                    feat_to_use = self.get_dinotxt_features(x)
                    if self.cfg.POINT_INFO.USE_CORRELATION:
                        new_metadata = get_points_using_correlation(self.cfg, feat_to_use)
                        metadata.update(new_metadata)
                else:
                    raise NotImplementedError('Feature extractor not implemented')

            few_shot_cfg = getattr(self.cfg, "FEW_SHOT", None)
            match_cfg = getattr(
                few_shot_cfg,
                "QUERY_CLASS_MATCHABILITY",
                None,
            )
            matchability_source = str(
                getattr(match_cfg, "EVIDENCE_SOURCE", "post")
            ).lower()
            absolute_mass_enabled = bool(
                getattr(match_cfg, "ABSOLUTE_MASS_ENABLE", False)
            )
            absolute_mass_source = str(
                getattr(match_cfg, "ABSOLUTE_MASS_SOURCE", "raw")
            ).lower()
            raw_evidence_requested = (
                matchability_source == "raw"
                or (absolute_mass_enabled and absolute_mass_source == "raw")
            )
            if (
                bool(getattr(match_cfg, "ENABLE", False))
                and raw_evidence_requested
            ):
                if self.feat_extractor_type != "dinotxt_vitl14_reg4":
                    raise ValueError(
                        "Raw matchability evidence requires dinotxt_vitl14_reg4."
                    )
                if self.cfg.POINT_INFO.ENABLE:
                    matchability_evidence_tokens = self._sample_point_features(
                        feat_to_use,
                        metadata['pred_tracks'],
                        add_pt_pos_embed=False,
                    )
                else:
                    matchability_evidence_tokens = rearrange(
                        feat_to_use,
                        'b t p q d -> b t (p q) d',
                    )

            if self.cfg.MF.USE_BASE_POS_EMBED:
                feat_to_use = self.add_st_pos_embeddings(feat_to_use)


            if self.cfg.POINT_INFO.ENABLE:
                pred_tracks = metadata['pred_tracks']
                sampled_feat = self._sample_point_features(
                    feat_to_use,
                    pred_tracks,
                    add_pt_pos_embed=(
                        self.cfg.MF.USE_PT_SPACE_POS_EMBED
                        and self.cfg.FEW_SHOT.USE_MODEL
                        and not self.cfg.MF.USE_BASE_POS_EMBED
                    ),
                )

            else:
                sampled_feat = rearrange(feat_to_use, 'b t p q d -> b t (p q) d')
                self.point_grid_size = int(sampled_feat.shape[2] ** 0.5)
        else:
            sampled_feat = 0

        app_feat = sampled_feat if torch.is_tensor(sampled_feat) else None
        intra_feat = None
        inter_feat = None
        if self.cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE:
            intra_feat = self.hod_motion_module(metadata['hod_feat'].float())

        if self.cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE:
            inter_feat = self.cross_motion_module(
                metadata['pred_tracks'], metadata['pred_visibility'])

        reference_feat = app_feat
        if reference_feat is None:
            reference_feat = intra_feat if intra_feat is not None else inter_feat
        if app_feat is None:
            app_feat = torch.zeros_like(reference_feat)
        if intra_feat is None:
            intra_feat = torch.zeros_like(reference_feat)
        if inter_feat is None:
            inter_feat = torch.zeros_like(reference_feat)
        fused_feat = app_feat + intra_feat + inter_feat

        cls_x, patch_x = self.pt_forward(fused_feat, metadata)
        if 'support_mask' in metadata and 'episode_positive_labels' in metadata:
            if self.use_text_alignment:
                text_align_aux = self._build_support_text_alignment(
                    patch_x,
                    metadata,
                )
                if text_align_aux is not None:
                    few_shot_aux.update(text_align_aux)
            if self.use_query_partial_q2s:
                frame_softmax_aux = self._build_frame_softmax_q2s_aux(
                    patch_x,
                    metadata,
                    pred_tracks=metadata.get('pred_tracks'),
                    matchability_evidence_tokens=matchability_evidence_tokens,
                )
                if frame_softmax_aux is not None:
                    few_shot_aux.update(frame_softmax_aux)
        if not few_shot_aux:
            few_shot_aux = None
        # x = self.forward_features(x, metadata) # [BS, d]
        x = self.head_drop(cls_x)


        x = self.head(x)
        # previously there was a softmax here for validation which messed up the loss computation
        if self.cfg.TASK == 'few_shot':
            return x, patch_x, few_shot_aux
        return x


def compute_correlation_map(features):
    """
    Compute correlation map between consecutive frames
    Args:
        features: Tensor of shape [bs, num_frames, num_patches, d]
    Returns:
        correlation: Tensor of shape [bs, num_frames-1, num_patches, num_patches]
    """
    bs, num_frames, num_patches, d = features.shape

    # Get features for all frames except last one
    feat1 = features[:, :-1]  # [bs, num_frames-1, num_patches, d]

    # Get features for all frames except first one
    feat2 = features[:, 1:]   # [bs, num_frames-1, num_patches, d]

    # Normalize the features
    feat1_norm = F.normalize(feat1, p=2, dim=-1)
    feat2_norm = F.normalize(feat2, p=2, dim=-1)

    # Compute correlation through matrix multiplication
    # Reshape for batch matrix multiplication
    feat1_reshaped = feat1_norm.view(bs * (num_frames-1), num_patches, d)
    feat2_reshaped = feat2_norm.view(bs * (num_frames-1), num_patches, d)

    # Compute correlation [bs*(num_frames-1), num_patches, num_patches]
    correlation = torch.bmm(feat1_reshaped, feat2_reshaped.transpose(1, 2))

    # Reshape back to [bs, num_frames-1, num_patches, num_patches]
    correlation = correlation.view(bs, num_frames-1, num_patches, num_patches)

    return correlation


def get_points_using_correlation(cfg, features):
    """
    Get points using correlation
    Args:
        cfg (dict): Configuration dictionary
        features (torch.Tensor): Features of shape [BS, T, P, Q, D]

    Returns:
        dict: Metadata containing prediction tracks and visibility
    """
    features = rearrange(features, 'b t p q d -> b t (p q) d')
    bs, num_frames, num_patches, _ = features.shape

    # Compute basic correlation first
    correlation = compute_correlation_map(features)

    # Apply mutual matching
    corr_b = correlation.view(bs*(num_frames-1), num_patches, num_patches)
    corr_a = corr_b

    # Get max values
    corr_b_max, _ = torch.max(corr_b, dim=1, keepdim=True)
    corr_a_max, _ = torch.max(corr_a, dim=2, keepdim=True)

    # Normalize by max values
    eps = 1e-5
    corr_b = corr_b / (corr_b_max + eps)
    corr_a = corr_a / (corr_a_max + eps)

    # Compute mutual correlation
    mutual_correlation = correlation * (corr_a.view_as(correlation) * corr_b.view_as(correlation))
    _, max_indices = torch.max(mutual_correlation, dim=-1)
    extra_indices = torch.arange(max_indices.shape[-1]).to(max_indices.device)
    extra_indices = extra_indices.unsqueeze(0).unsqueeze(0).expand(max_indices.shape[0], -1, -1)
    max_indices = torch.cat([extra_indices, max_indices], dim=1)
    if cfg.MODEL.FEAT_EXTRACTOR == "dino" and cfg.MODEL.DINO_CONFIG == "dinov2_vitb14":
        grid_points = create_normalized_grid(image_size=224, grid_size=16)
    elif cfg.MODEL.FEAT_EXTRACTOR == "clip_vit_b16":
        grid_points = create_normalized_grid(image_size=224, grid_size=14)
    elif cfg.MODEL.FEAT_EXTRACTOR == "dinotxt_vitl14_reg4":
        grid_points = create_normalized_grid(image_size=224, grid_size=16)
    else:
        raise NotImplementedError(
            f'Grid points dim not set for extractor {cfg.MODEL.FEAT_EXTRACTOR}'
        )
    grid_points = grid_points.to(max_indices.device)
    grid_points = rearrange(grid_points, 'n d -> 1 1 n d')
    grid_points = repeat(grid_points, '1 1 n d -> b t n d', b=bs, t=num_frames)
    new_metadata = sample_grid_points_with_indices(cfg, grid_points, max_indices)

    return new_metadata

def create_normalized_grid(image_size=224, grid_size=16):
    """
    Create a normalized grid of points from an image
    Args:
        image_size (int): Size of the image (assuming square image)
        grid_size (int): Size of the grid (e.g., 16 for 16x16 grid)
    Returns:
        grid_points: Tensor of shape [grid_size*grid_size, 2] containing normalized x,y coordinates
    """
    # Create linear spaces for x and y coordinates
    grid_step = image_size // grid_size
    points = torch.arange(grid_step // 2, image_size, grid_step)

    # Create meshgrid
    y, x = torch.meshgrid(points, points, indexing='ij')

    # Reshape to [N, 2] where N = grid_size * grid_size
    grid_points = torch.stack([x, y], dim=-1).reshape(-1, 2)

    # Normalize to [-1, 1]
    grid_points = 2 * (grid_points / (image_size - 1)) - 1

    return grid_points


def sample_grid_points_with_indices(cfg, grid_points, max_indices):
    """
    Sample grid points using max correlation indices
    Args:
        grid_points: Tensor of shape [bs, num_frames, num_patches, 2]
        max_indices: Tensor of shape [bs, num_frames, num_patches]
    Returns:
        sampled_points: Tensor of shape [bs, num_frames, num_patches, 2]
    """
    bs, num_frames, num_patches, _ = grid_points.shape

    # Create batch indices for gathering
    batch_indices = torch.arange(bs).view(-1, 1, 1).expand(-1, num_frames, num_patches)
    frame_indices = torch.arange(num_frames).view(1, -1, 1).expand(bs, -1, num_patches)

    # Gather points using indices
    sampled_points = grid_points[batch_indices, frame_indices, max_indices]
    new_metadata = {}
    if cfg.POINT_INFO.HOD.GET_FEAT:
        sampled_points_for_hod = rearrange(sampled_points, 'b t n d -> b n t d')
        hod_feats = []
        for i in range(bs):
            hod_feat = torch.tensor(get_orientation_hist(
                                        sampled_points_for_hod[i].cpu().numpy(),
                                        cfg.POINT_INFO.HOD.NUM_BINS,
                                        preserve_temporal=True))
            hod_feats.append(hod_feat.unsqueeze(0))
        hod_feats = torch.cat(hod_feats, dim=0).to(grid_points.device)
        new_metadata['hod_feat'] = hod_feats
    new_metadata['pred_tracks'] = sampled_points
    new_metadata['pred_visibility'] = torch.ones_like(
                            sampled_points).bool()[...,0].to(grid_points.device)
    new_metadata['pred_query_mask'] = torch.ones_like(
                            sampled_points).bool()[...,0].to(grid_points.device)


    return new_metadata
