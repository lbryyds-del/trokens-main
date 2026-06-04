"""Pointformer model."""
import json
import os
import sys
import time
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
from trokens.models.branches.motion_blocks import (
    CrossMotionModule,
    HODMotionModule
)
from trokens.datasets.hod import get_orientation_hist
from .build import MODEL_REGISTRY

# pylint: disable=unused-argument,redefined-builtin

class LabelAwareCMWCostNet(nn.Module):
    """Label-aware CMW-style private reliability estimator."""

    def __init__(
        self,
        token_evidence_dim=6,
        label_context_dim=7,
        hidden_dim=128,
        num_families=5,
    ):
        super().__init__()
        self.num_families = int(num_families)
        self.token_branch = nn.Sequential(
            nn.Linear(token_evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_families),
        )
        self.label_branch = nn.Sequential(
            nn.Linear(label_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_families),
        )

    def forward(
        self,
        token_evidence,
        label_context,
        point_mask,
        min_reliability=0.02,
    ):
        """Return private reliability over the raw label axis."""
        token_evidence = torch.nan_to_num(
            token_evidence.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        label_context = torch.nan_to_num(
            label_context.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        candidate_reliability = torch.sigmoid(self.token_branch(token_evidence))
        family_prob = torch.softmax(self.label_branch(label_context), dim=-1)
        reliability = (
            candidate_reliability * family_prob[:, None, None, :]
        ).sum(dim=-1)
        min_reliability = max(float(min_reliability), 0.0)
        reliability = torch.nan_to_num(
            reliability,
            nan=min_reliability,
            posinf=1.0,
            neginf=min_reliability,
        ).clamp(min=min_reliability, max=1.0)
        point_mask = point_mask.to(device=reliability.device).bool()
        reliability = reliability * point_mask.unsqueeze(0).to(reliability.dtype)
        return reliability, {
            "candidate_reliability": candidate_reliability,
            "family_prob": family_prob,
        }


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
        self._pot_debug_call_count = 0
        self._pot_debug_record_count = 0
        self._pot_debug_io_failed = False
        self.is_multilabel_few_shot = (
            cfg.TASK == 'few_shot'
            and cfg.DATA.MULTI_LABEL
            and not cfg.MODEL.APPEARANCE_MODULE_DISABLE
        )
        self.use_pot_support_route = (
            self.is_multilabel_few_shot
            and self.feat_extractor_type == "dinotxt_vitl14_reg4"
            and self.pot_route_cfg.ENABLE
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
                "POT support routing currently requires the dinotxt_vitl14_reg4 backbone."
            )
        if (
            self.is_multilabel_few_shot
            and self.text_align_cfg.ENABLE
            and self.feat_extractor_type != "dinotxt_vitl14_reg4"
        ):
            raise NotImplementedError(
                "TEXT_ALIGN currently requires the dinotxt_vitl14_reg4 backbone."
            )
        self.use_label_text_features = (
            self.use_pot_support_route or self.use_text_alignment
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
            if self.use_pot_support_route or self.use_text_alignment:
                self.atomic_label_names = self._load_atomic_label_names()

        self.use_cmw_cost = self.use_pot_support_route
        if self.use_cmw_cost:
            self.cmw_cost_net = LabelAwareCMWCostNet(
                token_evidence_dim=6,
                label_context_dim=7,
                hidden_dim=int(getattr(self.pot_route_cfg, "CMW_COST_HIDDEN_DIM", 128)),
                num_families=int(getattr(self.pot_route_cfg, "CMW_COST_NUM_FAMILIES", 5)),
            )

        # Initialize weights
        self.init_weights()
        self.apply(self._init_weights)
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
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

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
        """Return prompt variants for SAV labels that need finer visual grounding."""
        base_prompt = label_name.replace("_", " ")
        prompt_bank = {
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
            "take_notes": [
                "take notes",
                "a student writes notes on paper",
                "hand holding a pen and writing",
                "small repetitive hand motion on notebook",
                "writing on a desk with paper",
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

    def _masked_frame_mean(self, feat, point_mask):
        """Average each frame over valid trajectories: [T,N,C] -> [T,C]."""
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if point_mask is None:
            point_mask = torch.ones(
                feat.shape[:2],
                device=feat.device,
                dtype=torch.bool,
            )
        else:
            point_mask = point_mask.to(device=feat.device).bool()
        weights = point_mask.to(feat.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        frame_feat = (feat * weights).sum(dim=1) / denom
        return torch.nan_to_num(frame_feat, nan=0.0, posinf=0.0, neginf=0.0)

    def _masked_traj_mean(self, feat, point_mask):
        """Average each trajectory over time: [T,N,C] -> [N,C]."""
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if point_mask is None:
            point_mask = torch.ones(
                feat.shape[:2],
                device=feat.device,
                dtype=torch.bool,
            )
        else:
            point_mask = point_mask.to(device=feat.device).bool()
        weights = point_mask.to(feat.dtype).unsqueeze(-1)
        denom = weights.sum(dim=0).clamp_min(1e-6)
        traj_feat = (feat * weights).sum(dim=0) / denom
        return torch.nan_to_num(traj_feat, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalized_distribution_entropy(self, probs, valid_count=None):
        """Return entropy normalized to [0, 1] for the last dimension."""
        probs = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)
        support_size = probs.shape[-1] if valid_count is None else int(valid_count)
        normalizer = max(float(np.log(max(support_size, 2))), 1e-6)
        return entropy / entropy.new_tensor(normalizer)

    def _compute_sharedness_3d(self, sim, point_mask, return_components=False):
        """Estimate shared evidence over [T,N] from label entropy and semantic strength."""
        num_labels = sim.shape[0]
        sharedness = sim.new_zeros(sim.shape[1:])
        components = {
            "label_entropy": sharedness,
            "semantic_strength": sharedness,
        }
        if num_labels <= 1:
            return (sharedness, components) if return_components else sharedness

        route_cfg = self.pot_route_cfg
        tau_label = max(float(getattr(route_cfg, "SHARED_TAU_LABEL", 0.07)), 1e-6)
        theta_shared = float(getattr(route_cfg, "SHARED_THETA", 0.2))
        tau_strength = max(
            float(getattr(route_cfg, "SHARED_TAU_STRENGTH", 0.1)),
            1e-6,
        )

        point_mask = point_mask.to(device=sim.device).bool()
        sim = torch.nan_to_num(
            sim.float(),
            nan=-1e4,
            posinf=1.0,
            neginf=-1e4,
        ).clamp(-1.0, 1.0)

        label_prob = torch.softmax(sim / tau_label, dim=0).clamp_min(1e-12)
        label_entropy = -(label_prob * label_prob.log()).sum(dim=0)
        normalizer = max(float(np.log(max(num_labels, 2))), 1e-6)
        label_entropy = label_entropy / label_entropy.new_tensor(normalizer)

        max_sim = sim.max(dim=0).values
        semantic_strength = torch.sigmoid((max_sim - theta_shared) / tau_strength)
        sharedness = label_entropy * semantic_strength

        zeros = torch.zeros_like(sharedness)
        label_entropy = torch.where(point_mask, label_entropy, zeros)
        semantic_strength = torch.where(point_mask, semantic_strength, zeros)
        sharedness = torch.where(point_mask, sharedness, zeros)

        sharedness = torch.nan_to_num(
            sharedness.clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        if not return_components:
            return sharedness

        components = {
            "label_entropy": torch.nan_to_num(
                label_entropy.clamp(0.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            "semantic_strength": torch.nan_to_num(
                semantic_strength.clamp(0.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
        }
        return sharedness, components

    def _normalized_token_norm(self, tokens, point_mask):
        """Return a [T,N] min-max normalized token norm map."""
        point_mask = point_mask.bool()
        if tokens is None:
            return point_mask.new_zeros(point_mask.shape, dtype=torch.float32)
        if tokens.shape[:2] != point_mask.shape:
            return point_mask.new_zeros(point_mask.shape, dtype=torch.float32)

        tokens = torch.nan_to_num(
            tokens.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if tokens.ndim == 2:
            strength = tokens.abs()
        else:
            strength = torch.norm(tokens, dim=-1)
        strength = torch.nan_to_num(
            strength,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        normalized = torch.zeros_like(strength)
        if point_mask.any():
            valid_strength = strength[point_mask]
            value_range = valid_strength.max() - valid_strength.min()
            if float(value_range.item()) > 1e-6:
                normalized = (strength - valid_strength.min()) / value_range.clamp_min(1e-6)
        normalized = torch.where(point_mask, normalized, torch.zeros_like(normalized))
        return torch.nan_to_num(
            normalized.clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def _build_cmw_token_evidence(
        self,
        sim,
        sim01,
        sharedness,
        point_mask,
        intra_tokens=None,
        inter_tokens=None,
        semantic_strength=None,
    ):
        """Build CMW token evidence over the raw label axis."""
        route_cfg = self.pot_route_cfg
        num_labels = sim.shape[0]
        tau_margin = max(
            float(getattr(route_cfg, "CMW_COST_MARGIN_TAU", 0.1)),
            1e-6,
        )
        if num_labels > 1:
            best_other_sim = []
            for label_idx in range(num_labels):
                other_indices = [
                    other_idx
                    for other_idx in range(num_labels)
                    if other_idx != label_idx
                ]
                other_sim = sim.index_select(
                    0,
                    torch.as_tensor(
                        other_indices,
                        device=sim.device,
                        dtype=torch.long,
                    ),
                ).amax(dim=0)
                best_other_sim.append(other_sim.unsqueeze(0))
            best_other_sim = torch.cat(best_other_sim, dim=0)
            margin = sim - best_other_sim
        else:
            margin = sim

        margin_score = torch.sigmoid(margin / tau_margin)
        sharedness = torch.nan_to_num(
            sharedness.to(device=sim.device).float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        shared_map = sharedness.unsqueeze(0).expand_as(margin_score)
        conflict_shared = torch.maximum(1.0 - margin_score, shared_map)

        motion_strength = self._normalized_token_norm(intra_tokens, point_mask).to(
            device=sim.device,
        )
        inter_strength = self._normalized_token_norm(inter_tokens, point_mask).to(
            device=sim.device,
        )
        if semantic_strength is None or semantic_strength.shape != point_mask.shape:
            semantic_strength = sim01.max(dim=0).values
        semantic_strength = torch.nan_to_num(
            semantic_strength.to(device=sim.device).float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        point_mask = point_mask.to(device=sim.device).bool()
        valid3 = point_mask.unsqueeze(0).expand_as(sim01)
        text_sim = torch.where(valid3, sim01, torch.zeros_like(sim01))
        margin_score = torch.where(valid3, margin_score, torch.zeros_like(margin_score))
        conflict_shared = torch.where(
            valid3,
            conflict_shared,
            torch.zeros_like(conflict_shared),
        )
        motion_map = motion_strength.unsqueeze(0).expand_as(text_sim)
        inter_map = inter_strength.unsqueeze(0).expand_as(text_sim)
        semantic_map = semantic_strength.unsqueeze(0).expand_as(text_sim)

        token_evidence = torch.stack(
            [
                text_sim,
                margin_score,
                conflict_shared,
                motion_map,
                inter_map,
                semantic_map,
            ],
            dim=-1,
        )
        token_evidence = torch.nan_to_num(
            token_evidence.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        components = {
            "text_sim": text_sim,
            "margin_score": margin_score,
            "conflict_shared": conflict_shared,
            "motion_strength": motion_strength,
            "inter_strength": inter_strength,
            "semantic_strength": semantic_strength,
        }
        return token_evidence, components

    def _build_cmw_label_context(
        self,
        episode_positive_labels,
        support_mask,
        episode_label_text,
        sample_positive_labels,
        support_global,
        positive_text,
        raw_positive_labels=None,
    ):
        """Build [K,7] label context features for CMW private reliability."""
        del episode_positive_labels, episode_label_text
        device = positive_text.device
        dtype = positive_text.dtype
        num_labels = positive_text.shape[0]
        if num_labels == 0:
            return positive_text.new_zeros(0, 7)

        support_count = positive_text.new_zeros(num_labels)
        effective_support = positive_text.new_zeros(num_labels)
        cooccur_degree = positive_text.new_zeros(num_labels)
        num_support = 0
        total_label_dim = max(num_labels, 1)
        if raw_positive_labels is not None:
            raw_positive_labels = raw_positive_labels.to(device=device).float()
            if support_mask is None:
                support_mask = torch.ones(
                    raw_positive_labels.shape[0],
                    device=device,
                    dtype=torch.bool,
                )
            else:
                support_mask = support_mask.to(device=device).bool()
            support_labels = raw_positive_labels[support_mask]
            num_support = int(support_mask.sum().item())
            total_label_dim = max(int(raw_positive_labels.shape[-1]), 1)
            if support_labels.numel() > 0:
                sample_positive_labels = sample_positive_labels.to(
                    device=device,
                    dtype=torch.long,
                ).flatten()
                valid_label_ids = (
                    (sample_positive_labels >= 0)
                    & (sample_positive_labels < support_labels.shape[-1])
                )
                safe_label_ids = sample_positive_labels.clamp(
                    min=0,
                    max=max(int(support_labels.shape[-1]) - 1, 0),
                )
                support_subset = support_labels.index_select(1, safe_label_ids)
                support_subset = support_subset * valid_label_ids.to(dtype).unsqueeze(0)
                support_count = support_subset.sum(dim=0)

                cardinality = support_labels.sum(dim=1).clamp_min(1.0)
                effective_support = (
                    support_subset / cardinality.unsqueeze(1)
                ).sum(dim=0)
                if num_labels > 1:
                    label_cardinality = support_subset.sum(dim=1, keepdim=True)
                    cooccur_count = (
                        support_subset * (label_cardinality - support_subset)
                    ).sum(dim=0)
                    cooccur_degree = cooccur_count / (
                        support_count.clamp_min(1.0) * float(num_labels - 1)
                    )

        support_denom = max(float(num_support), 1.0)
        log_support = torch.log1p(support_count) / max(float(np.log1p(support_denom)), 1e-6)
        normalized_support_count = support_count / support_denom
        normalized_effective_support = effective_support / support_denom

        text_norm = F.normalize(positive_text.float(), dim=-1)
        if num_labels > 1:
            text_sim = torch.matmul(text_norm, text_norm.transpose(0, 1)).clamp(-1.0, 1.0)
            offdiag_mask = ~torch.eye(num_labels, device=device, dtype=torch.bool)
            ambiguity = text_sim.masked_select(offdiag_mask).view(num_labels, num_labels - 1)
            ambiguity = ambiguity.mean(dim=-1)
            ambiguity = ((ambiguity + 1.0) * 0.5).clamp(0.0, 1.0)
        else:
            ambiguity = positive_text.new_zeros(num_labels)

        support_global = F.normalize(
            torch.nan_to_num(
                support_global.to(device=device).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            dim=-1,
        )
        global_text_sim = torch.matmul(text_norm, support_global).clamp(-1.0, 1.0)
        global_text_sim = ((global_text_sim + 1.0) * 0.5).clamp(0.0, 1.0)
        positive_label_count = positive_text.new_full(
            (num_labels,),
            min(float(num_labels) / max(float(total_label_dim), 1.0), 1.0),
        )

        label_context = torch.stack(
            [
                log_support,
                normalized_support_count,
                normalized_effective_support,
                ambiguity,
                cooccur_degree,
                global_text_sim,
                positive_label_count,
            ],
            dim=-1,
        )
        return torch.nan_to_num(
            label_context.to(dtype=dtype).clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def _solve_avg_3d_uot(
        self,
        cost,
        prior_frame,
        prior_traj,
        prior_vis,
        valid_mask,
    ):
        """
        Target-conditioned 3D-UOT with three soft plane priors.

        cost: [R,T,N]
        prior_frame: [R,T], target for sum_n Gamma[r,t,n]
        prior_traj:  [R,N], target for sum_t Gamma[r,t,n]
        prior_vis:   [T,N], target for sum_r Gamma[r,t,n]
        valid_mask:  [T,N]
        return gamma: [R,T,N]
        """
        route_cfg = self.pot_route_cfg

        entropic_eps = max(
            float(getattr(route_cfg, "UOT3D_ENTROPIC_EPS", route_cfg.ENTROPIC_EPS)),
            1e-6,
        )
        max_iters = max(int(route_cfg.MAX_ITERS), 1)
        stop_tol = max(float(route_cfg.STOP_TOL), 0.0)

        alpha_frame = float(getattr(
            route_cfg,
            "UOT3D_RHO_FRAME",
            getattr(route_cfg, "UOT3D_RHO_LT", 0.3),
        ))
        alpha_traj = float(getattr(
            route_cfg,
            "UOT3D_RHO_TRAJ",
            getattr(route_cfg, "UOT3D_RHO_LN", 0.5),
        ))
        alpha_vis = float(getattr(
            route_cfg,
            "UOT3D_RHO_VIS",
            getattr(route_cfg, "UOT3D_RHO_TN", 0.5),
        ))

        alpha_frame = min(max(alpha_frame, 0.0), 1.0)
        alpha_traj = min(max(alpha_traj, 0.0), 1.0)
        alpha_vis = min(max(alpha_vis, 0.0), 1.0)

        cost = torch.nan_to_num(cost.float(), nan=1e4, posinf=1e4, neginf=0.0)
        prior_frame = torch.nan_to_num(
            prior_frame.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        prior_traj = torch.nan_to_num(
            prior_traj.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        prior_vis = torch.nan_to_num(
            prior_vis.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)

        if cost.numel() == 0:
            return cost.new_zeros(cost.shape)

        valid_mask = valid_mask.to(device=cost.device).bool()
        valid3 = valid_mask.unsqueeze(0).expand_as(cost)
        valid_cost = cost.masked_fill(~valid3, 1e4)

        row_min = valid_cost.flatten(1).amin(dim=1).view(-1, 1, 1)
        row_min = torch.where(
            torch.isfinite(row_min),
            row_min,
            torch.zeros_like(row_min),
        )
        shifted_cost = valid_cost - row_min

        gamma = torch.exp((-shifted_cost / entropic_eps).clamp(min=-80.0, max=0.0))
        gamma = torch.where(valid3, gamma, torch.zeros_like(gamma))

        if float(gamma.sum().item()) <= 0.0:
            return cost.new_zeros(cost.shape)

        target_mass = prior_vis.sum()
        if float(target_mass.item()) <= 0.0:
            target_mass = prior_frame.sum()
        if float(target_mass.item()) <= 0.0:
            target_mass = prior_traj.sum()
        if float(target_mass.item()) <= 0.0:
            target_mass = gamma.new_tensor(1.0)

        gamma = gamma * (target_mass / gamma.sum().clamp_min(1e-12))

        for _ in range(max_iters):
            prev_gamma = gamma

            cur_frame = gamma.sum(dim=2)
            scale_frame = (
                prior_frame / cur_frame.clamp_min(1e-12)
            ).clamp_min(0.0).pow(alpha_frame)
            scale_frame = torch.nan_to_num(
                scale_frame,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            gamma = gamma * scale_frame.unsqueeze(2)

            cur_traj = gamma.sum(dim=1)
            scale_traj = (
                prior_traj / cur_traj.clamp_min(1e-12)
            ).clamp_min(0.0).pow(alpha_traj)
            scale_traj = torch.nan_to_num(
                scale_traj,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            gamma = gamma * scale_traj.unsqueeze(1)

            cur_vis = gamma.sum(dim=0)
            scale_vis = (
                prior_vis / cur_vis.clamp_min(1e-12)
            ).clamp_min(0.0).pow(alpha_vis)
            scale_vis = torch.nan_to_num(
                scale_vis,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            gamma = gamma * scale_vis.unsqueeze(0)

            gamma = torch.where(valid3, gamma, torch.zeros_like(gamma))
            gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            delta = torch.max(torch.abs(gamma - prev_gamma))
            if float(delta.item()) <= stop_tol:
                break

        return gamma

    def _compute_avg_3d_uot_transport(
        self,
        st_tokens,
        point_mask,
        support_global,
        positive_text,
        intra_tokens=None,
        inter_tokens=None,
        episode_positive_labels=None,
        support_mask=None,
        sample_positive_labels=None,
        episode_label_text=None,
        raw_positive_labels=None,
        return_debug=False,
        target_label_indices=None,
    ):
        """
        Target-conditioned 3D-UOT over label-time-trajectory tensor.

        st_tokens: [T,N,C]
        point_mask: [T,N]
        support_global: [C]
        positive_text: [K,C], where K is the support sample raw-label axis
        target_label_indices: label-axis rows that should produce episode prototypes
        """
        route_cfg = self.pot_route_cfg

        def _normalize_masked_map(score, mask):
            score = torch.nan_to_num(
                score.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            mask = mask.to(device=score.device).bool()
            score = score * mask.to(score.dtype)
            if float(score.sum().item()) <= 0.0:
                score = mask.to(score.dtype)
            return score / score.sum().clamp_min(1e-12)

        def _normalize_masked_vector(score, mask):
            score = torch.nan_to_num(
                score.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            mask = mask.to(device=score.device).bool()
            score = score * mask.to(score.dtype)
            if float(score.sum().item()) <= 0.0:
                score = mask.to(score.dtype)
            return score / score.sum().clamp_min(1e-12)

        st_tokens = torch.nan_to_num(st_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        support_global = torch.nan_to_num(
            support_global,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        positive_text = torch.nan_to_num(
            positive_text,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if point_mask is None:
            point_mask = torch.ones(
                st_tokens.shape[:2],
                device=st_tokens.device,
                dtype=torch.bool,
            )
        else:
            point_mask = point_mask.to(device=st_tokens.device).bool()

        temporal_dim, num_points = st_tokens.shape[:2]
        num_labels = positive_text.shape[0]
        if target_label_indices is None:
            target_label_indices = torch.arange(
                num_labels,
                device=st_tokens.device,
                dtype=torch.long,
            )
        else:
            target_label_indices = target_label_indices.to(
                device=st_tokens.device,
                dtype=torch.long,
            ).flatten()
            valid_target_indices = (
                (target_label_indices >= 0)
                & (target_label_indices < num_labels)
            )
            target_label_indices = target_label_indices[valid_target_indices]
        num_targets = int(target_label_indices.numel())

        if (
            num_labels == 0
            or num_targets == 0
            or temporal_dim == 0
            or num_points == 0
            or not point_mask.any()
        ):
            empty_label = st_tokens.new_zeros(num_labels, temporal_dim, num_points)
            empty_target = st_tokens.new_zeros(num_targets, temporal_dim, num_points)
            return {
                "st_transport": empty_target,
                "sim": empty_label,
                "cost": empty_label,
                "cost_ext": st_tokens.new_zeros(num_labels + 1, temporal_dim, num_points),
                "sharedness": st_tokens.new_zeros(temporal_dim, num_points),
                "label_entropy": st_tokens.new_zeros(temporal_dim, num_points),
                "semantic_strength": st_tokens.new_zeros(temporal_dim, num_points),
                "prob_frame": st_tokens.new_zeros(num_labels, temporal_dim),
                "prob_traj": st_tokens.new_zeros(num_labels, num_points),
                "prob_vis": empty_label,
                "shared_transport": empty_target,
                "transport_mass": st_tokens.new_zeros(num_targets),
                "shared_transport_mass": st_tokens.new_zeros(num_targets),
                "target_label_indices": target_label_indices,
            }

        valid_frame_mask = point_mask.any(dim=1)
        valid_traj_mask = point_mask.any(dim=0)

        st_norm = F.normalize(st_tokens.float(), dim=-1)
        text_norm = F.normalize(positive_text.float(), dim=-1)
        support_global = F.normalize(support_global.float(), dim=-1)

        sim = torch.einsum("kc,tnc->ktn", text_norm, st_norm)
        sim = torch.nan_to_num(
            sim,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

        sim01 = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)

        shared_enabled = bool(getattr(route_cfg, "UOT3D_SHARED_ENABLE", False))
        sharedness = st_tokens.new_zeros(temporal_dim, num_points)
        shared_components = {
            "label_entropy": st_tokens.new_zeros(temporal_dim, num_points),
            "semantic_strength": st_tokens.new_zeros(temporal_dim, num_points),
        }
        if shared_enabled and num_labels > 1:
            sharedness, shared_components = self._compute_sharedness_3d(
                sim,
                point_mask,
                return_components=True,
            )

        if not hasattr(self, "cmw_cost_net"):
            raise RuntimeError(
                "LabelAwareCMWCostNet is required for POT support routing."
            )

        semantic_for_cmw = (
            shared_components["semantic_strength"]
            if shared_enabled and num_labels > 1
            else None
        )
        cmw_token_evidence, cmw_evidence_components = self._build_cmw_token_evidence(
            sim,
            sim01,
            sharedness,
            point_mask,
            intra_tokens=intra_tokens,
            inter_tokens=inter_tokens,
            semantic_strength=semantic_for_cmw,
        )
        if sample_positive_labels is None:
            sample_positive_labels = torch.arange(
                num_labels,
                device=st_tokens.device,
                dtype=torch.long,
            )
        cmw_label_context = self._build_cmw_label_context(
            episode_positive_labels,
            support_mask,
            episode_label_text,
            sample_positive_labels,
            support_global,
            positive_text,
            raw_positive_labels=raw_positive_labels,
        )
        cmw_min_reliability = float(getattr(
            route_cfg,
            "CMW_COST_MIN_RELIABILITY",
            0.02,
        ))
        cmw_private_reliability, cmw_aux = self.cmw_cost_net(
            cmw_token_evidence,
            cmw_label_context,
            point_mask,
            min_reliability=cmw_min_reliability,
        )
        cmw_private_reliability = torch.nan_to_num(
            cmw_private_reliability.to(device=sim.device).float(),
            nan=cmw_min_reliability,
            posinf=1.0,
            neginf=cmw_min_reliability,
        ).clamp(min=max(cmw_min_reliability, 0.0), max=1.0)
        cmw_private_reliability = (
            cmw_private_reliability
            * point_mask.unsqueeze(0).to(cmw_private_reliability.dtype)
        )
        private_cost = 1.0 - cmw_private_reliability
        private_cost = torch.nan_to_num(
            private_cost,
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        private_cost = private_cost.masked_fill(~point_mask.unsqueeze(0), 1e4)
        cost = private_cost
        cost_source = "cmw_private_cost"

        shared_cost = torch.nan_to_num(
            1.0 - sharedness,
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        cost_ext = torch.cat(
            [private_cost, shared_cost.unsqueeze(0)],
            dim=0,
        )
        cost_ext = cost_ext.masked_fill(~point_mask.unsqueeze(0), 1e4)
        cost_ext = torch.nan_to_num(cost_ext, nan=1e4, posinf=1e4, neginf=0.0)
        cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=0.0)

        base_mu_logit_scale = float(getattr(route_cfg, "MU_LOGIT_SCALE", 10.0))
        mu_logit_scale = max(
            float(getattr(route_cfg, "UOT3D_MU_LOGIT_SCALE", base_mu_logit_scale)),
            1e-6,
        )
        mu_logits = mu_logit_scale * torch.matmul(text_norm, support_global)
        mu = torch.softmax(mu_logits, dim=0)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        if float(mu.sum().item()) <= 0.0:
            mu = torch.full_like(mu, 1.0 / max(float(num_labels), 1.0))

        target_mix = float(getattr(route_cfg, "UOT3D_TARGET_MIX", 0.85))
        target_mix = min(max(target_mix, 0.0), 1.0)

        total_mass = float(getattr(route_cfg, "UOT3D_TOTAL_MASS", 1.0))
        total_mass = max(total_mass, 1e-6)
        debug_topk = max(int(getattr(route_cfg, "DEBUG_TOPK", 8)), 1)
        shared_ratio = float(getattr(route_cfg, "UOT3D_SHARED_RATIO", 0.2))
        shared_ratio = min(max(shared_ratio, 0.0), 0.5)
        use_shared_transport = (
            shared_enabled
            and num_labels > 1
            and shared_ratio > 0.0
            and float(sharedness.sum().item()) > 0.0
        )
        if not use_shared_transport:
            shared_ratio = 0.0
        shared_total_mass = total_mass * shared_ratio
        private_total_mass = total_mass - shared_total_mass
        alpha_private_vis = float(getattr(route_cfg, "UOT3D_VIS_PRIVATE_WEIGHT", 1.0))
        alpha_shared_vis = float(getattr(route_cfg, "UOT3D_VIS_SHARED_WEIGHT", 1.0))

        frame_feat = self._masked_frame_mean(st_tokens, point_mask)
        frame_feat = F.normalize(frame_feat.float(), dim=-1)
        sim_frame = torch.matmul(text_norm, frame_feat.transpose(0, 1))
        sim_frame = torch.nan_to_num(
            sim_frame,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

        tau_frame = max(
            float(getattr(
                route_cfg,
                "UOT3D_TAU_FRAME",
                getattr(route_cfg, "UOT3D_TAU_LT", route_cfg.AFFINITY_TAU),
            )),
            1e-6,
        )
        mask_frame = valid_frame_mask.unsqueeze(0).expand(num_labels, temporal_dim)
        prob_frame = self._masked_softmax_1d(
            sim_frame,
            mask_frame,
            dim=-1,
            tau=tau_frame,
        )

        traj_feat = self._masked_traj_mean(st_tokens, point_mask)
        traj_feat = F.normalize(traj_feat.float(), dim=-1)
        sim_traj = torch.matmul(text_norm, traj_feat.transpose(0, 1))
        sim_traj = torch.nan_to_num(
            sim_traj,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

        tau_traj = max(
            float(getattr(
                route_cfg,
                "UOT3D_TAU_TRAJ",
                getattr(route_cfg, "UOT3D_TAU_LN", route_cfg.AFFINITY_TAU),
            )),
            1e-6,
        )
        mask_traj = valid_traj_mask.unsqueeze(0).expand(num_labels, num_points)
        prob_traj = self._masked_softmax_1d(
            sim_traj,
            mask_traj,
            dim=-1,
            tau=tau_traj,
        )

        tau_vis = max(
            float(getattr(
                route_cfg,
                "UOT3D_TAU_VIS",
                getattr(route_cfg, "UOT3D_TAU_TN", route_cfg.AFFINITY_TAU),
            )),
            1e-6,
        )
        sim_flat = sim.reshape(num_labels, temporal_dim * num_points)
        flat_mask = point_mask.reshape(-1)
        mask_vis = flat_mask.unsqueeze(0).expand(num_labels, temporal_dim * num_points)
        prob_vis_flat = self._masked_softmax_1d(
            sim_flat,
            mask_vis,
            dim=-1,
            tau=tau_vis,
        )
        prob_vis = prob_vis_flat.view(num_labels, temporal_dim, num_points)

        st_transports = []
        shared_transports = []
        transport_masses = []
        shared_transport_masses = []
        target_debug = []
        for target_output_idx, target_idx_tensor in enumerate(target_label_indices):
            target_idx = int(target_idx_tensor.detach().cpu().item())
            one_hot = torch.zeros_like(mu)
            one_hot[target_idx] = 1.0

            label_mass_private = target_mix * one_hot + (1.0 - target_mix) * mu
            label_mass_private = label_mass_private / label_mass_private.sum().clamp_min(1e-12)

            if use_shared_transport:
                label_mass = private_total_mass * label_mass_private
                prior_frame_private = label_mass[:, None] * prob_frame
                prior_traj_private = label_mass[:, None] * prob_traj

                shared_frame = sharedness.sum(dim=1)
                shared_frame = _normalize_masked_vector(shared_frame, valid_frame_mask)
                shared_frame = shared_total_mass * shared_frame

                shared_traj = sharedness.sum(dim=0)
                shared_traj = _normalize_masked_vector(shared_traj, valid_traj_mask)
                shared_traj = shared_total_mass * shared_traj

                prior_frame = torch.cat(
                    [prior_frame_private, shared_frame.unsqueeze(0)],
                    dim=0,
                )
                prior_traj = torch.cat(
                    [prior_traj_private, shared_traj.unsqueeze(0)],
                    dim=0,
                )
                token_score = (
                    alpha_private_vis * prob_vis[target_idx]
                    + alpha_shared_vis * sharedness
                )
                prior_vis = total_mass * _normalize_masked_map(token_score, point_mask)
                solve_cost = cost_ext
            else:
                label_mass = total_mass * label_mass_private
                prior_frame = label_mass[:, None] * prob_frame
                prior_traj = label_mass[:, None] * prob_traj
                prior_vis = label_mass[target_idx] * prob_vis[target_idx]
                solve_cost = cost

            gamma = self._solve_avg_3d_uot(
                cost=solve_cost,
                prior_frame=prior_frame,
                prior_traj=prior_traj,
                prior_vis=prior_vis,
                valid_mask=point_mask,
            )

            target_weight = gamma[target_idx]
            target_weight = target_weight * point_mask.to(target_weight.dtype)
            shared_weight = (
                gamma[-1] * point_mask.to(gamma.dtype)
                if use_shared_transport
                else target_weight.new_zeros(target_weight.shape)
            )
            st_transports.append(target_weight.unsqueeze(0))
            shared_transports.append(shared_weight.unsqueeze(0))
            transport_masses.append(target_weight.sum().view(1))
            shared_transport_masses.append(shared_weight.sum().view(1))
            if return_debug:
                target_weight_sum = target_weight.sum()
                shared_weight_sum = shared_weight.sum()
                gamma_total = gamma.sum()
                gamma_row_mass = gamma.sum(dim=(1, 2))
                gamma_frame = gamma.sum(dim=2)
                gamma_traj = gamma.sum(dim=1)
                gamma_vis = gamma.sum(dim=0)

                target_flat = target_weight.reshape(-1)
                sim_target_flat = sim[target_idx].reshape(-1)
                weighted_target_sim = (
                    target_flat * sim_target_flat
                ).sum() / target_weight_sum.clamp_min(1e-12)
                if num_labels > 1:
                    other_indices = [
                        label_idx
                        for label_idx in range(num_labels)
                        if label_idx != target_idx
                    ]
                    other_sim = sim.index_select(
                        0,
                        torch.as_tensor(
                            other_indices,
                            device=sim.device,
                            dtype=torch.long,
                        ),
                    ).amax(dim=0)
                    weighted_best_other_sim = (
                        target_weight * other_sim
                    ).sum() / target_weight_sum.clamp_min(1e-12)
                else:
                    weighted_best_other_sim = target_weight.new_zeros(())

                frame_mass = target_weight.sum(dim=1)
                traj_mass = target_weight.sum(dim=0)
                target_debug_item = {
                    "target_output_idx": int(target_output_idx),
                    "target_idx": int(target_idx),
                    "label_axis_idx": int(target_idx),
                    "label_mass": self._pot_debug_list(label_mass),
                    "target_label_mass": self._pot_debug_scalar(label_mass[target_idx]),
                    "private_mass": self._pot_debug_scalar(private_total_mass),
                    "shared_mass": self._pot_debug_scalar(shared_total_mass),
                    "gamma_total_mass": self._pot_debug_scalar(gamma_total),
                    "gamma_row_mass": self._pot_debug_list(gamma_row_mass),
                    "target_row_mass": self._pot_debug_scalar(target_weight_sum),
                    "target_row_ratio": self._pot_debug_scalar(
                        target_weight_sum / gamma_total.clamp_min(1e-12)
                    ),
                    "competitor_row_mass": self._pot_debug_scalar(
                        gamma_total - gamma_row_mass[target_idx]
                    ),
                    "weighted_sim_target": self._pot_debug_scalar(weighted_target_sim),
                    "weighted_sim_best_other": self._pot_debug_scalar(weighted_best_other_sim),
                    "weighted_sim_margin": self._pot_debug_scalar(
                        weighted_target_sim - weighted_best_other_sim
                    ),
                    "private_weighted_sim_margin": self._pot_debug_scalar(
                        weighted_target_sim - weighted_best_other_sim
                    ),
                    "frame_plane_l1_to_prior": self._pot_debug_scalar(
                        torch.abs(gamma_frame - prior_frame).sum()
                    ),
                    "traj_plane_l1_to_prior": self._pot_debug_scalar(
                        torch.abs(gamma_traj - prior_traj).sum()
                    ),
                    "vis_plane_l1_to_prior": self._pot_debug_scalar(
                        torch.abs(gamma_vis - prior_vis).sum()
                    ),
                    "target_frame_l1_to_prior": self._pot_debug_scalar(
                        torch.abs(frame_mass - prior_frame[target_idx]).sum()
                    ),
                    "target_traj_l1_to_prior": self._pot_debug_scalar(
                        torch.abs(traj_mass - prior_traj[target_idx]).sum()
                    ),
                    "target_token_summary": self._pot_debug_weight_summary(
                        target_weight.reshape(-1),
                        point_mask.reshape(-1),
                        debug_topk,
                    ),
                    "target_frame_summary": self._pot_debug_weight_summary(
                        frame_mass,
                        valid_frame_mask,
                        min(debug_topk, temporal_dim),
                    ),
                    "target_traj_summary": self._pot_debug_weight_summary(
                        traj_mass,
                        valid_traj_mask,
                        min(debug_topk, num_points),
                    ),
                    "top_tokens": self._pot_debug_top_st_tokens(
                        target_weight,
                        point_mask,
                        sim=sim[target_idx],
                        topk=debug_topk,
                    ),
                    "target_weighted_sharedness": self._pot_debug_scalar(
                        self._pot_debug_weighted_map_mean(
                            sharedness,
                            target_weight,
                            point_mask,
                        )
                    ),
                    "shared_row_mass": self._pot_debug_scalar(shared_weight_sum),
                    "shared_row_ratio": self._pot_debug_scalar(
                        shared_weight_sum / gamma_total.clamp_min(1e-12)
                    ),
                    "shared_absorption_ratio": self._pot_debug_scalar(
                        shared_weight_sum / gamma_total.clamp_min(1e-12)
                    ),
                    "shared_weighted_sharedness": self._pot_debug_scalar(
                        self._pot_debug_weighted_map_mean(
                            sharedness,
                            shared_weight,
                            point_mask,
                        )
                    ),
                    "target_vs_shared_overlap": self._pot_debug_pair_st_overlap(
                        target_weight,
                        shared_weight,
                        point_mask,
                        topk=debug_topk,
                    ),
                    "shared_top_tokens": self._pot_debug_top_tokens(
                        shared_weight.reshape(-1),
                        point_mask.reshape(-1),
                        point_mask,
                        sharedness.reshape(-1),
                        sim[target_idx].reshape(-1),
                        debug_topk,
                    ),
                }
                family_prob = cmw_aux.get("family_prob")
                if family_prob is not None:
                    target_debug_item.update({
                        "cmw_target_reliability_summary": (
                            self._pot_debug_masked_summary(
                                cmw_private_reliability[target_idx],
                                point_mask,
                            )
                        ),
                        "cmw_target_family_prob": (
                            self._pot_debug_list(family_prob[target_idx])
                            if family_prob is not None
                            else []
                        ),
                    })
                if use_shared_transport:
                    target_debug_item.update({
                        "shared_frame_l1_to_prior": self._pot_debug_scalar(
                            torch.abs(gamma_frame[-1] - prior_frame[-1]).sum()
                        ),
                        "shared_traj_l1_to_prior": self._pot_debug_scalar(
                            torch.abs(gamma_traj[-1] - prior_traj[-1]).sum()
                        ),
                    })
                target_debug.append(target_debug_item)

        st_transport = (
            torch.cat(st_transports, dim=0)
            if st_transports
            else st_tokens.new_zeros(num_targets, temporal_dim, num_points)
        )
        transport_mass = (
            torch.cat(transport_masses, dim=0)
            if transport_masses
            else st_tokens.new_zeros(num_targets)
        )

        st_transport = torch.nan_to_num(
            st_transport,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        shared_transport = (
            torch.cat(shared_transports, dim=0)
            if shared_transports
            else st_tokens.new_zeros(num_targets, temporal_dim, num_points)
        )
        shared_transport_mass = (
            torch.cat(shared_transport_masses, dim=0)
            if shared_transport_masses
            else st_tokens.new_zeros(num_targets)
        )
        shared_transport = torch.nan_to_num(
            shared_transport,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        result = {
            "st_transport": st_transport,
            "sim": sim,
            "cost": cost,
            "cost_ext": cost_ext,
            "private_cost": private_cost,
            "shared_cost": shared_cost,
            "sharedness": sharedness,
            "label_entropy": shared_components["label_entropy"],
            "semantic_strength": shared_components["semantic_strength"],
            "prob_frame": prob_frame,
            "prob_traj": prob_traj,
            "prob_vis": prob_vis,
            "shared_transport": shared_transport,
            "transport_mass": transport_mass,
            "shared_transport_mass": shared_transport_mass,
            "target_label_indices": target_label_indices,
        }
        result["cmw_private_reliability"] = cmw_private_reliability
        result["cmw_family_prob"] = cmw_aux.get("family_prob")
        if return_debug:
            valid_cost_mask = point_mask.unsqueeze(0).expand_as(private_cost)
            shared_cost_mask = point_mask
            max_sim = sim.max(dim=0).values
            result["debug"] = {
                "debug_type": (
                    "psr_3d_uot_soft"
                    if use_shared_transport
                    else "target_conditioned_3d_uot"
                ),
                "shape": {
                    "num_labels": int(num_labels),
                    "num_targets": int(num_targets),
                    "temporal_dim": int(temporal_dim),
                    "num_points": int(num_points),
                    "feature_dim": int(st_tokens.shape[-1]),
                },
                "valid": {
                    "points": int(point_mask.sum().item()),
                    "frames": int(valid_frame_mask.sum().item()),
                    "trajectories": int(valid_traj_mask.sum().item()),
                },
                "config": {
                    "target_mix": round(float(target_mix), 6),
                    "total_mass": round(float(total_mass), 6),
                    "tau_frame": round(float(tau_frame), 6),
                    "tau_traj": round(float(tau_traj), 6),
                    "tau_vis": round(float(tau_vis), 6),
                    "shared_enable": bool(shared_enabled),
                    "shared_effective": bool(use_shared_transport),
                    "shared_ratio": round(float(shared_ratio), 6),
                    "vis_private_weight": round(float(alpha_private_vis), 6),
                    "vis_shared_weight": round(float(alpha_shared_vis), 6),
                    "shared_tau_label": round(float(getattr(route_cfg, "SHARED_TAU_LABEL", 0.07)), 6),
                    "shared_theta": round(float(getattr(route_cfg, "SHARED_THETA", 0.2)), 6),
                    "shared_tau_strength": round(float(getattr(route_cfg, "SHARED_TAU_STRENGTH", 0.1)), 6),
                    "private_cost_source": "cmw",
                },
                "cost_source": cost_source,
                "target_label_indices": [
                    int(target_idx)
                    for target_idx in target_label_indices.detach().cpu().tolist()
                ],
                "mu": self._pot_debug_list(mu),
                "transport_mass": self._pot_debug_list(transport_mass),
                "shared_transport_mass": self._pot_debug_list(shared_transport_mass),
                "sharedness_summary": self._pot_debug_masked_summary(
                    sharedness,
                    point_mask,
                ),
                "label_entropy_summary": self._pot_debug_masked_summary(
                    shared_components["label_entropy"],
                    point_mask,
                ),
                "semantic_strength_summary": self._pot_debug_masked_summary(
                    shared_components["semantic_strength"],
                    point_mask,
                ),
                "private_cost_summary": self._pot_debug_masked_summary(
                    private_cost,
                    valid_cost_mask,
                ),
                "shared_cost_summary": self._pot_debug_masked_summary(
                    shared_cost,
                    shared_cost_mask,
                ),
                "shared_top_tokens": self._pot_debug_top_tokens(
                    sharedness.reshape(-1),
                    point_mask.reshape(-1),
                    point_mask,
                    sharedness.reshape(-1),
                    max_sim.reshape(-1),
                    debug_topk,
                ),
                "prob_frame_entropy": self._pot_debug_list(
                    self._normalized_distribution_entropy(
                        prob_frame,
                        valid_count=int(valid_frame_mask.sum().item()),
                    )
                ),
                "prob_traj_entropy": self._pot_debug_list(
                    self._normalized_distribution_entropy(
                        prob_traj,
                        valid_count=int(valid_traj_mask.sum().item()),
                    )
                ),
                "prob_vis_entropy": self._pot_debug_list(
                    self._normalized_distribution_entropy(
                        prob_vis.reshape(num_labels, -1),
                        valid_count=int(point_mask.sum().item()),
                    )
                ),
                "transport_overlap": self._pot_debug_transport_overlap(
                    st_transport,
                    point_mask,
                    topk=debug_topk,
                ),
                "shared_transport_overlap": self._pot_debug_transport_overlap(
                    shared_transport,
                    point_mask,
                    topk=debug_topk,
                ),
                "targets": target_debug,
            }
            cmw_valid_mask = point_mask.unsqueeze(0).expand_as(
                cmw_private_reliability
            )
            cmw_family_prob = cmw_aux.get("family_prob")
            result["debug"].update({
                "cmw_private_reliability_summary": self._pot_debug_masked_summary(
                    cmw_private_reliability,
                    cmw_valid_mask,
                ),
                "cmw_family_prob": (
                    [
                        self._pot_debug_list(cmw_family_prob[label_idx])
                        for label_idx in range(cmw_family_prob.shape[0])
                    ]
                    if cmw_family_prob is not None
                    else []
                ),
                "cmw_evidence_mean": {
                    name: self._pot_debug_scalar(value[point_mask].mean())
                    if value.ndim == 2
                    else self._pot_debug_scalar(
                        value[point_mask.unsqueeze(0).expand_as(value)].mean()
                    )
                    for name, value in cmw_evidence_components.items()
                },
            })
        return result

    def _solve_joint_relaxed_transport(
        self,
        cost,
        row_mass,
        col_cap,
        force_total_mass=True,
    ):
        """Solve a joint relaxed POT problem over positive labels and trajectories."""
        route_cfg = self.pot_route_cfg
        entropic_eps = max(float(route_cfg.ENTROPIC_EPS), 1e-6)
        max_iters = max(int(route_cfg.MAX_ITERS), 1)
        stop_tol = max(float(route_cfg.STOP_TOL), 0.0)

        cost = torch.nan_to_num(cost.float(), nan=1e4, posinf=1e4, neginf=0.0)
        row_mass = torch.nan_to_num(
            row_mass.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        col_cap = torch.nan_to_num(
            col_cap.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        total_mass = row_mass.sum()
        if cost.numel() == 0 or float(total_mass.item()) <= 0.0:
            return cost.new_zeros(cost.shape)

        cap_sum = col_cap.sum()
        if float(cap_sum.item()) <= 0.0:
            return cost.new_zeros(cost.shape)
        if force_total_mass and float(cap_sum.item()) < float(total_mass.item()):
            col_cap = col_cap * (total_mass / cap_sum.clamp_min(1e-12))

        valid_entries = (
            (cost < 1e3)
            & (row_mass[:, None] > 0.0)
            & (col_cap[None, :] > 0.0)
        )
        row_min = cost.masked_fill(~valid_entries, float("inf")).amin(dim=1)
        has_valid_row = torch.isfinite(row_min)
        if not has_valid_row.any():
            return cost.new_zeros(cost.shape)

        row_min = torch.where(has_valid_row, row_min, torch.zeros_like(row_min))
        shifted_cost = cost - row_min[:, None]
        log_kernel = (-shifted_cost / entropic_eps).clamp(min=-80.0, max=0.0)
        transport = torch.where(
            valid_entries,
            torch.exp(log_kernel),
            torch.zeros_like(cost),
        )
        if float(transport.sum().item()) <= 0.0:
            return cost.new_zeros(cost.shape)
        transport = transport * (
            total_mass / transport.sum().clamp_min(1e-12)
        )

        for _ in range(max_iters):
            prev_transport = transport

            row_sum = prev_transport.sum(dim=1)
            row_scale = torch.minimum(
                row_mass / row_sum.clamp_min(1e-12),
                torch.ones_like(row_sum),
            )
            transport = row_scale[:, None] * prev_transport

            col_sum = transport.sum(dim=0)
            col_scale = torch.minimum(
                col_cap / col_sum.clamp_min(1e-12),
                torch.ones_like(col_sum),
            )
            transport = transport * col_scale[None, :]

            if force_total_mass:
                current_mass = transport.sum()
                if float(current_mass.item()) > 0.0:
                    transport = transport * (
                        total_mass / current_mass.clamp_min(1e-12)
                    )
            delta = torch.max(torch.abs(transport - prev_transport))
            if float(delta.item()) <= stop_tol:
                break

        return torch.nan_to_num(transport, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_joint_positive_traj_transport(
        self,
        st_tokens,
        point_mask,
        support_global,
        positive_text,
    ):
        """Solve positive-label POT over full trajectories, then refine over time."""
        route_cfg = self.pot_route_cfg
        st_tokens = torch.nan_to_num(st_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        support_global = torch.nan_to_num(
            support_global,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        positive_text = torch.nan_to_num(
            positive_text,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if point_mask is None:
            point_mask = torch.ones(
                st_tokens.shape[:2],
                device=st_tokens.device,
                dtype=torch.bool,
            )
        else:
            point_mask = point_mask.to(device=st_tokens.device).bool()
        num_labels = positive_text.shape[0]
        temporal_dim, num_points = st_tokens.shape[:2]
        valid_traj_mask = point_mask.any(dim=0) if num_points > 0 else point_mask.new_zeros(0)
        if (
            num_labels == 0
            or temporal_dim == 0
            or num_points == 0
            or not valid_traj_mask.any()
        ):
            traj_empty = st_tokens.new_zeros(num_labels, num_points)
            st_empty = st_tokens.new_zeros(num_labels, temporal_dim, num_points)
            return {
                "transport": traj_empty,
                "temporal_weights": st_empty,
                "st_transport": st_empty,
                "affinity": traj_empty,
                "traj_sim": traj_empty,
            }

        st_tokens = F.normalize(st_tokens.float(), dim=-1)
        support_global = F.normalize(support_global.float(), dim=-1)
        positive_text = F.normalize(positive_text.float(), dim=-1)

        sim = torch.einsum("ld,tnd->ltn", positive_text, st_tokens)
        sim = torch.nan_to_num(
            sim,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

        valid_time_mask = point_mask.unsqueeze(0)
        masked_sim = sim.masked_fill(~valid_time_mask, -1e4)
        temporal_tau = max(
            float(getattr(route_cfg, "TEMPORAL_TAU", route_cfg.AFFINITY_TAU)),
            1e-6,
        )
        temporal_weights = torch.softmax(masked_sim / temporal_tau, dim=1)
        temporal_weights = temporal_weights * valid_time_mask.to(temporal_weights.dtype)
        temporal_weights = temporal_weights / temporal_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
        temporal_weights = torch.where(
            valid_traj_mask.view(1, 1, num_points),
            temporal_weights,
            torch.zeros_like(temporal_weights),
        )
        temporal_weights = torch.nan_to_num(
            temporal_weights,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        traj_sim = (temporal_weights * sim).sum(dim=1)
        traj_sim = torch.nan_to_num(
            traj_sim,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        traj_sim = traj_sim.masked_fill(~valid_traj_mask.unsqueeze(0), -1e4)

        cost = 1.0 - traj_sim
        cost = cost.masked_fill(~valid_traj_mask.unsqueeze(0), 1e4)

        mu_logits = max(float(route_cfg.MU_LOGIT_SCALE), 1e-6) * torch.matmul(
            positive_text,
            support_global,
        )
        mu = torch.softmax(mu_logits, dim=0)

        affinity_tau = max(float(route_cfg.AFFINITY_TAU), 1e-6)
        affinity = torch.softmax(traj_sim / affinity_tau, dim=-1)
        affinity = affinity * valid_traj_mask.unsqueeze(0).to(affinity.dtype)
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        affinity = torch.nan_to_num(
            affinity,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        entropy = self._normalized_distribution_entropy(
            affinity,
            valid_count=int(valid_traj_mask.sum().item()),
        )
        rho_min = min(max(float(route_cfg.RHO_MIN), 0.0), 1.0)
        rho_max = min(max(float(route_cfg.RHO_MAX), rho_min), 1.0)
        rho = rho_min + (rho_max - rho_min) * entropy
        row_mass = mu * rho

        nu_shared = affinity.mean(dim=0) * valid_traj_mask.to(affinity.dtype)
        if float(nu_shared.sum().item()) <= 0.0:
            nu_shared = valid_traj_mask.to(affinity.dtype)
        nu_shared = nu_shared / nu_shared.sum().clamp_min(1e-6)
        kappa = max(float(route_cfg.KAPPA), 1.0)
        col_cap = kappa * nu_shared

        transport = self._solve_joint_relaxed_transport(
            cost,
            row_mass,
            col_cap,
            force_total_mass=True,
        )
        st_transport = transport[:, None, :] * temporal_weights
        st_transport = st_transport * valid_time_mask.to(st_transport.dtype)
        st_transport = torch.nan_to_num(
            st_transport,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return {
            "transport": transport,
            "temporal_weights": temporal_weights,
            "st_transport": st_transport,
            "affinity": affinity,
            "traj_sim": traj_sim,
        }

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

    def _pot_debug_rank(self):
        """Return the distributed rank used for POT debug logging."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return int(os.environ.get("RANK", "0"))

    def _should_log_pot_debug(self):
        """Return whether this forward should write sampled POT debug records."""
        route_cfg = self.pot_route_cfg
        if not bool(getattr(route_cfg, "DEBUG_ENABLE", False)):
            return False
        if bool(getattr(route_cfg, "DEBUG_RANK0_ONLY", True)):
            if self._pot_debug_rank() != 0:
                return False

        call_count = getattr(self, "_pot_debug_call_count", 0) + 1
        self._pot_debug_call_count = call_count
        log_period = max(int(getattr(route_cfg, "DEBUG_LOG_PERIOD", 10)), 1)
        if (call_count - 1) % log_period != 0:
            return False

        max_records = int(getattr(route_cfg, "DEBUG_MAX_RECORDS", 1000))
        if max_records <= 0:
            return True
        return getattr(self, "_pot_debug_record_count", 0) < max_records

    def _pot_debug_scalar(self, value, default=0.0):
        """Convert a scalar tensor-like value to a JSON-friendly float."""
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            value = torch.nan_to_num(
                value.detach().float(),
                nan=default,
                posinf=default,
                neginf=default,
            ).mean()
            return round(float(value.cpu().item()), 6)
        return round(float(value), 6)

    def _pot_debug_list(self, value, max_items=None):
        """Convert a tensor-like vector to JSON-friendly floats."""
        if not torch.is_tensor(value):
            values = list(value)
        else:
            values = torch.nan_to_num(
                value.detach().flatten().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).cpu().tolist()
        if max_items is not None:
            values = values[:max_items]
        return [round(float(item), 6) for item in values]

    def _pot_debug_masked_values(self, value, mask):
        """Return finite values selected by a boolean mask."""
        mask = mask.to(device=value.device).bool()
        if value.numel() == 0 or not mask.any():
            return value.new_zeros(0)
        return torch.nan_to_num(
            value.detach().float()[mask],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _pot_debug_masked_summary(self, value, mask):
        """Summarize masked tensor values."""
        values = self._pot_debug_masked_values(value, mask)
        if values.numel() == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return {
            "mean": self._pot_debug_scalar(values.mean()),
            "std": self._pot_debug_scalar(values.std(unbiased=False)),
            "min": self._pot_debug_scalar(values.min()),
            "max": self._pot_debug_scalar(values.max()),
        }

    def _pot_debug_weight_summary(self, weights, valid_mask, topk):
        """Summarize how concentrated a transport row is over valid tokens."""
        weights = torch.nan_to_num(
            weights.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        valid_mask = valid_mask.to(device=weights.device).bool()
        valid_weights = weights[valid_mask]
        num_valid = int(valid_mask.sum().item())
        invalid_weight_sum = weights[~valid_mask].sum() if (~valid_mask).any() else weights.new_zeros(())
        weight_sum = valid_weights.sum()
        summary = {
            "weight_sum": self._pot_debug_scalar(weight_sum),
            "invalid_weight_sum": self._pot_debug_scalar(invalid_weight_sum),
            "entropy": 0.0,
            "effective_token_count": 0.0,
            "top8_mass_ratio": 0.0,
            "topk_mass_ratio": 0.0,
        }
        if num_valid == 0 or float(weight_sum.item()) <= 0.0:
            return summary

        normalized = valid_weights / weight_sum.clamp_min(1e-12)
        entropy = -(
            normalized.clamp_min(1e-12) * normalized.clamp_min(1e-12).log()
        ).sum()
        entropy = entropy / entropy.new_tensor(max(float(np.log(max(num_valid, 2))), 1e-6))
        effective_count = 1.0 / normalized.pow(2).sum().clamp_min(1e-12)
        top8 = min(8, num_valid)
        topk = min(max(int(topk), 1), num_valid)
        summary.update({
            "entropy": self._pot_debug_scalar(entropy),
            "effective_token_count": self._pot_debug_scalar(effective_count),
            "top8_mass_ratio": self._pot_debug_scalar(torch.topk(normalized, k=top8).values.sum()),
            "topk_mass_ratio": self._pot_debug_scalar(torch.topk(normalized, k=topk).values.sum()),
        })
        return summary

    def _pot_debug_top_st_tokens(self, weights, point_mask, sim=None, topk=8):
        """Return top weighted frame/trajectory coordinates for 3D-UOT."""
        weights = torch.nan_to_num(
            weights.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        point_mask = point_mask.to(device=weights.device).bool()
        if weights.numel() == 0 or not point_mask.any():
            return []

        temporal_dim, num_points = point_mask.shape
        flat_weights = weights.reshape(-1)
        flat_mask = point_mask.reshape(-1)
        masked_weights = flat_weights.masked_fill(~flat_mask, -1.0)
        topk = min(max(int(topk), 1), int(flat_mask.sum().item()))
        values, indices = torch.topk(masked_weights, k=topk)

        sim_flat = None
        if sim is not None:
            sim_flat = torch.nan_to_num(
                sim.detach().float().reshape(-1).to(device=weights.device),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )

        tokens = []
        for weight, token_idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            if weight < 0.0:
                continue
            frame_idx = int(token_idx // num_points)
            point_idx = int(token_idx % num_points)
            item = {
                "token_idx": int(token_idx),
                "frame": frame_idx,
                "point": point_idx,
                "weight": round(float(weight), 6),
            }
            if sim_flat is not None:
                item["text_sim"] = self._pot_debug_scalar(sim_flat[token_idx])
            tokens.append(item)
        return tokens

    def _pot_debug_transport_overlap(self, st_transport, point_mask, topk=8):
        """Summarize pairwise overlap among target-conditioned transport maps."""
        st_transport = torch.nan_to_num(
            st_transport.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        point_mask = point_mask.to(device=st_transport.device).bool()
        num_labels = st_transport.shape[0]
        flat_mask = point_mask.reshape(-1)
        if num_labels <= 1 or not flat_mask.any():
            return {
                "pair_count": 0,
                "cosine_mean": 0.0,
                "cosine_max": 0.0,
                "topk_overlap_mean": 0.0,
                "topk_overlap_max": 0.0,
                "pairs": [],
            }

        flat = st_transport.reshape(num_labels, -1)
        flat = flat * flat_mask.unsqueeze(0).to(flat.dtype)
        l1 = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
        l2 = flat / flat.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
        cosine = torch.matmul(l2, l2.transpose(0, 1)).clamp(0.0, 1.0)

        valid_count = int(flat_mask.sum().item())
        topk = min(max(int(topk), 1), valid_count)
        top_indices = []
        masked_l1 = l1.masked_fill(~flat_mask.unsqueeze(0), -1.0)
        for label_idx in range(num_labels):
            top_indices.append(set(torch.topk(masked_l1[label_idx], k=topk).indices.detach().cpu().tolist()))

        pairs = []
        cos_values = []
        overlap_values = []
        for left_idx in range(num_labels):
            for right_idx in range(left_idx + 1, num_labels):
                overlap = len(top_indices[left_idx] & top_indices[right_idx]) / max(float(topk), 1.0)
                cos_value = cosine[left_idx, right_idx]
                pairs.append({
                    "left": int(left_idx),
                    "right": int(right_idx),
                    "cosine": self._pot_debug_scalar(cos_value),
                    "topk_overlap": round(float(overlap), 6),
                })
                cos_values.append(float(cos_value.detach().cpu().item()))
                overlap_values.append(float(overlap))

        return {
            "pair_count": len(pairs),
            "cosine_mean": round(float(np.mean(cos_values)), 6) if cos_values else 0.0,
            "cosine_max": round(float(np.max(cos_values)), 6) if cos_values else 0.0,
            "topk_overlap_mean": round(float(np.mean(overlap_values)), 6) if overlap_values else 0.0,
            "topk_overlap_max": round(float(np.max(overlap_values)), 6) if overlap_values else 0.0,
            "pairs": pairs,
        }

    def _pot_debug_pair_st_overlap(self, left_weights, right_weights, point_mask, topk=8):
        """Summarize overlap between two spatio-temporal transport maps."""
        left_weights = torch.nan_to_num(
            left_weights.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        right_weights = torch.nan_to_num(
            right_weights.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        point_mask = point_mask.to(device=left_weights.device).bool()
        flat_mask = point_mask.reshape(-1)
        if left_weights.numel() == 0 or right_weights.numel() == 0 or not flat_mask.any():
            return {
                "cosine": 0.0,
                "topk_overlap": 0.0,
                "left_mass": 0.0,
                "right_mass": 0.0,
                "shared_min_mass": 0.0,
            }

        left_flat = left_weights.reshape(-1) * flat_mask.to(left_weights.dtype)
        right_flat = right_weights.reshape(-1) * flat_mask.to(right_weights.dtype)
        left_sum = left_flat.sum()
        right_sum = right_flat.sum()
        if float(left_sum.item()) <= 0.0 or float(right_sum.item()) <= 0.0:
            return {
                "cosine": 0.0,
                "topk_overlap": 0.0,
                "left_mass": self._pot_debug_scalar(left_sum),
                "right_mass": self._pot_debug_scalar(right_sum),
                "shared_min_mass": 0.0,
            }
        denom = left_flat.pow(2).sum().sqrt() * right_flat.pow(2).sum().sqrt()
        cosine = (left_flat * right_flat).sum() / denom.clamp_min(1e-12)

        valid_count = int(flat_mask.sum().item())
        topk = min(max(int(topk), 1), valid_count)
        left_norm = left_flat / left_sum.clamp_min(1e-12)
        right_norm = right_flat / right_sum.clamp_min(1e-12)
        left_top = set(
            torch.topk(left_norm.masked_fill(~flat_mask, -1.0), k=topk)
            .indices.detach().cpu().tolist()
        )
        right_top = set(
            torch.topk(right_norm.masked_fill(~flat_mask, -1.0), k=topk)
            .indices.detach().cpu().tolist()
        )
        return {
            "cosine": self._pot_debug_scalar(cosine.clamp(0.0, 1.0)),
            "topk_overlap": round(len(left_top & right_top) / max(float(topk), 1.0), 6),
            "left_mass": self._pot_debug_scalar(left_sum),
            "right_mass": self._pot_debug_scalar(right_sum),
            "shared_min_mass": self._pot_debug_scalar(torch.minimum(left_flat, right_flat).sum()),
        }

    def _pot_debug_weighted_map_mean(self, values, weights, valid_mask):
        """Mean of values under non-negative map weights on valid tokens."""
        values = torch.nan_to_num(
            values.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        weights = torch.nan_to_num(
            weights.detach().float().to(device=values.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        valid_mask = valid_mask.to(device=values.device).bool()
        weights = weights * valid_mask.to(weights.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-12)

    def _pot_debug_top_value_mean(self, values, weights, valid_mask, topk):
        """Mean value over top-weighted valid tokens."""
        weights = torch.nan_to_num(
            weights.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        values = torch.nan_to_num(
            values.detach().float().to(device=weights.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid_mask = valid_mask.to(device=weights.device).bool()
        if not valid_mask.any():
            return 0.0
        masked_weights = weights.masked_fill(~valid_mask, -1.0)
        topk = min(max(int(topk), 1), int(valid_mask.sum().item()))
        indices = torch.topk(masked_weights, k=topk).indices
        return self._pot_debug_scalar(values[indices].mean())

    def _pot_debug_top_tokens(
        self,
        weights,
        valid_mask,
        point_mask,
        sharedness,
        sim_row,
        topk,
    ):
        """Return top-weighted token coordinates for a transport row."""
        if not bool(getattr(self.pot_route_cfg, "DEBUG_SAVE_TOP_TOKENS", True)):
            return []
        weights = torch.nan_to_num(
            weights.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid_mask = valid_mask.to(device=weights.device).bool()
        if not valid_mask.any():
            return []

        masked_weights = weights.masked_fill(~valid_mask, -1.0)
        topk = min(max(int(topk), 1), int(valid_mask.sum().item()))
        values, indices = torch.topk(masked_weights, k=topk)
        num_points = point_mask.shape[1]
        tokens = []
        for weight, token_idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            if weight < 0.0:
                continue
            frame_idx = int(token_idx // num_points)
            point_idx = int(token_idx % num_points)
            tokens.append({
                "token_idx": int(token_idx),
                "frame": frame_idx,
                "point": point_idx,
                "weight": round(float(weight), 6),
                "sharedness": self._pot_debug_scalar(sharedness[token_idx]),
                "text_sim": self._pot_debug_scalar(sim_row[token_idx]),
            })
        return tokens

    def _pot_debug_label_summary(self, D_label):
        """Summarize label-side FGW structure."""
        num_labels = D_label.shape[0] - 1
        if num_labels <= 1:
            return {
                "private_offdiag_mean": 0.0,
                "shared_tau_mean": self._pot_debug_scalar(D_label[-1, :-1].mean()) if num_labels > 0 else 0.0,
            }
        private = D_label[:num_labels, :num_labels]
        offdiag_mask = ~torch.eye(
            num_labels,
            device=D_label.device,
            dtype=torch.bool,
        )
        return {
            "private_offdiag_mean": self._pot_debug_scalar(private[offdiag_mask].mean()),
            "shared_tau_mean": self._pot_debug_scalar(D_label[-1, :-1].mean()),
        }

    def _write_pot_debug_record(self, record):
        """Append one POT debug record to the standalone JSONL file."""
        max_records = int(getattr(self.pot_route_cfg, "DEBUG_MAX_RECORDS", 1000))
        if max_records > 0 and getattr(self, "_pot_debug_record_count", 0) >= max_records:
            return
        if getattr(self, "_pot_debug_io_failed", False):
            return

        debug_file = str(getattr(self.pot_route_cfg, "DEBUG_FILE", "pot_3d_uot_debug.jsonl"))
        if os.path.isabs(debug_file):
            debug_path = Path(debug_file)
        else:
            debug_path = Path(str(self.cfg.OUTPUT_DIR)) / debug_file
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        record = dict(record)
        record["rank"] = self._pot_debug_rank()
        record["mode"] = "train" if self.training else "eval"
        record["debug_call"] = getattr(self, "_pot_debug_call_count", 0)
        record["wall_time"] = round(time.time(), 3)

        if (
            getattr(self, "_pot_debug_record_count", 0) == 0
            and bool(getattr(self.pot_route_cfg, "DEBUG_RESET_FILE", True))
        ):
            try:
                debug_path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            with debug_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._pot_debug_record_count = getattr(self, "_pot_debug_record_count", 0) + 1
        except OSError as err:
            self._pot_debug_io_failed = True
            print(f"WARNING: failed to write POT debug log {debug_path}: {err}")

    def _flatten_st_tokens(self, feat, point_mask):
        """Flatten spatio-temporal point tokens and masks."""
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if point_mask is None:
            point_mask = torch.ones(
                feat.shape[:2],
                device=feat.device,
                dtype=torch.bool,
            )
        else:
            point_mask = point_mask.to(device=feat.device).bool()
        flat_feat = rearrange(feat, 't n c -> (t n) c')
        flat_mask = rearrange(point_mask, 't n -> (t n)')
        return flat_feat, flat_mask

    def _compute_sharedness(self, sim_matrix, valid_mask):
        """Estimate how much each token is shared across positive labels."""
        valid_mask = valid_mask.to(device=sim_matrix.device).bool()
        num_labels = sim_matrix.shape[0]
        if num_labels <= 1:
            return sim_matrix.new_zeros(sim_matrix.shape[-1])

        route_cfg = self.pot_route_cfg
        tau_label = max(float(getattr(route_cfg, "SHARED_TAU_LABEL", 0.07)), 1e-6)
        theta_shared = float(getattr(route_cfg, "SHARED_THETA", 0.2))
        tau_strength = max(
            float(getattr(route_cfg, "SHARED_TAU_STRENGTH", 0.1)),
            1e-6,
        )

        sim_matrix = torch.nan_to_num(
            sim_matrix.float(),
            nan=-1e4,
            posinf=1.0,
            neginf=-1e4,
        )
        label_prob = torch.softmax(sim_matrix / tau_label, dim=0).clamp_min(1e-12)
        entropy = -(label_prob * label_prob.log()).sum(dim=0)
        normalizer = max(float(np.log(max(num_labels, 2))), 1e-6)
        entropy = entropy / entropy.new_tensor(normalizer)
        max_sim = sim_matrix.max(dim=0).values
        semantic_strength = torch.sigmoid((max_sim - theta_shared) / tau_strength)
        sharedness = entropy * semantic_strength
        sharedness = torch.where(valid_mask, sharedness, torch.zeros_like(sharedness))
        return torch.nan_to_num(
            sharedness.clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def _compute_intra_evidence(self, pred_tracks, valid_mask):
        """Compute intra-motion evidence from velocity and acceleration."""
        pred_tracks = torch.nan_to_num(
            pred_tracks.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid_mask = valid_mask.to(device=pred_tracks.device).bool()
        if valid_mask.ndim == 1:
            valid_mask_2d = valid_mask.view(pred_tracks.shape[:2])
        else:
            valid_mask_2d = valid_mask

        velocity = pred_tracks.new_zeros(pred_tracks.shape[:2])
        if pred_tracks.shape[0] > 1:
            velocity[1:] = torch.norm(pred_tracks[1:] - pred_tracks[:-1], dim=-1)

        acceleration = pred_tracks.new_zeros(pred_tracks.shape[:2])
        if pred_tracks.shape[0] > 1:
            acceleration[1:] = torch.abs(velocity[1:] - velocity[:-1])

        evidence = velocity + 0.5 * acceleration
        evidence_flat = rearrange(evidence, 't n -> (t n)')
        valid_flat = rearrange(valid_mask_2d, 't n -> (t n)')
        normalized = torch.zeros_like(evidence_flat)
        if valid_flat.any():
            valid_evidence = evidence_flat[valid_flat]
            if float((valid_evidence.max() - valid_evidence.min()).item()) > 1e-6:
                median = valid_evidence.median()
                mad = (valid_evidence - median).abs().median().clamp_min(1e-6)
                normalized = torch.sigmoid((evidence_flat - median) / mad)
                normalized = torch.where(
                    valid_flat,
                    normalized,
                    torch.zeros_like(normalized),
                )
        return torch.nan_to_num(
            normalized.clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def _compute_inter_relation(self, pred_tracks, valid_mask, topk=16):
        """Build a same-frame sparse-like dense relation matrix from point tracks."""
        pred_tracks = torch.nan_to_num(
            pred_tracks.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid_mask = valid_mask.to(device=pred_tracks.device).bool()
        temporal_dim, num_points = pred_tracks.shape[:2]
        num_tokens = temporal_dim * num_points
        relation_matrix = pred_tracks.new_zeros(num_tokens, num_tokens)
        if num_points <= 1:
            return relation_matrix

        route_cfg = self.pot_route_cfg
        sigma = max(float(getattr(route_cfg, "INTER_SIGMA", 0.5)), 1e-6)
        topk = max(int(topk), 1)
        topk = min(topk, num_points - 1)

        prev_dist = None
        for frame_idx in range(temporal_dim):
            frame_tracks = pred_tracks[frame_idx]
            dist = torch.cdist(frame_tracks, frame_tracks, p=2)
            if prev_dist is None:
                delta_dist = torch.zeros_like(dist)
            else:
                delta_dist = torch.abs(dist - prev_dist)

            frame_valid = valid_mask[frame_idx]
            pair_valid = frame_valid[:, None] & frame_valid[None, :]
            if frame_idx > 0:
                prev_valid = valid_mask[frame_idx - 1]
                pair_valid = pair_valid & prev_valid[:, None] & prev_valid[None, :]
            pair_valid.fill_diagonal_(False)

            delta_norm = torch.zeros_like(delta_dist)
            if pair_valid.any():
                max_delta = delta_dist[pair_valid].max().clamp_min(1e-6)
                delta_norm = delta_dist / max_delta

            relation = torch.exp(-(dist ** 2) / (sigma ** 2)) * delta_norm
            relation = torch.where(pair_valid, relation, torch.zeros_like(relation))
            if topk < num_points:
                values, indices = torch.topk(relation, k=topk, dim=-1)
                sparse_relation = torch.zeros_like(relation)
                sparse_relation.scatter_(dim=-1, index=indices, src=values)
                relation = sparse_relation

            start_idx = frame_idx * num_points
            end_idx = start_idx + num_points
            relation_matrix[start_idx:end_idx, start_idx:end_idx] = relation
            prev_dist = dist

        return torch.nan_to_num(
            relation_matrix.clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def _build_shared_private_cost(
        self,
        sim_matrix,
        sharedness,
        valid_mask,
    ):
        """Construct unary private-label and shared-row transport costs."""
        route_cfg = self.pot_route_cfg
        lambda_shared = float(getattr(route_cfg, "LAMBDA_SHARED", 0.5))
        valid_mask = valid_mask.to(device=sim_matrix.device).bool()

        sim_matrix = torch.nan_to_num(
            sim_matrix.float(),
            nan=-1e4,
            posinf=1.0,
            neginf=-1e4,
        )
        sharedness = torch.nan_to_num(
            sharedness.to(device=sim_matrix.device).float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        private_cost = 1.0 - sim_matrix + lambda_shared * sharedness.unsqueeze(0)
        shared_cost = 1.0 - sharedness
        cost = torch.cat([private_cost, shared_cost.unsqueeze(0)], dim=0)
        cost = torch.where(
            valid_mask.unsqueeze(0),
            cost,
            cost.new_full(cost.shape, 1e4),
        )
        return torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=0.0)

    def _build_label_structure(self, positive_text):
        """Build label-side distances for private labels plus one shared row."""
        route_cfg = self.pot_route_cfg
        beta_text = float(getattr(route_cfg, "FGW_BETA_TEXT", 0.5))
        beta_sep = float(getattr(route_cfg, "FGW_BETA_SEP", 0.5))
        shared_tau = float(getattr(route_cfg, "FGW_SHARED_TAU", 0.5))

        positive_text = torch.nan_to_num(
            positive_text.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        num_labels = positive_text.shape[0]
        label_structure = positive_text.new_zeros(num_labels + 1, num_labels + 1)
        if num_labels == 0:
            return label_structure

        text = F.normalize(positive_text, dim=-1)
        text_sim = torch.matmul(text, text.transpose(0, 1)).clamp(-1.0, 1.0)
        text_dist = (1.0 - text_sim) / 2.0
        sep_dist = torch.ones_like(text_dist)
        sep_dist.fill_diagonal_(0.0)
        private_dist = beta_text * text_dist + beta_sep * sep_dist
        private_dist.fill_diagonal_(0.0)

        label_structure[:num_labels, :num_labels] = private_dist
        label_structure[num_labels, :num_labels] = shared_tau
        label_structure[:num_labels, num_labels] = shared_tau
        return torch.nan_to_num(
            label_structure.clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def _build_visual_structure(
        self,
        value_flat,
        flat_mask,
        pred_tracks,
        point_mask,
        S_inter=None,
        return_debug=False,
    ):
        """Build token-side distances from value features and same-frame relations."""
        del pred_tracks, point_mask
        route_cfg = self.pot_route_cfg
        w_feat = float(getattr(route_cfg, "FGW_W_FEAT", 0.6))
        w_inter = float(getattr(route_cfg, "FGW_W_INTER", 0.4))

        value_flat = torch.nan_to_num(
            value_flat.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        flat_mask = flat_mask.to(device=value_flat.device).bool()
        pair_valid = flat_mask[:, None] & flat_mask[None, :]

        value_norm = F.normalize(value_flat, dim=-1)
        feat_sim = torch.matmul(value_norm, value_norm.transpose(0, 1)).clamp(-1.0, 1.0)
        D_feat = ((1.0 - feat_sim) / 2.0).clamp(0.0, 1.0)

        if S_inter is None:
            R_inter = torch.zeros_like(D_feat)
            D_inter = torch.full_like(D_feat, 0.5)
        else:
            R_inter = torch.nan_to_num(
                S_inter.to(device=value_flat.device, dtype=D_feat.dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            D_inter = torch.full_like(D_feat, 0.5)
            D_inter = torch.where(R_inter > 0.0, 1.0 - R_inter, D_inter)

        D_feat = torch.where(pair_valid, D_feat, torch.ones_like(D_feat))
        D_inter = torch.where(pair_valid, D_inter, torch.ones_like(D_inter))
        D_feat.fill_diagonal_(0.0)
        D_inter.fill_diagonal_(0.0)

        D_visual = w_feat * D_feat + w_inter * D_inter
        D_visual = 0.5 * (D_visual + D_visual.transpose(0, 1))
        D_visual = torch.where(pair_valid, D_visual, torch.ones_like(D_visual))
        D_visual.fill_diagonal_(0.0)
        D_visual = torch.nan_to_num(
            D_visual.clamp(0.0, 1.0),
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        if not return_debug:
            return D_visual

        offdiag_mask = ~torch.eye(
            D_visual.shape[0],
            device=D_visual.device,
            dtype=torch.bool,
        )
        pair_valid = pair_valid & offdiag_mask
        edge_mask = (R_inter > 0.0) & pair_valid
        debug = {
            "D_feat": self._pot_debug_masked_summary(D_feat, pair_valid),
            "D_inter": self._pot_debug_masked_summary(D_inter, pair_valid),
            "D_visual": self._pot_debug_masked_summary(D_visual, pair_valid),
            "D_visual_sym_error": self._pot_debug_scalar(
                torch.abs(D_visual - D_visual.transpose(0, 1)).mean()
            ),
            "D_visual_diag_max": self._pot_debug_scalar(torch.diag(D_visual).abs().max()),
            "S_inter_edge_density": self._pot_debug_scalar(
                edge_mask.to(D_visual.dtype).sum() / pair_valid.to(D_visual.dtype).sum().clamp_min(1.0)
            ),
            "S_inter_edge_mean": (
                self._pot_debug_scalar(R_inter[edge_mask].mean())
                if edge_mask.any()
                else 0.0
            ),
        }
        return D_visual, debug

    def _compute_fgw_cost(self, D_label, D_visual, transport, valid_mask):
        """Compute the square-loss GW cost induced by the current transport."""
        D_label = torch.nan_to_num(
            D_label.to(device=transport.device).float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        D_visual = torch.nan_to_num(
            D_visual.to(device=transport.device).float(),
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        transport = torch.nan_to_num(
            transport.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid_mask = valid_mask.to(device=transport.device).bool()

        row_mass = transport.sum(dim=1)
        col_mass = transport.sum(dim=0)
        term1 = torch.matmul(D_label.pow(2), row_mass).unsqueeze(1)
        term2 = torch.matmul(D_visual.pow(2), col_mass).unsqueeze(0)
        middle = torch.matmul(torch.matmul(D_label, transport), D_visual.transpose(0, 1))
        fgw_cost = (term1 + term2 - 2.0 * middle).clamp_min(0.0)

        shared_weight = float(getattr(self.pot_route_cfg, "FGW_SHARED_WEIGHT", 1.0))
        if shared_weight != 1.0 and fgw_cost.shape[0] > 0:
            fgw_cost[-1] = shared_weight * fgw_cost[-1]
        fgw_cost = torch.where(
            valid_mask.unsqueeze(0),
            fgw_cost,
            fgw_cost.new_full(fgw_cost.shape, 1e4),
        )
        return torch.nan_to_num(fgw_cost, nan=1e4, posinf=1e4, neginf=0.0)

    def _scale_fgw_cost_for_unary(self, fgw_cost, unary_cost, valid_columns):
        """Center and scale FGW costs so they can affect the OT solve."""
        route_cfg = self.pot_route_cfg
        if not bool(getattr(route_cfg, "FGW_NORMALIZE_COST", True)):
            return fgw_cost

        cost_scale = max(float(getattr(route_cfg, "FGW_COST_SCALE", 1.0)), 0.0)
        min_std = max(float(getattr(route_cfg, "FGW_COST_MIN_STD", 1e-6)), 1e-12)
        valid_columns = valid_columns.to(device=fgw_cost.device).bool()
        valid_mask = valid_columns.unsqueeze(0).expand_as(fgw_cost)

        fgw_cost = torch.nan_to_num(
            fgw_cost.float(),
            nan=1e4,
            posinf=1e4,
            neginf=0.0,
        )
        unary_cost = torch.nan_to_num(
            unary_cost.to(device=fgw_cost.device).float(),
            nan=1e4,
            posinf=1e4,
            neginf=0.0,
        )
        centered_fgw = torch.zeros_like(fgw_cost)

        for row_idx in range(fgw_cost.shape[0]):
            row_mask = valid_mask[row_idx] & (fgw_cost[row_idx] < 1e3)
            if not row_mask.any():
                continue

            row_fgw = fgw_cost[row_idx, row_mask]
            row_unary = unary_cost[row_idx, row_mask]
            fgw_std = row_fgw.std(unbiased=False)
            unary_std = row_unary.std(unbiased=False)
            if float(fgw_std.item()) <= min_std:
                continue

            if float(unary_std.item()) <= min_std:
                unary_std = row_unary.abs().mean().clamp_min(min_std)
            normalized = (row_fgw - row_fgw.mean()) / fgw_std.clamp_min(min_std)
            centered_fgw[row_idx, row_mask] = normalized * unary_std * cost_scale

        centered_fgw = torch.where(
            valid_mask,
            centered_fgw,
            centered_fgw.new_full(centered_fgw.shape, 1e4),
        )
        return torch.nan_to_num(centered_fgw, nan=1e4, posinf=1e4, neginf=-1e4)

    def _compute_private_row_mass(self, sim_matrix, valid_mask):
        """Compute partial row mass for private label rows."""
        route_cfg = self.pot_route_cfg
        valid_mask = valid_mask.to(device=sim_matrix.device).bool()
        num_labels = sim_matrix.shape[0]
        if num_labels == 0 or not valid_mask.any():
            return sim_matrix.new_zeros(num_labels)

        tau_mass = max(float(getattr(route_cfg, "MASS_TAU", 0.07)), 1e-6)
        mass_topk = max(int(getattr(route_cfg, "MASS_TOPK", 16)), 1)
        theta_mass = float(getattr(route_cfg, "MASS_THETA", 0.2))
        tau_quality = max(
            float(getattr(route_cfg, "MASS_TAU_QUALITY", 0.1)),
            1e-6,
        )
        rho_min = min(max(float(getattr(route_cfg, "RHO_MIN", 0.2)), 0.0), 1.0)
        rho_max = min(max(float(getattr(route_cfg, "RHO_MAX", 0.9)), rho_min), 1.0)

        sim_matrix = torch.nan_to_num(
            sim_matrix.float(),
            nan=-1e4,
            posinf=1.0,
            neginf=-1e4,
        )
        masked_sim = sim_matrix.masked_fill(~valid_mask.unsqueeze(0), -1e4)
        affinity = torch.softmax(masked_sim / tau_mass, dim=-1)
        affinity = affinity * valid_mask.unsqueeze(0).to(affinity.dtype)
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(affinity.clamp_min(1e-12) * affinity.clamp_min(1e-12).log()).sum(
            dim=-1
        )

        num_valid = int(valid_mask.sum().item())
        entropy_normalizer = max(float(np.log(max(num_valid, 2))), 1e-6)
        entropy = entropy / entropy.new_tensor(entropy_normalizer)
        effective_area = torch.exp(
            entropy * entropy.new_tensor(float(np.log(max(num_valid, 1))))
        ) / max(float(num_valid), 1.0)

        valid_sim = sim_matrix[:, valid_mask]
        topk = min(mass_topk, valid_sim.shape[-1])
        topk_sim = torch.topk(valid_sim, k=topk, dim=-1).values.mean(dim=-1)
        quality = torch.sigmoid((topk_sim - theta_mass) / tau_quality)

        rho = rho_min + (rho_max - rho_min) * effective_area * quality
        row_mass = rho / max(float(num_labels), 1.0)
        return torch.nan_to_num(row_mass, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_shared_private_column_cap(self, sim_matrix, sharedness, valid_mask):
        """Compute column capacity that allows shared tokens without favoring private rows."""
        route_cfg = self.pot_route_cfg
        valid_mask = valid_mask.to(device=sim_matrix.device).bool()
        if not valid_mask.any():
            return sim_matrix.new_zeros(sim_matrix.shape[-1])

        affinity_tau = max(float(getattr(route_cfg, "AFFINITY_TAU", 0.07)), 1e-6)
        alpha_shared_cap = float(getattr(route_cfg, "ALPHA_SHARED_CAP", 1.0))
        kappa = max(float(getattr(route_cfg, "KAPPA", 1.5)), 1.0)

        sim_matrix = torch.nan_to_num(
            sim_matrix.float(),
            nan=-1e4,
            posinf=1.0,
            neginf=-1e4,
        )
        masked_sim = sim_matrix.masked_fill(~valid_mask.unsqueeze(0), -1e4)
        affinity = torch.softmax(masked_sim / affinity_tau, dim=-1)
        affinity = affinity * valid_mask.unsqueeze(0).to(affinity.dtype)
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        max_affinity = affinity.max(dim=0).values
        sharedness = sharedness.to(device=sim_matrix.device).float().clamp(0.0, 1.0)
        private_candidate = max_affinity * (1.0 - sharedness)
        shared_candidate = sharedness
        nu = private_candidate + alpha_shared_cap * shared_candidate
        nu = nu * valid_mask.to(nu.dtype)
        if float(nu.sum().item()) <= 0.0:
            nu = valid_mask.to(nu.dtype)
        nu = nu / nu.sum().clamp_min(1e-12)
        col_cap = kappa * nu
        return torch.nan_to_num(col_cap, nan=0.0, posinf=0.0, neginf=0.0)

    def _solve_shared_private_fgw_transport(
        self,
        unary_cost,
        row_mass,
        col_cap,
        D_label,
        D_visual,
        valid_mask,
        return_debug=False,
    ):
        """Solve shared-private partial FGW transport."""
        route_cfg = self.pot_route_cfg
        outer_iters = max(int(getattr(route_cfg, "FGW_OUTER_ITERS", 2)), 0)
        lambda_gw = float(getattr(route_cfg, "FGW_LAMBDA", 0.03))
        force_total_mass = bool(getattr(route_cfg, "FORCE_TOTAL_MASS", False))

        unary_cost = torch.nan_to_num(
            unary_cost.float(),
            nan=1e4,
            posinf=1e4,
            neginf=0.0,
        )
        transport = self._solve_joint_relaxed_transport(
            unary_cost,
            row_mass,
            col_cap,
            force_total_mass=force_total_mass,
        )
        initial_transport = transport.clone() if return_debug else None
        debug_iters = []
        valid_columns = valid_mask.to(device=unary_cost.device).bool()
        valid_cost_mask = valid_columns.unsqueeze(0).expand_as(unary_cost)

        for outer_idx in range(outer_iters):
            fgw_cost = self._compute_fgw_cost(
                D_label,
                D_visual,
                transport,
                valid_mask,
            )
            scaled_fgw_cost = self._scale_fgw_cost_for_unary(
                fgw_cost,
                unary_cost,
                valid_columns,
            )
            effective_cost = unary_cost + lambda_gw * scaled_fgw_cost
            if return_debug:
                delta_cost = lambda_gw * scaled_fgw_cost
                debug_iters.append({
                    "iter": outer_idx,
                    "fgw_cost": self._pot_debug_masked_summary(
                        fgw_cost,
                        valid_cost_mask,
                    ),
                    "scaled_fgw_cost": self._pot_debug_masked_summary(
                        scaled_fgw_cost,
                        valid_cost_mask,
                    ),
                    "effective_delta": self._pot_debug_masked_summary(
                        delta_cost,
                        valid_cost_mask,
                    ),
                    "transport_sum_before": self._pot_debug_scalar(transport.sum()),
                })
            effective_cost = torch.where(
                valid_columns.unsqueeze(0),
                effective_cost,
                effective_cost.new_full(effective_cost.shape, 1e4),
            )
            effective_cost = torch.nan_to_num(
                effective_cost,
                nan=1e4,
                posinf=1e4,
                neginf=0.0,
            )
            transport = self._solve_joint_relaxed_transport(
                effective_cost,
                row_mass,
                col_cap,
                force_total_mass=force_total_mass,
            )
            if return_debug:
                debug_iters[-1]["transport_sum_after"] = self._pot_debug_scalar(
                    transport.sum()
                )

        transport = torch.nan_to_num(transport, nan=0.0, posinf=0.0, neginf=0.0)
        if not return_debug:
            return transport

        delta = torch.abs(transport - initial_transport)
        valid_delta = delta[valid_cost_mask] if valid_cost_mask.any() else delta.new_zeros(1)
        debug = {
            "outer_iters": outer_iters,
            "lambda_gw": round(float(lambda_gw), 6),
            "force_total_mass": bool(force_total_mass),
            "initial_transport_sum": self._pot_debug_scalar(initial_transport.sum()),
            "final_transport_sum": self._pot_debug_scalar(transport.sum()),
            "transport_delta_l1": self._pot_debug_scalar(valid_delta.mean()),
            "transport_delta_max": self._pot_debug_scalar(valid_delta.max()),
            "initial_private_row_sum": self._pot_debug_list(initial_transport[:-1].sum(dim=1)),
            "final_private_row_sum": self._pot_debug_list(transport[:-1].sum(dim=1)),
            "initial_shared_row_sum": self._pot_debug_scalar(initial_transport[-1].sum()),
            "final_shared_row_sum": self._pot_debug_scalar(transport[-1].sum()),
            "iters": debug_iters,
        }
        return transport, debug

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

    def _build_pot_support_prototypes(
        self,
        app_tokens,
        intra_tokens,
        inter_tokens,
        value_tokens,
        metadata,
    ):
        """Build support prototypes with target-conditioned 3D-UOT routing."""
        del app_tokens
        support_mask = metadata['support_mask'].bool()
        episode_positive_labels = metadata['episode_positive_labels'].to(
            device=value_tokens.device,
        ).bool()
        base_pt_mask = (
            metadata['pred_query_mask']
            if self.cfg.POINT_INFO.USE_PT_QUERY_MASK
            else metadata['pred_visibility']
        ).bool()
        episode_class_ids = metadata['episode_class_ids'].to(
            device=value_tokens.device,
        ).long()
        episode_class_ids = (
            episode_class_ids[0]
            if episode_class_ids.ndim == 2
            else episode_class_ids
        )

        raw_positive_labels = metadata.get('raw_positive_labels')
        if raw_positive_labels is None:
            max_episode_id = (
                int(episode_class_ids.max().item()) + 1
                if episode_class_ids.numel() > 0
                else 0
            )
            fallback_num_classes = (
                int(self.num_classes)
                if isinstance(self.num_classes, int)
                else max_episode_id
            )
            fallback_num_classes = max(fallback_num_classes, max_episode_id)
            raw_positive_labels = value_tokens.new_zeros(
                value_tokens.shape[0],
                fallback_num_classes,
            )
            raw_positive_labels[:, episode_class_ids] = episode_positive_labels.to(
                dtype=raw_positive_labels.dtype,
            )
        raw_positive_labels = raw_positive_labels.to(
            device=value_tokens.device,
        ).bool()
        raw_positive_labels = raw_positive_labels.view(value_tokens.shape[0], -1)

        support_indices = torch.nonzero(support_mask, as_tuple=False).flatten()
        support_raw_positive_labels = raw_positive_labels.index_select(
            0,
            support_indices.to(device=value_tokens.device),
        )
        raw_label_union = torch.nonzero(
            support_raw_positive_labels.any(dim=0),
            as_tuple=False,
        ).flatten()
        if raw_label_union.numel() > 0:
            label_text_class_ids = torch.unique(
                torch.cat([episode_class_ids, raw_label_union.to(episode_class_ids.device)])
            )
        else:
            label_text_class_ids = episode_class_ids
        label_text_class_ids = label_text_class_ids.to(
            device=value_tokens.device,
            dtype=torch.long,
        )
        label_text_features = self._get_pot_label_text_features(
            label_text_class_ids,
            value_tokens.dtype,
        )
        text_lookup_size = raw_positive_labels.shape[1]
        if label_text_class_ids.numel() > 0:
            text_lookup_size = max(
                int(text_lookup_size),
                int(label_text_class_ids.max().item()) + 1,
            )
        global_to_text_index = torch.full(
            (text_lookup_size,),
            -1,
            device=value_tokens.device,
            dtype=torch.long,
        )
        global_to_text_index[label_text_class_ids] = torch.arange(
            label_text_class_ids.numel(),
            device=value_tokens.device,
            dtype=torch.long,
        )

        branch_tokens = []
        branch_weights = []
        branch_class_indices = []
        branch_sample_indices = []
        log_pot_debug = self._should_log_pot_debug()
        debug_max_samples = max(
            int(getattr(self.pot_route_cfg, "DEBUG_MAX_SAMPLES_PER_CALL", 2)),
            1,
        )
        debug_topk = max(int(getattr(self.pot_route_cfg, "DEBUG_TOPK", 8)), 1)
        debug_records_this_call = 0
        for sample_idx in support_indices.tolist():
            sample_episode_positive_labels = torch.nonzero(
                episode_positive_labels[sample_idx],
                as_tuple=False,
            ).flatten()
            label_axis_global_labels = torch.nonzero(
                raw_positive_labels[sample_idx],
                as_tuple=False,
            ).flatten().long()
            if label_axis_global_labels.numel() == 0:
                label_axis_global_labels = episode_class_ids.index_select(
                    0,
                    sample_episode_positive_labels,
                )
            if label_axis_global_labels.numel() == 0:
                continue

            label_axis_global_labels = label_axis_global_labels.to(
                device=value_tokens.device,
                dtype=torch.long,
            )
            axis_episode_match = (
                label_axis_global_labels[:, None] == episode_class_ids[None, :]
            )
            label_axis_proto_indices, proto_episode_labels = torch.nonzero(
                axis_episode_match,
                as_tuple=True,
            )
            if proto_episode_labels.numel() == 0:
                continue

            sample_value_tokens = value_tokens[sample_idx]
            sample_point_mask = base_pt_mask[sample_idx]
            support_global = self._masked_space_time_mean(
                sample_value_tokens.unsqueeze(0),
                sample_point_mask.unsqueeze(0),
            ).squeeze(0)

            text_indices = global_to_text_index.index_select(
                0,
                label_axis_global_labels,
            )
            valid_text = text_indices >= 0
            if not valid_text.all():
                label_axis_global_labels = label_axis_global_labels[valid_text]
                text_indices = text_indices[valid_text]
                axis_episode_match = (
                    label_axis_global_labels[:, None] == episode_class_ids[None, :]
                )
                label_axis_proto_indices, proto_episode_labels = torch.nonzero(
                    axis_episode_match,
                    as_tuple=True,
                )
                if proto_episode_labels.numel() == 0:
                    continue

            positive_text = label_text_features.index_select(
                0,
                text_indices,
            )
            sample_intra_tokens = (
                intra_tokens[sample_idx]
                if torch.is_tensor(intra_tokens)
                else None
            )
            sample_inter_tokens = (
                inter_tokens[sample_idx]
                if torch.is_tensor(inter_tokens)
                else None
            )
            debug_this_sample = (
                log_pot_debug
                and debug_records_this_call < debug_max_samples
            )
            joint_stats = self._compute_avg_3d_uot_transport(
                sample_value_tokens,
                sample_point_mask,
                support_global,
                positive_text,
                intra_tokens=sample_intra_tokens,
                inter_tokens=sample_inter_tokens,
                episode_positive_labels=episode_positive_labels,
                support_mask=support_mask,
                sample_positive_labels=label_axis_global_labels,
                episode_label_text=label_text_features,
                raw_positive_labels=raw_positive_labels,
                return_debug=debug_this_sample,
                target_label_indices=label_axis_proto_indices,
            )
            st_transport = joint_stats["st_transport"]
            debug_branch_summaries = []

            for target_output_idx, (label_axis_idx, proto_episode_label) in enumerate(
                zip(label_axis_proto_indices, proto_episode_labels)
            ):
                st_weights = st_transport[target_output_idx]
                st_weights = st_weights * sample_point_mask.to(st_weights.dtype)
                weight_sum = st_weights.sum()
                if float(weight_sum.item()) <= 0.0:
                    st_weights = sample_point_mask.to(sample_value_tokens.dtype)
                    weight_sum = st_weights.sum()
                st_weights = st_weights / weight_sum.clamp_min(1e-6)
                st_weights = torch.nan_to_num(
                    st_weights,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                sample_proto = self._aggregate_weighted_st_support_tokens(
                    sample_value_tokens,
                    sample_point_mask,
                    st_weights,
                )
                branch_tokens.append(sample_proto.unsqueeze(1))
                branch_weights.append(st_weights.reshape(-1))
                branch_class_indices.append(proto_episode_label.view(1))
                branch_sample_indices.append(sample_idx)
                if debug_this_sample:
                    global_class_id = label_axis_global_labels[label_axis_idx]
                    debug_branch_summaries.append({
                        "target_idx": int(label_axis_idx.detach().cpu().item()),
                        "target_output_idx": int(target_output_idx),
                        "label_axis_idx": int(label_axis_idx.detach().cpu().item()),
                        "episode_class_idx": int(proto_episode_label.detach().cpu().item()),
                        "global_class_id": int(global_class_id.detach().cpu().item()),
                        "normalized_weight_summary": self._pot_debug_weight_summary(
                            st_weights.reshape(-1),
                            sample_point_mask.reshape(-1),
                            debug_topk,
                        ),
                        "frame_mass": self._pot_debug_list(st_weights.sum(dim=1)),
                        "top_tokens": self._pot_debug_top_st_tokens(
                            st_weights,
                            sample_point_mask,
                            sim=joint_stats["sim"][label_axis_idx],
                            topk=debug_topk,
                        ),
                    })

            if debug_this_sample and "debug" in joint_stats:
                label_axis_episode_indices = torch.full(
                    (label_axis_global_labels.numel(),),
                    -1,
                    device=value_tokens.device,
                    dtype=torch.long,
                )
                label_axis_episode_indices[label_axis_proto_indices] = proto_episode_labels
                proto_global_labels = label_axis_global_labels.index_select(
                    0,
                    label_axis_proto_indices,
                )
                auxiliary_global_labels = label_axis_global_labels[
                    label_axis_episode_indices < 0
                ]

                label_axis_global_ids = [
                    int(class_id)
                    for class_id in label_axis_global_labels.detach().cpu().tolist()
                ]
                label_axis_names = [
                    (
                        self.atomic_label_names[class_id]
                        if 0 <= class_id < len(self.atomic_label_names)
                        else str(class_id)
                    )
                    for class_id in label_axis_global_ids
                ]
                proto_global_ids = [
                    int(class_id)
                    for class_id in proto_global_labels.detach().cpu().tolist()
                ]
                proto_label_names = [
                    (
                        self.atomic_label_names[class_id]
                        if 0 <= class_id < len(self.atomic_label_names)
                        else str(class_id)
                    )
                    for class_id in proto_global_ids
                ]
                auxiliary_global_ids = [
                    int(class_id)
                    for class_id in auxiliary_global_labels.detach().cpu().tolist()
                ]
                auxiliary_label_names = [
                    (
                        self.atomic_label_names[class_id]
                        if 0 <= class_id < len(self.atomic_label_names)
                        else str(class_id)
                    )
                    for class_id in auxiliary_global_ids
                ]
                self._write_pot_debug_record({
                    "debug_type": joint_stats["debug"].get(
                        "debug_type",
                        "target_conditioned_3d_uot",
                    ),
                    "sample_idx": int(sample_idx),
                    "positive_episode_class_indices": [
                        int(class_idx)
                        for class_idx in label_axis_episode_indices.detach().cpu().tolist()
                    ],
                    "positive_global_class_ids": label_axis_global_ids,
                    "positive_label_names": label_axis_names,
                    "label_axis_global_class_ids": label_axis_global_ids,
                    "label_axis_episode_class_indices": [
                        int(class_idx)
                        for class_idx in label_axis_episode_indices.detach().cpu().tolist()
                    ],
                    "label_axis_label_names": label_axis_names,
                    "prototype_episode_class_indices": [
                        int(class_idx)
                        for class_idx in proto_episode_labels.detach().cpu().tolist()
                    ],
                    "target_label_axis_indices": [
                        int(class_idx)
                        for class_idx in label_axis_proto_indices.detach().cpu().tolist()
                    ],
                    "prototype_global_class_ids": proto_global_ids,
                    "prototype_label_names": proto_label_names,
                    "auxiliary_global_class_ids": auxiliary_global_ids,
                    "auxiliary_label_names": auxiliary_label_names,
                    "support_valid_points": int(sample_point_mask.sum().item()),
                    "uot3d": joint_stats["debug"],
                    "branches": debug_branch_summaries,
                })
                debug_records_this_call += 1

        if not branch_tokens:
            return None

        support_tokens = torch.stack(branch_tokens, dim=0)
        support_tokens = torch.nan_to_num(
            support_tokens,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        support_branch_point_weights = torch.stack(branch_weights, dim=0)
        class_indices = torch.cat(branch_class_indices, dim=0)
        aux = {
            'support_conditioned_patch_tokens': support_tokens,
            'support_branch_point_weights': support_branch_point_weights.to(
                device=support_tokens.device,
                dtype=support_tokens.dtype,
            ),
            'support_branch_class_indices': class_indices.to(
                device=support_tokens.device,
                dtype=torch.long,
            ),
            'support_branch_sample_indices': torch.tensor(
                branch_sample_indices,
                device=support_tokens.device,
                dtype=torch.long,
            ),
        }
        return aux

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
            if self.use_pot_support_route:
                route_aux = self._build_pot_support_prototypes(
                    app_tokens=app_feat,
                    intra_tokens=intra_feat,
                    inter_tokens=inter_feat,
                    value_tokens=patch_x,
                    metadata=metadata,
                )
                if route_aux is not None:
                    few_shot_aux.update(route_aux)
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
