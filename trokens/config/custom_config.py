#!/usr/bin/env python3

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode

#pylint: disable=line-too-long

def add_custom_config(cfg):
    """Add custom configs."""
    cfg.DATA.PATH_TO_TROKEN_PT_DATA = '/fs/cfar-projects/actionloc/bounce_back/camera_ready/data/trokens_pt_data/'

    #wandb config
    cfg.WANDB = CfgNode()
    cfg.WANDB.PROJECT = 'trokens'
    cfg.WANDB.ENTITY = 'act_seg_pi_umd'
    cfg.WANDB.ID = ''
    cfg.WANDB.EXP_NAME = ''

    # few shot config
    cfg.FEW_SHOT = CfgNode()
    cfg.FEW_SHOT.N_WAY = 5
    cfg.FEW_SHOT.K_SHOT = 1
    cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS = 6
    cfg.FEW_SHOT.TEST_QUERY_PER_CLASS = 1
    cfg.FEW_SHOT.TRAIN_EPISODES = 100000
    cfg.FEW_SHOT.TEST_EPISODES = 10000
    cfg.FEW_SHOT.PATCH_TOKENS_AGG = 'spatial'
    cfg.FEW_SHOT.USE_MODEL = True
    cfg.FEW_SHOT.DIST_NORM = 'none'
    cfg.FEW_SHOT.TRAIN_OG_EPISODES = False
    cfg.FEW_SHOT.CLASS_LOSS_LAMBDA = 1.0
    cfg.FEW_SHOT.Q2S_LOSS_LAMBDA = 1.0
    # Opt-in single-path Label -> Patch priors inside serial temporal/spatial attention.
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL = CfgNode()
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ENABLE = False
    # projected_cosine: learned label/Raw projections (v5-v8 control).
    # cosine: original-space cosine, no learned routing Q/K (v9-v12).
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.SCORING = "projected_cosine"
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ROUTE_TEMPERATURE = 0.04
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.GATE_STRENGTH = 0.25
    # Fixed lambda in softmax(QK/sqrt(d) + lambda * log(g + eps)); zero ablates
    # only that attention bias, not the downstream class-conditioned readout.
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.TEMPORAL_BIAS_SCALE = 1.0
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.SPATIAL_BIAS_SCALE = 1.0
    # Keep Raw Support+text for encoding; optionally use Post matching Support
    # prototypes+text ONLY for the final Query readout. False preserves v5/v6.
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.POST_SUPPORT_READOUT = False
    # Fold independent classes into the video batch; 1 keeps serial execution.
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.CLASS_CHUNK_SIZE = 1
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.VIDEO_CHUNK_SIZE = 4
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ACTIVATION_CHECKPOINT = True
    # Raw quality is class-conditioned but independent of Post encodings.
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.BATCH_RAW_QUALITY = False
    # Faster high-memory alternative to checkpointing the entire encoder/readout.
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.MLP_CHECKPOINT = False
    cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.READOUT_CHECKPOINT = False
    # Per-rank episodes accumulated before one optimizer update.
    cfg.SOLVER.GRAD_ACCUM_STEPS = 1
    # Opt-in floor; zero disables the floor, not non-finite scale checks.
    cfg.SOLVER.AMP_MIN_SCALE = 0.0
    cfg.FEW_SHOT.SUPPORT_TEXT_FUSION = CfgNode()
    cfg.FEW_SHOT.SUPPORT_TEXT_FUSION.ENABLE = False
    cfg.FEW_SHOT.SUPPORT_TEXT_FUSION.TEXT_WEIGHT = 1.0
    cfg.FEW_SHOT.SUPPORT_TEXT_FUSION.VISUAL_WEIGHT = 1.0
    cfg.FEW_SHOT.SUPPORT_TEXT_FUSION.VISUAL_DETACH = True
    cfg.FEW_SHOT.POT_ROUTE = CfgNode()
    cfg.FEW_SHOT.POT_ROUTE.ENABLE = False
    cfg.FEW_SHOT.POT_ROUTE.FRAME_SOFTMAX_TAU = 0.07
    cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_ENABLE = False
    cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_LOGIT_ALPHA = 10.0
    cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_LOGIT_BIAS = -2.0


    # Raw semantic quality and Post metric uniqueness verify Query frames.
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY = CfgNode()
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE = False
    verifier = CfgNode()
    # Preserve v3 for old configs; SAV opts into the direct learned decision.
    verifier.QUALITY_MODE = "semantic_relation"
    verifier.QUALITY_HIDDEN_DIM = 64
    verifier.QUALITY_DECISION_LOCAL_DIM = 32
    verifier.QUALITY_DECISION_READOUT_DIM = 16
    verifier.QUALITY_PATCH_TEMPERATURE = 0.07
    verifier.QUALITY_REFERENCE_TEMPERATURE = 0.10
    verifier.QUALITY_MAX_POOL_DELTA = 0.5
    verifier.QUALITY_MAX_REFERENCE_DELTA = 0.5
    verifier.QUALITY_MAX_EVIDENCE_DELTA = 0.10
    verifier.QUALITY_SCALE_INIT = 10.0
    verifier.QUALITY_BIAS_INIT = 0.0
    # Inference mechanism control: 0 removes only the final relation correction.
    verifier.QUALITY_RELATION_STRENGTH = 1.0
    verifier.METRIC_TEMPERATURE_INIT = 0.10
    verifier.METRIC_SCALE_INIT = 10.0
    verifier.DETACH_INPUTS = True
    verifier.WEIGHT_DECAY = 0.01
    verifier.SEMANTIC_MIL_WEIGHT = 0.05
    verifier.RELATION_MIL_WEIGHT = 0.05
    verifier.TRANSPORT_WARMUP_FRACTION = 0.05
    verifier.QUERY_CHUNK_SIZE = 4
    verifier.ACTIVATION_CHECKPOINT = True
    # Follow outer CUDA autocast for Linear/MLP/attention GEMMs when opted in;
    # normalization, softmax, pooling and gates remain FP32.
    verifier.AMP_ENABLE = False
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER = verifier
    # Independent quality and uniqueness ablations.
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.QUALITY_ENABLE = True
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.UNIQUENESS_ENABLE = True
    # Shared hypothesis transport and base/verified training objectives.
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_TRANSPORT_STRENGTH = 1.0
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_UNMATCHED_COST = 0.0
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_ONE_SIDED = True
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.DUAL_LOGIT_LOSS_ENABLE = False
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.BASE_LOGIT_LOSS_WEIGHT = 0.50
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.VERIFIED_LOGIT_LOSS_WEIGHT = 0.50

    # point info config
    cfg.POINT_INFO = CfgNode()
    cfg.POINT_INFO.ENABLE = True
    cfg.POINT_INFO.GRID_SIZE = 16
    cfg.POINT_INFO.NAME = ''
    cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE = 256
    cfg.POINT_INFO.SAMPLING_TYPE = 'random'
    cfg.POINT_INFO.PT_FIX_SAMPLING_TRAIN = False
    cfg.POINT_INFO.PT_FIX_SAMPLING_TEST = False
    cfg.POINT_INFO.USE_PT_QUERY_MASK = False
    # seed: forward-tracking start mask; visibility: observed points including
    # backward tracks; all: explicit historical unmasked-route ablation.
    cfg.POINT_INFO.QUERY_MASK_MODE = "seed"
    cfg.POINT_INFO.OBJ_ID_KEY = 'obj_ids'
    cfg.POINT_INFO.HOD = CfgNode()
    cfg.POINT_INFO.HOD.NUM_BINS = 32
    cfg.POINT_INFO.HOD.NUM_CLUSTERS = 16
    cfg.POINT_INFO.HOD_MIN = True
    cfg.POINT_INFO.HOD.GET_FEAT = True
    cfg.POINT_INFO.HOD.PRESERVE_TEMPORAL = True

    # motion module config
    cfg.MODEL.MOTION_MODULE = CfgNode()
    cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE = False
    cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE = False
    cfg.MODEL.APPEARANCE_MODULE_DISABLE = False

    return cfg
