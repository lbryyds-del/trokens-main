"""Resolve supported Query/Support conditioning and rho verification settings."""
import math


def resolve_ablation_cfg(cfg):
    """Resolve after YAML/CLI merge; keep legacy rho/no-verifier routes intact."""
    fs = cfg.FEW_SHOT
    switches = fs.DECOUPLING
    names = ('QUERY_ENABLE', 'SUPPORT_ENABLE', 'RHO_ENABLE')
    values = [getattr(switches, name, None) for name in names]
    if all(value is None for value in values):
        from .entity_change import validate_entity_change_cfg
        validate_entity_change_cfg(cfg)
        return cfg
    if not all(type(value) is bool for value in values):
        raise ValueError('DECOUPLING requires all three switches to be True or False.')
    query, support, rho = values
    if not query and rho:
        raise ValueError('RHO_ENABLE requires QUERY_ENABLE=True (K:K matching).')
    if fs.LABEL_PRE_TEMPORAL.ENABLE or fs.POT_ROUTE.ENABLE or fs.SUPPORT_TEXT_FUSION.ENABLE:
        raise ValueError('DECOUPLING cannot be combined with legacy early routing or Support/text fusion.')
    from .entity_change import validate_entity_change_cfg
    validate_entity_change_cfg(cfg)
    matching = fs.MATCHING
    decoupled = query or support
    for name in ('LOGIT_SCALE', 'LOGIT_BIAS', 'BASE_LOSS_WEIGHT', 'FINAL_LOSS_WEIGHT',
                 'RHO_AUX_WEIGHT', 'TRANSPORT_STRENGTH'):
        value = float(getattr(matching, name))
        if not math.isfinite(value) or (name != 'LOGIT_BIAS' and value < 0):
            raise ValueError(f'MATCHING.{name} must be finite and nonnegative (bias may be negative).')
    if rho and any(getattr(matching, name) <= 0 for name in
                   ('BASE_LOSS_WEIGHT', 'FINAL_LOSS_WEIGHT', 'TRANSPORT_STRENGTH', 'RHO_AUX_WEIGHT')):
        raise ValueError('Enabled rho requires positive base/final/auxiliary weights and transport strength.')
    frozen = cfg.is_frozen()
    if frozen:
        cfg.defrost()
    try:
        fs.TEXT_CONCAT_ST.ENABLE = decoupled
        fs.TEXT_CONCAT_ST.SHARED_QUERY_ENABLE = not query
        fs.TEXT_CONCAT_ST.SHARED_SUPPORT_ENABLE = not support
        fs.POT_ROUTE.QUERY_PARTIAL_ENABLE = decoupled
        # Both roles off preserves the original negative-distance scorer.
        fs.POT_ROUTE.QUERY_PARTIAL_LOGIT_ALPHA = matching.LOGIT_SCALE if decoupled else 1.0
        fs.POT_ROUTE.QUERY_PARTIAL_LOGIT_BIAS = matching.LOGIT_BIAS if decoupled else 0.0
        match = fs.QUERY_CLASS_MATCHABILITY
        match.ENABLE = decoupled
        match.UNIQUENESS_ENABLE = rho
        match.BASE_ONLY_LOSS_ENABLE = decoupled and not rho
        match.DUAL_LOGIT_LOSS_ENABLE = rho
        match.BASE_LOGIT_LOSS_WEIGHT = matching.BASE_LOSS_WEIGHT if rho else 1.0
        match.VERIFIED_LOGIT_LOSS_WEIGHT = matching.FINAL_LOSS_WEIGHT if rho else 0.0
        match.ABSOLUTE_MASS_TRANSPORT_STRENGTH = matching.TRANSPORT_STRENGTH if rho else 0.0
        match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT = matching.RHO_AUX_WEIGHT if rho else 0.0
    finally:
        if frozen:
            cfg.freeze()
    return cfg


def ablation_summary(cfg):
    switches = cfg.FEW_SHOT.DECOUPLING
    if switches.QUERY_ENABLE is None:
        return 'Legacy profile: using explicit role/rho/loss settings.'
    match = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    decoupled = switches.QUERY_ENABLE or switches.SUPPORT_ENABLE
    scoring = ('calibrated bidirectional similarity' if decoupled else
               'original Trokens pt_forward + ReLU + negative bidirectional distance (no scale/bias)')
    loss = 'L_base' if decoupled else 'L_original_q2s_BCE'
    if switches.RHO_ENABLE:
        loss = (f'{match.BASE_LOGIT_LOSS_WEIGHT:g}*L_base + '
                f'{match.VERIFIED_LOGIT_LOSS_WEIGHT:g}*L_final + '
                f'{match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT:g}*L_rho')
    change = cfg.FEW_SHOT.ENTITY_CHANGE
    detail = ''
    if change.ENABLE:
        detail = (f'; trajectory-cluster evidence: state={change.STATE_ENABLE}, geo={change.GEO_ENABLE}, '
                  f'additive rescoring, scalar_lr_mult={change.LR_MULT:g}, '
                  f'aux={change.STATE_AUX_WEIGHT:g}/{change.GEO_AUX_WEIGHT:g}')
    return (f'Support text={switches.SUPPORT_ENABLE}, Query text={switches.QUERY_ENABLE}, '
            f'rho={switches.RHO_ENABLE}; '
            f'matching={"K:K (same-class)" if switches.QUERY_ENABLE else "1:K (shared Query)"}; '
            f'score={scoring}; L={cfg.FEW_SHOT.CLASS_LOSS_LAMBDA:g}*L_global + '
            f'{cfg.FEW_SHOT.Q2S_LOSS_LAMBDA:g}*({loss}){detail}')
