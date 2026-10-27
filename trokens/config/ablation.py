"""Single source of truth for SAV role, judgment, matching and loss switches."""
import math


def resolve_ablation_cfg(cfg):
    """Resolve after YAML + CLI merging, before logging/model/optimizer creation.

    The operation is idempotent. Legacy profiles with no DECOUPLING switches
    retain their exact behavior. Never silently disable a requested judgment.
    """
    fs = cfg.FEW_SHOT
    switches = getattr(fs, 'DECOUPLING', None)
    names = ('QUERY_ENABLE', 'SUPPORT_ENABLE', 'M_ENABLE', 'RHO_ENABLE')
    values = [getattr(switches, name, None) for name in names]
    if all(value is None for value in values):
        return cfg
    if not all(type(value) is bool for value in values):
        raise ValueError('DECOUPLING requires all four switches to be True or False.')
    query, support, quality, uniqueness = values
    if not query and (quality or uniqueness):
        raise ValueError('M_ENABLE/RHO_ENABLE require QUERY_ENABLE=True (K:K matching). '
                         'With shared Query (1:K), set both judgments to False.')
    if fs.LABEL_PRE_TEMPORAL.ENABLE or fs.POT_ROUTE.ENABLE or fs.SUPPORT_TEXT_FUSION.ENABLE:
        raise ValueError('DECOUPLING cannot be combined with legacy early routing or Support/text fusion.')
    matching = fs.MATCHING
    for name in ('LOGIT_SCALE', 'LOGIT_BIAS', 'BASE_LOSS_WEIGHT', 'FINAL_LOSS_WEIGHT',
                 'M_AUX_WEIGHT', 'RHO_AUX_WEIGHT', 'TRANSPORT_STRENGTH'):
        value = float(getattr(matching, name))
        if not math.isfinite(value) or (name != 'LOGIT_BIAS' and value < 0):
            raise ValueError(f'MATCHING.{name} must be finite and nonnegative (bias may be negative).')
    verified = quality or uniqueness
    if verified and (matching.BASE_LOSS_WEIGHT <= 0 or matching.FINAL_LOSS_WEIGHT <= 0
                     or matching.TRANSPORT_STRENGTH <= 0):
        raise ValueError('Enabled m/rho require positive base/final weights and transport strength.')
    if (quality and matching.M_AUX_WEIGHT <= 0) or (uniqueness and matching.RHO_AUX_WEIGHT <= 0):
        raise ValueError('Enabled m/rho require their corresponding positive auxiliary loss weight.')
    frozen = cfg.is_frozen()
    if frozen:
        cfg.defrost()
    try:
        fs.TEXT_CONCAT_ST.ENABLE = True
        fs.TEXT_CONCAT_ST.SHARED_QUERY_ENABLE = not query
        fs.TEXT_CONCAT_ST.SHARED_SUPPORT_ENABLE = not support
        fs.POT_ROUTE.QUERY_PARTIAL_ENABLE = True  # episode interface, not role encoding
        fs.POT_ROUTE.QUERY_PARTIAL_LOGIT_ALPHA = matching.LOGIT_SCALE
        fs.POT_ROUTE.QUERY_PARTIAL_LOGIT_BIAS = matching.LOGIT_BIAS
        match = fs.QUERY_CLASS_MATCHABILITY
        match.ENABLE = True  # also preserves unknown-label masking
        match.QUALITY_ENABLE, match.UNIQUENESS_ENABLE = quality, uniqueness
        match.BASE_ONLY_LOSS_ENABLE = not verified
        match.DUAL_LOGIT_LOSS_ENABLE = verified
        match.BASE_LOGIT_LOSS_WEIGHT = matching.BASE_LOSS_WEIGHT if verified else 1.0
        match.VERIFIED_LOGIT_LOSS_WEIGHT = matching.FINAL_LOSS_WEIGHT if verified else 0.0
        match.ABSOLUTE_MASS_TRANSPORT_STRENGTH = matching.TRANSPORT_STRENGTH if verified else 0.0
        match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT = matching.M_AUX_WEIGHT if quality else 0.0
        match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT = matching.RHO_AUX_WEIGHT if uniqueness else 0.0
    finally:
        if frozen:
            cfg.freeze()
    return cfg


def ablation_summary(cfg):
    """Concise effective settings for the launcher, including the actual loss."""
    switches = cfg.FEW_SHOT.DECOUPLING
    if switches.QUERY_ENABLE is None:
        return 'Legacy profile: using its explicit role/judgment/loss settings.'
    match = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    loss = 'L_base'
    if switches.M_ENABLE or switches.RHO_ENABLE:
        loss = f'{match.BASE_LOGIT_LOSS_WEIGHT:g}*L_base + {match.VERIFIED_LOGIT_LOSS_WEIGHT:g}*L_final'
        if switches.M_ENABLE:
            loss += f' + {match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT:g}*L_m'
        if switches.RHO_ENABLE:
            loss += f' + {match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT:g}*L_rho'
    return (f'Support text={switches.SUPPORT_ENABLE}, Query text={switches.QUERY_ENABLE}, '
            f'm={switches.M_ENABLE}, rho={switches.RHO_ENABLE}; '
            f'matching={"K:K (same-class)" if switches.QUERY_ENABLE else "1:K (shared Query)"}; '
            f'L={cfg.FEW_SHOT.CLASS_LOSS_LAMBDA:g}*L_global + '
            f'{cfg.FEW_SHOT.Q2S_LOSS_LAMBDA:g}*({loss})')
