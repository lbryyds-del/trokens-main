"""Validation shared by launcher/model; the disabled path is a no-op."""
import math


def validate_entity_change_cfg(cfg):
    c = cfg.FEW_SHOT.ENTITY_CHANGE
    if not c.ENABLE:
        return
    f = cfg.FEW_SHOT
    d = f.DECOUPLING
    if not (d.QUERY_ENABLE is True and d.SUPPORT_ENABLE is True
            and d.RHO_ENABLE is True):
        raise ValueError('ENTITY_CHANGE requires dual decoupling and rho ON.')
    if not c.DETACH_INPUTS or not f.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.DETACH_INPUTS:
        raise ValueError('ENTITY_CHANGE requires detached evidence inputs.')
    if not (c.STATE_ENABLE or c.GEO_ENABLE):
        raise ValueError('ENTITY_CHANGE needs at least one enabled channel.')
    if c.GRANULARITY not in ('cluster', 'frame', 'point'):
        raise ValueError('Invalid ENTITY_CHANGE.GRANULARITY.')
    if c.STRICT_COMMON_REPLAY and (c.GRANULARITY != 'frame' or c.GEO_ENABLE
                                  or c.CORRESPONDENCE_SHUFFLE):
        raise ValueError('STRICT_COMMON_REPLAY requires frame, state-only, no shuffle.')
    if c.STRICT_COMMON_REPLAY and cfg.TRAIN.ENABLE:
        raise ValueError('STRICT_COMMON_REPLAY is inference-only; set TRAIN.ENABLE=False.')
    if c.CORRESPONDENCE_SHUFFLE and cfg.TRAIN.ENABLE:
        raise ValueError('CORRESPONDENCE_SHUFFLE is an inference-only intervention.')
    for field in ('MAX_ENTITIES', 'MIN_POINTS', 'REF_TOPK', 'GEO_NUM_BINS', 'GEO_MIN_STEPS'):
        if type(getattr(c, field)) is not int or getattr(c, field) < 1:
            raise ValueError('ENTITY_CHANGE.' + field + ' must be a positive integer.')
    for field in ('NORM_EPS', 'GEO_SPEED_SCALE', 'LR_MULT', 'STATE_TEMPERATURE_INIT',
                  'GEO_TEMPERATURE_INIT', 'STATE_SCALE_INIT', 'GEO_SCALE_INIT'):
        if not math.isfinite(getattr(c, field)) or getattr(c, field) <= 0:
            raise ValueError('ENTITY_CHANGE.' + field + ' must be finite and positive.')
    for field in ('STATE_AUX_WEIGHT', 'GEO_AUX_WEIGHT', 'STATE_TEST_MULT', 'GEO_TEST_MULT'):
        if not math.isfinite(getattr(c, field)) or getattr(c, field) < 0:
            raise ValueError('ENTITY_CHANGE.' + field + ' must be finite and nonnegative.')
    for field in ('LAMBDA_STATE_INIT', 'LAMBDA_GEO_INIT'):
        if not math.isfinite(getattr(c, field)):
            raise ValueError('ENTITY_CHANGE.' + field + ' must be finite.')
    if not 0 <= c.GEO_AMPLITUDE_WEIGHT <= 1:
        raise ValueError('GEO_AMPLITUDE_WEIGHT must be in [0,1].')
    if cfg.DATA.NUM_FRAMES < 2:
        raise ValueError('ENTITY_CHANGE requires at least two frames.')
    if cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE or cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE:
        raise ValueError('ENTITY_CHANGE keeps HoD/CTD OFF.')


def entity_change_signature(c):
    """Training semantics; runtime inference interventions are excluded."""
    return [9, 1, int(c.STATE_ENABLE), int(c.GEO_ENABLE), int(c.DETACH_INPUTS),
            {'cluster': 0, 'frame': 1, 'point': 2}[c.GRANULARITY],
            int(c.STRICT_COMMON_REPLAY), c.MAX_ENTITIES, c.MIN_POINTS, c.REF_TOPK,
            c.GEO_NUM_BINS, c.GEO_MIN_STEPS]
