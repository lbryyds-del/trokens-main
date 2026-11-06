"""CPU-only audit of saved SAV trajectory-cluster pilot predictions.

No fitting, checkpoint selection, inference, or modification of model/configs.
Epoch-conditioned knockouts are not independently retrained baselines.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def class_episode_ap(y, s):
    # Precision at the END of each tied-score group, matching sklearn AP.
    ge = s[:, None, :, :] >= s[:, :, None, :]
    precision = (ge * y[:, None, :, :]).sum(2) / ge.sum(2)
    count = y.sum(1)
    return np.divide((precision * y).sum(1), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def episode_metrics(ap, ids, cfg):
    result = []
    for key in ('SEEN_LABELS', 'NOVEL_LABELS'):
        valid = np.isin(ids, cfg['TEST'][key]) & np.isfinite(ap)
        count = valid.sum(1)
        result.append(np.divide(np.where(valid, ap, 0).sum(1), count,
                                out=np.full(len(ap), np.nan), where=count > 0))
    return np.stack(result, axis=1)


def aggregate(ep):
    b, n = np.nanmean(ep, axis=0) * 100
    return np.array([b, n, 2 * b * n / (b + n)])


def metric_dict(values):
    return dict(zip(('base_map', 'novel_map', 'hm_map'), map(float, values)))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_run(folder, cfg):
    df = pd.read_csv(folder / 'multilabel_predictions.csv')
    assert len(df) == 250000, 'Expected 10000 x 5 Queries x 5 classes'
    ids = df.class_id.to_numpy().reshape(-1, 5, 5)
    videos = df.video_name.to_numpy().reshape(-1, 5, 5)
    assert (ids == ids[:, :1, :]).all()
    assert (videos == videos[:, :, :1]).all()
    assert all(len(set(row)) == 5 for row in ids[:, 0])
    y = df.label.to_numpy(np.float32).reshape(-1, 5, 5)
    assert np.isin(y, [0, 1]).all()
    base = df.query_class_base_score.to_numpy(np.float32).reshape(y.shape)
    rho = base + df.query_frame_temporal_logit_delta.to_numpy(np.float32).reshape(y.shape)
    scores = dict(full=df.score.to_numpy(np.float32).reshape(y.shape), rho_path_only=rho)
    aps, eps = {}, {}
    for name, s in scores.items():
        assert np.isfinite(s).all()
        aps[name] = class_episode_ap(y, s)
        eps[name] = episode_metrics(aps[name], ids[:, 0], cfg)
    history = json.loads((folder / 'test_history.jsonl').read_text().splitlines()[-1])
    np.testing.assert_allclose(aggregate(eps['full']),
        [history[k] for k in ('base_map', 'novel_map', 'hm_map')], atol=1e-8, rtol=0)
    # Independent sklearn check, including ties in the quantized inputs.
    for e in range(30):
        for k in range(5):
            if y[e, :, k].any():
                for name, s in scores.items():
                    np.testing.assert_allclose(aps[name][e, k],
                        average_precision_score(y[e, :, k], s[e, :, k]), atol=1e-10)
    return dict(df=df, y=y, ids=ids[:, 0], videos=videos[:, :, 0],
                scores=scores, aps=aps, eps=eps, history=history,
                rho=df.query_class_matchability.to_numpy(np.float32).reshape(y.shape))


def bootstrap(differences, repetitions):
    rng = np.random.default_rng(20261106)
    values = {key: [] for key in differences}
    n = len(next(iter(differences.values()))[0])
    for _ in range(repetitions):
        take = rng.integers(n, size=n)
        for key, (a, b) in differences.items():
            values[key].append(aggregate(a[take]) - aggregate(b[take]))
    return {key: dict(delta_pp=metric_dict(aggregate(a) - aggregate(b)),
                     ci95_pp={m: [float(q[0]), float(q[1])] for m, q in zip(
                         ('base_map', 'novel_map', 'hm_map'),
                         np.quantile(values[key], [.025, .975], axis=0).T)})
            for key, (a, b) in differences.items()}


def subgroup_row(run, mask, name):
    y, full, old = run['y'][mask], run['scores']['full'][mask], run['scores']['rho_path_only'][mask]
    delta = full - old
    row = dict(group=name, n=len(y), positive_n=int(y.sum()), negative_n=int((y == 0).sum()))
    if not len(y):
        return row
    row.update(mean_joint_logit_delta=float(delta.mean()),
               mean_abs_joint_logit_delta=float(np.abs(delta).mean()),
               upward_fraction=float((delta > 1e-6).mean()),
               downward_fraction=float((delta < -1e-6).mean()),
               positive_demoted_below_zero=int(((y == 1) & (old >= 0) & (full < 0)).sum()),
               positive_promoted_above_zero=int(((y == 1) & (old < 0) & (full >= 0)).sum()),
               negative_corrected_below_zero=int(((y == 0) & (old >= 0) & (full < 0)).sum()),
               negative_new_false_positive=int(((y == 0) & (old < 0) & (full >= 0)).sum()))
    for label, s in [('full', full), ('rho_path_only', old)]:
        row[label + '_pred_positive_rate'] = float((s >= 0).mean())
        if len(np.unique(y)) == 2:
            row[label + '_auc'] = float(roc_auc_score(y, s))
    return row


def parameters(root, epoch):
    checkpoint = root / 'checkpoints' / f'checkpoint_epoch_{epoch:05d}.pyth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
    out = {}
    for key, value in state.items():
        if 'entity_change_verifier.' not in key:
            continue
        name = key.split('entity_change_verifier.', 1)[1]
        if name.endswith('_raw'):
            out[name[:-4]] = float(F.softplus(value.float()) + 1e-4)
        else:
            out[name] = float(value)
    return out


def audit(args):
    root, out = args.root.resolve(), args.output.resolve()
    cfg = yaml.safe_load((root / 'effective.yaml').read_text())
    repo = Path(__file__).resolve().parents[1]
    blocks = re.findall(r'item\s*\{(.*?)\}', (repo / 'data/sav/education_first_label.pbtxt').read_text(), re.S)
    names = {int(re.search(r'id:\s*(\d+)', b)[1]) - 1: re.search(r'name:\s*"([^"]+)"', b)[1] for b in blocks}
    folders = {3: root / 'test_e3_reserved_20261106', 10: root}
    runs = {e: load_run(p, cfg) for e, p in folders.items()}
    columns = ['video_name', 'class_id', 'label']
    same_stream = runs[3]['df'][columns].equals(runs[10]['df'][columns])
    assert same_stream
    comparisons = {f'e{e}_joint_vs_off': (r['eps']['full'], r['eps']['rho_path_only']) for e, r in runs.items()}
    comparisons['e3_full_vs_e10_full'] = (runs[3]['eps']['full'], runs[10]['eps']['full'])
    intervals = bootstrap(comparisons, args.bootstrap)
    print('BOOTSTRAP', json.dumps(intervals), flush=True)

    metrics, classes, groups, correlations, examples = [], [], [], [], []
    # Ground truth used strictly for retrospective grouping, not scoring.
    dataset = pd.read_csv(repo / cfg['TEST']['DATA_CSV'])
    all_labels = {row.vid_id: set(int(x) - 1 for x in str(row.label_id).split('_'))
                  for row in dataset.itertuples()}
    for epoch, run in runs.items():
        y, ids = run['y'], run['ids']
        full, old = run['scores']['full'], run['scores']['rho_path_only']
        delta = full - old
        for arm, ep in run['eps'].items():
            metrics.append(dict(epoch=epoch, arm=arm, **metric_dict(aggregate(ep))))
        for c, name in names.items():
            mask = run['df'].class_id.to_numpy() == c
            yy = y.reshape(-1)[mask]
            row = dict(epoch=epoch, class_id=c, class_name=name,
                       split='Base' if c in cfg['TEST']['SEEN_LABELS'] else 'Novel',
                       n=int(mask.sum()), positive_n=int(yy.sum()))
            for arm, s in run['scores'].items():
                vals = s.reshape(-1)[mask]
                row[arm + '_pooled_ap'] = 100 * average_precision_score(yy, vals)
                row[arm + '_pooled_auc'] = roc_auc_score(yy, vals)
                row[arm + '_episode_class_ap'] = 100 * np.nanmean(run['aps'][arm][ids == c])
            row['joint_delta_pooled_ap_pp'] = row['full_pooled_ap'] - row['rho_path_only_pooled_ap']
            row['joint_delta_episode_class_ap_pp'] = row['full_episode_class_ap'] - row['rho_path_only_episode_class_ap']
            classes.append(row)
        known = np.array([[len(all_labels[str(v)]) for v in row] for row in run['videos']])
        # Every saved target label must agree with the dataset's full annotation.
        for e, row in enumerate(run['videos']):
            for q, v in enumerate(row):
                assert [float(c in all_labels[str(v)]) for c in ids[e]] == y[e, q].tolist()
        high = run['rho'] >= .8
        subsets = dict(all=np.ones_like(y, dtype=bool), positive=y == 1, negative=y == 0,
            high_rho=high, high_rho_negative=high & (y == 0), high_rho_positive=high & (y == 1),
            rho_path_false_positive=(y == 0) & (old >= 0),
            rho_path_true_positive=(y == 1) & (old >= 0),
            query_known_multilabel_positive=(y == 1) & (known[:, :, None] > 1),
            query_known_singlelabel_positive=(y == 1) & (known[:, :, None] == 1))
        for name, mask in subsets.items():
            groups.append(dict(epoch=epoch, **subgroup_row(run, mask, name)))
        correlations.append(dict(epoch=epoch,
            joint_delta_vs_mean_rho_pearson=float(np.corrcoef(delta.ravel(), run['rho'].ravel())[0, 1]),
            joint_delta_vs_mean_rho_spearman=float(spearmanr(delta.ravel(), run['rho'].ravel()).statistic),
            joint_delta_vs_original_final_pearson=float(np.corrcoef(delta.ravel(), old.ravel())[0, 1]),
            note='Combined residual only, not individual state/geo evidence. Correlation does not prove complementarity.'))
        for kind, select, sign in [('corrected_negative', (y == 0) & (old >= 0) & (full < 0), -1),
                                   ('harmed_positive', (y == 1) & (old >= 0) & (full < 0), -1),
                                   ('rescued_positive', (y == 1) & (old < 0) & (full >= 0), 1)]:
            ix = np.flatnonzero(select.ravel())
            ix = ix[np.argsort(sign * delta.ravel()[ix])[::-1]][:20]
            for i in ix:
                e, q, k = np.unravel_index(i, y.shape)
                examples.append(dict(epoch=epoch, kind=kind, csv_episode_index=e, query_index=q,
                    video=run['videos'][e, q], class_id=int(ids[e, k]), class_name=names[int(ids[e, k])],
                    label=int(y[e, q, k]), original_rho=float(run['rho'][e, q, k]),
                    rho_path_logit=float(old[e, q, k]), full_logit=float(full[e, q, k]), joint_delta=float(delta[e, q, k])))

    # Complete training curve and exact checkpoint scalars; never treat
    # training valid fractions as evaluation coverage.
    curve = []
    for line in (root / 'stdout.log').read_text().splitlines():
        if 'json_stats: ' in line:
            d = json.loads(line.split('json_stats: ', 1)[1])
            if d.get('_type') == 'train_epoch':
                row = {k: v for k, v in d.items() if k.startswith('q2s_')}
                row.update(epoch=int(d['epoch'].split('/')[0]), learning_rate=d['lr']['lr'])
                curve.append(row)
        m = re.search(r'train_few_shot\.py:\s*861: base mAP: ([\d.]+), novel mAP: ([\d.]+), hm mAP: ([\d.]+)', line)
        if m:
            curve[-1].update(zip(('dev_base_map', 'dev_novel_map', 'dev_hm_map'), map(float, m.groups())))
    assert len(curve) == 10 and all('dev_hm_map' in x for x in curve)
    params = {str(e): parameters(root, e) for e in runs}
    required = [
        dict(item='Original rho-only epoch 10 on identical reserved pool', status='MISSING', why='Primary retrained-baseline comparison; historical full-pool metrics are not comparable.'),
        dict(item='Original rho-only epoch 3 on identical reserved pool', status='MISSING', why='Supplementary best/early-checkpoint comparison, not a replacement for prespecified epoch-10 endpoint.'),
        dict(item='state-only / geo-only inference knockouts', status='MISSING', why='CSV exports only their sum; separate scores require another forward pass.'),
        dict(item='test validity and reference counts for each channel', status='MISSING', why='Training fractions exist; test masks and Support identities are not exported.'),
        dict(item='Fixed historical 2088 difficult pairs', status='NOT_REPLAYABLE_FROM_CURRENT_CSV', why='Those pairs belong to the development pool and require frame-level rho validity/values; current CSV is reserved and aggregated.'),
        dict(item='Query/positive-Support cooccurrence and positive-reference count groups', status='MISSING', why='Query-only labels are available, but Support IDs/labels are not exported.'),
        dict(item='Within-cluster correspondence shuffle', status='NOT_RUN', why='Inference switch exists; shuffle also changes attention-feature association, so interpretation must be bounded.'),
        dict(item='Frame-order control', status='NOT_RUN', why='Requires matched input intervention; output permutation of set matching is not a valid sequence test.'),
        dict(item='Text injection strength / frame-level readout', status='NOT_RUN', why='Dedicated intervention harness needed; do not silently load incompatible checkpoints.'),
        dict(item='Support reliability weighting', status='DEFERRED', why='Not included in the trained model. Fit only on development A, evaluate B; never tune on reserved.'),
        dict(item='Multi-seed stability', status='OUT_OF_SCOPE', why='User requested one seed; episode bootstrap cannot establish seed stability.'),
    ]
    report = dict(protocol=dict(pool=cfg['TEST']['DATA_CSV'], episodes=10000, query_per_episode=5,
        classes_per_episode=5, seed=cfg['RNG_SEED'], ranks=2, checkpoint_epochs=[3,10],
        primary_epoch=10, development_selected_supplementary_epoch=3,
        matching_query_class_label_sequence=same_stream,
        support_and_patch_identity_verification='Not exported; equality is not independently verified.',
        query_records=50000, score_records=250000,
        unique_query_videos=int(runs[3]['df'].video_name.nunique()),
        unique_query_video_class_pairs=int(len(runs[3]['df'][['video_name','class_id']].drop_duplicates())),
        bootstrap_repetitions=args.bootstrap, bootstrap_seed=20261106,
        bootstrap_scope='Paired resampling of episodes, conditional on this pool/checkpoint; repeated videos/recordings and seed variation are not captured.',
        knockout='float32(base_logit + original_temporal_delta); no retraining. May have sub-ULP reconstruction rounding.',
        subgroup_thresholds='Descriptive only: mean rho >= 0.8; predicted positive iff final logit >= 0. No threshold fitting.'),
        metrics=metrics, paired_comparisons=intervals, parameters=params,
        correlations=correlations, pending=required,
        artifacts={f'e{e}':dict(folder=str(p), prediction_sha256=sha(p/'multilabel_predictions.csv'),
            history=runs[e]['history']) for e,p in folders.items()})
    # Only this utility's derived reports may be refreshed, never raw inputs.
    allowed = {'metrics.csv', 'class_metrics.csv', 'subgroups.csv', 'training_curve.csv',
               'examples.csv', 'pending_checks.csv', 'statistics.json', 'REPORT.md'}
    if out.exists() and any(p.name not in allowed for p in out.iterdir()):
        raise FileExistsError('Refusing to write over a directory with unrelated artifacts')
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in [('metrics.csv',metrics), ('class_metrics.csv',classes),
                           ('subgroups.csv',groups), ('training_curve.csv',curve),
                           ('examples.csv',examples), ('pending_checks.csv',required)]:
        pd.DataFrame(data).to_csv(out/filename,index=False)
    (out/'statistics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines = ['# SAV 轨迹簇双通道：已完成统计', '',
        '仅分析已有预测与权重；没有新训练、没有新 GPU 推理、没有拟合任何融合参数。', '',
        '## 正式测试与同权重联合关闭', '',
        '| Epoch | 设置 | Base mAP | Novel mAP | HM |', '|---|---|---:|---:|---:|']
    for row in metrics:
        lines.append(f"| {row['epoch']} | {row['arm']} | {row['base_map']:.4f} | {row['novel_map']:.4f} | {row['hm_map']:.4f} |")
    lines += ['', '`rho_path_only` 是联合训练权重中同时移除新增残差，不是独立训练的原始 baseline。',
        '完整模型指标逐项复现正式 test_history；每轮 25 万条分数，1 万个 episodes。', '',
        '## 配对区间', '', '| 对照 | HM 差值/百分点 | episode bootstrap 95%区间 |', '|---|---:|---|']
    for name, value in intervals.items():
        lo, hi = value['ci95_pp']['hm_map']
        lines.append(f"| {name} | {value['delta_pp']['hm_map']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '区间仅反映固定模型/数据池的 episode 抽样变化，不包含重复视频相关性、多 seed 或选择不确定性。',
        '第3轮是开发集选择的补充结果；预设主比较仍然是第10轮。历史测试池并非未被使用过的全新留出池。', '',
        '## 第3轮逐类贡献', '',
        '以下是跨 episode 汇总的 pooled AP，不等同于主指标的逐 episode Base/Novel mAP。', '',
        '| 类别 | Base/Novel | 两路关闭 AP | 完整 AP | 变化/百分点 |', '|---|---|---:|---:|---:|']
    for row in classes:
        if row['epoch'] == 3:
            lines.append(f"| {row['class_name']} | {row['split']} | {row['rho_path_only_pooled_ap']:.2f} | {row['full_pooled_ap']:.2f} | {row['joint_delta_pooled_ap_pp']:+.2f} |")
    lines += ['', '完整逐类结果和逐 episode 类别 AP 见 `class_metrics.csv`。逐类结论是描述性的，未进行多重比较检验。', '',
        '## 错误与共现', '',
        '`subgroups.csv` 统计高ρ正/负、零阈值误报修正/新增、真阳性损伤/恢复、Query已知多标签子集。',
        'Query多标签不等同于Query与正Support共享共现动作；后者缺少Support信息，不能判断。',
        '`examples.csv` 每类选最多20个logit改变量较大的案例，是事后检查线索，不是已核验视觉机制。',
        '`correlations` 只测联合残差与原ρ的相关性，不能分离两路，也不能由低相关性推出有效。', '',
        '### 第3轮固定阈值描述性统计', '',
        '统一用 logit >= 0（sigmoid概率>=0.5），没有在测试集上调阈值。记录可重复来自同一视频，不是独立视频数。', '',
        '| 项目 | 原ρ路径 | 加入两路证据 |', '|---|---:|---:|']
    g3 = {r['group']:r for r in groups if r['epoch']==3}
    for label, group in [('负样本误报率','negative'),('正样本召回率','positive'),
                          ('高ρ负样本误报率','high_rho_negative'),('高ρ正样本召回率','high_rho_positive')]:
        r = g3[group]
        lines.append(f"| {label} | {100*r['rho_path_only_pred_positive_rate']:.2f}% | {100*r['full_pred_positive_rate']:.2f}% |")
    r = g3['all']
    lines += ['', f"修正误报 {r['negative_corrected_below_zero']} 条；新增误报 {r['negative_new_false_positive']} 条。"
        f"损伤真阳性 {r['positive_demoted_below_zero']} 条；恢复真阳性 {r['positive_promoted_above_zero']} 条。",
        '结论：降低误报，但有轻微召回代价，不能宣称不伤害真实多动作。全部89233条正记录都来自已知多标签Query，没有单标签正Query对照。',
        '高ρ组的AUC为跨类别汇总、未做ρ匹配；不能当成历史2088个相近对的检验。高ρ误报大部分仍未被否决。', '',
        '## 参数与训练曲线', '', '精确第3/10轮标量见 `statistics.json -> parameters`；完整十轮均值见 `training_curve.csv`。',
        '有效比例与辅助loss来自训练日志，不是测试覆盖率；lambda大小不能直接比较分支贡献。', '',
        '| Epoch | λ_state | λ_geo | 温度_state | 温度_geo |', '|---|---:|---:|---:|---:|']
    for epoch, p in params.items():
        lines.append(f"| {epoch} | {p['lambda_state']:.6f} | {p['lambda_geo']:.6f} | {p['state_temperature']:.6f} | {p['geo_temperature']:.6f} |")
    lines += ['', '第3→10轮：完整模型HM下降0.4141；同权重原ρ路径HM下降0.6137，附加分支贡献从0.5139增至0.7135。',
        '因此，后期下降并非表现为新增两路的推理增益消失；但主路径也参与了联合训练，不能据此判定训练因果或断言新分支不导致过拟合。',
        '温度后期显著变小，说明证据聚合更尖锐；是否导致过拟合尚未检验。', '',
        '## 尚缺的对照', '', '| 项目 | 状态 | 原因 |', '|---|---|---|']
    for row in required:
        lines.append(f"| {row['item']} | {row['status']} | {row['why']} |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('SAVED',out,flush=True)
    print('METRICS',json.dumps(metrics),flush=True)
    print('PARAMETERS',json.dumps(params),flush=True)
    print('FOCUS_CLASSES',json.dumps([r for r in classes if r['epoch']==3 and r['class_name'] in ('read','take_notes','flip_books','talk_with_others')]),flush=True)
    print('E3_SUBGROUPS',json.dumps([r for r in groups if r['epoch']==3]),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('output/sav/entity_change_s1'))
    parser.add_argument('--output',type=Path,default=Path('output/sav/entity_change_s1/statistics_20261106'))
    parser.add_argument('--bootstrap',type=int,default=3000)
    audit(parser.parse_args())
