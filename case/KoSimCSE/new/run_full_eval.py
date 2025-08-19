import os
import sys
import types
import contextlib
from importlib.machinery import SourceFileLoader
from datetime import datetime
import json
import pandas as pd
import numpy as np


def stub_streamlit():
    st = types.SimpleNamespace(
        cache_resource=lambda f: f,
        cache_data=lambda f: f,
        spinner=lambda *a, **k: contextlib.nullcontext(),
        markdown=lambda *a, **k: None,
        success=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        write=lambda *a, **k: None,
        subheader=lambda *a, **k: None,
        columns=lambda *a, **k: [types.SimpleNamespace()],
        plotly_chart=lambda *a, **k: None,
        set_page_config=lambda *a, **k: None,
    )
    mod = types.ModuleType('streamlit')
    mod.__dict__.update(st.__dict__)
    sys.modules['streamlit'] = mod


def weighted_vote(labels, weights):
    scores = {}
    for lbl, w in zip(labels, weights):
        scores[lbl] = scores.get(lbl, 0.0) + float(w)
    if not scores:
        return None, {}
    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, scores


def build_query_case(row):
    return {
        '수입국': row.get('수입국'),
        '보험종목': row.get('보험종목'),
        '사고유형명': row.get('사고유형명'),
        '원화사고금액': row.get('원화사고금액'),
        '사고설명': row.get('사고설명'),
        '부보율': row.get('부보율'),
        '상품분류명': row.get('상품분류명'),
        '상품분류그룹명': row.get('상품분류그룹명'),
        '결제방법': row.get('결제방법'),
        '결제조건': row.get('결제조건'),
        '향후결제전망': row.get('향후결제전망'),
    }


def main():
    # Project root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(repo_root)

    # Stub streamlit (use real transformers/torch installed in env)
    if 'streamlit' not in sys.modules:
        stub_streamlit()

    # Imports
    from KoSimCSE.new.evaluation_system import InsuranceEvaluationSystem
    improved_path = os.path.join(repo_root, 'KoSimCSE', 'new', 'improved_insurance_system.py')
    Improved = SourceFileLoader('improved', improved_path).load_module()

    data_path = os.path.join(repo_root, 'KoSimCSE', 'new', 'data', 'design.csv')

    # Build pipeline
    eval_sys = InsuranceEvaluationSystem(data_path, preserve_labels=True, min_support_for_test=2)
    eval_sys.load_and_prepare_data()
    eval_sys.create_train_valid_test_split()
    eval_sys.prepare_features_for_modeling()

    sim = Improved.ImprovedInsuranceSystem()  # real KoSimCSE embeddings

    # Search pool limits
    max_candidates = 1000  # per-query cap for performance

    test_df = eval_sys.test_df.copy().reset_index(drop=True)
    train_df = eval_sys.train_df.copy().reset_index(drop=True)
    train_exemption_df = train_df[train_df['판정구분'] == '면책'].reset_index(drop=True)

    # Outputs
    rows = []

    for i, row in test_df.iterrows():
        q = build_query_case(row)

        # Binary decision using full train
        pool = train_df
        if len(pool) > max_candidates:
            pool = pool.sample(n=max_candidates, random_state=42)

        sims = sim.calculate_similarity_scores(q, pool)
        if not sims:
            pred_decision = None
            decision_conf = 0.0
            decision_avg_sim = 0.0
        else:
            top_k = sims[:5]
            scores = [t[0] for t in top_k]
            labels = [t[3]['판정구분'] for t in top_k]
            pred_decision, score_map = weighted_vote(labels, scores)
            # Override: if top1 is exemption and score high
            top1_score, _, _, top1_case = top_k[0]
            if top1_case['판정구분'] == '면책' and top1_score >= 0.65:
                pred_decision = '면책'
            sum_w = sum(scores) if scores else 1.0
            decision_conf = (score_map.get(pred_decision, 0.0) / sum_w) if sum_w else 0.0
            decision_avg_sim = float(np.mean(scores)) if scores else 0.0

        # Reason prediction (two ways)
        pred_reason_true_ex = None  # reason using true-exemption pool (only computed when true is exemption)
        pred_reason_pred_ex = None  # reason using predicted-exemption pool (computed when pred is exemption)

        # A) For true exemption rows, predict reason using train_exemption_df
        if row.get('판정구분') == '면책' and len(train_exemption_df) > 0:
            pool_ex = train_exemption_df
            if len(pool_ex) > max_candidates:
                pool_ex = pool_ex.sample(n=max_candidates, random_state=42)
            sims_ex = sim.calculate_similarity_scores(q, pool_ex)
            if sims_ex:
                top_k_ex = sims_ex[:5]
                scores_ex = [t[0] for t in top_k_ex]
                labels_ex = [t[3]['판정사유'] for t in top_k_ex]
                pred_reason_true_ex, _ = weighted_vote(labels_ex, scores_ex)

        # B) For predicted exemption rows, also output reason prediction
        if pred_decision == '면책' and len(train_exemption_df) > 0:
            pool_ex = train_exemption_df
            if len(pool_ex) > max_candidates:
                pool_ex = pool_ex.sample(n=max_candidates, random_state=42)
            sims_ex = sim.calculate_similarity_scores(q, pool_ex)
            if sims_ex:
                top_k_ex = sims_ex[:5]
                scores_ex = [t[0] for t in top_k_ex]
                labels_ex = [t[3]['판정사유'] for t in top_k_ex]
                pred_reason_pred_ex, _ = weighted_vote(labels_ex, scores_ex)

        rows.append({
            'row_id': int(i),
            'true_decision': row.get('판정구분'),
            'pred_decision': pred_decision,
            'true_reason': row.get('판정사유'),
            'pred_reason_true_ex_only': pred_reason_true_ex,
            'pred_reason_on_predicted_ex': pred_reason_pred_ex,
            'decision_confidence': float(decision_conf),
            'decision_avg_similarity': float(decision_avg_sim),
        })

    out_df = pd.DataFrame(rows)

    # Metrics
    def simple_acc(a, b):
        ok = (a == b) & a.notna() & b.notna()
        return float(ok.mean()) if len(ok) else 0.0

    # Decision accuracy (overall)
    acc_decision = simple_acc(out_df['true_decision'], out_df['pred_decision'])

    # Reason accuracy on true exemption subset
    mask_true_ex = out_df['true_decision'] == '면책'
    acc_reason_true_ex = simple_acc(
        out_df.loc[mask_true_ex, 'true_reason'],
        out_df.loc[mask_true_ex, 'pred_reason_true_ex_only']
    ) if mask_true_ex.any() else 0.0

    # Reason accuracy on predicted exemption subset (evaluate only where true is also exemption)
    mask_pred_ex = out_df['pred_decision'] == '면책'
    mask_both_ex = mask_true_ex & mask_pred_ex
    acc_reason_on_pred = simple_acc(
        out_df.loc[mask_both_ex, 'true_reason'],
        out_df.loc[mask_both_ex, 'pred_reason_on_predicted_ex']
    ) if mask_both_ex.any() else 0.0

    summary = {
        'num_test_rows': int(len(out_df)),
        'decision_accuracy': acc_decision,
        'num_true_exemptions': int(mask_true_ex.sum()),
        'reason_accuracy_true_ex_only': acc_reason_true_ex,
        'num_pred_exemptions': int(mask_pred_ex.sum()),
        'reason_accuracy_on_predicted_ex_and_true_ex': acc_reason_on_pred,
    }

    # Save
    out_dir = os.path.join(repo_root, 'KoSimCSE', 'new', 'output')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(out_dir, f'full_eval_rows_{ts}.csv')
    out_json = os.path.join(out_dir, f'full_eval_summary_{ts}.json')

    out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print concise summary
    print('Saved rows to:', out_csv)
    print('Saved summary to:', out_json)
    print('Summary:', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()


