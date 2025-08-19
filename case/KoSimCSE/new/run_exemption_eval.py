import os
import sys
import types
import contextlib
from importlib.machinery import SourceFileLoader
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
    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, scores


def main(sample_size=300, top1_exemption_threshold=0.65):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(repo_root)

    if 'streamlit' not in sys.modules:
        stub_streamlit()

    # Import core
    from KoSimCSE.new.evaluation_system import InsuranceEvaluationSystem
    improved_path = os.path.join(repo_root, 'KoSimCSE', 'new', 'improved_insurance_system.py')
    Improved = SourceFileLoader('improved', improved_path).load_module()

    data_path = os.path.join(repo_root, 'KoSimCSE', 'new', 'data', 'design.csv')

    # Build pipeline (labels preserved)
    eval_sys = InsuranceEvaluationSystem(data_path, preserve_labels=True, min_support_for_test=2)
    eval_sys.load_and_prepare_data()
    eval_sys.create_train_valid_test_split()
    eval_sys.prepare_features_for_modeling()

    sim = Improved.ImprovedInsuranceSystem()  # real embeddings

    # Exemption-only sample from test
    test_ex = eval_sys.test_df[eval_sys.test_df['판정구분'] == '면책']
    if len(test_ex) == 0:
        print('면책 케이스가 테스트셋에 없습니다.')
        return

    test_sample = test_ex.sample(n=min(sample_size, len(test_ex)), random_state=42)

    predictions = []
    avg_sims = []
    confidences = []

    # Search pool
    search_pool = eval_sys.train_df.copy()

    for _, row in test_sample.iterrows():
        case_data = {
            '수입국': row.get('수입국'),
            '보험종목': row.get('보험종목'),
            '사고유형명': row.get('사고유형명'),
            '원화사고금액': row.get('원화사고금액'),
            '사고설명': row.get('사고설명'),
        }

        sims = sim.calculate_similarity_scores(case_data, search_pool.head(1000))
        if not sims:
            continue

        top_k = sims[:5]
        top_scores = [t[0] for t in top_k]
        labels = [t[3]['판정구분'] for t in top_k]
        pred, score_map = weighted_vote(labels, top_scores)

        top1_score, _, _, top1_case = top_k[0]
        if top1_case['판정구분'] == '면책' and top1_score >= top1_exemption_threshold:
            pred = '면책'

        sum_w = sum(top_scores) if top_scores else 1.0
        conf = (score_map.get(pred, 0.0) / sum_w) if sum_w else 0.0
        avg_sim = float(np.mean(top_scores)) if top_scores else 0.0

        predictions.append(pred)
        confidences.append(conf)
        avg_sims.append(avg_sim)

    if not predictions:
        print('예측 결과가 없습니다.')
        return

    tp = sum(1 for p in predictions if p == '면책')
    total = len(predictions)
    recall = tp / total

    from collections import Counter
    pred_dist = dict(Counter(predictions))

    print('\n🛡️ 면책 전용 평가 결과:')
    print(f'   - Recall(면책): {recall:.3f} ({tp}/{total})')
    print(f'   - 예측 분포: {pred_dist}')
    print(f'   - 평균 유사도: {float(np.mean(avg_sims)):.3f} ± {float(np.std(avg_sims)):.3f}')
    print(f'   - 평균 신뢰도: {float(np.mean(confidences)):.3f} ± {float(np.std(confidences)):.3f}')


if __name__ == '__main__':
    main()




