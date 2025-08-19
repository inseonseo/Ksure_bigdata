import os
import sys
import types
import contextlib
from importlib.machinery import SourceFileLoader


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
        expander=lambda *a, **k: contextlib.nullcontext(),
        form=lambda *a, **k: contextlib.nullcontext(),
        button=lambda *a, **k: False,
    )
    mod = types.ModuleType('streamlit')
    mod.__dict__.update(st.__dict__)
    sys.modules['streamlit'] = mod


def main():
    # Ensure project root in path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(repo_root)

    # Only stub streamlit (we want real transformers/torch)
    if 'streamlit' not in sys.modules:
        stub_streamlit()

    # Import systems
    from KoSimCSE.new.evaluation_system import InsuranceEvaluationSystem
    improved_path = os.path.join(repo_root, 'KoSimCSE', 'new', 'improved_insurance_system.py')
    Improved = SourceFileLoader('improved', improved_path).load_module()

    # Data path
    data_path = os.path.join(repo_root, 'KoSimCSE', 'new', 'data', 'design.csv')

    # Build pipeline
    eval_sys = InsuranceEvaluationSystem(data_path, preserve_labels=True, min_support_for_test=2)
    eval_sys.load_and_prepare_data()
    eval_sys.create_train_valid_test_split()
    eval_sys.prepare_features_for_modeling()

    sim = Improved.ImprovedInsuranceSystem()  # uses real KoSimCSE embeddings

    results = eval_sys.evaluate_similarity_system(sim, sample_size=100)

    if results is None:
        print('No results produced.')
        return

    # Summary
    print('\n=== Real-model summary ===')
    for key in ['judgment', 'reason']:
        r = results[key]
        print(f"[{key}] accuracy={r['accuracy']:.3f}, balanced_acc={r['balanced_accuracy']:.3f}, f1_macro={r['f1_macro']:.3f}, n={r['sample_size']}")


if __name__ == '__main__':
    main()


