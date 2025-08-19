import sys
import types
import contextlib
from importlib.machinery import SourceFileLoader


def install_streamlit_stub():
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
        tabs=lambda *a, **k: [types.SimpleNamespace()],
        expander=lambda *a, **k: contextlib.nullcontext(),
        form=lambda *a, **k: contextlib.nullcontext(),
        text_area=lambda *a, **k: None,
        selectbox=lambda *a, **k: None,
        number_input=lambda *a, **k: None,
        slider=lambda *a, **k: None,
        checkbox=lambda *a, **k: None,
        button=lambda *a, **k: False,
        metric=lambda *a, **k: None,
        dataframe=lambda *a, **k: None,
        progress=lambda *a, **k: None,
        info=lambda *a, **k: None,
    )
    mod = types.ModuleType('streamlit')
    mod.__dict__.update(st.__dict__)
    sys.modules['streamlit'] = mod


def install_torch_stub():
    class _NoGrad:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    torch = types.SimpleNamespace(no_grad=lambda: _NoGrad(), device=lambda *a, **k: None)
    sys.modules['torch'] = torch


def install_transformers_stub():
    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            # Return a dummy object with attributes used in code, but we won't call it
            return object()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *a, **k):
            # Return a dummy tokenizer object
            tok = types.SimpleNamespace(
                __call__=lambda *a, **k: {}
            )
            return tok

    transformers = types.SimpleNamespace(AutoModel=_AutoModel, AutoTokenizer=_AutoTokenizer)
    sys.modules['transformers'] = transformers


def main():
    # Ensure project root is on sys.path
    # Assumes this script is executed from project root
    # If not, append parent of 'KoSimCSE' manually
    
    # Install stubs to bypass UI/ML heavy deps
    if 'streamlit' not in sys.modules:
        install_streamlit_stub()
    if 'torch' not in sys.modules:
        install_torch_stub()
    if 'transformers' not in sys.modules:
        install_transformers_stub()

    # Import evaluation system
    from KoSimCSE.new.evaluation_system import InsuranceEvaluationSystem

    # Import improved_insurance_system via loader to avoid package import issues
    improved_path = 'KoSimCSE/new/improved_insurance_system.py'
    Improved = SourceFileLoader('improved', improved_path).load_module()

    # Build evaluation pipeline
    eval_sys = InsuranceEvaluationSystem('KoSimCSE/new/data/design.csv', preserve_labels=True, min_support_for_test=2)
    eval_sys.load_and_prepare_data()
    eval_sys.create_train_valid_test_split()
    eval_sys.prepare_features_for_modeling()

    sim = Improved.ImprovedInsuranceSystem()

    # Disable heavy embeddings to run offline/fast
    sim.get_text_embeddings = lambda texts, batch_size=4: None

    results = eval_sys.evaluate_similarity_system(sim, sample_size=200)

    if results is None:
        print('No results produced.')
        return

    # Print concise summary
    print('\n=== Summary ===')
    for key in ['judgment', 'reason']:
        r = results[key]
        print(f"[{key}] accuracy={r['accuracy']:.3f}, balanced_acc={r['balanced_accuracy']:.3f}, f1_macro={r['f1_macro']:.3f}, n={r['sample_size']}")


if __name__ == '__main__':
    main()




