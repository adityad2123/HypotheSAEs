import numpy as np
from hypothesaes import get_local_embeddings, train_sae
from hypothesaes import NeuronInterpreter, InterpretConfig, LLMConfig, SamplingConfig
from hypothesaes.annotate import annotate

from .sentences import BLUE_SENTENCES, RED_SENTENCES, ALL_TEST_SENTENCES

LOCAL_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_LLM = "mistral:7b"

def test_ollama_interpretation():
    texts = BLUE_SENTENCES + RED_SENTENCES
    activations = np.stack([
        np.concatenate([np.ones(len(BLUE_SENTENCES)), np.zeros(len(RED_SENTENCES))]),
        np.concatenate([np.zeros(len(BLUE_SENTENCES)), np.ones(len(RED_SENTENCES))])
    ], axis=1)

    interpreter = NeuronInterpreter(interpreter_model=OLLAMA_LLM)
    config = InterpretConfig(
        sampling=SamplingConfig(n_examples=20),
        llm=LLMConfig(max_interpretation_tokens=50, temperature=0.7, tokenizer_kwargs={"enable_thinking": False}),
    )
    results = interpreter.interpret_neurons(texts, activations, neuron_indices=[0, 1], config=config)
    print(f"Interpretations:\n - Neuron 0: {results[0][0]}\n - Neuron 1: {results[1][0]}")
    for idx in [0, 1]:
        assert idx in results
        assert isinstance(results[idx][0], str)
        assert len(results[idx][0]) > 0

def test_ollama_annotation():
    blue_concept = "contains words associated with the color blue"
    red_concept = "contains words associated with the color red"
    positive_tasks = [(text, blue_concept) for text in BLUE_SENTENCES] + [(text, red_concept) for text in RED_SENTENCES]
    negative_tasks = [(text, blue_concept) for text in RED_SENTENCES] + [(text, red_concept) for text in BLUE_SENTENCES]
    results = annotate(positive_tasks + negative_tasks, model=OLLAMA_LLM, show_progress=True)
    
    print(results) # I added this print statement so I could just see what's going on under the hood.
    
    # Calculate precision and recall for each concept
    for concept, concept_dict in results.items():
        if concept == blue_concept:
            tp = sum(1 for text in BLUE_SENTENCES if concept_dict.get(text, 0) == 1)
            fp = sum(1 for text in RED_SENTENCES if concept_dict.get(text, 0) == 1)
            fn = sum(1 for text in BLUE_SENTENCES if concept_dict.get(text, 0) == 0)
        else:  # red_concept
            tp = sum(1 for text in RED_SENTENCES if concept_dict.get(text, 0) == 1)
            fp = sum(1 for text in BLUE_SENTENCES if concept_dict.get(text, 0) == 1)
            fn = sum(1 for text in RED_SENTENCES if concept_dict.get(text, 0) == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Concept: '{concept}' - precision: {precision:.2f}, recall: {recall:.2f}")

    for concept_dict in results.values():
        for val in concept_dict.values():
            assert val in (0, 1)