"""
End-to-end test of the semiotic pipeline using the Python base model.

This is a standalone Python test that simulates what the Julia pipeline does,
using numpy for the semiotic computations. It validates that the full data flow
works: text → base model embeddings → (semiotic processing placeholder) → conditioned generation.

Once PythonCall.jl is available, the Julia pipeline replaces the numpy placeholder.
"""

import numpy as np
from base_model import BaseModel


def softmax(x, axis=0):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def test_sign_formation(embeddings: np.ndarray, tokens: list[str], num_slots: int = 8):
    """Simulate the sign-formation module in numpy."""
    d_model, T = embeddings.shape
    print(f"\n--- Sign Formation ---")
    print(f"Input: {T} tokens, d_model={d_model}, forming {num_slots} signs")

    # 1. Learned segmentation (random weights as placeholder)
    seg_logits = np.random.randn(num_slots, d_model) @ embeddings  # num_slots × T
    segment_weights = softmax(seg_logits, axis=0)  # num_slots × T
    print(f"Segment weights shape: {segment_weights.shape}")

    # Show which tokens belong primarily to which slot
    assignments = np.argmax(segment_weights, axis=0)
    for k in range(num_slots):
        slot_tokens = [tokens[i] for i in range(T) if assignments[i] == k]
        if slot_tokens:
            print(f"  Slot {k}: {slot_tokens}")

    # 2. Slot attention (simplified: weighted mean of token embeddings per slot)
    sign_content = np.zeros((d_model, num_slots), dtype=np.float32)
    for k in range(num_slots):
        weights = segment_weights[k, :]  # T
        sign_content[:, k] = embeddings @ weights / (weights.sum() + 1e-8)
    print(f"Sign content shape: {sign_content.shape}")

    # 3. Sign-type classification (random classifier as placeholder)
    type_logits = np.random.randn(10, d_model) @ sign_content  # 10 × num_slots
    sign_types = softmax(type_logits, axis=0)
    print(f"Sign types shape: {sign_types.shape}")

    sign_class_names = [
        "rhematic iconic qualisign",
        "rhematic iconic sinsign",
        "rhematic indexical sinsign",
        "dicent indexical sinsign",
        "rhematic iconic legisign",
        "rhematic indexical legisign",
        "dicent indexical legisign",
        "rhematic symbolic legisign",
        "dicent symbolic legisign",
        "argument symbolic legisign",
    ]
    for k in range(num_slots):
        top = np.argmax(sign_types[:, k])
        conf = sign_types[top, k]
        slot_tokens = [tokens[i] for i in range(T) if assignments[i] == k]
        if slot_tokens:
            print(f"  Slot {k} ({slot_tokens}): {sign_class_names[top]} ({conf:.2f})")

    # 4. Truth values (initialized neutral)
    truth = np.full((3, num_slots), 0.5, dtype=np.float32)

    return sign_content, sign_types, truth


def test_inference_graph(sign_content: np.ndarray, sign_types: np.ndarray):
    """Simulate inference graph operations."""
    d_model, K = sign_content.shape
    print(f"\n--- Inference Graph ---")
    print(f"Adding {K} nodes")

    # Type-conditional edge creation
    SIGN_CLASSES = [
        (1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2), (3, 1, 1),
        (3, 2, 1), (3, 2, 2), (3, 3, 1), (3, 3, 2), (3, 3, 3),
    ]

    edges = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            src_class = np.argmax(sign_types[:, i])
            tgt_class = np.argmax(sign_types[:, j])
            src_triple = SIGN_CLASSES[src_class]
            tgt_triple = SIGN_CLASSES[tgt_class]

            # Commitment: source interpretant >= 2, target <= source
            if src_triple[2] >= 2 and tgt_triple[2] <= src_triple[2]:
                weight = float(np.dot(sign_content[:, i], sign_content[:, j]))
                weight = 1.0 / (1.0 + np.exp(-weight))  # sigmoid
                if weight > 0.3:
                    edges.append((i, j, "COMMITMENT", weight))

    print(f"Valid edges found: {len(edges)}")
    for src, tgt, etype, w in edges[:5]:
        print(f"  {src} → {tgt}: {etype} (weight={w:.3f})")
    if len(edges) > 5:
        print(f"  ... and {len(edges) - 5} more")

    return edges


def test_generation(bm: BaseModel, sign_content: np.ndarray, prompt: str):
    """Simulate conditioned generation with ToM reranking."""
    print(f"\n--- Realization ---")
    print(f"Generating candidates for: '{prompt}'")

    candidates = bm.generate_conditioned(prompt, max_new_tokens=32, num_beams=4, num_return_sequences=4)

    print(f"Candidates:")
    for i, c in enumerate(candidates):
        print(f"  {i+1}: {c[:80]}{'...' if len(c) > 80 else ''}")

    # Simulate ToM reranking: score each candidate by re-embedding and comparing
    print(f"\n--- Theory of Mind Reranking ---")
    scores = []
    target_mean = sign_content.mean(axis=1)  # mean sign embedding as target

    for i, candidate in enumerate(candidates):
        embeds, _ = bm.get_embeddings(candidate)
        candidate_mean = embeds.mean(axis=1)

        # Cosine similarity as proxy for sign structure match
        cos_sim = np.dot(target_mean, candidate_mean) / (
            np.linalg.norm(target_mean) * np.linalg.norm(candidate_mean) + 1e-8
        )
        scores.append(cos_sim)
        print(f"  Candidate {i+1}: cosine_sim={cos_sim:.4f}")

    best = np.argmax(scores)
    print(f"\nSelected candidate {best+1}: {candidates[best][:80]}{'...' if len(candidates[best]) > 80 else ''}")


def main():
    print("=== Semiotic Pipeline End-to-End Test ===\n")

    bm = BaseModel()
    config = bm.get_config()
    print(f"Base model: {config['model_name']}, d_model={config['d_model']}")

    text = "Water is H2O. The cat sat on the mat. Therefore, the mat is under the cat."
    print(f"\nInput text: {text}")

    # Step 1: Get embeddings from base model
    embeddings, tokens = bm.get_embeddings(text)
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Step 2: Sign formation
    sign_content, sign_types, truth = test_sign_formation(embeddings, tokens, num_slots=8)

    # Step 3: Inference graph
    edges = test_inference_graph(sign_content, sign_types)

    # Step 4: Generation with ToM
    test_generation(bm, sign_content, text)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
