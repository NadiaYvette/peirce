# Peirce

A Large Semiotic Model — an LLM-adjacent system that operates over Peircean sign structures rather than raw tokens.

## What This Is

Instead of treating language as sequences of tokens with attention-derived meaning, Peirce processes text through a semiotic pipeline grounded in Charles Sanders Peirce's theory of signs. The system:

- **Segments** token embeddings into **sign slots** via learned prototypes (CLIP-style logit scaling)
- **Classifies** each sign into one of Peirce's **10 sign classes** as a continuous distribution (e.g., rhematic-iconic-qualisign through argument-symbolic-legisign)
- **Enforces trichotomy constraints** (the algebraic relationships I ≥ II ≥ III between Peirce's three trichotomies)
- **Maintains a Brandomian inference graph** tracking commitments, entitlements, and incompatibilities between signs
- **Computes multi-valued truth** (semiotic confidence, truth estimate, inferential weight)
- **Conditions generation** on semiotic structure via soft-prompt injection into the base model

The base model (Qwen2.5-0.5B) provides token embeddings and generation; the semiotic pipeline provides structured meaning.

## Architecture

```
Text → Qwen Embeddings (896d) → Orthogonal Projection (128d)
  → Learned Segmentation (tokens → sign slots)
  → Sign Formation (content, object, interpretant, sign-type, truth)
  → Sign-Typed Attention (type-aware self-attention)
  → Inference Graph (GNN message passing)
  → Realization (discourse planning, Gricean filtering, conditioned generation)
```

**Julia** is the primary language (Flux.jl, Zygote.jl for autodiff). **Python** handles the base model ecosystem (HuggingFace transformers) via a subprocess bridge.

## Project Structure

```
src/
  Peirce.jl            — module definition and exports
  sign_types.jl        — Peirce's 10 sign classes, trichotomy algebra
  sign_formation.jl    — segmentation, slot formation, type classification
  sign_attention.jl    — sign-type-aware attention layers
  rigid_designation.jl — Kripkean rigid designators with learned rigidity
  inference_graph.jl   — Brandomian inference graph (commitments/entitlements)
  realization.jl       — discourse planning, Gricean filter, ToM reranking
  pipeline.jl          — full comprehension/generation pipeline
  base_model_bridge.jl — Julia ↔ Python subprocess bridge
  training.jl          — training loop, losses, reconstruction decoder

python/
  base_model.py        — Qwen2.5-0.5B wrapper (embeddings + conditioned generation)
  bridge_server.py     — JSON-over-stdio bridge server
  data_loader.py       — original training data (hardcoded texts)
  corpus_loader.py     — corpus-based training data preparation

corpus/                — training texts by category (public domain)
train.jl               — Phase 1 training (distillation)
train_phase2.jl        — Phase 2 training (contrastive + structure)
evaluate.jl            — downstream evaluation suite
end_to_end.jl          — full pipeline integration test
```

## Setup

### Prerequisites
- Julia 1.11+
- Python 3.12 with `torch`, `transformers`, `numpy`
- ~1GB disk for Qwen2.5-0.5B weights

### Install
```bash
# Download Qwen2.5-0.5B
mkdir -p models/qwen2.5-0.5b
# (download from HuggingFace: Qwen/Qwen2.5-0.5B)

# Julia dependencies
JULIA_PKG_SERVER="" julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Prepare training data
python3.12 python/corpus_loader.py

# Train
JULIA_PKG_SERVER="" julia --project=. train.jl
JULIA_PKG_SERVER="" julia --project=. train_phase2.jl

# Evaluate
JULIA_PKG_SERVER="" julia --project=. evaluate.jl

# End-to-end test
JULIA_PKG_SERVER="" julia --project=. end_to_end.jl
```

## Current Status

- Full pipeline working end-to-end with live Qwen embeddings
- Conditioned generation functional (semiotic soft-prompt injection)
- Phase 1 + Phase 2 training complete on small corpus
- Evaluation: 0% trichotomy violations, semantic similarity separation, conditioned output differentiation
- Interaction boundary detection (anthropomorphisation, emotional escalation, dependency patterns)
- Multilingual corpus assembly in progress

## Theoretical Foundation

- **Peirce's semiotics**: 10 sign classes from three trichotomies (representamen, object, interpretant)
- **Kripke's rigid designation**: some terms designate the same referent in all possible worlds
- **Brandom's inferentialism**: meaning is constituted by inferential relationships (commitments, entitlements, incompatibilities)
- **Grice's maxims**: quantity, quality, relation, manner as generation constraints

## License

TBD
