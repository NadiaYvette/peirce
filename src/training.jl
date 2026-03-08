# Training Pipeline — Phase 1: Distillation
#
# Objective: Learn sign-structured representations that preserve enough information
# to reconstruct original token embeddings.
#
# Loss = reconstruction_loss + λ_sparse * sparsity_loss + λ_entropy * entropy_loss + λ_consist * consistency_loss
#
# Reconstruction: sign representations → decoder → predicted token embeddings → MSE with originals
# Sparsity: each token should belong predominantly to one sign slot
# Entropy: sign-type distributions should use all 10 classes across the batch (not collapse)
# Consistency: same expression in similar contexts should get similar sign-type distributions

using Flux.Optimise

"""
    ReconstructionDecoder(d_model, max_tokens, num_slots)

Decodes sign representations back to token embedding space.
Cross-attention: each token position attends to sign slots to reconstruct its embedding.
Uses both content and interpretant slots for richer reconstruction.
"""
struct ReconstructionDecoder
    position_embed::AbstractMatrix{Float32}  # d_model × max_tokens (learned positions)
    to_q::Dense       # position -> query
    to_k::Dense       # sign content -> key
    to_v::Dense       # sign content -> value
    interp_gate::Dense  # gates interpretant contribution
    out_proj::Dense   # projected attention output -> reconstructed embedding
    norm::LayerNorm
    max_tokens::Int
end

Flux.@layer ReconstructionDecoder

function ReconstructionDecoder(d_model::Int; max_tokens::Int=512)
    ReconstructionDecoder(
        randn(Float32, d_model, max_tokens) .* Float32(0.02),
        Dense(d_model => d_model, bias=false),
        Dense(d_model => d_model, bias=false),
        Dense(d_model => d_model, bias=false),
        Dense(d_model => 1),
        Dense(d_model => d_model),
        LayerNorm(d_model),
        max_tokens
    )
end

function (dec::ReconstructionDecoder)(signs::SignBatch, num_tokens::Int)
    # Reconstruct token embeddings from sign representations
    # signs.content: d_model × K, signs.interpretant_slots: d_model × K
    # output: d_model × T (reconstructed token embeddings)

    T = min(num_tokens, dec.max_tokens)
    d = size(signs.content, 1)
    scale = Float32(sqrt(d))

    # Blend content and interpretant for richer sign representation
    gate = sigmoid.(dec.interp_gate(signs.interpretant_slots))  # 1 × K
    blended = signs.content .+ gate .* signs.interpretant_slots  # d_model × K

    # Position queries
    pos = dec.position_embed[:, 1:T]        # d_model × T
    queries = dec.to_q(pos)                  # d_model × T

    # Sign keys and values from blended representation
    keys = dec.to_k(blended)                 # d_model × K
    vals = dec.to_v(blended)                 # d_model × K

    # Cross-attention: each token position attends to sign slots
    attn = softmax((queries' * keys) ./ scale; dims=2)  # T × K
    attended = vals * attn'                               # d_model × T

    dec.norm(dec.out_proj(attended) .+ pos)               # d_model × T
end

# --- Loss Functions ---

"""
    reconstruction_loss(reconstructed, original)

MSE between reconstructed and original token embeddings.
"""
function reconstruction_loss(reconstructed::AbstractMatrix, original::AbstractMatrix)
    T = min(size(reconstructed, 2), size(original, 2))
    Flux.mse(reconstructed[:, 1:T], original[:, 1:T])
end

"""
    consistency_loss(sign_types_a, sign_types_b)

KL divergence between sign-type distributions of the same text processed twice
(with different dropout / noise). Encourages stable classifications.
"""
function consistency_loss(a::AbstractMatrix, b::AbstractMatrix)
    eps = Float32(1e-8)
    kl = sum(a .* log.((a .+ eps) ./ (b .+ eps))) / size(a, 2)
    # Symmetrize
    kl_rev = sum(b .* log.((b .+ eps) ./ (a .+ eps))) / size(b, 2)
    (kl + kl_rev) / 2f0
end

"""
    TrainingState

Holds all trainable components and optimizer state.
"""
mutable struct TrainingState
    pipeline::SemioticPipeline
    decoder::ReconstructionDecoder
    rigid_table::RigidDesignationTable
    opt_state::Any  # Flux optimizer state

    # Loss weights
    λ_sparse::Float32
    λ_entropy::Float32
    λ_consist::Float32

    # Tracking
    step::Int
    losses::Vector{Dict{String, Float32}}
end

function TrainingState(d_model::Int;
                        num_slots::Int=32,
                        num_standard_layers::Int=4,
                        num_semiotic_layers::Int=8,
                        max_tokens::Int=512,
                        learning_rate::Float64=1e-4,
                        λ_sparse::Float32=Float32(0.1),
                        λ_entropy::Float32=Float32(0.01),
                        λ_consist::Float32=Float32(0.05))
    pipeline = SemioticPipeline(d_model; num_slots, num_standard_layers, num_semiotic_layers)
    decoder = ReconstructionDecoder(d_model; max_tokens)
    rigid_table = RigidDesignationTable(d_model)

    # Collect all trainable parameters
    params = Flux.params(pipeline, decoder)
    opt_state = Flux.setup(Adam(learning_rate), (pipeline, decoder))

    TrainingState(pipeline, decoder, rigid_table, opt_state,
                  λ_sparse, λ_entropy, λ_consist, 0, Dict{String, Float32}[])
end

"""
    train_step!(state, token_embeddings, token_strings; τ=1.0f0)

One training step: forward pass, compute loss, backward pass, update parameters.
τ controls Gumbel-Softmax temperature (anneal from high to low during training).

Returns dict of individual loss components.
"""
function train_step!(state::TrainingState,
                     token_embeddings::AbstractMatrix{Float32},
                     token_strings::Vector{String};
                     τ::Float32=Float32(1.0))
    T = size(token_embeddings, 2)

    loss_dict = Dict{String, Float32}()

    # Pre-create graph outside gradient (its constructor uses fill! which Zygote can't handle)
    graph = InferenceGraph(size(token_embeddings, 1))

    # Gradient computation
    (loss_val, loss_dict), grads = Flux.withgradient(state.pipeline, state.decoder) do pipeline, decoder
        # Forward pass with Gumbel-Softmax hard assignment
        signs, _, seg_weights = comprehend(pipeline, token_embeddings, state.rigid_table, token_strings;
                                            graph=graph, τ=τ, hard=true)

        # Decoder reconstruction (secondary — position-based)
        reconstructed = decoder(signs, T)
        l_recon = reconstruction_loss(reconstructed, token_embeddings)

        # Segmentation-based reconstruction (PRIMARY)
        l_seg_recon = segmentation_reconstruction_loss(signs, seg_weights, token_embeddings)

        # Regularizers
        l_entropy = sign_type_entropy_loss(signs.sign_type)
        l_diversity = slot_content_diversity_loss(signs)
        l_interp_div = interpretant_diversity_loss(signs)
        l_occupancy = slot_occupancy_loss(seg_weights)
        l_proto_div = prototype_diversity_loss(pipeline.sign_formation.segmentation.prototypes)

        # Truth value loss (targets computed from graph + sign structure)
        truth_targets = Zygote.ignore() do
            compute_truth_targets(signs, graph, seg_weights)
        end
        l_truth = truth_value_loss(signs, truth_targets)

        # Consistency: run sign formation again (different Gumbel noise draw)
        signs2, _ = forward(pipeline.sign_formation, token_embeddings, state.rigid_table, token_strings;
                            τ=τ, hard=true)
        l_consist = consistency_loss(signs.sign_type, signs2.sign_type)

        total = Float32(0.5) * l_recon +                   # decoder reconstruction (reduced)
                Float32(5.0) * l_seg_recon +                # segmentation reconstruction (PRIMARY)
                state.λ_entropy * l_entropy +
                state.λ_consist * l_consist +
                Float32(50.0) * l_diversity +               # strong slot differentiation
                Float32(50.0) * l_interp_div +              # interpretant differentiation
                Float32(0.5) * l_occupancy +                # activate all slots
                Float32(20.0) * l_proto_div +               # prevent prototype collapse
                Float32(1.0) * l_truth                      # truth value prediction

        local_dict = Dict{String, Float32}(
            "total" => total,
            "reconstruction" => l_recon,
            "seg_recon" => l_seg_recon,
            "entropy" => l_entropy,
            "consistency" => l_consist,
            "diversity" => l_diversity,
            "interp_div" => l_interp_div,
            "occupancy" => l_occupancy,
            "proto_div" => l_proto_div,
            "truth" => l_truth,
        )

        (total, local_dict)
    end

    # Update parameters
    Flux.update!(state.opt_state, (state.pipeline, state.decoder), grads)

    state.step += 1
    push!(state.losses, loss_dict)

    loss_dict
end

"""
    train_epoch!(state, data_iterator; log_every=10)

Train for one epoch over a data iterator.
Each element of data_iterator should be (token_embeddings::Matrix{Float32}, token_strings::Vector{String}).
"""
function train_epoch!(state::TrainingState, data_iterator;
                      log_every::Int=10, τ::Float32=Float32(1.0))
    epoch_losses = Dict{String, Float64}(
        "total" => 0.0, "reconstruction" => 0.0,
        "entropy" => 0.0, "consistency" => 0.0,
        "diversity" => 0.0, "seg_recon" => 0.0,
        "occupancy" => 0.0
    )
    n_batches = 0

    for (token_embeddings, token_strings) in data_iterator
        loss_dict = train_step!(state, token_embeddings, token_strings; τ=τ)

        for (k, v) in loss_dict
            epoch_losses[k] = get(epoch_losses, k, 0.0) + v
        end
        n_batches += 1

        if state.step % log_every == 0
            println("Step $(state.step): total=$(round(loss_dict["total"]; digits=4)), " *
                    "recon=$(round(loss_dict["reconstruction"]; digits=4)), " *
                    "seg_r=$(round(get(loss_dict, "seg_recon", 0f0); digits=4)), " *
                    "div=$(round(get(loss_dict, "diversity", 0f0); digits=4)), " *
                    "τ=$(round(τ; digits=3))")
        end
    end

    # Average
    for k in keys(epoch_losses)
        epoch_losses[k] /= max(n_batches, 1)
    end

    println("Epoch avg: total=$(round(epoch_losses["total"]; digits=4)), " *
            "recon=$(round(epoch_losses["reconstruction"]; digits=4)), " *
            "seg_r=$(round(get(epoch_losses, "seg_recon", 0.0); digits=4))")

    epoch_losses
end

# --- Phase 2 Losses ---
# These losses encourage semiotically meaningful structure beyond reconstruction.

"""
    slot_content_diversity_loss(signs)

Penalizes slot content embeddings being too similar.
Uses soft-max penalty: heavily penalizes ANY pair with high similarity.
This prevents the 2-cluster collapse where mean similarity is low but all slots
within each cluster are identical.
"""
function slot_content_diversity_loss(signs::SignBatch)
    content = signs.content  # d_model × K
    K = size(content, 2)
    K <= 1 && return Float32(0)
    eps = Float32(1e-8)

    c_norms = sqrt.(sum(content .^ 2; dims=1) .+ eps)
    c_normed = content ./ c_norms
    sim = c_normed' * c_normed  # K × K

    # Zero out diagonal
    mask = Zygote.dropgrad(Float32.(1 .- I(K)))
    abs_off = abs.(sim .* mask)  # K × K, diagonal = 0

    # Three-part penalty:
    # 1. Soft-max over all off-diagonal similarities (differentiable max approximation)
    β = Float32(20.0)
    soft_max = log.(sum(exp.(abs_off .* β .* mask); dims=2) .+ eps) ./ β  # K × 1
    worst_pair_penalty = mean(soft_max)

    # 2. Mean absolute similarity — prevents overall clustering
    mean_penalty = sum(abs_off) / (K * (K - 1))

    # 3. Hard repulsion for near-duplicates: quadratic penalty above threshold
    #    Pairs with |sim| > 0.4 get heavily penalized, scaling with (sim - 0.4)^2
    threshold = Float32(0.4)
    excess = max.(abs_off .- threshold, Float32(0)) .* mask
    hard_repulsion = sum(excess .^ 2) / (K * (K - 1))

    worst_pair_penalty + mean_penalty + Float32(100.0) * hard_repulsion
end

"""
    interpretant_diversity_loss(signs)

Penalizes interpretant slots collapsing to the same vector.
Same structure as slot_content_diversity_loss but for interpretants.
"""
function interpretant_diversity_loss(signs::SignBatch)
    interp = signs.interpretant_slots  # d_model × K
    K = size(interp, 2)
    K <= 1 && return Float32(0)
    eps = Float32(1e-8)

    i_norms = sqrt.(sum(interp .^ 2; dims=1) .+ eps)
    i_normed = interp ./ i_norms
    sim = i_normed' * i_normed  # K × K

    mask = Zygote.dropgrad(Float32.(1 .- I(K)))
    abs_off = abs.(sim .* mask)

    # Mean similarity + hard repulsion above threshold
    mean_penalty = sum(abs_off) / (K * (K - 1))
    threshold = Float32(0.4)
    excess = max.(abs_off .- threshold, Float32(0)) .* mask
    hard_repulsion = sum(excess .^ 2) / (K * (K - 1))

    mean_penalty + Float32(100.0) * hard_repulsion
end

"""
    truth_value_loss(signs, truth_targets)

MSE between predicted truth values and computed targets.
Targets are detached (no gradient flows through them).
"""
function truth_value_loss(signs::SignBatch, truth_targets::AbstractMatrix)
    targets = Zygote.dropgrad(truth_targets)
    Flux.mse(signs.truth, targets)
end

"""
    slot_occupancy_loss(segment_weights)

Penalizes empty slots. Each slot should receive at least some token mass.
Uses log-barrier: -log(load + eps) penalizes slots approaching zero load.
"""
function slot_occupancy_loss(segment_weights::AbstractMatrix)
    # segment_weights: K × T
    K, T = size(segment_weights)
    eps = Float32(1e-8)

    # Total mass per slot (should be > 0 for all slots)
    slot_loads = sum(segment_weights; dims=2) ./ T  # K × 1, each ∈ [0, 1]

    # Log-barrier: heavily penalizes near-zero loads, mild for healthy loads
    -mean(log.(slot_loads .+ eps))
end

"""
    segmentation_reconstruction_loss(signs, segment_weights, token_embeddings)

Each token should be well-reconstructed by its assigned slot's content.
This creates a direct gradient path: better segmentation → better per-slot content → lower loss.
"""
function segmentation_reconstruction_loss(signs::SignBatch,
                                           segment_weights::AbstractMatrix,
                                           token_embeddings::AbstractMatrix)
    # signs.content: d_model × K
    # segment_weights: K × T
    # token_embeddings: d_model × T

    # Reconstruct each token as weighted combination of slot contents
    # using the segmentation weights
    reconstructed = signs.content * segment_weights  # d_model × T

    # MSE between segmentation-weighted reconstruction and originals
    Flux.mse(reconstructed, token_embeddings)
end

"""
    sign_type_prediction_loss(pipeline, signs, token_embeddings)

Can sign types be predicted from local token context alone?
If yes, the classifier has learned context-dependent type assignment.
Loss: cross-entropy between sign types predicted from full pipeline vs.
a lightweight probe on just the slot content (before attention layers).
"""
function sign_type_prediction_loss(pipeline::SemioticPipeline,
                                   token_embeddings::AbstractMatrix,
                                   rigid_table::RigidDesignationTable,
                                   token_strings::Vector{String})
    # Get slot embeddings from just sign formation (no attention layers)
    seg_weights = pipeline.sign_formation.segmentation(token_embeddings)
    slot_content = pipeline.sign_formation.slot_attention(token_embeddings, seg_weights)

    # Predict sign types from raw slot content
    predicted_types = pipeline.sign_formation.sign_classifier(slot_content)  # 10 × K

    # Get sign types from full pipeline (target)
    graph = Zygote.ignore() do
        InferenceGraph(size(token_embeddings, 1))
    end
    signs, _, _ = comprehend(pipeline, token_embeddings, rigid_table, token_strings; graph=graph)
    target_types = Zygote.dropgrad(signs.sign_type)  # detach target

    # Cross-entropy
    eps = Float32(1e-8)
    -sum(target_types .* log.(predicted_types .+ eps)) / size(predicted_types, 2)
end

"""
    slot_contrastive_loss(signs)

One-sided contrastive: penalize slots that have DIFFERENT sign types but SIMILAR content.
Does NOT reward content similarity for similar types (that would fight the diversity loss).
"""
function slot_contrastive_loss(signs::SignBatch)
    content = signs.content      # d_model × K
    types = signs.sign_type      # 10 × K
    K = size(content, 2)
    K <= 1 && return Float32(0)
    eps = Float32(1e-8)

    # Content cosine similarity
    c_norms = sqrt.(sum(content .^ 2; dims=1) .+ eps)
    c_normed = content ./ c_norms
    content_sim = c_normed' * c_normed  # K × K

    # Type dissimilarity: 1 - cosine_sim(type_i, type_j)
    t_norms = sqrt.(sum(types .^ 2; dims=1) .+ eps)
    t_normed = types ./ t_norms
    type_dissim = Float32(1) .- (t_normed' * t_normed)  # K × K, 0=same types, ~1=different

    mask = Zygote.dropgrad(Float32.(1 .- I(K)))

    # Penalize: high content similarity * high type dissimilarity
    # Only fires when content is similar BUT types are different
    violation = max.(content_sim, Float32(0)) .* type_dissim .* mask
    sum(violation .^ 2) / (K * (K - 1))
end

"""
    trichotomy_consistency_loss(signs)

The three trichotomy marginals should obey Peirce's validity constraint:
for each sign, trichotomy II value <= trichotomy I value, and
trichotomy III value <= trichotomy II value.

This is enforced softly: penalize mass on invalid configurations.
"""
function trichotomy_consistency_loss(signs::SignBatch)
    trich1, trich2, trich3 = decompose_trichotomies(signs.sign_type)
    K = size(signs.sign_type, 2)

    # For each sign, compute expected trichotomy level
    # Level = sum(i * p(i)) for i in 1:3
    levels = Float32[1, 2, 3]
    e1 = sum(levels .* trich1; dims=1)  # 1 × K
    e2 = sum(levels .* trich2; dims=1)  # 1 × K
    e3 = sum(levels .* trich3; dims=1)  # 1 × K

    # Penalty for violations: E[trich2] > E[trich1] or E[trich3] > E[trich2]
    violation_21 = max.(e2 .- e1, Float32(0))
    violation_32 = max.(e3 .- e2, Float32(0))

    mean(violation_21 .^ 2 .+ violation_32 .^ 2)
end

"""
    trichotomy_balance_loss(signs)

Encourages variety across ALL THREE trichotomies, not just trichotomy I.
For each trichotomy: pushes the across-slot mean toward uniform(3) and
encourages different slots to have different profiles.
"""
function trichotomy_balance_loss(signs::SignBatch)
    trichotomies = decompose_trichotomies(signs.sign_type)
    K = size(signs.sign_type, 2)
    eps = Float32(1e-8)
    uniform = Float32(1.0 / 3.0)

    total = Float32(0)
    for trich in trichotomies
        # trich: 3 × K
        # Push mean distribution toward uniform(3)
        mean_t = mean(trich; dims=2)  # 3 × 1
        kl = sum(mean_t .* log.((mean_t .+ eps) ./ uniform))
        total += kl

        # Push slots to have different profiles within this trichotomy
        if K > 1
            t_norms = sqrt.(sum(trich .^ 2; dims=1) .+ eps)
            t_normed = trich ./ t_norms
            sim = t_normed' * t_normed  # K × K
            slot_sim = (sum(sim) - K) / (K * (K - 1))
            total += slot_sim
        end
    end
    total / Float32(3)  # average over trichotomies
end

# --- Data utilities ---

"""
    chunk_text(token_embeddings, token_strings, chunk_size)

Split long sequences into chunks for training.
Returns vector of (embeddings_chunk, tokens_chunk) tuples.
"""
function chunk_text(token_embeddings::AbstractMatrix{Float32},
                    token_strings::Vector{String},
                    chunk_size::Int)
    T = size(token_embeddings, 2)
    chunks = Tuple{Matrix{Float32}, Vector{String}}[]
    for start in 1:chunk_size:T
        stop = min(start + chunk_size - 1, T)
        if stop - start + 1 >= 4  # minimum chunk size
            push!(chunks, (token_embeddings[:, start:stop], token_strings[start:stop]))
        end
    end
    chunks
end
