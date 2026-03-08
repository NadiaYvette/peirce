# Sign Formation Module
#
# Converts token embeddings (d_model × T) into sign representations (SignBatch of K signs).
#
# Architecture:
#   1. Learned segmentation: each token gets a soft membership distribution over K sign slots
#   2. Slot attention: K sign slots attend to token embeddings, weighted by segment membership
#   3. Sign-type classification: each slot gets a 10-dim distribution over Peircean classes
#   4. Object slot population: provisional, via rigid designation table + soft attention over prior signs
#   5. Interpretant slots: initialized to zero (deferred)
#
# Training: reconstruction loss + regularizers (entropy, sparsity, consistency)

"""
    LearnedSegmentation(d_model, num_slots)

Prototype-based segmentation: each slot has a learned prototype vector.
Tokens are assigned to slots by cosine similarity with prototypes.
Hard assignment via straight-through estimator during training.
"""
struct LearnedSegmentation
    prototypes::AbstractMatrix{Float32}  # d_model × K (learned slot prototypes)
    logit_scale::AbstractVector{Float32} # learned scalar to amplify cosine similarities
    num_slots::Int
end

Flux.@layer LearnedSegmentation

function LearnedSegmentation(d_model::Int, num_slots::Int)
    # Orthogonal initialization — maximally different prototypes
    proto_raw = randn(Float32, d_model, num_slots)
    prototypes = Float32.(Matrix(qr(proto_raw).Q)[:, 1:num_slots])
    # Learned logit scale (like CLIP): amplifies cosine similarities before softmax
    # Initialized to 10.0 so that cos_sim * scale gives meaningful logits
    logit_scale = Float32[log(10.0)]  # stored as log, applied as exp
    LearnedSegmentation(prototypes, logit_scale, num_slots)
end

function (seg::LearnedSegmentation)(token_embeddings::AbstractMatrix;
                                     τ::Float32=Float32(1.0), hard::Bool=true)
    # token_embeddings: d_model × T
    # output: num_slots × T
    eps = Float32(1e-8)

    # Normalize both for cosine similarity
    t_norms = sqrt.(sum(token_embeddings .^ 2; dims=1) .+ eps)  # 1 × T
    t_normed = token_embeddings ./ t_norms

    p_norms = sqrt.(sum(seg.prototypes .^ 2; dims=1) .+ eps)  # 1 × K
    p_normed = seg.prototypes ./ p_norms

    # Cosine similarity: K × T, scaled by learned temperature
    scale = clamp(exp(seg.logit_scale[1]), Float32(1.0), Float32(100.0))
    logits = scale .* (p_normed' * t_normed)  # K × T

    if hard
        y_soft = softmax(logits ./ τ; dims=1)
        y_hard = Zygote.ignore() do
            idx = argmax(logits; dims=1)
            one_hot = zeros(Float32, size(logits))
            for i in idx
                one_hot[i] = Float32(1)
            end
            one_hot
        end
        y_hard .- Zygote.dropgrad(y_soft) .+ y_soft
    else
        softmax(logits ./ τ; dims=1)
    end
end

"""
    SlotAttention(d_model, num_slots, num_iterations)

Iterative slot attention mechanism. K learned slot vectors attend to token embeddings,
refined over multiple iterations.

Each iteration:
  1. Slots compute attention over tokens (weighted by segment membership)
  2. Attended values update slot representations via gated residual (Zygote-safe)
"""
struct SlotAttention
    slot_init::AbstractMatrix{Float32}   # d_model × K (learned initial slot vectors)
    to_q::Dense          # slot -> query
    to_k::Dense          # token -> key
    to_v::Dense          # token -> value
    update_proj::Dense   # projects update for gated residual
    gate_proj::Dense     # computes gate for update blending
    norm_slots::LayerNorm
    norm_inputs::LayerNorm
    num_iterations::Int
    scale::Float32
end

Flux.@layer SlotAttention

function SlotAttention(d_model::Int, num_slots::Int; num_iterations::Int=3)
    scale = Float32(sqrt(d_model))
    # Orthogonal slot initialization — ensures slots start maximally different
    slot_init_raw = randn(Float32, d_model, num_slots)
    slot_init = Float32.(Matrix(qr(slot_init_raw).Q)[:, 1:num_slots]) .* Float32(0.1)
    SlotAttention(
        slot_init,
        Dense(d_model => d_model, bias=false),  # to_q
        Dense(d_model => d_model, bias=false),  # to_k
        Dense(d_model => d_model, bias=false),  # to_v
        Dense(d_model => d_model),              # update_proj
        Dense(d_model => d_model),              # gate_proj
        LayerNorm(d_model),
        LayerNorm(d_model),
        num_iterations,
        scale
    )
end

function (sa::SlotAttention)(token_embeddings::AbstractMatrix, segment_weights::AbstractMatrix)
    # token_embeddings: d_model × T
    # segment_weights: K × T (from LearnedSegmentation)
    # output: d_model × K (refined slot embeddings)

    T = size(token_embeddings, 2)
    K = size(sa.slot_init, 2)

    inputs = sa.norm_inputs(token_embeddings)    # d_model × T
    keys = sa.to_k(inputs)                       # d_model × T
    vals = sa.to_v(inputs)                       # d_model × T

    slots = sa.slot_init                         # d_model × K

    for _ in 1:sa.num_iterations
        slots_normed = sa.norm_slots(slots)      # d_model × K
        queries = sa.to_q(slots_normed)          # d_model × K

        # Attention: K × T
        # Each slot attends to tokens, modulated by segment membership
        attn_logits = (queries' * keys) ./ sa.scale   # K × T
        attn_logits = attn_logits .+ log.(segment_weights .+ Float32(1e-8))  # bias toward assigned segments
        attn = softmax(attn_logits; dims=2)           # K × T, normalized over tokens

        # Weighted aggregation: d_model × K
        updates = vals * attn'                         # d_model × K

        # Gated residual update (all slots at once — Zygote-safe, no per-column loop)
        projected = sa.update_proj(updates)            # d_model × K
        gate = sigmoid.(sa.gate_proj(updates .+ slots))  # d_model × K
        slots = gate .* projected .+ (1f0 .- gate) .* slots
    end

    slots  # d_model × K
end

"""
    SignTypeClassifier(d_model)

Classifies each sign slot into a distribution over the 10 Peircean sign classes.
Learned temperature per slot allows variable confidence.
"""
struct SignTypeClassifier
    proj::Dense              # d_model -> 10
    temperature::Dense       # d_model -> 1 (positive, per-slot temperature)
end

Flux.@layer SignTypeClassifier

function SignTypeClassifier(d_model::Int)
    SignTypeClassifier(
        Dense(d_model => NUM_SIGN_CLASSES),
        Dense(d_model => 1)
    )
end

function (cls::SignTypeClassifier)(slot_embeddings::AbstractMatrix)
    # slot_embeddings: d_model × K
    # output: 10 × K (softmax distribution per slot)
    logits = cls.proj(slot_embeddings)                    # 10 × K
    temp = softplus.(cls.temperature(slot_embeddings))    # 1 × K (positive)
    temp = clamp.(temp, Float32(0.1), Float32(10.0))
    softmax(logits ./ temp; dims=1)
end

"""
    ObjectSlotPopulator(d_model, rigid_table)

Populates object slots provisionally:
  1. If the sign matches a rigid designator, use its referent embedding (weighted by rigidity)
  2. Otherwise, soft attention over previously formed signs provides a referential context
  3. Unresolved signs get a learned null embedding
"""
struct ObjectSlotPopulator
    ref_proj::Dense          # projects slot embedding for reference query
    null_embedding::AbstractVector{Float32}  # learned "unresolved" vector
    blend_gate::Dense        # d_model -> 1, gates between rigid and attentional
end

Flux.@layer ObjectSlotPopulator

function ObjectSlotPopulator(d_model::Int)
    ObjectSlotPopulator(
        Dense(d_model => d_model),
        randn(Float32, d_model) .* Float32(0.02),
        Dense(d_model => 1)
    )
end

function (pop::ObjectSlotPopulator)(
    slot_embeddings::AbstractMatrix,      # d_model × K
    rigid_results::Vector,                # length K: each is (referent, rigidity) or nothing
    prior_signs::Union{AbstractMatrix, Nothing}  # d_model × M (previously formed signs) or nothing
)
    d, K = size(slot_embeddings)

    # Functional (non-mutating) construction for Zygote compatibility
    columns = map(1:K) do k
        slot = slot_embeddings[:, k]
        rigid = rigid_results[k]

        if !isnothing(rigid)
            referent, rigidity = rigid
            gate = sigmoid(pop.blend_gate(slot))[1] * rigidity
            contextual = _attend_to_prior(pop, slot, prior_signs)
            gate .* referent .+ (1f0 - gate) .* contextual
        elseif !isnothing(prior_signs) && size(prior_signs, 2) > 0
            _attend_to_prior(pop, slot, prior_signs)
        else
            pop.null_embedding
        end
    end

    reduce(hcat, columns)  # d_model × K
end

function _attend_to_prior(pop::ObjectSlotPopulator, slot::AbstractVector,
                          prior_signs::Union{AbstractMatrix, Nothing})
    isnothing(prior_signs) && return pop.null_embedding
    M = size(prior_signs, 2)
    query = pop.ref_proj(slot)                          # d_model
    scores = (prior_signs' * query) ./ Float32(sqrt(length(query)))  # M
    weights = softmax(scores)                            # M
    prior_signs * weights                                # d_model
end

"""
    InterpretantInit(d_model)

Initializes interpretant slots from sign content and sign type.
The interpretant is what the sign produces in the interpreter —
initialized as a learned projection of content, conditioned on type.
"""
struct InterpretantInit
    content_proj::Dense    # content → interpretant basis
    type_proj::Dense       # sign_type (10-dim) → modulation
    blend::Dense           # combines content + type info → interpretant
    norm::LayerNorm
end

Flux.@layer InterpretantInit

function InterpretantInit(d_model::Int)
    InterpretantInit(
        Dense(d_model => d_model),
        Dense(NUM_SIGN_CLASSES => d_model),
        Dense(2 * d_model => d_model),
        LayerNorm(d_model)
    )
end

function (ii::InterpretantInit)(content::AbstractMatrix, sign_types::AbstractMatrix)
    # content: d_model × K, sign_types: 10 × K
    c = ii.content_proj(content)         # d_model × K
    t = ii.type_proj(sign_types)         # d_model × K
    ii.norm(ii.blend(vcat(c, t)))        # d_model × K
end

"""
    SignFormationModule

Complete sign-formation pipeline: tokens → signs.
"""
struct SignFormationModule
    segmentation::LearnedSegmentation
    slot_attention::SlotAttention
    sign_classifier::SignTypeClassifier
    object_populator::ObjectSlotPopulator
    interpretant_init::InterpretantInit
    d_model::Int
    num_slots::Int
end

Flux.@layer SignFormationModule

function SignFormationModule(d_model::Int, num_slots::Int; num_iterations::Int=3)
    SignFormationModule(
        LearnedSegmentation(d_model, num_slots),
        SlotAttention(d_model, num_slots; num_iterations),
        SignTypeClassifier(d_model),
        ObjectSlotPopulator(d_model),
        InterpretantInit(d_model),
        d_model,
        num_slots
    )
end

"""
    forward(mod, token_embeddings, rigid_table, token_strings; ...)

Returns: (SignBatch, segment_weights) — segment_weights are K × T (hard or soft).
"""
function forward(
    mod::SignFormationModule,
    token_embeddings::AbstractMatrix{Float32},
    rigid_table::RigidDesignationTable,
    token_strings::Vector{String};
    prior_signs::Union{AbstractMatrix{Float32}, Nothing}=nothing,
    τ::Float32=Float32(1.0),
    hard::Bool=true
)
    T = size(token_embeddings, 2)
    K = mod.num_slots

    # 1. Gumbel-Softmax segmentation: hard token-to-slot assignment (differentiable)
    segment_weights = mod.segmentation(token_embeddings; τ=τ, hard=hard)   # K × T

    # 2. Slot attention: form sign content embeddings
    content = mod.slot_attention(token_embeddings, segment_weights)  # d_model × K

    # 3. Sign-type classification
    sign_types = mod.sign_classifier(content)               # 10 × K

    # 4. Object slot population (provisional)
    rigid_results = Zygote.ignore() do
        _lookup_rigid_for_slots(rigid_table, token_strings, segment_weights)
    end
    object_slots = mod.object_populator(content, rigid_results, prior_signs)

    # 5. Interpretant slots: initialized from content + sign type
    interpretant_slots = mod.interpretant_init(content, sign_types)

    # 6. Truth values: initialized to neutral
    truth = fill(Float32(0.5), 3, K)

    (SignBatch(content, sign_types, object_slots, interpretant_slots, truth), segment_weights)
end

"""
Look up rigid designators for each sign slot by finding the highest-membership
token per slot and querying the table.
"""
function _lookup_rigid_for_slots(
    table::RigidDesignationTable,
    token_strings::Vector{String},
    segment_weights::AbstractMatrix  # K × T
)
    K = size(segment_weights, 1)
    results = Vector{Union{Nothing, Tuple{Vector{Float32}, Float32}}}(undef, K)

    for k in 1:K
        # Find the token with highest membership in this slot
        max_idx = argmax(segment_weights[k, :])
        result = lookup(table, token_strings[max_idx])
        results[k] = result
    end

    results
end

# --- Training losses ---

"""
    segmentation_sparsity_loss(segment_weights)

Encourages each token to belong predominantly to one slot.
Two components:
  1. Entropy: lower entropy = sparser assignment
  2. Load balancing: each slot should get roughly equal total mass (prevents dead slots)
"""
function segmentation_sparsity_loss(segment_weights::AbstractMatrix)
    # segment_weights: K × T, softmax over K per token
    K, T = size(segment_weights)

    # 1. Per-token entropy (minimize — encourage sharp assignments)
    ent = -sum(segment_weights .* log.(segment_weights .+ Float32(1e-8)); dims=1)  # 1 × T
    sparsity = mean(ent)

    # 2. Load balance: slot usage should be roughly uniform
    # Total mass per slot across all tokens
    slot_loads = sum(segment_weights; dims=2) ./ T  # K × 1, should be ~1/K each
    target_load = Float32(1.0 / K)
    load_balance = sum((slot_loads .- target_load) .^ 2) * K  # normalized

    # Combined: sparsity + balance (balance prevents dead slots from being optimal)
    sparsity + Float32(5.0) * load_balance
end

"""
    sign_type_entropy_loss(sign_types)

Balances two pressures on sign-type distributions:
  1. Population diversity: mean distribution across slots should be near-uniform (KL penalty)
  2. Per-slot specialization: each slot should have a peaked (low-entropy) distribution
  3. Slot diversity: different slots should have different type profiles

The net effect: each slot picks a preferred sign class, but different slots pick different ones.
"""
function sign_type_entropy_loss(sign_types::AbstractMatrix)
    # sign_types: 10 × K
    K = size(sign_types, 2)
    eps = Float32(1e-8)

    # 1. Population diversity: mean distribution should be close to uniform
    mean_dist = mean(sign_types; dims=2)  # 10 × 1
    uniform = Float32(1.0 / NUM_SIGN_CLASSES)
    kl_from_uniform = sum(mean_dist .* log.((mean_dist .+ eps) ./ uniform))

    # 2. Per-slot specialization: penalize high per-slot entropy
    # Each slot should have low entropy (peaked distribution)
    per_slot_entropy = -sum(sign_types .* log.(sign_types .+ eps); dims=1)  # 1 × K
    max_entropy = log(Float32(NUM_SIGN_CLASSES))
    # Normalize and penalize: mean normalized entropy (want this to be low)
    mean_normalized_entropy = mean(per_slot_entropy) / max_entropy

    # 3. Slot diversity: penalize slots having similar type distributions
    slot_diversity = if K > 1
        t_norms = sqrt.(sum(sign_types .^ 2; dims=1) .+ eps)
        t_normed = sign_types ./ t_norms
        sim_matrix = t_normed' * t_normed  # K × K
        (sum(sim_matrix) - K) / (K * (K - 1))
    else
        Float32(0)
    end

    # Balance: diversity keeps population uniform, specialization sharpens each slot
    kl_from_uniform + Float32(2.0) * mean_normalized_entropy + Float32(0.5) * slot_diversity
end

"""
    prototype_diversity_loss(prototypes)

Penalizes segmentation prototypes that point in similar directions.
When two prototypes are near-identical, the corresponding slots will capture
the same tokens and produce collapsed content embeddings.
Hard repulsion above cosine similarity 0.5.
"""
function prototype_diversity_loss(prototypes::AbstractMatrix)
    K = size(prototypes, 2)
    K <= 1 && return Float32(0)
    eps = Float32(1e-8)

    p_norms = sqrt.(sum(prototypes .^ 2; dims=1) .+ eps)
    p_normed = prototypes ./ p_norms
    sim = p_normed' * p_normed  # K × K

    mask = Zygote.dropgrad(Float32.(1 .- I(K)))
    abs_off = abs.(sim .* mask)

    # Hard repulsion above threshold
    threshold = Float32(0.5)
    excess = max.(abs_off .- threshold, Float32(0)) .* mask
    sum(excess .^ 2) / (K * (K - 1))
end
