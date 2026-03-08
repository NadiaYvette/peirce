# Full Semiotic Processing Pipeline
#
# Chains: sign formation → N layers of (sign-typed attention + FFN + GNN message passing) → realization
#
# Layer structure (for layers after transition point N):
#   1. Sign-typed attention (with residual + norm)
#   2. Feed-forward network (with residual + norm)
#   3. GNN message passing on inference graph
#   4. Sync: updated graph node embeddings feed back into sign representations
#
# Layers before transition point N use standard multi-head attention instead of sign-typed.

"""
    StandardAttentionLayer(d_model; num_heads=8)

Standard multi-head attention for early layers (before sign-typed transition).
"""
struct StandardAttentionLayer
    to_q::Dense
    to_k::Dense
    to_v::Dense
    out_proj::Dense
    norm::LayerNorm
    d_model::Int
    num_heads::Int
    d_head::Int
end

Flux.@layer StandardAttentionLayer

function StandardAttentionLayer(d_model::Int; num_heads::Int=8)
    d_head = d_model ÷ num_heads
    StandardAttentionLayer(
        Dense(d_model => d_model, bias=false),
        Dense(d_model => d_model, bias=false),
        Dense(d_model => d_model, bias=false),
        Dense(d_model => d_model),
        LayerNorm(d_model),
        d_model,
        num_heads,
        d_head
    )
end

function (layer::StandardAttentionLayer)(x::AbstractMatrix)
    # x: d_model × K
    K = size(x, 2)
    d_h = layer.d_head
    H = layer.num_heads
    scale = Float32(sqrt(d_h))

    q = layer.to_q(x)  # d_model × K
    k = layer.to_k(x)
    v = layer.to_v(x)

    # Reshape to (d_head, K, num_heads) and compute attention per head
    q_h = reshape(q, d_h, H, K)  # d_head × H × K
    k_h = reshape(k, d_h, H, K)
    v_h = reshape(v, d_h, H, K)

    # Attention scores per head: for each head h, compute (K × K) attention
    head_outputs = map(1:H) do h
        qh = q_h[:, h, :]  # d_head × K
        kh = k_h[:, h, :]
        vh = v_h[:, h, :]
        attn = softmax((qh' * kh) ./ scale; dims=2)  # K × K
        vh * attn'  # d_head × K
    end

    out = reduce(vcat, head_outputs)  # d_model × K
    layer.norm(layer.out_proj(out) .+ x)
end

"""
    FeedForward(d_model; expansion=4)

Standard transformer feed-forward block with residual + norm.
"""
struct FeedForward
    net::Chain
    norm::LayerNorm
end

Flux.@layer FeedForward

function FeedForward(d_model::Int; expansion::Int=4)
    FeedForward(
        Chain(
            Dense(d_model => d_model * expansion, gelu),
            Dense(d_model * expansion => d_model)
        ),
        LayerNorm(d_model)
    )
end

function (ff::FeedForward)(x::AbstractMatrix)
    ff.norm(ff.net(x) .+ x)
end

"""
    TruthValueModule(d_model)

Predicts 3-dim truth values from sign content + sign type:
  1. Semiotic confidence: how certain is this sign's classification?
  2. Truth estimate: is this sign's content true?
  3. Inferential weight: how much should this sign influence inference?

Output is sigmoid-bounded to [0, 1].
"""
struct TruthValueModule
    predictor::Chain
end

Flux.@layer TruthValueModule

function TruthValueModule(d_model::Int)
    TruthValueModule(
        Chain(
            Dense(d_model + NUM_SIGN_CLASSES => d_model ÷ 2, gelu),
            Dense(d_model ÷ 2 => 3)
        )
    )
end

function (tvm::TruthValueModule)(content::AbstractMatrix, sign_types::AbstractMatrix)
    # content: d_model × K, sign_types: 10 × K
    input = vcat(content, sign_types)       # (d_model + 10) × K
    sigmoid.(tvm.predictor(input))          # 3 × K, each in [0, 1]
end

"""
    compute_truth_targets(signs, graph, segment_weights)

Compute target truth values from available signals (non-differentiable).
Called inside Zygote.ignore() blocks.
Returns 3 × K target matrix.
"""
function compute_truth_targets(signs::SignBatch, graph::InferenceGraph,
                                segment_weights::AbstractMatrix)
    K = size(signs.content, 2)
    targets = zeros(Float32, 3, K)
    eps = Float32(1e-8)

    for k in 1:K
        # 1. Semiotic confidence: inverse entropy of sign-type distribution
        st = signs.sign_type[:, k]
        entropy = -sum(st .* log.(st .+ eps))
        max_entropy = log(Float32(NUM_SIGN_CLASSES))
        targets[1, k] = 1f0 - entropy / max_entropy

        # 2. Truth estimate: proportion of incoming commitment edges (vs total incoming)
        #    Dicisigns and arguments make truth claims; others get neutral
        trich = decompose_trichotomies(signs.sign_type[:, k:k])
        interp_level = sum(Float32[1, 2, 3] .* trich[3])  # expected interpretant level
        if interp_level >= 1.5  # dicisign or higher
            incoming = filter(e -> e.target == k || (e.target > K && false), graph.edges)
            if !isempty(incoming)
                commit_weight = sum(e.weight for e in incoming if e.edge_type == COMMITMENT; init=0f0)
                incompat_weight = sum(e.weight for e in incoming if e.edge_type == INCOMPATIBILITY; init=0f0)
                total = commit_weight + incompat_weight + eps
                targets[2, k] = commit_weight / total
            else
                targets[2, k] = 0.5f0  # no evidence either way
            end
        else
            targets[2, k] = 0.5f0  # rhemes don't assert truth
        end

        # 3. Inferential weight: slot occupancy (how many tokens this slot covers)
        slot_load = sum(segment_weights[k, :]) / size(segment_weights, 2)
        targets[3, k] = clamp(slot_load * Float32(K), 0f0, 1f0)  # normalized to [0, 1]
    end

    targets
end

"""
    InterpretantUpdate(d_model)

Refines interpretant slots via cross-attention to content.
Each sign's interpretant attends to all signs' content to capture
what this sign "means to" the other signs in context.
"""
struct InterpretantUpdate
    to_q::Dense       # interpretant → query
    to_k::Dense       # content → key
    to_v::Dense       # content → value
    update_proj::Dense
    gate_proj::Dense
    norm::LayerNorm
end

Flux.@layer InterpretantUpdate

function InterpretantUpdate(d_model::Int)
    InterpretantUpdate(
        Dense(d_model => d_model, bias=false),
        Dense(d_model => d_model, bias=false),
        Dense(d_model => d_model, bias=false),
        Dense(d_model => d_model),
        Dense(d_model => d_model),
        LayerNorm(d_model)
    )
end

function (iu::InterpretantUpdate)(interpretants::AbstractMatrix, content::AbstractMatrix)
    # interpretants: d_model × K, content: d_model × K
    d = size(content, 1)
    scale = Float32(sqrt(d))

    q = iu.to_q(interpretants)    # d_model × K
    k = iu.to_k(content)          # d_model × K
    v = iu.to_v(content)          # d_model × K

    attn = softmax((q' * k) ./ scale; dims=2)  # K × K
    update = v * attn'                          # d_model × K

    # Gated residual
    projected = iu.update_proj(update)
    gate = sigmoid.(iu.gate_proj(update .+ interpretants))
    iu.norm(gate .* projected .+ (1f0 .- gate) .* interpretants)
end

"""
    SemioticLayer

One processing layer after the sign-typed transition point.
Sign-typed attention → FFN → interpretant update → GNN message passing → sync.
"""
struct SemioticLayer
    attention::SignTypedAttentionLayer
    ffn::FeedForward
    interp_update::InterpretantUpdate
end

Flux.@layer SemioticLayer

function SemioticLayer(d_model::Int; heads_per_type::Int=2, ffn_expansion::Int=4)
    SemioticLayer(
        SignTypedAttentionLayer(d_model; heads_per_type),
        FeedForward(d_model; expansion=ffn_expansion),
        InterpretantUpdate(d_model)
    )
end

function (layer::SemioticLayer)(signs::SignBatch, graph::InferenceGraph)
    # 1. Sign-typed attention
    attended_content = layer.attention(signs)  # returns d_model × K (after residual+norm)

    # 2. FFN
    processed = layer.ffn(attended_content)  # d_model × K

    # 3. Interpretant update: cross-attention from interpretant to content
    updated_interp = layer.interp_update(signs.interpretant_slots, processed)

    # 4. GNN message passing (non-differentiable — graph is infrastructure)
    Zygote.ignore() do
        _sync_signs_to_graph!(graph, processed, signs.sign_type)
        predict_edges!(graph)
        message_pass!(graph)
    end

    # 5. Sync back: blend graph embeddings with processed signs
    updated_content = _sync_graph_to_signs(graph, processed, signs)

    # 6. Decorrelate: push apart similar slot embeddings and interpretants
    updated_content = decorrelate_slots(updated_content)
    updated_interp = decorrelate_slots(updated_interp)

    SignBatch(updated_content, signs.sign_type, signs.object_slots,
             updated_interp, signs.truth)
end

"""
    StandardLayer

One processing layer before the sign-typed transition point.
Standard attention → FFN → GNN message passing → sync.
"""
struct StandardLayer
    attention::StandardAttentionLayer
    ffn::FeedForward
end

Flux.@layer StandardLayer

function StandardLayer(d_model::Int; num_heads::Int=8, ffn_expansion::Int=4)
    StandardLayer(
        StandardAttentionLayer(d_model; num_heads),
        FeedForward(d_model; expansion=ffn_expansion)
    )
end

function (layer::StandardLayer)(signs::SignBatch, graph::InferenceGraph)
    # 1. Standard attention on content only
    attended = layer.attention(signs.content)

    # 2. FFN
    processed = layer.ffn(attended)

    # 3. GNN message passing (non-differentiable)
    Zygote.ignore() do
        _sync_signs_to_graph!(graph, processed, signs.sign_type)
        predict_edges!(graph)
        message_pass!(graph)
    end

    # 4. Sync back
    updated_content = _sync_graph_to_signs(graph, processed, signs)

    # 5. Decorrelate: push apart similar slot embeddings
    updated_content = decorrelate_slots(updated_content)

    SignBatch(updated_content, signs.sign_type, signs.object_slots,
             signs.interpretant_slots, signs.truth)
end

# --- Slot decorrelation ---

"""
    decorrelate_slots(content; strength=0.1f0)

Differentiable decorrelation: for each pair of slots with cosine similarity
above threshold, subtract a fraction of their shared component. This directly
counteracts the convergence pressure from self-attention over slots.
"""
function decorrelate_slots(content::AbstractMatrix; strength::Float32=Float32(0.15))
    d, K = size(content)
    K <= 1 && return content
    eps = Float32(1e-8)

    c_norms = sqrt.(sum(content .^ 2; dims=1) .+ eps)  # 1 × K
    c_normed = content ./ c_norms                        # d × K
    sim = c_normed' * c_normed                           # K × K

    # Only decorrelate pairs above threshold
    threshold = Float32(0.5)
    mask = Zygote.dropgrad(Float32.(1 .- I(K)))
    excess_sim = max.(sim .- threshold, Float32(0)) .* mask  # K × K

    # Each slot gets corrected by subtracting excess-similar slot projections
    # correction_k = sum_j [excess_sim(k,j) * content_j]
    correction = content * excess_sim  # d × K

    content .- strength .* correction
end

# --- Graph-sign synchronization ---

"""
Push current sign embeddings into the inference graph as nodes.
"""
function _sync_signs_to_graph!(graph::InferenceGraph, content::AbstractMatrix,
                                sign_types::AbstractMatrix)
    K = size(content, 2)
    for k in 1:K
        add_node!(graph, content[:, k], sign_types[:, k])
    end
end

"""
Blend graph node embeddings back into sign representations.
Uses soft attention: each sign attends to active graph nodes.
"""
function _sync_graph_to_signs(graph::InferenceGraph, content::AbstractMatrix,
                               signs::SignBatch)
    # Get graph embeddings as detached constants (no gradient through graph state)
    graph_embeds, graph_indices = Zygote.ignore() do
        get_active_embeddings(graph)
    end
    isempty(graph_indices) && return content

    # Detach graph embeddings from computation graph
    graph_embeds_detached = Zygote.dropgrad(graph_embeds)

    d, K = size(content)
    scale = Float32(sqrt(d))

    # Each sign attends to graph nodes — gradients flow through content only
    attn_logits = (content' * graph_embeds_detached) ./ scale  # K × num_active
    attn = softmax(attn_logits; dims=2)
    graph_context = graph_embeds_detached * attn'               # d × K

    # Gated blend: use inferential weight (truth[3,:]) to modulate graph influence
    # Scale by 0.2 so max graph influence is 20% (inferential weight is in [0,1])
    gate = Float32(0.2) .* signs.truth[3:3, :]  # 1 × K
    content .* (1f0 .- gate) .+ graph_context .* gate
end

# --- Full Pipeline ---

"""
    SemioticPipeline

The complete semiotic processing pipeline.

Components:
  - Sign formation module (tokens → signs)
  - N standard layers (early processing)
  - M semiotic layers (sign-typed processing)
  - Inference graph (maintained across layers)
  - Realization module (signs → tokens)
"""
struct SemioticPipeline
    sign_formation::SignFormationModule
    standard_layers::Vector{StandardLayer}
    semiotic_layers::Vector{SemioticLayer}
    truth_module::TruthValueModule
    realization::RealizationModule
    d_model::Int
    num_slots::Int
end

Flux.@layer SemioticPipeline

function SemioticPipeline(d_model::Int;
                           num_slots::Int=32,
                           num_standard_layers::Int=4,
                           num_semiotic_layers::Int=8,
                           num_heads::Int=8,
                           heads_per_type::Int=2,
                           ffn_expansion::Int=4,
                           slot_attention_iters::Int=3)
    SemioticPipeline(
        SignFormationModule(d_model, num_slots; num_iterations=slot_attention_iters),
        [StandardLayer(d_model; num_heads, ffn_expansion) for _ in 1:num_standard_layers],
        [SemioticLayer(d_model; heads_per_type, ffn_expansion) for _ in 1:num_semiotic_layers],
        TruthValueModule(d_model),
        RealizationModule(d_model),
        d_model,
        num_slots
    )
end

"""
    comprehend(pipeline, token_embeddings, rigid_table, token_strings;
               prior_signs=nothing, graph=nothing)

Run the comprehension half of the pipeline: tokens → sign representations.
Returns (final SignBatch, inference graph state).
"""
function comprehend(pipeline::SemioticPipeline,
                    token_embeddings::AbstractMatrix{Float32},
                    rigid_table::RigidDesignationTable,
                    token_strings::Vector{String};
                    prior_signs::Union{AbstractMatrix{Float32}, Nothing}=nothing,
                    graph::Union{InferenceGraph, Nothing}=nothing,
                    τ::Float32=Float32(1.0),
                    hard::Bool=true)
    # Initialize inference graph if not provided
    # NOTE: When called inside Flux.withgradient, graph must be pre-created
    # and passed in to avoid Zygote trying to differentiate through constructor.
    if isnothing(graph)
        graph = InferenceGraph(pipeline.d_model)
    end

    # Sign formation: tokens → signs (with Gumbel-Softmax segmentation)
    signs, segment_weights = forward(pipeline.sign_formation, token_embeddings, rigid_table, token_strings;
                                      prior_signs, τ, hard)

    # Standard layers (early processing)
    for layer in pipeline.standard_layers
        signs = layer(signs, graph)
    end

    # Semiotic layers (sign-typed processing)
    for layer in pipeline.semiotic_layers
        signs = layer(signs, graph)
    end

    # Predict truth values from refined content + sign types
    truth = pipeline.truth_module(signs.content, signs.sign_type)
    signs = SignBatch(signs.content, signs.sign_type, signs.object_slots,
                      signs.interpretant_slots, truth)

    (signs, graph, segment_weights)
end

"""
    DiscourseState

Accumulates sign representations across multiple passes of discourse processing.
Tracks the inference graph, accumulated signs, and segment boundaries.
"""
mutable struct DiscourseState
    graph::InferenceGraph
    accumulated_signs::Vector{SignBatch}     # signs from each segment
    segment_boundaries::Vector{Int}          # cumulative sign count per segment
    d_model::Int
end

Flux.@layer DiscourseState trainable=()

function DiscourseState(d_model::Int)
    DiscourseState(
        InferenceGraph(d_model),
        SignBatch[],
        Int[],
        d_model
    )
end

"""
    prior_content(state)

Returns the content matrix of all accumulated signs so far, or nothing if empty.
Used as `prior_signs` for the next comprehension pass.
"""
function prior_content(state::DiscourseState)
    isempty(state.accumulated_signs) && return nothing
    reduce(hcat, [s.content for s in state.accumulated_signs])  # d_model × total_K
end

"""
    comprehend_segment!(state, pipeline, token_embeddings, rigid_table, token_strings;
                        τ=1.0f0, hard=true)

Process one segment of discourse, using accumulated prior signs for context.
Updates the discourse state with new signs and graph edges.
Returns the SignBatch for this segment.
"""
function comprehend_segment!(state::DiscourseState,
                              pipeline::SemioticPipeline,
                              token_embeddings::AbstractMatrix{Float32},
                              rigid_table::RigidDesignationTable,
                              token_strings::Vector{String};
                              τ::Float32=Float32(1.0),
                              hard::Bool=true)
    prior = prior_content(state)

    signs, graph, seg_weights = comprehend(pipeline, token_embeddings, rigid_table, token_strings;
                                            prior_signs=prior, graph=state.graph, τ=τ, hard=hard)

    push!(state.accumulated_signs, signs)
    prev_total = isempty(state.segment_boundaries) ? 0 : state.segment_boundaries[end]
    push!(state.segment_boundaries, prev_total + size(signs.content, 2))

    (signs, seg_weights)
end

"""
    comprehend_discourse(pipeline, segments, rigid_table; τ=1.0f0, hard=true)

Process a full discourse as a sequence of segments. Each segment is a tuple
(token_embeddings, token_strings). Signs from earlier segments provide context
for later ones via `prior_signs`, and the inference graph accumulates across segments.

Returns (DiscourseState, Vector{SignBatch}, Vector{segment_weights}).
"""
function comprehend_discourse(pipeline::SemioticPipeline,
                               segments::Vector{<:Tuple{AbstractMatrix{Float32}, Vector{String}}},
                               rigid_table::RigidDesignationTable;
                               τ::Float32=Float32(1.0),
                               hard::Bool=true)
    state = DiscourseState(pipeline.d_model)
    all_signs = SignBatch[]
    all_seg_weights = []

    for (embeds, tokens) in segments
        signs, seg_w = comprehend_segment!(state, pipeline, embeds, rigid_table, tokens; τ=τ, hard=hard)
        push!(all_signs, signs)
        push!(all_seg_weights, seg_w)
    end

    (state, all_signs, all_seg_weights)
end

"""
    generate(pipeline, signs, graph, rigid_table;
             beam_width=4, recipient=nothing)

Run the generation half: sign representations → target sign structure → conditioning signal.
If recipient is provided, performs theory-of-mind reranking.
"""
function generate(pipeline::SemioticPipeline,
                  signs::SignBatch,
                  graph::InferenceGraph,
                  rigid_table::RigidDesignationTable;
                  beam_width::Int=4,
                  recipient::Union{RecipientModel, Nothing}=nothing)
    # Context embedding: mean-pool the input signs
    context_embed = vec(mean(signs.content; dims=2))

    # Discourse planning + Gricean filtering → conditioning signal
    conditioning = prepare_conditioning(pipeline.realization, graph, context_embed)

    # The actual token generation happens in the base model decoder (external).
    # We return the conditioning signal for cross-attention.
    # If ToM reranking is requested, the caller must:
    #   1. Generate `beam_width` candidates from the base model
    #   2. Call theory_of_mind_rerank with the comprehension pipeline

    conditioning
end

"""
    comprehension_for_tom(pipeline, rigid_table, recipient)

Returns a closure suitable for theory_of_mind_rerank's comprehension_fn argument.
Configures the pipeline to simulate a recipient's interpretation.
"""
function comprehension_for_tom(pipeline::SemioticPipeline,
                                rigid_table::RigidDesignationTable,
                                recipient::RecipientModel)
    function(tokens::Vector{String}, _recipient::RecipientModel)
        # Simulate limited context window
        effective_tokens = if length(tokens) > _recipient.context_window
            tokens[end-_recipient.context_window+1:end]
        else
            tokens
        end

        # Create dummy embeddings (in production, base model provides these)
        d = pipeline.d_model
        dummy_embeds = randn(Float32, d, length(effective_tokens))

        # Run comprehension with potentially reduced ontology access
        # (expertise modulates rigid designation — low expertise means fewer lookups succeed)
        signs, _, _ = comprehend(pipeline, dummy_embeds, rigid_table, effective_tokens)
        signs
    end
end
