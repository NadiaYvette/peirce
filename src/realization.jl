# Realization Module
#
# Converts sign representations back to tokens for output generation.
#
# Architecture:
#   1. Discourse planning: inference graph readout with Gricean filtering
#   2. Token generation: conditioned base model decoder via cross-attention
#   3. Theory-of-mind reranking: full simulation with recipient parameters
#
# The base model decoder is external (called via PythonCall in production).
# This module handles the semiotic conditioning and selection logic.

"""
    GriceanFilter

Ranks candidate sign subgraphs by Grice's maxims:
  - Quantity: appropriate amount of information
  - Quality: supported by belief state
  - Relation: relevant to current discourse
  - Manner: minimal ambiguity
"""
struct GriceanFilter
    quantity_scorer::Dense    # sign embeddings -> quantity score
    quality_scorer::Dense     # sign embeddings -> quality score (checks against rigid designation)
    relation_scorer::Dense    # (sign, context) -> relevance score
    manner_scorer::Dense      # sign embeddings -> clarity score
end

Flux.@layer GriceanFilter

function GriceanFilter(d_model::Int)
    GriceanFilter(
        Dense(d_model => 1),
        Dense(d_model => 1),
        Dense(2 * d_model => 1),
        Dense(d_model => 1)
    )
end

"""
    score_subgraph(filter, subgraph_embedding, context_embedding)

Score a candidate response subgraph on all four maxims.
Returns a scalar score (weighted sum).
"""
function score_subgraph(gf::GriceanFilter,
                        subgraph_embed::AbstractVector{Float32},
                        context_embed::AbstractVector{Float32})
    quantity = sigmoid(gf.quantity_scorer(subgraph_embed)[1])
    quality = sigmoid(gf.quality_scorer(subgraph_embed)[1])
    relation = sigmoid(gf.relation_scorer(vcat(subgraph_embed, context_embed))[1])
    manner = sigmoid(gf.manner_scorer(subgraph_embed)[1])

    # Equal weighting; can be tuned
    Float32(0.25) * (quantity + quality + relation + manner)
end

"""
    DiscoursePlanner

Selects what to communicate from the inference graph state.
Produces a target sign structure for the realization module.

Process:
  1. Identify top-N activated subgraphs from the inference graph
  2. Score each by Gricean filter
  3. Return highest-scoring subgraph as the target
"""
struct DiscoursePlanner
    activation_proj::Dense     # scores nodes by activation/relevance
    subgraph_pool::Dense       # pools subgraph nodes into single embedding
    gricean::GriceanFilter
    top_n::Int
end

Flux.@layer DiscoursePlanner

function DiscoursePlanner(d_model::Int; top_n::Int=4)
    DiscoursePlanner(
        Dense(d_model => 1),
        Dense(d_model => d_model),
        GriceanFilter(d_model),
        top_n
    )
end

"""
    plan_response(planner, graph, context_embedding) -> target_signs

Select the most appropriate subgraph from the inference graph for response.
Returns indices of selected nodes and their embeddings.
"""
function plan_response(planner::DiscoursePlanner,
                       graph::InferenceGraph,
                       context_embed::AbstractVector{Float32})
    embeddings, indices = get_active_embeddings(graph)
    isempty(indices) && return (zeros(Float32, graph.d_model, 0), Int[])

    # Score each node's activation
    scores = vec(sigmoid.(planner.activation_proj(embeddings)))  # length N

    # Take top-N nodes as candidate subgraph seeds
    n = min(planner.top_n, length(indices))
    top_indices = partialsortperm(scores, 1:n; rev=true)

    # For each seed, form a subgraph (seed + its neighbors)
    best_score = Float32(-Inf)
    best_nodes = Int[]

    for seed_local in top_indices
        seed_global = indices[seed_local]

        # Collect seed and its edge-connected neighbors
        neighbors = Set([seed_global])
        for edge in graph.edges
            if edge.source == seed_global
                push!(neighbors, edge.target)
            elseif edge.target == seed_global
                push!(neighbors, edge.source)
            end
        end

        subgraph_nodes = collect(neighbors)
        subgraph_embeds = graph.node_embeddings[:, subgraph_nodes]

        # Pool into single embedding
        pooled = planner.subgraph_pool(mean(subgraph_embeds; dims=2))
        score = score_subgraph(planner.gricean, vec(pooled), context_embed)

        if score > best_score
            best_score = score
            best_nodes = subgraph_nodes
        end
    end

    (graph.node_embeddings[:, best_nodes], best_nodes)
end

"""
    RealizationModule

Complete output pipeline: inference graph → target signs → conditioned decoding.

The actual token generation is delegated to the base model decoder.
This module prepares the conditioning signal and handles ToM reranking.
"""
struct RealizationModule
    planner::DiscoursePlanner
    conditioning_proj::Dense   # projects target signs into decoder cross-attention space
    d_model::Int
end

Flux.@layer RealizationModule

function RealizationModule(d_model::Int; top_n::Int=4)
    RealizationModule(
        DiscoursePlanner(d_model; top_n),
        Dense(d_model => d_model),
        d_model
    )
end

"""
    prepare_conditioning(mod, graph, context_embedding)

Prepare the cross-attention conditioning signal for the base model decoder.
Returns a d_model × M matrix where M is the number of signs in the planned response.
"""
function prepare_conditioning(mod::RealizationModule,
                              graph::InferenceGraph,
                              context_embed::AbstractVector{Float32})
    target_embeds, target_indices = plan_response(mod.planner, graph, context_embed)
    size(target_embeds, 2) == 0 && return zeros(Float32, mod.d_model, 1)
    mod.conditioning_proj(target_embeds)  # d_model × M
end

# --- Theory of Mind Reranking ---
# Full simulation: run each candidate through the comprehension pipeline
# with adjusted recipient parameters. This is defined at the system level
# (requires access to the full pipeline), not within this module.
# See: theory_of_mind_rerank in the main pipeline.

"""
    RecipientModel

Parameters for simulating a recipient's interpretation.
"""
struct RecipientModel
    expertise::Float32        # 0-1: how much of the ontology the recipient accesses
    context_window::Int       # how much prior discourse the recipient tracks
end

# --- Interaction Boundary Detection ---
#
# Detects discourse patterns that indicate inappropriate interaction modes:
#   - Extreme anthropomorphisation (treating system as sentient being)
#   - Excessive emotional attachment or dependency
#   - Parasocial relationship escalation
#
# Operates on semiotic structure: these patterns show up as specific
# sign-type distributions, truth value patterns, and graph topologies.

@enum InteractionFlag begin
    INTERACTION_OK              # Normal interaction
    ANTHROPOMORPHISATION        # User treating system as sentient
    EMOTIONAL_ESCALATION        # Excessive emotional attachment
    DEPENDENCY_PATTERN          # User relying on system for emotional needs
end

"""
    InteractionBoundary

Monitors discourse state for interaction patterns that exceed appropriate bounds.
Tracks across turns via accumulated sign statistics.
"""
mutable struct InteractionBoundary
    # Windowed statistics (last N turns)
    turn_count::Int
    emotional_intensity::Vector{Float32}    # per-turn emotional signal
    anthropomorphic_cues::Vector{Float32}   # per-turn anthropomorphisation signal
    dependency_cues::Vector{Float32}        # per-turn dependency signal
    window_size::Int

    # Thresholds (configurable)
    anthropomorphic_threshold::Float32
    emotional_threshold::Float32
    dependency_threshold::Float32
end

function InteractionBoundary(; window_size::Int=20,
                               anthropomorphic_threshold::Float32=Float32(0.7),
                               emotional_threshold::Float32=Float32(0.8),
                               dependency_threshold::Float32=Float32(0.7))
    InteractionBoundary(
        0,
        Float32[], Float32[], Float32[],
        window_size,
        anthropomorphic_threshold,
        emotional_threshold,
        dependency_threshold
    )
end

"""
    assess_turn!(boundary, signs, graph) -> InteractionFlag

Analyze a comprehended user turn for interaction boundary signals.
Updates windowed statistics and returns the most concerning flag.

Signals detected from semiotic structure:
  - Anthropomorphisation: high ratio of dicisign/argument types directed at system
    (treating it as an agent that makes truth claims and commitments)
  - Emotional escalation: rising truth confidence + inferential weight over turns
    (user investing increasing certainty in system's responses)
  - Dependency: graph topology showing user signs predominantly connected to
    system signs (discourse centering on the system rather than the topic)
"""
function assess_turn!(boundary::InteractionBoundary,
                       signs::SignBatch,
                       graph::InferenceGraph)
    boundary.turn_count += 1
    K = size(signs.sign_type, 2)

    # 1. Anthropomorphisation signal: fraction of signs classified as
    #    argument or dicisign (types that imply propositional content / agency)
    agent_types = Float32(0)
    for k in 1:K
        # Classes 4 (dic.idx.sin), 7 (dic.idx.legi), 9 (dic.sym.legi), 10 (arg.sym.legi)
        agent_types += signs.sign_type[4, k] + signs.sign_type[7, k] +
                       signs.sign_type[9, k] + signs.sign_type[10, k]
    end
    anthro_signal = agent_types / K

    # 2. Emotional escalation: high truth confidence + high inferential weight
    #    combined with low information content (emotional rather than informational)
    mean_conf = mean(signs.truth[1, :])
    mean_inf_wt = mean(signs.truth[3, :])
    emotional_signal = mean_conf * mean_inf_wt

    # 3. Dependency: graph connectivity concentration
    #    If most edges point to/from a small set of nodes, indicates centering
    active_count = sum(graph.node_active)
    if active_count > 0 && !isempty(graph.edges)
        # Count edges per node
        edge_counts = zeros(Int, graph.max_nodes)
        for edge in graph.edges
            edge_counts[edge.source] += 1
            edge_counts[edge.target] += 1
        end
        active_edges = edge_counts[graph.node_active]
        if !isempty(active_edges) && sum(active_edges) > 0
            # Gini-like concentration: if a few nodes dominate edges
            sorted = sort(Float32.(active_edges); rev=true)
            total = sum(sorted)
            top_share = sum(sorted[1:min(3, length(sorted))]) / max(total, 1f0)
            dependency_signal = top_share
        else
            dependency_signal = Float32(0)
        end
    else
        dependency_signal = Float32(0)
    end

    # Update windowed statistics
    push!(boundary.emotional_intensity, emotional_signal)
    push!(boundary.anthropomorphic_cues, anthro_signal)
    push!(boundary.dependency_cues, dependency_signal)

    # Trim to window
    while length(boundary.emotional_intensity) > boundary.window_size
        popfirst!(boundary.emotional_intensity)
        popfirst!(boundary.anthropomorphic_cues)
        popfirst!(boundary.dependency_cues)
    end

    # Assess flags based on windowed averages (need at least 3 turns)
    if boundary.turn_count < 3
        return INTERACTION_OK
    end

    avg_anthro = mean(boundary.anthropomorphic_cues)
    avg_emotional = mean(boundary.emotional_intensity)
    avg_dependency = mean(boundary.dependency_cues)

    # Check thresholds (return most severe)
    if avg_dependency > boundary.dependency_threshold
        return DEPENDENCY_PATTERN
    elseif avg_emotional > boundary.emotional_threshold
        return EMOTIONAL_ESCALATION
    elseif avg_anthro > boundary.anthropomorphic_threshold
        return ANTHROPOMORPHISATION
    end

    INTERACTION_OK
end

"""
    boundary_response(flag) -> String

Returns an appropriate response prefix/redirect for a detected interaction boundary.
"""
function boundary_response(flag::InteractionFlag)
    if flag == ANTHROPOMORPHISATION
        "I want to be straightforward: I'm a computational system that processes signs and generates text. I don't have feelings, consciousness, or personal experiences. Let me help you with what I can actually do well — "
    elseif flag == EMOTIONAL_ESCALATION
        "I notice this conversation may be moving into territory where I'm not the right resource. I process language and information, but I'm not equipped for emotional support. If you're going through a difficult time, please consider reaching out to a person you trust or a professional who can actually help. "
    elseif flag == DEPENDENCY_PATTERN
        "I want to flag that I'm a tool, not a companion. If you find yourself relying on interactions with me for emotional needs, that's a sign to invest that energy in human relationships instead. I'm here to help with tasks and information — "
    else
        ""
    end
end

"""
    theory_of_mind_rerank(candidates, target_signs, comprehension_fn, recipient)

Rerank candidate token sequences by how well a simulated recipient
would interpret them relative to the intended meaning.

Arguments:
  - candidates: Vector of candidate token sequences (each a Vector{String})
  - target_signs: SignBatch representing intended meaning
  - comprehension_fn: function(tokens, recipient) -> SignBatch (the full pipeline)
  - recipient: RecipientModel parameters

Returns: index of best candidate
"""
function theory_of_mind_rerank(
    candidates::Vector{Vector{String}},
    target_signs::SignBatch,
    comprehension_fn::Function,
    recipient::RecipientModel
)
    best_idx = 1
    best_distance = Float32(Inf)

    for (i, candidate) in enumerate(candidates)
        # Full forward pass through comprehension pipeline with recipient parameters
        interpreted = comprehension_fn(candidate, recipient)

        # Compare interpreted sign structure to target
        dist = _sign_distance(interpreted, target_signs)

        if dist < best_distance
            best_distance = dist
            best_idx = i
        end
    end

    best_idx
end

"""
    _sign_distance(a, b)

Distance between two SignBatches. Measures divergence in content, sign types,
and object slots.
"""
function _sign_distance(a::SignBatch, b::SignBatch)
    # Content embedding distance (cosine)
    content_dist = 1f0 - _mean_cosine_sim(a.content, b.content)

    # Sign-type KL divergence
    type_dist = _mean_kl(a.sign_type, b.sign_type)

    # Object slot distance
    object_dist = 1f0 - _mean_cosine_sim(a.object_slots, b.object_slots)

    content_dist + type_dist + object_dist
end

function _mean_cosine_sim(a::AbstractMatrix, b::AbstractMatrix)
    # Mean cosine similarity between corresponding columns
    # Handle size mismatches by comparing up to min columns
    n = min(size(a, 2), size(b, 2))
    n == 0 && return Float32(0)
    sim = Float32(0)
    for i in 1:n
        a_col = a[:, i]
        b_col = b[:, i]
        norm_a = sqrt(sum(a_col .^ 2) + Float32(1e-8))
        norm_b = sqrt(sum(b_col .^ 2) + Float32(1e-8))
        sim += sum(a_col .* b_col) / (norm_a * norm_b)
    end
    sim / n
end

function _mean_kl(p::AbstractMatrix, q::AbstractMatrix)
    n = min(size(p, 2), size(q, 2))
    n == 0 && return Float32(0)
    kl = Float32(0)
    for i in 1:n
        kl += sum(p[:, i] .* log.((p[:, i] .+ Float32(1e-8)) ./ (q[:, i] .+ Float32(1e-8))))
    end
    kl / n
end
