# Brandomian Inference Graph (GNN)
#
# Nodes: propositions with continuous embeddings
# Edges: inferential relations (commitment, entitlement, incompatibility)
#   with continuous weights
# Edge creation: type-conditional (Peircean sign-type validity constrains valid pairs)
# Lifecycle: fixed window, oldest evicted with summary embedding
# Coupling: message passing after each transformer layer

# Edge types in the inference graph
@enum EdgeType COMMITMENT ENTITLEMENT INCOMPATIBILITY

struct InferenceEdge
    source::Int
    target::Int
    edge_type::EdgeType
    weight::Float32
end

mutable struct InferenceGraph
    node_embeddings::Matrix{Float32}     # d_model × max_nodes
    node_sign_types::Matrix{Float32}     # 10 × max_nodes
    node_active::BitVector               # which slots are in use
    edges::Vector{InferenceEdge}
    summary_embedding::Vector{Float32}   # compressed representation of evicted nodes
    max_nodes::Int
    d_model::Int
    oldest_idx::Int                      # circular buffer pointer

    # GNN parameters
    message_fn::Dense       # transforms neighbor embeddings for message passing
    update_fn::Flux.GRUv3Cell    # updates node from aggregated messages
    edge_predictor::Dense   # predicts edge weight for valid pairs
    norm::LayerNorm
end

Flux.@layer InferenceGraph trainable=(message_fn, update_fn, edge_predictor, norm)

function InferenceGraph(d_model::Int; max_nodes::Int=256)
    InferenceGraph(
        zeros(Float32, d_model, max_nodes),
        zeros(Float32, NUM_SIGN_CLASSES, max_nodes),
        falses(max_nodes),
        InferenceEdge[],
        zeros(Float32, d_model),
        max_nodes,
        d_model,
        1,
        Dense(d_model => d_model),
        Flux.GRUv3Cell(d_model => d_model),
        Dense(2 * d_model => 3),  # predicts weight for each of 3 edge types
        LayerNorm(d_model)
    )
end

"""
    reset!(graph)

Reset graph state (nodes, edges) without reconstructing GNN parameters.
Safe to call inside Zygote.ignore() blocks.
"""
function reset!(graph::InferenceGraph)
    fill!(graph.node_embeddings, 0f0)
    fill!(graph.node_sign_types, 0f0)
    fill!(graph.node_active, false)
    empty!(graph.edges)
    fill!(graph.summary_embedding, 0f0)
    graph.oldest_idx = 1
    graph
end

"""
    add_node!(graph, embedding, sign_type) -> node_index

Add a sign as a node. If the graph is full, evict the oldest node
(compressing it into the summary embedding).
"""
function add_node!(graph::InferenceGraph, embedding::AbstractVector{Float32},
                   sign_type::AbstractVector{Float32})
    idx = _next_slot!(graph)
    graph.node_embeddings[:, idx] = embedding
    graph.node_sign_types[:, idx] = sign_type
    graph.node_active[idx] = true
    idx
end

function _next_slot!(graph::InferenceGraph)
    # Find first inactive slot
    for i in 1:graph.max_nodes
        if !graph.node_active[i]
            return i
        end
    end

    # Full — evict oldest
    idx = graph.oldest_idx
    _evict_node!(graph, idx)
    graph.oldest_idx = (graph.oldest_idx % graph.max_nodes) + 1
    idx
end

function _evict_node!(graph::InferenceGraph, idx::Int)
    # Compress evicted node into running summary via exponential moving average
    alpha = Float32(0.05)
    graph.summary_embedding .= (1f0 - alpha) .* graph.summary_embedding .+
                                alpha .* graph.node_embeddings[:, idx]
    graph.node_active[idx] = false

    # Remove edges involving this node
    filter!(e -> e.source != idx && e.target != idx, graph.edges)
end

# --- Type-conditional edge creation ---

# Which sign-type pairs can have which edge types?
# Derived from Peircean theory:
#   - Arguments (class 10) can have COMMITMENT edges to dicisigns (classes 4,7,9)
#   - Dicisigns can have INCOMPATIBILITY edges with other dicisigns
#   - Any sign can have ENTITLEMENT edges to signs of equal or lower complexity

"""
    edge_type_affinity(source_type, target_type, edge_type) -> Float32

Compute the probability-weighted affinity for an edge type between two signs.
Uses the full sign-type distributions (not argmax) so that all edge types
get some weight even when distributions are uncertain.
"""
function edge_type_affinity(source_type::AbstractVector, target_type::AbstractVector, et::EdgeType)
    # Compute expected validity across all class pairs, weighted by probabilities
    affinity = Float32(0)
    for (i, src_triple) in enumerate(SIGN_CLASSES)
        for (j, tgt_triple) in enumerate(SIGN_CLASSES)
            valid = if et == COMMITMENT
                # Source must be dicisign or argument, target has lower interpretant
                src_triple[3] >= 2 && tgt_triple[3] < src_triple[3]
            elseif et == INCOMPATIBILITY
                # Both dicisign+, different object relation
                src_triple[3] >= 2 && tgt_triple[3] >= 2 && src_triple[2] != tgt_triple[2]
            elseif et == ENTITLEMENT
                # Relaxed: target not more complex than source overall
                sum(tgt_triple) <= sum(src_triple)
            else
                false
            end
            if valid
                affinity += source_type[i] * target_type[j]
            end
        end
    end
    affinity
end

"""
    predict_edges!(graph)

For all pairs of active nodes, predict edge weights modulated by type affinity.
Uses probability-weighted validity instead of hard argmax decisions.
"""
function predict_edges!(graph::InferenceGraph; threshold::Float32=Float32(0.3))
    active = findall(graph.node_active)
    new_edges = InferenceEdge[]

    for i in active, j in active
        i == j && continue

        src_type = graph.node_sign_types[:, i]
        tgt_type = graph.node_sign_types[:, j]

        # Predict edge weights for all types
        pair_embed = vcat(graph.node_embeddings[:, i], graph.node_embeddings[:, j])
        logits = graph.edge_predictor(pair_embed)  # 3-dim

        for et in instances(EdgeType)
            affinity = edge_type_affinity(src_type, tgt_type, et)
            affinity < Float32(0.01) && continue  # skip negligible

            # Weight = predicted strength * type affinity
            weight = sigmoid(logits[Int(et) + 1]) * affinity

            if weight > threshold
                push!(new_edges, InferenceEdge(i, j, et, weight))
            end
        end
    end

    graph.edges = new_edges
end

"""
    message_pass!(graph; num_rounds=1)

One round of GNN message passing over active nodes.
Each node aggregates messages from neighbors, weighted by edge weight,
then updates via GRU.
"""
function message_pass!(graph::InferenceGraph; num_rounds::Int=1)
    active = findall(graph.node_active)
    isempty(active) && return

    for _ in 1:num_rounds
        # Compute messages for each active node
        new_embeddings = copy(graph.node_embeddings)

        for idx in active
            # Gather incoming edges
            incoming = filter(e -> e.target == idx, graph.edges)

            if isempty(incoming)
                # Include summary embedding as a weak message from evicted context
                msg = graph.message_fn(graph.summary_embedding) .* Float32(0.1)
            else
                # Weighted sum of transformed neighbor embeddings
                msg = zeros(Float32, graph.d_model)
                total_weight = Float32(0)
                for edge in incoming
                    neighbor = graph.node_embeddings[:, edge.source]
                    # Incompatibility edges send negated messages
                    sign = edge.edge_type == INCOMPATIBILITY ? -1f0 : 1f0
                    msg .+= sign .* edge.weight .* graph.message_fn(neighbor)
                    total_weight += edge.weight
                end
                if total_weight > 0
                    msg ./= total_weight
                end
            end

            # GRU update
            new_embeddings[:, idx] = graph.update_fn(msg, graph.node_embeddings[:, idx])[1]
        end

        # Apply updates and normalize
        for idx in active
            graph.node_embeddings[:, idx] = graph.norm(new_embeddings[:, idx])
        end
    end
end

"""
    get_active_embeddings(graph) -> (embeddings, indices)

Return the embeddings and indices of all active nodes.
"""
function get_active_embeddings(graph::InferenceGraph)
    active = findall(graph.node_active)
    (graph.node_embeddings[:, active], active)
end
