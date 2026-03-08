# Sign-Typed Attention
#
# Three groups of attention heads, one per Peircean trichotomy:
#   Group 1 (sign-in-itself): qualisign/sinsign/legisign
#   Group 2 (sign-to-object): icon/index/symbol
#   Group 3 (sign-to-interpretant): rheme/dicisign/argument
#
# Each group has specialized heads with different attention kernels:
#   - Iconic heads: cosine similarity (structural resemblance)
#   - Indexical heads: distance-sensitive + referential similarity (causal/spatial proximity)
#   - Symbolic heads: learned key-query, no distance decay (convention-based)
#
# Within each group, the relevant trichotomy marginal from the sign-type distribution
# blends the specialized heads' outputs.

"""
    IconicAttentionHead(d_model, d_head)

Attention via raw embedding cosine similarity. No learned projections for scoring —
similarity is directly structural.
"""
struct IconicAttentionHead
    to_v::Dense       # value projection only
    d_head::Int
end

Flux.@layer IconicAttentionHead

function IconicAttentionHead(d_model::Int, d_head::Int)
    IconicAttentionHead(Dense(d_model => d_head, bias=false), d_head)
end

function (head::IconicAttentionHead)(signs::AbstractMatrix)
    # signs: d_model × K
    # Cosine similarity for attention scores
    norms = sqrt.(sum(signs .^ 2; dims=1) .+ Float32(1e-8))  # 1 × K
    normed = signs ./ norms                                     # d_model × K
    attn = normed' * normed                                     # K × K (cosine similarities)
    attn = softmax(attn; dims=2)
    vals = head.to_v(signs)                                     # d_head × K
    vals * attn'                                                # d_head × K
end

"""
    IndexicalAttentionHead(d_model, d_head, max_distance)

Distance-sensitive attention with referential similarity term.
Nearby signs attend more strongly; signs sharing object slot similarity get a bonus.
"""
struct IndexicalAttentionHead
    to_q::Dense
    to_k::Dense
    to_v::Dense
    distance_decay::AbstractVector{Float32}  # learned decay rates
    ref_proj::Dense   # projects object slots for referential similarity
    d_head::Int
end

Flux.@layer IndexicalAttentionHead

function IndexicalAttentionHead(d_model::Int, d_head::Int; max_distance::Int=128)
    IndexicalAttentionHead(
        Dense(d_model => d_head, bias=false),
        Dense(d_model => d_head, bias=false),
        Dense(d_model => d_head, bias=false),
        ones(Float32, max_distance) .* Float32(0.1),  # initial gentle decay
        Dense(d_model => d_head, bias=false),
        d_head
    )
end

function (head::IndexicalAttentionHead)(
    sign_content::AbstractMatrix,     # d_model × K
    object_slots::AbstractMatrix      # d_model × K
)
    K = size(sign_content, 2)
    scale = Float32(sqrt(head.d_head))

    queries = head.to_q(sign_content)    # d_head × K
    keys = head.to_k(sign_content)       # d_head × K
    vals = head.to_v(sign_content)       # d_head × K

    # Content-based attention
    content_attn = (queries' * keys) ./ scale  # K × K

    # Distance bias: positions i,j get a penalty proportional to |i-j|
    # Constructed functionally for Zygote compatibility
    max_dist = length(head.distance_decay)
    dist_bias = -[let d = min(abs(i - j), max_dist)
                      d > 0 ? head.distance_decay[d] : Float32(0)
                  end for i in 1:K, j in 1:K]

    # Referential similarity: bonus for signs sharing similar object slots
    ref_q = head.ref_proj(object_slots)   # d_head × K
    ref_norms = sqrt.(sum(ref_q .^ 2; dims=1) .+ Float32(1e-8))
    ref_normed = ref_q ./ ref_norms
    ref_sim = ref_normed' * ref_normed    # K × K

    attn = softmax(content_attn .+ dist_bias .+ ref_sim; dims=2)
    vals * attn'   # d_head × K
end

"""
    SymbolicAttentionHead(d_model, d_head)

Standard learned key-query attention with no distance decay.
Convention-based relations are position-independent.
"""
struct SymbolicAttentionHead
    to_q::Dense
    to_k::Dense
    to_v::Dense
    d_head::Int
end

Flux.@layer SymbolicAttentionHead

function SymbolicAttentionHead(d_model::Int, d_head::Int)
    SymbolicAttentionHead(
        Dense(d_model => d_head, bias=false),
        Dense(d_model => d_head, bias=false),
        Dense(d_model => d_head, bias=false),
        d_head
    )
end

function (head::SymbolicAttentionHead)(signs::AbstractMatrix)
    scale = Float32(sqrt(head.d_head))
    q = head.to_q(signs)   # d_head × K
    k = head.to_k(signs)   # d_head × K
    v = head.to_v(signs)   # d_head × K
    attn = softmax((q' * k) ./ scale; dims=2)  # K × K
    v * attn'   # d_head × K
end

"""
    SignTypedAttentionLayer(d_model; heads_per_type=2)

One layer of sign-typed attention with three head groups.
Each group has three specialized heads (iconic, indexical, symbolic)
blended by the relevant trichotomy marginal.
"""
struct SignTypedAttentionLayer
    # Group 1: sign-in-itself (qualisign=iconic-like, sinsign=indexical-like, legisign=symbolic-like)
    group1_iconic::Vector{IconicAttentionHead}
    group1_indexical::Vector{IndexicalAttentionHead}
    group1_symbolic::Vector{SymbolicAttentionHead}

    # Group 2: sign-to-object (icon, index, symbol)
    group2_iconic::Vector{IconicAttentionHead}
    group2_indexical::Vector{IndexicalAttentionHead}
    group2_symbolic::Vector{SymbolicAttentionHead}

    # Group 3: sign-to-interpretant (rheme≈iconic, dicisign≈indexical, argument≈symbolic)
    group3_iconic::Vector{IconicAttentionHead}
    group3_indexical::Vector{IndexicalAttentionHead}
    group3_symbolic::Vector{SymbolicAttentionHead}

    # Output projection: combines all heads
    out_proj::Dense
    norm::LayerNorm
    d_model::Int
    d_head::Int
end

Flux.@layer SignTypedAttentionLayer

function SignTypedAttentionLayer(d_model::Int; heads_per_type::Int=2)
    d_head = d_model ÷ (3 * heads_per_type)  # 3 groups × heads_per_type (blending reduces 3 types to 1)
    d_head = max(d_head, 8)  # minimum head dimension
    total_head_dim = 3 * heads_per_type * d_head

    make_iconic(n) = [IconicAttentionHead(d_model, d_head) for _ in 1:n]
    make_indexical(n) = [IndexicalAttentionHead(d_model, d_head) for _ in 1:n]
    make_symbolic(n) = [SymbolicAttentionHead(d_model, d_head) for _ in 1:n]

    SignTypedAttentionLayer(
        make_iconic(heads_per_type), make_indexical(heads_per_type), make_symbolic(heads_per_type),
        make_iconic(heads_per_type), make_indexical(heads_per_type), make_symbolic(heads_per_type),
        make_iconic(heads_per_type), make_indexical(heads_per_type), make_symbolic(heads_per_type),
        Dense(total_head_dim => d_model),
        LayerNorm(d_model),
        d_model,
        d_head
    )
end

function (layer::SignTypedAttentionLayer)(signs::SignBatch)
    content = signs.content              # d_model × K
    objects = signs.object_slots          # d_model × K
    interpretants = signs.interpretant_slots  # d_model × K
    sign_types = signs.sign_type         # 10 × K

    # Decompose sign-type distribution into trichotomy marginals
    trich1, trich2, trich3 = decompose_trichotomies(sign_types)  # each 3 × K

    # Process each group, blending by its trichotomy marginal
    # Group 1 (sign-in-itself): operates on content
    g1_out = _blend_group(layer.group1_iconic, layer.group1_indexical, layer.group1_symbolic,
                          content, objects, trich1)
    # Group 2 (sign-to-object): operates on content with object reference
    g2_out = _blend_group(layer.group2_iconic, layer.group2_indexical, layer.group2_symbolic,
                          content, objects, trich2)
    # Group 3 (sign-to-interpretant): operates on interpretant slots
    g3_out = _blend_group(layer.group3_iconic, layer.group3_indexical, layer.group3_symbolic,
                          interpretants, objects, trich3)

    # Concatenate all group outputs and project
    combined = vcat(g1_out, g2_out, g3_out)    # (3 * group_dim) × K
    output = layer.out_proj(combined)           # d_model × K

    # Residual + norm
    layer.norm(output .+ content)
end

function _blend_group(
    iconic_heads::Vector{<:IconicAttentionHead},
    indexical_heads::Vector{<:IndexicalAttentionHead},
    symbolic_heads::Vector{<:SymbolicAttentionHead},
    content::AbstractMatrix,
    objects::AbstractMatrix,
    trichotomy_dist::AbstractMatrix   # 3 × K
)
    # Run all heads
    iconic_out = reduce(vcat, [h(content) for h in iconic_heads])              # (n*d_head) × K
    indexical_out = reduce(vcat, [h(content, objects) for h in indexical_heads])  # (n*d_head) × K
    symbolic_out = reduce(vcat, [h(content) for h in symbolic_heads])            # (n*d_head) × K

    d = size(iconic_out, 1)

    # Blend by trichotomy weights: value 1 → iconic, value 2 → indexical, value 3 → symbolic
    w_iconic = trichotomy_dist[1:1, :]      # 1 × K
    w_indexical = trichotomy_dist[2:2, :]    # 1 × K
    w_symbolic = trichotomy_dist[3:3, :]     # 1 × K

    iconic_out .* w_iconic .+ indexical_out .* w_indexical .+ symbolic_out .* w_symbolic  # d × K
end
