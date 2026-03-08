# Peirce's 10 valid sign classes.
# Each is a combination of three trichotomies:
#   I.   Sign-in-itself:      qualisign (1), sinsign (2), legisign (3)
#   II.  Sign-to-object:      icon (1), index (2), symbol (3)
#   III. Sign-to-interpretant: rheme (1), dicisign (2), argument (3)
#
# Validity constraint: each trichotomy value <= the previous trichotomy value
# (using the numeric encoding above). This yields 10 of 27 combinations.

const NUM_SIGN_CLASSES = 10

# The 10 valid classes, ordered by their trichotomy triples.
const SIGN_CLASSES = [
    # (itself, object, interpretant)  — name
    (1, 1, 1),  # 1: rhematic iconic qualisign
    (2, 1, 1),  # 2: rhematic iconic sinsign
    (2, 2, 1),  # 3: rhematic indexical sinsign
    (2, 2, 2),  # 4: dicent indexical sinsign
    (3, 1, 1),  # 5: rhematic iconic legisign
    (3, 2, 1),  # 6: rhematic indexical legisign
    (3, 2, 2),  # 7: dicent indexical legisign
    (3, 3, 1),  # 8: rhematic symbolic legisign
    (3, 3, 2),  # 9: dicent symbolic legisign
    (3, 3, 3),  # 10: argument symbolic legisign
]

# Decompose the 10-class distribution back into three trichotomy distributions.
# Used by sign-typed attention for grouped routing.
#
# Each trichotomy marginal is obtained by summing the 10-class probabilities
# over classes that share a given trichotomy value.

# Precomputed masks: which of the 10 classes have trichotomy T = value V?
# trichotomy_masks[T][V] is a BitVector over the 10 classes.
const TRICHOTOMY_MASKS = let
    masks = [[falses(NUM_SIGN_CLASSES) for _ in 1:3] for _ in 1:3]
    for (i, (t1, t2, t3)) in enumerate(SIGN_CLASSES)
        masks[1][t1][i] = true
        masks[2][t2][i] = true
        masks[3][t3][i] = true
    end
    masks
end

"""
    decompose_trichotomies(sign_type_dist) -> (itself, object, interpretant)

Given a 10-dim sign-type distribution (or batch: 10×B matrix),
return three marginal distributions over each trichotomy (each 3-dim).
"""
function decompose_trichotomies(dist::AbstractVecOrMat)
    map(1:3) do t
        reduce(vcat, [sum(dist[TRICHOTOMY_MASKS[t][v], :], dims=1) for v in 1:3])
    end
end

"""
    SignRepresentation

The structured representation of a single sign. Batched as arrays of these
or as parallel arrays (see SignBatch).
"""
struct SignRepresentation
    content::AbstractVector{Float32}       # d_model-dim content embedding
    sign_type::AbstractVector{Float32}     # 10-dim softmax distribution
    object_slot::AbstractVector{Float32}   # d_model-dim reference embedding
    interpretant_slot::AbstractVector{Float32}  # d_model-dim (initially zeros)
    truth::NTuple{3, Float32}              # (semiotic_confidence, truth_estimate, inferential_weight)
end

"""
    SignBatch

Batched sign representations as parallel matrices. K signs, each d_model-dimensional.
"""
struct SignBatch
    content::AbstractMatrix{Float32}        # d_model × K
    sign_type::AbstractMatrix{Float32}      # 10 × K
    object_slots::AbstractMatrix{Float32}   # d_model × K
    interpretant_slots::AbstractMatrix{Float32}  # d_model × K
    truth::AbstractMatrix{Float32}          # 3 × K
end

num_signs(b::SignBatch) = size(b.content, 2)
embed_dim(b::SignBatch) = size(b.content, 1)
