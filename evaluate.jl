# Downstream Evaluation Suite
#
# Tests semiotic pipeline properties that should emerge from training:
#   1. Sign-type linguistic alignment — do sign types correlate with syntactic categories?
#   2. Semantic similarity — do slot representations capture meaning?
#   3. Discourse coherence — does multi-pass comprehension produce structured graphs?
#   4. Conditioned generation — does semiotic conditioning improve topical relevance?
#   5. Trichotomy validity — do predicted types obey Peircean constraints?
#   6. Rigid designation stability — are proper names consistent across contexts?
#
# Usage: JULIA_PKG_SERVER="" julia --project=. evaluate.jl [checkpoint_path]

include("fix7z.jl")
using Peirce
using Flux
using Statistics
using Printf
using LinearAlgebra: qr, norm
using BSON

const D_PROJECT = 128

# --- Test Data ---

# Minimal pairs for semantic similarity
const SIMILARITY_PAIRS = [
    # (text_a, text_b, expected_similarity)  — "high" or "low"
    ("The cat sat on the mat.", "The dog lay on the rug.", "high"),
    ("The cat sat on the mat.", "Markets rallied on strong earnings.", "low"),
    ("She promised to return the book.", "She vowed to bring back the novel.", "high"),
    ("She promised to return the book.", "The temperature dropped overnight.", "low"),
    ("Mathematics is the language of nature.", "Algebra describes natural patterns.", "high"),
    ("Mathematics is the language of nature.", "The chef prepared a delicious meal.", "low"),
]

# Texts with known syntactic categories for sign-type alignment
const SIGN_TYPE_PROBES = [
    # (text, token_index (1-based), expected_category)
    # Categories: "function" (articles, prepositions) or "content" (nouns, verbs, adjectives)
    ("The cat sat on the mat.", "The", "function"),
    ("The cat sat on the mat.", " cat", "content"),
    ("The cat sat on the mat.", " sat", "content"),
    ("The cat sat on the mat.", " on", "function"),
    ("The cat sat on the mat.", " mat", "content"),
    ("She promised to return the book tomorrow.", " promised", "content"),
    ("She promised to return the book tomorrow.", " to", "function"),
    ("She promised to return the book tomorrow.", " book", "content"),
    ("She promised to return the book tomorrow.", " tomorrow", "content"),
]

# Discourse pairs for coherence testing
const DISCOURSE_PAIRS = [
    ["Water is composed of hydrogen and oxygen.", "This makes it essential for all life."],
    ["The economy grew by three percent.", "Unemployment fell to historic lows."],
    ["She studied for months.", "The exam was difficult but she passed."],
]

# Texts for rigid designation (proper names should be stable)
const RIGID_DESIGNATION_TESTS = [
    ("Einstein developed the theory of relativity.", "Einstein"),
    ("Einstein was born in Germany.", "Einstein"),
    ("The theory of relativity was proposed by Einstein.", "Einstein"),
    ("Hamlet spoke to the ghost of his father.", "Hamlet"),
    ("The tragedy of Hamlet is set in Denmark.", "Hamlet"),
    ("Hamlet decided to feign madness.", "Hamlet"),
    ("Newton formulated the laws of motion.", "Newton"),
    ("Newton discovered gravity while observing an apple.", "Newton"),
    ("London is the capital of England.", "London"),
    ("The fog rolled over London at dusk.", "London"),
]

# Trichotomy constraint test texts
const TRICHOTOMY_TEXTS = [
    "The cat is on the mat.",
    "Therefore it follows that the mat supports the cat.",
    "Red is a warm color.",
    "This particular scratch proves that a cat was here.",
]

function load_model(checkpoint_path, d_orig)
    local proj_matrix, pipeline, decoder
    if isfile(checkpoint_path)
        try
            cp_data = BSON.load(checkpoint_path)
            proj_matrix = cp_data[:proj_matrix]
            pipeline = SemioticPipeline(D_PROJECT;
                num_slots=16, num_standard_layers=2, num_semiotic_layers=4)
            decoder = ReconstructionDecoder(D_PROJECT; max_tokens=128)
            Flux.loadmodel!(pipeline, cp_data[:pipeline])
            Flux.loadmodel!(decoder, cp_data[:decoder])
            println("Loaded checkpoint: $checkpoint_path (step $(cp_data[:step]))")
            return proj_matrix, pipeline, decoder
        catch e
            println("Checkpoint incompatible: $e")
        end
    end
    println("Using random initialization")
    proj_raw = randn(Float32, D_PROJECT, d_orig)
    proj_matrix = Float32.(Matrix(qr(proj_raw').Q)[:, 1:D_PROJECT]')
    pipeline = SemioticPipeline(D_PROJECT;
        num_slots=16, num_standard_layers=2, num_semiotic_layers=4)
    decoder = ReconstructionDecoder(D_PROJECT; max_tokens=128)
    return proj_matrix, pipeline, decoder
end

function comprehend_text(pipeline, base_model, proj_matrix, text, rigid_table)
    raw_emb, tokens = get_embeddings(base_model, text)
    embeddings = proj_matrix * raw_emb
    graph = InferenceGraph(D_PROJECT)
    signs, graph, seg_weights = comprehend(pipeline, embeddings, rigid_table,
                                            String.(tokens);
                                            graph=graph, hard=false, τ=Float32(0.3))
    (signs, graph, seg_weights, tokens, embeddings)
end

function content_cosine_sim(signs_a, signs_b)
    # Pool slot contents into single vector (weighted by inferential weight)
    function pool(signs)
        w = signs.truth[3:3, :]  # 1 × K
        w_norm = w ./ (sum(w) + 1f-8)
        vec(signs.content * w_norm')  # d_model
    end
    a = pool(signs_a)
    b = pool(signs_b)
    sum(a .* b) / (norm(a) * norm(b) + 1f-8)
end

# --- Evaluation 1: Semantic Similarity ---

function eval_semantic_similarity(pipeline, base_model, proj_matrix, rigid_table)
    println("\n" * "=" ^ 60)
    println("  EVAL 1: Semantic Similarity")
    println("=" ^ 60)

    high_sims = Float32[]
    low_sims = Float32[]

    for (text_a, text_b, expected) in SIMILARITY_PAIRS
        signs_a, _, _, _, _ = comprehend_text(pipeline, base_model, proj_matrix, text_a, rigid_table)
        signs_b, _, _, _, _ = comprehend_text(pipeline, base_model, proj_matrix, text_b, rigid_table)

        sim = content_cosine_sim(signs_a, signs_b)
        label = expected == "high" ? "✓ high" : "✗ low "
        @printf("  %s (%.3f): \"%s\" ↔ \"%s\"\n", label, sim,
                first(text_a, 30), first(text_b, 30))

        if expected == "high"
            push!(high_sims, sim)
        else
            push!(low_sims, sim)
        end
    end

    mean_high = mean(high_sims)
    mean_low = mean(low_sims)
    separation = mean_high - mean_low
    @printf("\n  Mean high-sim: %.3f, Mean low-sim: %.3f, Separation: %.3f\n",
            mean_high, mean_low, separation)
    @printf("  PASS: %s (separation > 0 means semantically similar pairs are closer)\n",
            separation > 0 ? "YES" : "NO")

    separation
end

# --- Evaluation 2: Sign-Type Linguistic Alignment ---

function eval_sign_type_alignment(pipeline, base_model, proj_matrix, rigid_table)
    println("\n" * "=" ^ 60)
    println("  EVAL 2: Sign-Type Linguistic Alignment")
    println("=" ^ 60)

    sign_class_names = [
        "rhem.icon.quali", "rhem.icon.sin", "rhem.idx.sin", "dic.idx.sin",
        "rhem.icon.legi", "rhem.idx.legi", "dic.idx.legi",
        "rhem.sym.legi", "dic.sym.legi", "arg.sym.legi",
    ]

    function_types = Dict{Int, Int}()  # sign class → count for function words
    content_types = Dict{Int, Int}()   # sign class → count for content words

    for (text, target_token, category) in SIGN_TYPE_PROBES
        signs, _, seg_weights, tokens, _ = comprehend_text(pipeline, base_model, proj_matrix, text, rigid_table)

        # Find the token index
        tok_idx = findfirst(==(target_token), tokens)
        isnothing(tok_idx) && continue

        # Find which slot this token belongs to
        slot = argmax(seg_weights[:, tok_idx])
        top_type = argmax(signs.sign_type[:, slot])
        conf = signs.sign_type[top_type, slot]

        dict = category == "function" ? function_types : content_types
        dict[top_type] = get(dict, top_type, 0) + 1

        @printf("  %-12s (%s) → slot %2d → %-20s (%.2f)\n",
                target_token, category, slot, sign_class_names[top_type], conf)
    end

    println("\n  Function word type distribution:")
    for (k, v) in sort(collect(function_types); by=last, rev=true)
        @printf("    %-20s: %d\n", sign_class_names[k], v)
    end
    println("  Content word type distribution:")
    for (k, v) in sort(collect(content_types); by=last, rev=true)
        @printf("    %-20s: %d\n", sign_class_names[k], v)
    end

    # Check if function and content words get different type distributions
    f_set = Set(keys(function_types))
    c_set = Set(keys(content_types))
    overlap = length(intersect(f_set, c_set))
    total = length(union(f_set, c_set))
    distinctness = 1.0 - overlap / max(total, 1)
    @printf("  Type distinctness: %.2f (1.0 = completely different types)\n", distinctness)

    distinctness
end

# --- Evaluation 3: Discourse Coherence ---

function eval_discourse_coherence(pipeline, base_model, proj_matrix, rigid_table)
    println("\n" * "=" ^ 60)
    println("  EVAL 3: Discourse Coherence")
    println("=" ^ 60)

    for (i, segments) in enumerate(DISCOURSE_PAIRS)
        println("\n  Discourse $i:")
        for s in segments
            println("    \"$s\"")
        end

        # Process each segment and check graph connectivity
        all_signs = SignBatch[]
        total_edges = 0
        edge_types = Dict{String, Int}()

        prev_content = nothing
        for (j, text) in enumerate(segments)
            signs, graph, _, tokens, _ = comprehend_text(pipeline, base_model, proj_matrix, text, rigid_table)
            push!(all_signs, signs)

            for edge in graph.edges
                k = string(edge.edge_type)
                edge_types[k] = get(edge_types, k, 0) + 1
            end
            total_edges += length(graph.edges)

            # Cross-segment similarity (should be higher for coherent discourse)
            if !isnothing(prev_content)
                cross_sim = content_cosine_sim(all_signs[end-1], signs)
                @printf("    Segment %d→%d cross-similarity: %.3f\n", j-1, j, cross_sim)
            end
            prev_content = signs.content
        end

        @printf("    Total edges: %d\n", total_edges)
        for (k, v) in sort(collect(edge_types); by=last, rev=true)
            @printf("      %-20s: %d (%.1f%%)\n", k, v, 100.0 * v / max(total_edges, 1))
        end
    end
end

# --- Evaluation 4: Trichotomy Validity ---

function eval_trichotomy_validity(pipeline, base_model, proj_matrix, rigid_table)
    println("\n" * "=" ^ 60)
    println("  EVAL 4: Trichotomy Validity")
    println("=" ^ 60)

    total_slots = 0
    violations = 0

    for text in TRICHOTOMY_TEXTS
        signs, _, _, tokens, _ = comprehend_text(pipeline, base_model, proj_matrix, text, rigid_table)
        K = size(signs.sign_type, 2)
        trich1, trich2, trich3 = decompose_trichotomies(signs.sign_type)

        levels = Float32[1, 2, 3]
        for k in 1:K
            e1 = sum(levels .* trich1[:, k])
            e2 = sum(levels .* trich2[:, k])
            e3 = sum(levels .* trich3[:, k])
            total_slots += 1

            # Peirce constraint: II ≤ I and III ≤ II (as expected values)
            v21 = max(e2 - e1, 0)
            v32 = max(e3 - e2, 0)
            if v21 > 0.1 || v32 > 0.1
                violations += 1
            end
        end

        println("  \"$(first(text, 50))\"")
        # Show one slot as example
        k = 1
        e1 = sum(levels .* trich1[:, k])
        e2 = sum(levels .* trich2[:, k])
        e3 = sum(levels .* trich3[:, k])
        @printf("    Slot 1: E[I]=%.2f E[II]=%.2f E[III]=%.2f %s\n",
                e1, e2, e3, e2 <= e1 + 0.1 && e3 <= e2 + 0.1 ? "✓" : "✗")
    end

    violation_rate = violations / max(total_slots, 1)
    @printf("\n  Trichotomy violations: %d/%d (%.1f%%)\n", violations, total_slots, 100 * violation_rate)
    @printf("  PASS: %s (< 20%% violations)\n", violation_rate < 0.2 ? "YES" : "NO")

    violation_rate
end

# --- Evaluation 5: Conditioned Generation ---

function eval_conditioned_generation(pipeline, base_model, proj_matrix, rigid_table)
    println("\n" * "=" ^ 60)
    println("  EVAL 5: Conditioned Generation")
    println("=" ^ 60)

    test_cases = [
        ("Complete the sentence:", "The scientist discovered a new species of"),
        ("Answer the question:", "What is the capital of France?"),
    ]

    for (context, prompt) in test_cases
        println("\n  Context: \"$context\"")
        println("  Prompt:  \"$prompt\"")

        # Get semiotic conditioning from context
        ctx_signs, ctx_graph, _, _, _ = comprehend_text(pipeline, base_model, proj_matrix, context, rigid_table)

        # Use pipeline's realization module to prepare conditioning
        context_embed = vec(mean(ctx_signs.content; dims=2))
        conditioning = prepare_conditioning(pipeline.realization, ctx_graph, context_embed)

        # Generate with and without conditioning
        uncond = generate_candidates(base_model, prompt; num_candidates=2, max_tokens=25)
        cond = generate_conditioned(base_model, prompt, Float32.(conditioning);
                                     proj_matrix=Float32.(proj_matrix),
                                     num_candidates=2, max_tokens=25)

        println("  Unconditioned:")
        for (i, c) in enumerate(uncond)
            println("    $i: $(first(c, 70))")
        end
        println("  Conditioned:")
        for (i, c) in enumerate(cond)
            println("    $i: $(first(c, 70))")
        end
    end
end

# --- Evaluation 6: Rigid Designation Stability ---

function eval_rigid_designation(pipeline, base_model, proj_matrix, rigid_table)
    println("\n" * "=" ^ 60)
    println("  EVAL 6: Rigid Designation Stability")
    println("=" ^ 60)

    # Group by designator
    designators = Dict{String, Vector{Tuple{String, Int}}}()
    for (text, name) in RIGID_DESIGNATION_TESTS
        lst = get!(designators, name, Tuple{String, Int}[])
        push!(lst, (text, 0))
    end

    for (name, entries) in designators
        println("\n  Designator: \"$name\"")
        embeddings_list = Vector{Float32}[]

        for (text, _) in entries
            signs, _, seg_weights, tokens, _ = comprehend_text(pipeline, base_model, proj_matrix, text, rigid_table)

            # Find token matching the designator
            tok_idx = findfirst(t -> startswith(strip(t), name), tokens)
            if isnothing(tok_idx)
                # Try partial match
                tok_idx = findfirst(t -> occursin(lowercase(first(name, 4)), lowercase(t)), tokens)
            end
            isnothing(tok_idx) && continue

            slot = argmax(seg_weights[:, tok_idx])
            content = signs.content[:, slot]
            push!(embeddings_list, vec(content))

            @printf("    \"%s\" → slot %d\n", first(text, 50), slot)
        end

        if length(embeddings_list) >= 2
            # Compute pairwise cosine similarities
            sims = Float32[]
            for i in 1:length(embeddings_list)
                for j in (i+1):length(embeddings_list)
                    a, b = embeddings_list[i], embeddings_list[j]
                    sim = sum(a .* b) / (norm(a) * norm(b) + 1f-8)
                    push!(sims, sim)
                end
            end
            @printf("    Cross-context similarity: mean=%.3f, min=%.3f\n",
                    mean(sims), minimum(sims))
        end
    end
end

# --- Main ---

function main()
    println("=== Peirce Downstream Evaluation ===\n")

    # Connect to base model
    base_model = BridgedBaseModel()
    cfg = model_config(base_model)
    println("Base model: $(cfg["model_name"]), d_model=$(cfg["d_model"])")

    # Load checkpoint
    checkpoint_path = length(ARGS) >= 1 ? ARGS[1] : joinpath("checkpoints", "phase1_epoch30.bson")
    proj_matrix, pipeline, decoder = load_model(checkpoint_path, cfg["d_model"])

    # Build rigid table from test texts
    rigid_table = RigidDesignationTable(D_PROJECT)
    all_emb = Matrix{Float32}[]
    all_tok = Vector{String}[]
    test_texts = vcat(
        [p[1] for p in SIMILARITY_PAIRS], [p[2] for p in SIMILARITY_PAIRS],
        [p[1] for p in SIGN_TYPE_PROBES],
        vcat(DISCOURSE_PAIRS...),
        TRICHOTOMY_TEXTS,
        [t[1] for t in RIGID_DESIGNATION_TESTS],
    ) |> unique
    for text in test_texts
        emb, tok = get_embeddings(base_model, text)
        push!(all_emb, proj_matrix * emb)
        push!(all_tok, tok)
    end
    populate_from_corpus!(rigid_table, all_emb, all_tok)
    println("Rigid table: $(length(rigid_table.entries)) entries\n")

    # Run evaluations
    sim_score = eval_semantic_similarity(pipeline, base_model, proj_matrix, rigid_table)
    type_score = eval_sign_type_alignment(pipeline, base_model, proj_matrix, rigid_table)
    eval_discourse_coherence(pipeline, base_model, proj_matrix, rigid_table)
    trich_violations = eval_trichotomy_validity(pipeline, base_model, proj_matrix, rigid_table)
    eval_conditioned_generation(pipeline, base_model, proj_matrix, rigid_table)
    eval_rigid_designation(pipeline, base_model, proj_matrix, rigid_table)

    # Summary
    println("\n" * "=" ^ 60)
    println("  EVALUATION SUMMARY")
    println("=" ^ 60)
    @printf("  Semantic similarity separation: %.3f %s\n", sim_score, sim_score > 0 ? "✓" : "✗")
    @printf("  Sign-type distinctness:         %.2f %s\n", type_score, type_score > 0.3 ? "✓" : "✗")
    @printf("  Trichotomy violation rate:       %.1f%% %s\n", 100*trich_violations, trich_violations < 0.2 ? "✓" : "✗")
    @printf("  Rigid designation entries:       %d\n", length(rigid_table.entries))
    println()

    shutdown!(base_model)
    println("Done!")
end

main()
