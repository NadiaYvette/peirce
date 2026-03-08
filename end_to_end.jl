# End-to-End Integration Test
#
# Runs the full pipeline: live Qwen embeddings → semiotic processing → generation
# Uses BridgedBaseModel for real-time Python ↔ Julia communication.
#
# Usage: JULIA_PKG_SERVER="" julia --project=. end_to_end.jl [checkpoint_path]

include("fix7z.jl")
using Peirce
using Flux
using Statistics
using Printf
using LinearAlgebra: qr
using BSON

const D_PROJECT = 128

function main()
    println("=== Peirce End-to-End Integration Test ===\n")

    # 1. Connect to base model
    println("1. Starting base model bridge...")
    base_model = BridgedBaseModel()
    cfg = model_config(base_model)
    println("   Connected: $(cfg["model_name"]), d_model=$(cfg["d_model"])")

    d_orig = cfg["d_model"]

    # 2. Load or create projection matrix and semiotic pipeline
    checkpoint_path = length(ARGS) >= 1 ? ARGS[1] : joinpath("checkpoints", "phase2_final.bson")
    if !isfile(checkpoint_path)
        checkpoint_path = joinpath("checkpoints", "phase1_epoch30.bson")
    end

    local proj_matrix, pipeline, decoder
    loaded = false
    if isfile(checkpoint_path)
        println("\n2. Loading checkpoint: $checkpoint_path")
        try
            cp_data = BSON.load(checkpoint_path)
            proj_matrix = cp_data[:proj_matrix]
            pipeline = SemioticPipeline(D_PROJECT;
                num_slots=16, num_standard_layers=2, num_semiotic_layers=4)
            decoder = ReconstructionDecoder(D_PROJECT; max_tokens=128)
            Flux.loadmodel!(pipeline, cp_data[:pipeline])
            Flux.loadmodel!(decoder, cp_data[:decoder])
            println("   Loaded pipeline from step $(cp_data[:step])")
            loaded = true
        catch e
            println("   Checkpoint incompatible ($(typeof(e))), using fresh initialization")
            println("   (Retrain to create a compatible checkpoint)")
        end
    end
    if !loaded
        println("\n2. Using random initialization")
        proj_raw = randn(Float32, D_PROJECT, d_orig)
        proj_matrix = Float32.(Matrix(qr(proj_raw').Q)[:, 1:D_PROJECT]')
        pipeline = SemioticPipeline(D_PROJECT;
            num_slots=16, num_standard_layers=2, num_semiotic_layers=4)
        decoder = ReconstructionDecoder(D_PROJECT; max_tokens=128)
    end

    # 3. Test with sample texts
    test_texts = [
        "The cat sat on the mat.",
        "Mathematics is the language of nature.",
        "She promised to return the book tomorrow.",
    ]

    rigid_table = RigidDesignationTable(D_PROJECT)

    # Build rigid table from test texts
    all_emb = Matrix{Float32}[]
    all_tok = Vector{String}[]
    for text in test_texts
        emb, tok = get_embeddings(base_model, text)
        projected = proj_matrix * emb
        push!(all_emb, projected)
        push!(all_tok, tok)
    end
    populate_from_corpus!(rigid_table, all_emb, all_tok)
    println("   Rigid designators: $(length(rigid_table.entries)) entries")

    println("\n3. Processing test texts through semiotic pipeline...\n")

    for (i, text) in enumerate(test_texts)
        println("=" ^ 60)
        println("  Input $i: \"$text\"")
        println("=" ^ 60)

        # Get live embeddings from Qwen
        raw_emb, tokens = get_embeddings(base_model, text)
        embeddings = proj_matrix * raw_emb
        T = length(tokens)

        println("  Tokens ($T): $(join(tokens, " | "))")
        println("  Raw embedding shape: $(size(raw_emb))")
        println("  Projected shape: $(size(embeddings))")

        # Run semiotic comprehension
        graph = InferenceGraph(D_PROJECT)
        signs, graph, seg_weights = comprehend(pipeline, embeddings, rigid_table,
                                                String.(tokens);
                                                graph=graph, hard=false, τ=Float32(0.5))

        # Sign-type analysis
        K = size(signs.sign_type, 2)
        sign_class_names = [
            "rhem.icon.quali", "rhem.icon.sin", "rhem.idx.sin", "dic.idx.sin",
            "rhem.icon.legi", "rhem.idx.legi", "dic.idx.legi",
            "rhem.sym.legi", "dic.sym.legi", "arg.sym.legi",
        ]

        println("\n  Slot assignments:")
        for t in 1:min(T, 20)
            col = seg_weights[:, t]
            top_slot = argmax(col)
            @printf("    %-12s → slot %2d (%.2f)\n", tokens[t], top_slot, col[top_slot])
        end

        println("\n  Sign types (top class per slot):")
        for k in 1:K
            dist = signs.sign_type[:, k]
            top_idx = argmax(dist)
            count = sum(argmax(seg_weights[:, t]) == k for t in 1:T)
            if count > 0
                @printf("    Slot %2d (%d tok): %-20s (%.2f)\n",
                        k, count, sign_class_names[top_idx], dist[top_idx])
            end
        end

        # Truth values
        println("\n  Truth values:")
        for k in 1:K
            count = sum(argmax(seg_weights[:, t]) == k for t in 1:T)
            count > 0 || continue
            conf = signs.truth[1, k]
            tru = signs.truth[2, k]
            inf_w = signs.truth[3, k]
            @printf("    Slot %2d: conf=%.3f  truth=%.3f  inf_wt=%.3f\n", k, conf, tru, inf_w)
        end

        # Reconstruction quality
        reconstructed = decoder(signs, T)
        mse = mean((reconstructed .- embeddings) .^ 2)
        @printf("\n  Reconstruction MSE: %.4f\n", mse)

        # Inference graph
        active = sum(graph.node_active)
        println("  Graph: $active active nodes, $(length(graph.edges)) edges")

        println()
    end

    # 4. Test generation (unconditioned vs conditioned)
    println("=" ^ 60)
    println("  4. Generation Test")
    println("=" ^ 60)

    prompt = "The meaning of the word 'sacrifice' is"
    println("  Prompt: \"$prompt\"\n")

    candidates = generate_candidates(base_model, prompt; num_candidates=2, max_tokens=25)
    println("  Unconditioned candidates:")
    for (i, c) in enumerate(candidates)
        println("    $i: $c")
    end

    # Get semiotic analysis and use as conditioning
    raw_emb, tokens = get_embeddings(base_model, prompt)
    embeddings = proj_matrix * raw_emb
    graph = InferenceGraph(D_PROJECT)
    signs, graph_out, seg_weights = comprehend(pipeline, embeddings, rigid_table,
                                                String.(tokens);
                                                graph=graph, hard=false, τ=Float32(0.5))
    K = size(signs.sign_type, 2)

    println("\n  Semiotic analysis of prompt:")
    for t in 1:length(tokens)
        col = seg_weights[:, t]
        top_slot = argmax(col)
        @printf("    %-15s → slot %2d (%.2f)\n", tokens[t], top_slot, col[top_slot])
    end

    # Conditioned generation: use sign content as conditioning signal
    cond_candidates = generate_conditioned(base_model, prompt, Float32.(signs.content);
                                            proj_matrix=Float32.(proj_matrix),
                                            num_candidates=2, max_tokens=25)
    println("\n  Conditioned candidates (semiotic soft-prompts):")
    for (i, c) in enumerate(cond_candidates)
        println("    $i: $c")
    end

    # Content similarity structure
    content = signs.content
    norms = sqrt.(sum(content .^ 2; dims=1) .+ 1f-8)
    normed = content ./ norms
    sim = normed' * normed
    avg_sim = (sum(sim) - K) / (K * (K - 1))
    @printf("\n  Slot content avg cosine sim: %.3f\n", avg_sim)

    @printf("  Mean truth: conf=%.3f truth=%.3f inf_wt=%.3f\n",
            mean(signs.truth[1, :]), mean(signs.truth[2, :]), mean(signs.truth[3, :]))

    # Edge type distribution
    edge_types = Dict{String, Int}()
    for edge in graph_out.edges
        k = string(edge.edge_type)
        edge_types[k] = get(edge_types, k, 0) + 1
    end
    println("\n  Edge type distribution:")
    total_edges = sum(values(edge_types))
    for (k, v) in sort(collect(edge_types); by=last, rev=true)
        @printf("    %-20s: %4d (%5.1f%%)\n", k, v, 100.0 * v / max(total_edges, 1))
    end

    # Cleanup
    println("\n--- Shutting down base model ---")
    shutdown!(base_model)
    println("Done!")
end

main()
