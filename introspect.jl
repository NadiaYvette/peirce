# Introspection Script
#
# Loads a trained checkpoint (or trains from scratch) and visualizes:
#   1. Token → slot assignments (segmentation)
#   2. Sign-type distributions per slot
#   3. Trichotomy marginals
#   4. Slot content similarity structure
#   5. Reconstruction quality per token
#
# Usage: JULIA_PKG_SERVER="" julia --project=. introspect.jl [checkpoint_path]

include("fix7z.jl")
using Peirce
using JSON3
using NPZ
using BSON
using Statistics
using Printf
using LinearAlgebra
using Flux: softmax

const DATA_DIR = joinpath(@__DIR__, "data", "embeddings_cache")

const SIGN_CLASS_NAMES = [
    "rhematic iconic qualisign",
    "rhematic iconic sinsign",
    "rhematic indexical sinsign",
    "dicent indexical sinsign",
    "rhematic iconic legisign",
    "rhematic indexical legisign",
    "dicent indexical legisign",
    "rhematic symbolic legisign",
    "dicent symbolic legisign",
    "argument symbolic legisign",
]

const TRICH1_NAMES = ["qualisign", "sinsign", "legisign"]
const TRICH2_NAMES = ["icon", "index", "symbol"]
const TRICH3_NAMES = ["rheme", "dicisign", "argument"]

function load_sample(proj_matrix)
    manifest_path = joinpath(DATA_DIR, "manifest.json")
    manifest = JSON3.read(read(manifest_path, String))
    # Load first chunk
    entry = manifest[1]
    embeddings = Float32.(npzread(joinpath(DATA_DIR, entry.embeddings_file)))
    tokens = String.(entry.tokens)
    projected = isnothing(proj_matrix) ? embeddings : proj_matrix * embeddings
    (projected, tokens)
end

function load_checkpoint(checkpoint_path)
    if !isnothing(checkpoint_path) && isfile(checkpoint_path)
        println("Loading checkpoint: $checkpoint_path")
        data = BSON.load(checkpoint_path)
        proj_matrix = haskey(data, :proj_matrix) ? data[:proj_matrix] : nothing
        return data[:pipeline], data[:decoder], proj_matrix
    end
    error("No checkpoint found at $checkpoint_path. Run train.jl first.")
end

function print_header(title)
    println("\n", "="^60)
    println("  ", title)
    println("="^60)
end

function visualize_segmentation(pipeline, embeddings, tokens)
    print_header("1. TOKEN → SLOT ASSIGNMENTS")

    # Use soft segmentation for visualization
    seg_weights = pipeline.sign_formation.segmentation(embeddings; τ=Float32(0.1), hard=false)
    K, T = size(seg_weights)

    # Show top slot per token
    println("\nToken assignments (top slot, weight):")
    for t in 1:min(T, 40)
        col = seg_weights[:, t]
        top_slot = argmax(col)
        top_weight = col[top_slot]
        second_slot = argmax([i == top_slot ? -Inf : col[i] for i in 1:K])
        second_weight = col[second_slot]
        token_str = rpad(tokens[t], 15)
        @printf("  %s → slot %2d (%.3f)  slot %2d (%.3f)\n",
                token_str, top_slot, top_weight, second_slot, second_weight)
    end
    if T > 40
        println("  ... ($(T - 40) more tokens)")
    end

    # Slot occupancy: how many tokens primarily assigned to each slot?
    println("\nSlot occupancy (tokens with highest weight in each slot):")
    assignments = [argmax(seg_weights[:, t]) for t in 1:T]
    for k in 1:K
        count = sum(assignments .== k)
        member_tokens = tokens[assignments .== k]
        preview = join(member_tokens[1:min(5, length(member_tokens))], ", ")
        if length(member_tokens) > 5
            preview *= ", ..."
        end
        @printf("  Slot %2d: %3d tokens  [%s]\n", k, count, preview)
    end

    # Entropy of per-token distributions (lower = sparser = more specialized)
    entropies = [-sum(seg_weights[:, t] .* log.(seg_weights[:, t] .+ 1f-8)) for t in 1:T]
    @printf("\nSegmentation entropy: mean=%.3f, min=%.3f, max=%.3f (max possible=%.3f)\n",
            mean(entropies), minimum(entropies), maximum(entropies), log(K))

    seg_weights
end

function visualize_sign_types(signs)
    print_header("2. SIGN-TYPE DISTRIBUTIONS")

    K = size(signs.sign_type, 2)

    for k in 1:K
        dist = signs.sign_type[:, k]
        top_idx = argmax(dist)
        println("\n  Slot $k: top class = $(SIGN_CLASS_NAMES[top_idx]) ($(round(dist[top_idx]; digits=3)))")

        # Show full distribution as bar chart
        sorted_indices = sortperm(dist; rev=true)
        for i in sorted_indices
            bar_len = round(Int, dist[i] * 40)
            bar = "█"^bar_len
            if dist[i] > 0.01
                @printf("    %2d %-35s %.3f %s\n", i, SIGN_CLASS_NAMES[i], dist[i], bar)
            end
        end
    end

    # Diversity: entropy of mean distribution across slots
    mean_dist = vec(mean(signs.sign_type; dims=2))
    diversity_entropy = -sum(mean_dist .* log.(mean_dist .+ 1f-8))
    max_entropy = log(10)
    @printf("\nSign-type diversity: entropy=%.3f / %.3f (%.1f%% of max)\n",
            diversity_entropy, max_entropy, 100 * diversity_entropy / max_entropy)
end

function visualize_trichotomies(signs)
    print_header("3. TRICHOTOMY MARGINALS")

    trich1, trich2, trich3 = decompose_trichotomies(signs.sign_type)
    K = size(signs.sign_type, 2)

    for k in 1:K
        t1 = trich1[:, k]
        t2 = trich2[:, k]
        t3 = trich3[:, k]

        println("\n  Slot $k:")
        @printf("    I.  Sign-in-itself:      ")
        for (i, name) in enumerate(TRICH1_NAMES)
            @printf("%-11s=%.3f  ", name, t1[i])
        end
        @printf("\n    II. Sign-to-object:      ")
        for (i, name) in enumerate(TRICH2_NAMES)
            @printf("%-11s=%.3f  ", name, t2[i])
        end
        @printf("\n    III.Sign-to-interpretant:")
        for (i, name) in enumerate(TRICH3_NAMES)
            @printf("%-11s=%.3f  ", name, t3[i])
        end
        println()
    end
end

function visualize_slot_similarity(signs)
    print_header("4. SLOT CONTENT SIMILARITY")

    content = signs.content  # d_model × K
    K = size(content, 2)

    # Cosine similarity between slot contents
    norms = sqrt.(sum(content .^ 2; dims=1))
    normed = content ./ (norms .+ 1f-8)
    sim = normed' * normed  # K × K

    println("\nCosine similarity matrix (slots × slots):")
    print("        ")
    for j in 1:K
        @printf("  S%02d ", j)
    end
    println()
    for i in 1:K
        @printf("  S%02d  ", i)
        for j in 1:K
            val = sim[i, j]
            if i == j
                @printf("  --- ")
            elseif val > 0.8
                @printf(" %.2f*", val)
            else
                @printf(" %.2f ", val)
            end
        end
        println()
    end

    avg_off_diag = (sum(sim) - K) / (K * (K - 1))
    @printf("\nAvg off-diagonal similarity: %.3f (lower = more differentiated slots)\n", avg_off_diag)
end

function visualize_interpretants(signs)
    print_header("4b. INTERPRETANT SLOT ANALYSIS")

    content = signs.content                # d_model × K
    interp = signs.interpretant_slots      # d_model × K
    K = size(content, 2)

    # Content-interpretant similarity per slot
    println("\nPer-slot content ↔ interpretant cosine similarity:")
    eps = 1f-8
    for k in 1:K
        c = content[:, k]
        i = interp[:, k]
        c_norm = sqrt(sum(c .^ 2) + eps)
        i_norm = sqrt(sum(i .^ 2) + eps)
        sim = sum(c .* i) / (c_norm * i_norm)
        bar_len = round(Int, abs(sim) * 30)
        bar = sim >= 0 ? "█"^bar_len : "-"^bar_len
        @printf("  Slot %2d: sim=%.3f  %s\n", k, sim, bar)
    end

    # Interpretant diversity (are interpretants more/less diverse than content?)
    c_norms = sqrt.(sum(content .^ 2; dims=1) .+ eps)
    c_normed = content ./ c_norms
    i_norms = sqrt.(sum(interp .^ 2; dims=1) .+ eps)
    i_normed = interp ./ i_norms

    c_sim = c_normed' * c_normed
    i_sim = i_normed' * i_normed
    c_avg = (sum(c_sim) - K) / (K * (K - 1))
    i_avg = (sum(i_sim) - K) / (K * (K - 1))

    @printf("\nContent avg similarity:      %.3f\n", c_avg)
    @printf("Interpretant avg similarity: %.3f\n", i_avg)
    @printf("Difference: %.3f (%s)\n", i_avg - c_avg,
            i_avg < c_avg ? "interpretants more diverse" : "content more diverse")

    # Interpretant norms (are they being used or dead?)
    i_norms_vec = vec(sqrt.(sum(interp .^ 2; dims=1)))
    c_norms_vec = vec(sqrt.(sum(content .^ 2; dims=1)))
    @printf("\nNorm ratio (interp/content): mean=%.3f, min=%.3f, max=%.3f\n",
            mean(i_norms_vec ./ (c_norms_vec .+ eps)),
            minimum(i_norms_vec ./ (c_norms_vec .+ eps)),
            maximum(i_norms_vec ./ (c_norms_vec .+ eps)))
end

function visualize_reconstruction(pipeline, decoder, embeddings, tokens, rigid_table)
    print_header("5. RECONSTRUCTION QUALITY")

    T = size(embeddings, 2)
    graph = InferenceGraph(size(embeddings, 1))
    signs, _, _ = comprehend(pipeline, embeddings, rigid_table, String.(tokens); graph=graph, hard=false, τ=Float32(0.5))
    reconstructed = decoder(signs, T)

    # Per-token MSE
    per_token_mse = vec(mean((reconstructed .- embeddings) .^ 2; dims=1))

    println("\nPer-token reconstruction error (MSE):")
    sorted_idx = sortperm(per_token_mse)

    println("  Best reconstructed:")
    for i in sorted_idx[1:min(10, T)]
        @printf("    %-15s  MSE=%.4f\n", tokens[i], per_token_mse[i])
    end

    println("  Worst reconstructed:")
    for i in sorted_idx[max(1, T-9):T]
        @printf("    %-15s  MSE=%.4f\n", tokens[i], per_token_mse[i])
    end

    @printf("\nOverall: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n",
            mean(per_token_mse), std(per_token_mse),
            minimum(per_token_mse), maximum(per_token_mse))
end

function visualize_inference_graph(pipeline, embeddings, tokens, rigid_table)
    print_header("6. INFERENCE GRAPH STATE")

    graph = InferenceGraph(size(embeddings, 1))
    signs, graph, _ = comprehend(pipeline, embeddings, rigid_table, String.(tokens); graph=graph, hard=false, τ=Float32(0.5))

    active_count = sum(graph.node_active)
    println("\nActive nodes: $active_count / $(length(graph.node_active))")
    println("Edges: $(length(graph.edges))")

    if !isempty(graph.edges)
        println("\nEdge types:")
        edge_types = Dict{Peirce.EdgeType, Int}()
        for edge in graph.edges
            edge_types[edge.edge_type] = get(edge_types, edge.edge_type, 0) + 1
        end
        for (rel, count) in sort(collect(edge_types); by=last, rev=true)
            println("  $rel: $count")
        end

        println("\nSample edges (top 10 by weight):")
        sorted_edges = sort(graph.edges; by=e -> e.weight, rev=true)
        for edge in sorted_edges[1:min(10, length(sorted_edges))]
            @printf("  node %d →[%s]→ node %d  (weight=%.3f)\n",
                    edge.source, edge.edge_type, edge.target, edge.weight)
        end
    end
end

function main()
    checkpoint_path = length(ARGS) >= 1 ? ARGS[1] : joinpath("checkpoints", "phase1_epoch30.bson")

    println("=== Peirce Model Introspection ===\n")

    # Load model
    pipeline, decoder, proj_matrix = load_checkpoint(checkpoint_path)

    # Load sample data
    embeddings, tokens = load_sample(proj_matrix)
    d_model = size(embeddings, 1)
    T = size(embeddings, 2)
    println("Sample: $(T) tokens, d_model=$d_model")
    println("First tokens: ", join(tokens[1:min(20, T)], " "))

    # Populate rigid designation table
    rigid_table = RigidDesignationTable(d_model)
    populate_from_corpus!(rigid_table, [embeddings], [String.(tokens)])
    println("Rigid designators: $(length(rigid_table.entries)) entries")

    # Run introspection
    seg_weights = visualize_segmentation(pipeline, embeddings, tokens)

    # Get signs for visualization
    graph = InferenceGraph(d_model)
    signs, _, _ = comprehend(pipeline, embeddings, rigid_table, String.(tokens); graph=graph, hard=false, τ=Float32(0.5))

    visualize_sign_types(signs)
    visualize_trichotomies(signs)
    visualize_slot_similarity(signs)
    visualize_interpretants(signs)
    visualize_reconstruction(pipeline, decoder, embeddings, tokens, rigid_table)
    visualize_inference_graph(pipeline, embeddings, tokens, rigid_table)

    # Truth values
    print_header("TRUTH VALUES")
    println()
    println("Per-slot truth values (confidence, truth_estimate, inferential_weight):")
    truth = signs.truth  # 3 × K
    for k in 1:size(truth, 2)
        conf = truth[1, k]
        tru = truth[2, k]
        inf_w = truth[3, k]
        bar_c = repeat("█", round(Int, conf * 20))
        bar_t = repeat("█", round(Int, tru * 20))
        bar_i = repeat("█", round(Int, inf_w * 20))
        @printf("  Slot %2d: conf=%.3f %s  truth=%.3f %s  inf_wt=%.3f %s\n",
                k, conf, bar_c, tru, bar_t, inf_w, bar_i)
    end
    @printf("\n  Mean:    conf=%.3f  truth=%.3f  inf_wt=%.3f\n",
            mean(truth[1, :]), mean(truth[2, :]), mean(truth[3, :]))
    @printf("  Std:     conf=%.3f  truth=%.3f  inf_wt=%.3f\n",
            std(truth[1, :]), std(truth[2, :]), std(truth[3, :]))

    print_header("SUMMARY")
    K = size(signs.sign_type, 2)
    seg_entropies = [-sum(seg_weights[:, t] .* log.(seg_weights[:, t] .+ 1f-8)) for t in 1:T]
    mean_dist = vec(mean(signs.sign_type; dims=2))
    type_entropy = -sum(mean_dist .* log.(mean_dist .+ 1f-8))

    println()
    @printf("  Slots: %d\n", K)
    @printf("  Segmentation entropy: %.3f / %.3f (%.0f%% of max)\n",
            mean(seg_entropies), log(K), 100 * mean(seg_entropies) / log(K))
    @printf("  Sign-type diversity:  %.3f / %.3f (%.0f%% of max)\n",
            type_entropy, log(10), 100 * type_entropy / log(10))

    content = signs.content
    norms = sqrt.(sum(content .^ 2; dims=1))
    normed = content ./ (norms .+ 1f-8)
    sim = normed' * normed
    avg_sim = (sum(sim) - K) / (K * (K - 1))
    @printf("  Slot differentiation: avg cosine sim = %.3f\n", avg_sim)
    println()
end

main()
