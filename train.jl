# Phase 1 Training Script
#
# Usage:
#   1. First prepare data: python3.12 python/data_loader.py
#   2. Then train: JULIA_PKG_SERVER="" julia --project=. train.jl
#
# Projects 896-dim Qwen embeddings to smaller dimension for fast iteration,
# then trains the semiotic pipeline with reconstruction + regularizer losses.

include("fix7z.jl")
using Peirce
using JSON3
using NPZ
using BSON
using Random: randperm
using Statistics
using LinearAlgebra: qr

const DATA_DIR = joinpath(@__DIR__, "data", "embeddings_cache")
const CHECKPOINT_DIR = joinpath(@__DIR__, "checkpoints")

# Project dimension: 896 → this value for fast gradient compilation
const D_PROJECT = 128

function load_manifest()
    manifest_path = joinpath(DATA_DIR, "manifest.json")
    isfile(manifest_path) || error("No manifest found at $manifest_path. Run: python3.12 python/data_loader.py")
    JSON3.read(read(manifest_path, String))
end

function load_chunk(entry)
    embeddings = npzread(joinpath(DATA_DIR, entry.embeddings_file))
    tokens = String.(entry.tokens)
    (Float32.(embeddings), tokens)
end

function make_data_iterator(manifest, proj_matrix; shuffle::Bool=true)
    indices = collect(1:length(manifest))
    if shuffle
        indices = indices[randperm(length(indices))]
    end
    (let (emb, tok) = load_chunk(manifest[i])
        (proj_matrix * emb, tok)  # D_PROJECT × T
    end for i in indices)
end

function save_checkpoint(state::TrainingState, proj_matrix, epoch::Int, tag::String="phase1")
    mkpath(CHECKPOINT_DIR)
    path = joinpath(CHECKPOINT_DIR, "$(tag)_epoch$(epoch).bson")
    BSON.@save path pipeline=state.pipeline decoder=state.decoder step=state.step proj_matrix=proj_matrix
    println("  Saved checkpoint: $path")
end

function main()
    println("=== Peirce Phase 1 Training ===\n")

    # Load data
    manifest = load_manifest()
    println("Loaded $(length(manifest)) training chunks from cache")

    # Check d_model from first chunk
    test_embed, _ = load_chunk(manifest[1])
    d_orig = size(test_embed, 1)
    println("Original d_model = $d_orig, projecting to $D_PROJECT")

    # Random projection matrix (fixed, orthogonalized)
    proj_raw = randn(Float32, D_PROJECT, d_orig)
    # Orthogonalize rows via QR
    proj_matrix = Float32.(Matrix(qr(proj_raw').Q)[:, 1:D_PROJECT]')  # D_PROJECT × d_orig
    println("Projection matrix: $(size(proj_matrix))")

    # Create training state with projected dimension
    # High entropy weight to prevent sign-type collapse
    # High sparsity weight to encourage token specialization
    state = TrainingState(D_PROJECT;
        num_slots=16,
        num_standard_layers=2,
        num_semiotic_layers=4,
        max_tokens=128,
        learning_rate=3e-4,
        λ_sparse=Float32(1.0),
        λ_entropy=Float32(1.0),
        λ_consist=Float32(0.05)
    )
    println("Pipeline created: $(D_PROJECT)d, 16 slots, 2+4 layers")

    # Populate rigid designation table from corpus
    all_embeddings = Matrix{Float32}[]
    all_tokens = Vector{String}[]
    for entry in manifest
        emb, tok = load_chunk(entry)
        push!(all_embeddings, proj_matrix * emb)
        push!(all_tokens, tok)
    end
    populate_from_corpus!(state.rigid_table, all_embeddings, all_tokens)
    println("Rigid designation table: $(length(state.rigid_table.entries)) entries\n")

    # Training loop with Gumbel-Softmax temperature annealing
    # Start warm (τ=2.0, soft assignments) → anneal to cold (τ=0.3, nearly hard)
    num_epochs = 30
    τ_start = Float32(2.0)
    τ_end = Float32(0.3)

    for epoch in 1:num_epochs
        # Exponential temperature annealing
        progress = Float32((epoch - 1) / max(num_epochs - 1, 1))
        τ = τ_start * (τ_end / τ_start) ^ progress

        println("--- Epoch $epoch/$num_epochs (τ=$(round(τ; digits=3))) ---")
        data = make_data_iterator(manifest, proj_matrix)
        epoch_losses = train_epoch!(state, data; log_every=5, τ=τ)

        if epoch % 10 == 0
            save_checkpoint(state, proj_matrix, epoch)
        end
        println()
    end

    # Final checkpoint
    save_checkpoint(state, proj_matrix, num_epochs)
    println("Training complete. $(state.step) total steps.")
end

main()
