# Phase 2 Training Script
#
# Builds on Phase 1 checkpoint. Adds semiotic structure losses:
#   - Slot contrastive: different sign types → different content embeddings
#   - Trichotomy consistency: marginals obey Peirce's validity constraints
#   - Sign-type prediction: types should be predictable from local context
#
# Usage:
#   JULIA_PKG_SERVER="" julia --project=. train_phase2.jl [checkpoint_path]

include("fix7z.jl")
using Peirce
using JSON3
using NPZ
using BSON
using Flux
using Random: randperm
using Zygote

using LinearAlgebra: qr

const DATA_DIR = joinpath(@__DIR__, "data", "embeddings_cache")
const CHECKPOINT_DIR = joinpath(@__DIR__, "checkpoints")
const D_PROJECT = 128

function load_manifest()
    manifest_path = joinpath(DATA_DIR, "manifest.json")
    isfile(manifest_path) || error("No manifest found. Run: python3.12 python/data_loader.py")
    JSON3.read(read(manifest_path, String))
end

function load_chunk(entry)
    embeddings = npzread(joinpath(DATA_DIR, entry.embeddings_file))
    tokens = String.(entry.tokens)
    (Float32.(embeddings), tokens)
end

function train_step_phase2!(state::TrainingState,
                            token_embeddings::AbstractMatrix{Float32},
                            token_strings::Vector{String};
                            λ_contrastive::Float32=Float32(0.1),
                            λ_trichotomy::Float32=Float32(0.2),
                            τ::Float32=Float32(0.5))
    T = size(token_embeddings, 2)
    graph = InferenceGraph(size(token_embeddings, 1))

    (loss_val, loss_dict), grads = Flux.withgradient(state.pipeline, state.decoder) do pipeline, decoder
        signs, _, seg_weights = comprehend(pipeline, token_embeddings, state.rigid_table, token_strings;
                                              graph=graph, τ=τ, hard=true)

        # Reconstruction losses
        reconstructed = decoder(signs, T)
        l_recon = reconstruction_loss(reconstructed, token_embeddings)

        l_seg_recon = segmentation_reconstruction_loss(signs, seg_weights, token_embeddings)
        l_entropy = sign_type_entropy_loss(signs.sign_type)

        # Phase 2 losses
        l_contrastive = slot_contrastive_loss(signs)
        l_trichotomy = trichotomy_consistency_loss(signs)
        l_diversity = slot_content_diversity_loss(signs)
        l_interp_div = interpretant_diversity_loss(signs)
        l_occupancy = slot_occupancy_loss(seg_weights)
        l_trich_balance = trichotomy_balance_loss(signs)
        l_proto_div = prototype_diversity_loss(pipeline.sign_formation.segmentation.prototypes)

        # Truth value loss
        truth_targets = Zygote.ignore() do
            compute_truth_targets(signs, graph, seg_weights)
        end
        l_truth = truth_value_loss(signs, truth_targets)

        total = Float32(0.5) * l_recon +
                Float32(5.0) * l_seg_recon +
                Float32(0.5) * l_entropy +
                λ_contrastive * l_contrastive +
                λ_trichotomy * l_trichotomy +
                Float32(50.0) * l_diversity +
                Float32(50.0) * l_interp_div +
                Float32(0.5) * l_occupancy +
                Float32(2.0) * l_trich_balance +
                Float32(20.0) * l_proto_div +
                Float32(2.0) * l_truth

        local_dict = Dict{String, Float32}(
            "total" => total,
            "reconstruction" => l_recon,
            "seg_recon" => l_seg_recon,
            "entropy" => l_entropy,
            "contrastive" => l_contrastive,
            "trichotomy" => l_trichotomy,
            "diversity" => l_diversity,
            "interp_div" => l_interp_div,
            "occupancy" => l_occupancy,
            "trich_balance" => l_trich_balance,
            "proto_div" => l_proto_div,
            "truth" => l_truth,
        )
        (total, local_dict)
    end

    Flux.update!(state.opt_state, (state.pipeline, state.decoder), grads)
    state.step += 1
    push!(state.losses, loss_dict)
    loss_dict
end

function main()
    println("=== Peirce Phase 2 Training ===\n")

    checkpoint_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(CHECKPOINT_DIR, "phase1_epoch30.bson")

    manifest = load_manifest()
    println("Loaded $(length(manifest)) training chunks")

    test_embed, _ = load_chunk(manifest[1])
    d_orig = size(test_embed, 1)

    # Load Phase 1 checkpoint to get projection matrix
    local proj_matrix
    if isfile(checkpoint_path)
        println("Loading Phase 1 checkpoint: $checkpoint_path")
        cp_data = BSON.load(checkpoint_path)
        proj_matrix = cp_data[:proj_matrix]
    else
        println("No Phase 1 checkpoint found. Creating projection matrix.")
        proj_raw = randn(Float32, D_PROJECT, d_orig)
        proj_matrix = Float32.(Matrix(qr(proj_raw').Q)[:, 1:D_PROJECT]')
    end

    # Create state with projected dimension
    state = TrainingState(D_PROJECT;
        num_slots=16, num_standard_layers=2, num_semiotic_layers=4,
        max_tokens=128, learning_rate=1e-4,
        λ_sparse=Float32(1.0), λ_entropy=Float32(1.0), λ_consist=Float32(0.0))

    # Load Phase 1 weights if available
    if isfile(checkpoint_path)
        Flux.loadmodel!(state.pipeline, cp_data[:pipeline])
        Flux.loadmodel!(state.decoder, cp_data[:decoder])
        println("Loaded. Continuing from step $(cp_data[:step]).")
        state.step = cp_data[:step]
    end

    # Populate rigid designation table from corpus
    all_embeddings = Matrix{Float32}[]
    all_tokens = Vector{String}[]
    for entry in manifest
        emb, tok = load_chunk(entry)
        push!(all_embeddings, proj_matrix * emb)
        push!(all_tokens, tok)
    end
    populate_from_corpus!(state.rigid_table, all_embeddings, all_tokens)
    println("Rigid designation table: $(length(state.rigid_table.entries)) entries")

    # Phase 2 hyperparameters — ramp up semiotic losses over epochs
    num_epochs = 30
    for epoch in 1:num_epochs
        # Ramp contrastive and trichotomy losses
        progress = Float32(min(epoch / 10, 1.0))
        λ_c = Float32(0.1) * progress
        λ_t = Float32(0.2) * progress

        println("--- Phase 2 Epoch $epoch/$num_epochs (λ_contrastive=$(round(λ_c; digits=3)), λ_trichotomy=$(round(λ_t; digits=3))) ---")

        indices = randperm(length(manifest))
        epoch_losses = Dict{String, Float64}()
        n = 0

        for i in indices
            raw_emb, tokens = load_chunk(manifest[i])
            embeddings = proj_matrix * raw_emb
            loss_dict = train_step_phase2!(state, embeddings, tokens;
                                           λ_contrastive=λ_c, λ_trichotomy=λ_t,
                                           τ=Float32(0.3))
            for (k, v) in loss_dict
                epoch_losses[k] = get(epoch_losses, k, 0.0) + v
            end
            n += 1

            if state.step % 5 == 0
                println("  Step $(state.step): total=$(round(loss_dict["total"]; digits=4)), " *
                        "recon=$(round(loss_dict["reconstruction"]; digits=4)), " *
                        "contrastive=$(round(loss_dict["contrastive"]; digits=4)), " *
                        "truth=$(round(loss_dict["truth"]; digits=4)), " *
                        "trich_bal=$(round(loss_dict["trich_balance"]; digits=4))")
            end
        end

        for k in keys(epoch_losses)
            epoch_losses[k] /= max(n, 1)
        end
        println("  Epoch avg: recon=$(round(epoch_losses["reconstruction"]; digits=4)), " *
                "contrastive=$(round(get(epoch_losses, "contrastive", 0.0); digits=4)), " *
                "trich_bal=$(round(get(epoch_losses, "trich_balance", 0.0); digits=4)), " *
                "proto=$(round(get(epoch_losses, "proto_div", 0.0); digits=4))")

        if epoch % 10 == 0
            mkpath(CHECKPOINT_DIR)
            path = joinpath(CHECKPOINT_DIR, "phase2_epoch$(epoch).bson")
            BSON.@save path pipeline=state.pipeline decoder=state.decoder step=state.step proj_matrix=proj_matrix
            println("  Saved: $path")
        end
        println()
    end

    # Final save
    mkpath(CHECKPOINT_DIR)
    path = joinpath(CHECKPOINT_DIR, "phase2_final.bson")
    BSON.@save path pipeline=state.pipeline decoder=state.decoder step=state.step proj_matrix=proj_matrix
    println("Phase 2 complete. $(state.step) total steps. Saved: $path")
end

main()
