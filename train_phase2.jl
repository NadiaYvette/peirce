# Phase 2 Training Script
#
# Builds on Phase 1 checkpoint. Adds semiotic structure losses:
#   - Slot contrastive: different sign types → different content embeddings
#   - Trichotomy consistency: marginals obey Peirce's validity constraints
#   - Sign-type prediction: types should be predictable from local context
#
# Usage:
#   JULIA_PKG_SERVER="" julia --project=. train_phase2.jl [checkpoint_path]
#   JULIA_PKG_SERVER="" julia --project=. train_phase2.jl --resume

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
    isfile(manifest_path) || error("No manifest found. Run: python3.12 python/corpus_loader.py")
    JSON3.read(read(manifest_path, String))
end

function load_chunk(entry)
    embeddings = npzread(joinpath(DATA_DIR, entry.embeddings_file))
    tokens = String.(entry.tokens)
    (Float32.(embeddings), tokens)
end

"""Memory bank for text-level contrastive learning."""
mutable struct ContrastiveMemoryBank
    representations::Matrix{Float32}  # d_model × max_size
    sources::Vector{String}           # source document for each entry
    categories::Vector{String}        # corpus category for each entry
    max_size::Int
    count::Int                        # number of entries filled
    idx::Int                          # next write position (circular)
end

function ContrastiveMemoryBank(d_model::Int; max_size::Int=256)
    ContrastiveMemoryBank(
        zeros(Float32, d_model, max_size),
        fill("", max_size),
        fill("", max_size),
        max_size, 0, 1
    )
end

function push_to_bank!(bank::ContrastiveMemoryBank, rep::Vector{Float32},
                       source::String, category::String)
    bank.representations[:, bank.idx] .= rep
    bank.sources[bank.idx] = source
    bank.categories[bank.idx] = category
    bank.idx = mod1(bank.idx + 1, bank.max_size)
    bank.count = min(bank.count + 1, bank.max_size)
end

function bank_data(bank::ContrastiveMemoryBank)
    n = bank.count
    n == 0 && return (zeros(Float32, size(bank.representations, 1), 0), String[], String[])
    (bank.representations[:, 1:n], bank.sources[1:n], bank.categories[1:n])
end

function train_step_phase2!(state::TrainingState,
                            token_embeddings::AbstractMatrix{Float32},
                            token_strings::Vector{String};
                            λ_contrastive::Float32=Float32(0.5),
                            λ_trichotomy::Float32=Float32(0.2),
                            λ_text_contrastive::Float32=Float32(1.0),
                            τ::Float32=Float32(0.5),
                            bank_reps::AbstractMatrix{Float32}=zeros(Float32, 0, 0),
                            bank_sources::Vector{String}=String[],
                            anchor_source::String="")
    T = size(token_embeddings, 2)
    graph = InferenceGraph(size(token_embeddings, 1))
    pooled_ref = Ref{Vector{Float32}}(Float32[])

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

        # Text-level contrastive loss (InfoNCE over pooled representations)
        l_text_contrastive = Float32(0)
        anchor_rep = pool_signs(signs)
        Zygote.ignore() do
            pooled_ref[] = Vector{Float32}(anchor_rep)
        end
        if size(bank_reps, 2) > 0 && !isempty(anchor_source)
            l_text_contrastive = text_level_contrastive_loss(
                anchor_rep, bank_reps, bank_sources, anchor_source)
        end

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
                Float32(2.0) * l_truth +
                λ_text_contrastive * l_text_contrastive

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
            "text_contrastive" => l_text_contrastive,
        )
        (total, local_dict)
    end

    Flux.update!(state.opt_state, (state.pipeline, state.decoder), grads)
    state.step += 1
    push!(state.losses, loss_dict)
    (loss_dict, pooled_ref[])
end

function find_latest_checkpoint(tag::String)
    mkpath(CHECKPOINT_DIR)
    candidates = filter(f -> startswith(f, "$(tag)_epoch") && endswith(f, ".bson"), readdir(CHECKPOINT_DIR))
    isempty(candidates) && return nothing

    # Use file modification time to find the most recently saved checkpoint
    # Matches both epoch-end (phase2_epoch5.bson) and mid-epoch (phase2_epoch5_step5000.bson)
    best_mtime = 0.0
    best_file = ""
    for f in candidates
        path = joinpath(CHECKPOINT_DIR, f)
        mt = mtime(path)
        if mt > best_mtime
            best_mtime = mt
            best_file = f
        end
    end
    best_mtime > 0 ? joinpath(CHECKPOINT_DIR, best_file) : nothing
end

function main()
    println("=== Peirce Phase 2 Training ===\n")

    resume = "--resume" in ARGS || "-r" in ARGS
    # Filter out flags from ARGS for checkpoint path
    positional = filter(a -> !startswith(a, "-"), ARGS)

    manifest = load_manifest()
    println("Loaded $(length(manifest)) training chunks")

    test_embed, _ = load_chunk(manifest[1])
    d_orig = size(test_embed, 1)

    num_epochs = 30
    start_epoch = 1

    local proj_matrix, state

    if resume
        cp_path = find_latest_checkpoint("phase2")
        if cp_path !== nothing
            println("Resuming Phase 2 from: $cp_path")
            cp_data = BSON.load(cp_path)
            proj_matrix = cp_data[:proj_matrix]

            state = TrainingState(D_PROJECT;
                num_slots=16, num_standard_layers=2, num_semiotic_layers=4,
                max_tokens=128, learning_rate=1e-4,
                λ_sparse=Float32(1.0), λ_entropy=Float32(1.0), λ_consist=Float32(0.0))

            Flux.loadmodel!(state.pipeline, cp_data[:pipeline])
            Flux.loadmodel!(state.decoder, cp_data[:decoder])
            state.step = cp_data[:step]
            if haskey(cp_data, :epoch)
                # Mid-epoch checkpoints include _step in filename; resume same epoch
                is_mid_epoch = occursin("_step", basename(cp_path))
                start_epoch = is_mid_epoch ? cp_data[:epoch] : cp_data[:epoch] + 1
            else
                start_epoch = (state.step ÷ length(manifest)) + 1
            end
            println("Resuming from epoch $start_epoch, step $(state.step)")
        else
            println("No Phase 2 checkpoint found, falling back to Phase 1")
            resume = false
        end
    end

    if !resume
        # Load from Phase 1 or specified checkpoint
        checkpoint_path = !isempty(positional) ? positional[1] : joinpath(CHECKPOINT_DIR, "phase1_epoch30.bson")

        # Also try phase1_final if the epoch30 doesn't exist
        if !isfile(checkpoint_path)
            alt = joinpath(CHECKPOINT_DIR, "phase1_final_epoch30.bson")
            if isfile(alt)
                checkpoint_path = alt
            else
                # Try to find the latest phase1 checkpoint
                latest = find_latest_checkpoint("phase1")
                if latest !== nothing
                    checkpoint_path = latest
                end
            end
        end

        if isfile(checkpoint_path)
            println("Loading Phase 1 checkpoint: $checkpoint_path")
            cp_data = BSON.load(checkpoint_path)
            proj_matrix = cp_data[:proj_matrix]
        else
            println("No Phase 1 checkpoint found. Creating projection matrix.")
            proj_raw = randn(Float32, D_PROJECT, d_orig)
            proj_matrix = Float32.(Matrix(qr(proj_raw').Q)[:, 1:D_PROJECT]')
        end

        state = TrainingState(D_PROJECT;
            num_slots=16, num_standard_layers=2, num_semiotic_layers=4,
            max_tokens=128, learning_rate=1e-4,
            λ_sparse=Float32(1.0), λ_entropy=Float32(1.0), λ_consist=Float32(0.0))

        if isfile(checkpoint_path)
            Flux.loadmodel!(state.pipeline, cp_data[:pipeline])
            Flux.loadmodel!(state.decoder, cp_data[:decoder])
            println("Loaded. Continuing from step $(cp_data[:step]).")
            state.step = cp_data[:step]
        end
    end

    # Populate rigid designation table from corpus (single-pass, memory-efficient)
    # Pass 1: lightweight scan for auto-discovery (only track term→doc counts, no embeddings)
    println("Auto-discovering rigid designators (lightweight scan)...")
    rigid_terms = Peirce.default_rigid_terms()
    foundational_terms = Peirce.default_foundational_terms()
    term_docs = Dict{String, Set{String}}()
    stopwords = Set(["The", "A", "An", "In", "On", "At", "To", "For", "Of",
                     "It", "He", "She", "We", "They", "But", "And", "Or",
                     "So", "If", "As", "By", "Is", "Was", "Are", "Were",
                     "No", "Not", "This", "That", "With", "From", "His",
                     "Her", "My", "Your", "Its", "All", "One", "Two",
                     "What", "When", "Where", "Who", "How", "Why",
                     "There", "Here", "Each", "Every", "Some", "Any",
                     "Do", "Did", "Has", "Had", "Have", "Will", "Would",
                     "Could", "Should", "May", "Might", "Must", "Shall",
                     "I", "You", "Me", "Him", "Us", "Them",
                     "Mr", "Mrs", "Ms", "Dr", "Sir", "Lord", "Lady",
                     "Chapter", "Act", "Scene", "Part", "Book", "Vol",
                     "Project", "Section", "Page", "Yes", "Oh", "Now",
                     "Then", "Well", "Come", "Let", "Yet", "Thus", "Still"])
    for entry in manifest
        source = String(entry.source)
        for tok in entry.tokens
            clean = strip(String(tok))
            length(clean) < 3 && continue
            fc = first(clean)
            !isuppercase(fc) && continue
            all(isuppercase, clean) && length(clean) > 3 && continue
            clean in stopwords && continue
            !any(islowercase, clean) && continue
            if !haskey(term_docs, clean)
                term_docs[clean] = Set{String}()
            end
            push!(term_docs[clean], source)
        end
    end
    for (term, docs) in term_docs
        length(docs) >= 3 || continue
        if !haskey(rigid_terms, term)
            rigid_terms[term] = clamp(Float32(0.70 + 0.02 * length(docs)), 0.70f0, 0.92f0)
        end
    end
    n_discovered = length(rigid_terms) - length(Peirce.default_rigid_terms())
    println("  Discovered $n_discovered terms, total rigid terms: $(length(rigid_terms))")
    term_docs = nothing  # free

    # Pass 2: collect embeddings only for rigid terms (streaming, one chunk at a time)
    println("Collecting rigid term embeddings...")
    term_embeds = Dict{String, Vector{Vector{Float32}}}()
    for (i, entry) in enumerate(manifest)
        emb, tok = load_chunk(entry)
        projected = proj_matrix * emb
        for (t, token) in enumerate(tok)
            clean = strip(token)
            if haskey(rigid_terms, clean)
                if !haskey(term_embeds, clean)
                    term_embeds[clean] = Vector{Float32}[]
                end
                push!(term_embeds[clean], Vector{Float32}(projected[:, t]))
            end
        end
        if i % 5000 == 0
            println("  Scanned $i/$(length(manifest)) chunks")
        end
    end
    for (term, embeds) in term_embeds
        isempty(embeds) && continue
        avg_embed = Vector{Float32}(sum(embeds) ./ length(embeds))
        rigidity = rigid_terms[term]
        foundational = term in foundational_terms
        insert!(state.rigid_table, term, avg_embed, rigidity; foundational)
        insert!(state.rigid_table, " $term", avg_embed, rigidity; foundational)
    end
    term_embeds = nothing
    GC.gc()
    println("Rigid designation table: $(length(state.rigid_table.entries)) entries")

    # Initialize contrastive memory bank
    bank = ContrastiveMemoryBank(D_PROJECT; max_size=256)
    println("Contrastive memory bank initialized (size=256)")

    # Phase 2 training loop
    for epoch in start_epoch:num_epochs
        # Ramp contrastive and trichotomy losses
        progress = Float32(min(epoch / 10, 1.0))
        λ_c = Float32(0.5) * progress      # slot contrastive: increased from 0.1 to 0.5
        λ_t = Float32(0.2) * progress      # trichotomy
        λ_tc = Float32(1.0) * progress     # text-level contrastive: ramps to 1.0

        println("--- Phase 2 Epoch $epoch/$num_epochs (λ_slot_contr=$(round(λ_c; digits=3)), λ_text_contr=$(round(λ_tc; digits=3)), λ_trich=$(round(λ_t; digits=3))) ---")

        indices = randperm(length(manifest))
        epoch_losses = Dict{String, Float64}()
        n = 0

        for i in indices
            raw_emb, tokens = load_chunk(manifest[i])
            embeddings = proj_matrix * raw_emb
            source = String(manifest[i].source)
            category = String(manifest[i].category)

            # Get current memory bank state (detached — no gradients)
            b_reps, b_sources, b_categories = bank_data(bank)

            loss_dict, pooled_rep = train_step_phase2!(state, embeddings, tokens;
                                           λ_contrastive=λ_c, λ_trichotomy=λ_t,
                                           λ_text_contrastive=λ_tc,
                                           τ=Float32(0.3),
                                           bank_reps=b_reps,
                                           bank_sources=b_sources,
                                           anchor_source=source)

            # Update memory bank with current chunk's pooled representation
            push_to_bank!(bank, pooled_rep, source, category)

            for (k, v) in loss_dict
                epoch_losses[k] = get(epoch_losses, k, 0.0) + v
            end
            n += 1

            if state.step % 5 == 0
                println("  Step $(state.step): total=$(round(loss_dict["total"]; digits=4)), " *
                        "recon=$(round(loss_dict["reconstruction"]; digits=4)), " *
                        "slot_c=$(round(loss_dict["contrastive"]; digits=4)), " *
                        "text_c=$(round(loss_dict["text_contrastive"]; digits=4)), " *
                        "truth=$(round(loss_dict["truth"]; digits=4)), " *
                        "trich=$(round(loss_dict["trich_balance"]; digits=4))")
            end

            # Mid-epoch checkpoint every 5000 steps (~1 hour) to limit OOM loss
            if state.step % 5000 == 0
                mkpath(CHECKPOINT_DIR)
                mid_path = joinpath(CHECKPOINT_DIR, "phase2_epoch$(epoch)_step$(state.step).bson")
                BSON.@save mid_path pipeline=state.pipeline decoder=state.decoder step=state.step proj_matrix=proj_matrix epoch=epoch
                println("  Mid-epoch checkpoint: $mid_path")
            end
        end

        for k in keys(epoch_losses)
            epoch_losses[k] /= max(n, 1)
        end
        println("  Epoch avg: recon=$(round(epoch_losses["reconstruction"]; digits=4)), " *
                "slot_c=$(round(get(epoch_losses, "contrastive", 0.0); digits=4)), " *
                "text_c=$(round(get(epoch_losses, "text_contrastive", 0.0); digits=4)), " *
                "trich_bal=$(round(get(epoch_losses, "trich_balance", 0.0); digits=4)), " *
                "proto=$(round(get(epoch_losses, "proto_div", 0.0); digits=4))")

        # Save every epoch
        mkpath(CHECKPOINT_DIR)
        path = joinpath(CHECKPOINT_DIR, "phase2_epoch$(epoch).bson")
        BSON.@save path pipeline=state.pipeline decoder=state.decoder step=state.step proj_matrix=proj_matrix epoch=epoch
        println("  Saved: $path")
        println()
    end

    # Final save
    mkpath(CHECKPOINT_DIR)
    path = joinpath(CHECKPOINT_DIR, "phase2_final.bson")
    BSON.@save path pipeline=state.pipeline decoder=state.decoder step=state.step proj_matrix=proj_matrix
    println("Phase 2 complete. $(state.step) total steps. Saved: $path")
end

main()
