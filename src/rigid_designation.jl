# Rigid Designation Table
#
# Maps terms to referent embeddings with a learned rigidity score.
# Rigidity near 1.0: binding resists contextual override (names, natural kinds).
# Rigidity near 0.0: binding is fully defeasible (descriptions).

struct RigidDesignationEntry
    referent::Vector{Float32}   # d_model-dim embedding of the referent
    rigidity::Float32           # learned rigidity score in [0, 1]
    is_foundational::Bool       # if true, this is a pinned belief (Section 8)
end

mutable struct RigidDesignationTable
    entries::Dict{String, RigidDesignationEntry}
    d_model::Int
end

function RigidDesignationTable(d_model::Int)
    RigidDesignationTable(Dict{String, RigidDesignationEntry}(), d_model)
end

"""
    lookup(table, term) -> (referent_embedding, rigidity) or nothing

Look up a term in the rigid designation table.
Returns the referent embedding and rigidity score, or nothing if not found.
"""
function lookup(table::RigidDesignationTable, term::String)
    entry = get(table.entries, term, nothing)
    isnothing(entry) && return nothing
    (entry.referent, entry.rigidity)
end

"""
    update_rigidity!(table, term, new_rigidity)

Update the rigidity score of a non-foundational entry.
Foundational entries are immutable.
"""
function update_rigidity!(table::RigidDesignationTable, term::String, new_rigidity::Float32)
    entry = get(table.entries, term, nothing)
    isnothing(entry) && return
    entry.is_foundational && return  # pinned — no update
    table.entries[term] = RigidDesignationEntry(entry.referent, clamp(new_rigidity, 0f0, 1f0), false)
end

"""
    insert!(table, term, referent, rigidity; foundational=false)

Add or overwrite a term in the table.
"""
function Base.insert!(table::RigidDesignationTable, term::String, referent::Vector{Float32},
                      rigidity::Float32; foundational::Bool=false)
    @assert length(referent) == table.d_model
    table.entries[term] = RigidDesignationEntry(referent, clamp(rigidity, 0f0, 1f0), foundational)
end

"""
    populate_from_corpus!(table, token_embeddings_list, token_strings_list;
                          rigid_terms, foundational_terms)

Populate the rigid designation table from training data.
For each rigid term found in the corpus, averages its embeddings across occurrences
and registers it with an initial rigidity score.

- `rigid_terms`: Dict of term => rigidity (e.g., "H2O" => 0.95)
- `foundational_terms`: Set of terms that are pinned (rigidity immutable)
"""
function populate_from_corpus!(table::RigidDesignationTable,
                               token_embeddings_list::Vector{<:AbstractMatrix{Float32}},
                               token_strings_list::Vector{Vector{String}};
                               rigid_terms::Dict{String, Float32}=default_rigid_terms(),
                               foundational_terms::Set{String}=default_foundational_terms())
    # Accumulate embeddings for each rigid term
    term_embeds = Dict{String, Vector{Vector{Float32}}}()

    for (embeddings, tokens) in zip(token_embeddings_list, token_strings_list)
        for (t, tok) in enumerate(tokens)
            # Strip leading/trailing whitespace for matching
            clean = strip(tok)
            if haskey(rigid_terms, clean)
                if !haskey(term_embeds, clean)
                    term_embeds[clean] = Vector{Float32}[]
                end
                push!(term_embeds[clean], Vector{Float32}(embeddings[:, t]))
            end
        end
    end

    # Register averaged embeddings
    for (term, embeds) in term_embeds
        avg_embed = Vector{Float32}(sum(embeds) ./ length(embeds))
        rigidity = rigid_terms[term]
        foundational = term in foundational_terms
        insert!(table, term, avg_embed, rigidity; foundational)
        # Also register with leading space (tokenizer convention)
        insert!(table, " $term", avg_embed, rigidity; foundational)
    end
end

"""Default set of rigid designators for the training corpus."""
function default_rigid_terms()
    Dict{String, Float32}(
        # Chemical/physical constants (highest rigidity)
        "H2O" => 0.98f0, "H" => 0.95f0, "O" => 0.95f0, "Au" => 0.95f0,
        "DNA" => 0.95f0, "CO2" => 0.95f0,
        # Natural kind terms
        "water" => 0.90f0, "gold" => 0.90f0, "oxygen" => 0.90f0,
        "hydrogen" => 0.90f0, "iron" => 0.90f0,
        # Proper names
        "Copernicus" => 0.95f0, "Kepler" => 0.95f0, "Galileo" => 0.95f0,
        "Watson" => 0.90f0, "Crick" => 0.90f0, "Newton" => 0.90f0,
        "Gutenberg" => 0.90f0, "Pompeii" => 0.90f0, "Vesuvius" => 0.90f0,
        "Everest" => 0.90f0, "Napoleon" => 0.90f0, "Waterloo" => 0.90f0,
        "Armstrong" => 0.90f0, "Descartes" => 0.90f0, "Theseus" => 0.90f0,
        "Hume" => 0.90f0, "Rawls" => 0.90f0, "Wittgenstein" => 0.90f0,
        "Austin" => 0.85f0, "Grice" => 0.85f0, "Cantor" => 0.90f0,
        "Lakoff" => 0.85f0, "Johnson" => 0.85f0,
        # Numerical/physical constants
        "299,792,458" => 0.99f0, "79" => 0.95f0, "1440" => 0.90f0,
        "1953" => 0.90f0, "1969" => 0.90f0, "1983" => 0.90f0,
        # Geographic
        "Earth" => 0.95f0, "Sun" => 0.95f0, "Moon" => 0.95f0,
        "Hawaii" => 0.90f0, "Europe" => 0.90f0,
    )
end

"""Default foundational terms (pinned beliefs)."""
function default_foundational_terms()
    Set{String}(["H2O", "DNA", "Au", "water", "gold", "oxygen", "hydrogen",
                  "Earth", "Sun", "Moon"])
end
