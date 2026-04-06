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
        # Proper names — scientists/philosophers
        "Copernicus" => 0.95f0, "Kepler" => 0.95f0, "Galileo" => 0.95f0,
        "Watson" => 0.90f0, "Crick" => 0.90f0, "Newton" => 0.90f0,
        "Gutenberg" => 0.90f0, "Pompeii" => 0.90f0, "Vesuvius" => 0.90f0,
        "Everest" => 0.90f0, "Napoleon" => 0.90f0, "Waterloo" => 0.90f0,
        "Armstrong" => 0.90f0, "Descartes" => 0.90f0, "Theseus" => 0.90f0,
        "Hume" => 0.90f0, "Rawls" => 0.90f0, "Wittgenstein" => 0.90f0,
        "Austin" => 0.85f0, "Grice" => 0.85f0, "Cantor" => 0.90f0,
        "Lakoff" => 0.85f0, "Johnson" => 0.85f0,
        # Literary characters — English
        "Alice" => 0.90f0, "Holmes" => 0.90f0, "Hamlet" => 0.95f0,
        "Ophelia" => 0.90f0, "Polonius" => 0.90f0, "Horatio" => 0.90f0,
        "Laertes" => 0.90f0, "Claudius" => 0.90f0, "Gertrude" => 0.90f0,
        "Gulliver" => 0.90f0, "Socrates" => 0.95f0, "Plato" => 0.95f0,
        "Pygmalion" => 0.90f0, "Eliza" => 0.90f0,
        # Literary characters — German
        "Faust" => 0.90f0, "Mephisto" => 0.90f0, "Gretchen" => 0.90f0,
        "Werther" => 0.90f0, "Woyzeck" => 0.90f0, "Danton" => 0.90f0,
        "Nora" => 0.90f0,
        # Literary characters — French
        "Bovary" => 0.90f0, "Emma" => 0.85f0, "Phedre" => 0.90f0,
        "Alceste" => 0.90f0,
        # Literary characters — Spanish
        "Quijote" => 0.95f0, "Sancho" => 0.90f0, "Dulcinea" => 0.90f0,
        # Literary characters — Russian
        "Raskolnikov" => 0.90f0,
        # Authors (appear frequently in corpus)
        "Shakespeare" => 0.95f0, "Goethe" => 0.95f0, "Kafka" => 0.95f0,
        "Cervantes" => 0.95f0, "Moliere" => 0.90f0, "Racine" => 0.90f0,
        "Schiller" => 0.90f0, "Dostoevsky" => 0.90f0, "Tolstoy" => 0.90f0,
        "Dickens" => 0.90f0, "Austen" => 0.90f0, "Flaubert" => 0.90f0,
        "Baudelaire" => 0.90f0, "Rilke" => 0.90f0, "Pushkin" => 0.90f0,
        "Zamenhof" => 0.90f0,
        # Geographic — expanded with corpus locations
        "Earth" => 0.95f0, "Sun" => 0.95f0, "Moon" => 0.95f0,
        "Hawaii" => 0.90f0, "Europe" => 0.90f0,
        "London" => 0.90f0, "Paris" => 0.90f0, "England" => 0.90f0,
        "France" => 0.90f0, "Germany" => 0.90f0, "Spain" => 0.90f0,
        "Russia" => 0.90f0, "Denmark" => 0.90f0, "Elsinore" => 0.90f0,
        "Mancha" => 0.85f0,
        # Numerical/physical constants
        "299,792,458" => 0.99f0, "79" => 0.95f0, "1440" => 0.90f0,
        "1953" => 0.90f0, "1969" => 0.90f0, "1983" => 0.90f0,
    )
end

"""Default foundational terms (pinned beliefs)."""
function default_foundational_terms()
    Set{String}(["H2O", "DNA", "Au", "water", "gold", "oxygen", "hydrogen",
                  "Earth", "Sun", "Moon"])
end

"""
    auto_discover_rigid_terms(token_strings_list, source_list; min_docs=3)

Auto-discover potential rigid designators from the corpus.
Finds capitalized words appearing across multiple documents, filtering out
common sentence-starters and short words.

Returns Dict{String, Float32} of discovered term => rigidity score.
"""
function auto_discover_rigid_terms(token_strings_list::Vector{Vector{String}},
                                    source_list::Vector{String};
                                    min_docs::Int=3)
    # Common words to exclude (sentence-starters, articles, etc.)
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

    # Track: term → set of source documents it appears in
    term_docs = Dict{String, Set{String}}()

    for (tokens, source) in zip(token_strings_list, source_list)
        for tok in tokens
            clean = strip(tok)
            length(clean) < 3 && continue
            # Check if capitalized (first char uppercase, not all uppercase)
            first_char = first(clean)
            !isuppercase(first_char) && continue
            all(isuppercase, clean) && length(clean) > 3 && continue  # skip ALL CAPS
            clean in stopwords && continue
            # Must contain at least one lowercase letter
            !any(islowercase, clean) && continue

            if !haskey(term_docs, clean)
                term_docs[clean] = Set{String}()
            end
            push!(term_docs[clean], source)
        end
    end

    # Keep terms appearing in min_docs+ different documents
    discovered = Dict{String, Float32}()
    for (term, docs) in term_docs
        n = length(docs)
        n >= min_docs || continue
        # Rigidity scales with document coverage (more docs = more rigid)
        rigidity = clamp(Float32(0.70 + 0.02 * n), 0.70f0, 0.92f0)
        discovered[term] = rigidity
    end
    discovered
end
