#!/bin/bash
# Download public domain texts for training corpus
# Usage: bash scripts/fetch_corpus.sh

set -e
CORPUS_DIR="$(dirname "$0")/../corpus"
mkdir -p "$CORPUS_DIR"/{conversational,argumentative,narrative,allegory,poetry,drama,factual}

fetch() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  SKIP (exists): $(basename "$dest")"
        return
    fi
    echo "  Fetching: $(basename "$dest")"
    curl -sL "$url" -o "$dest.tmp" && mv "$dest.tmp" "$dest"
}

strip_gutenberg() {
    # Strip Project Gutenberg header/footer from a file in-place
    local file="$1"
    [ -f "$file" ] || return
    local start end
    start=$(grep -n '^\*\*\* START OF' "$file" | head -1 | cut -d: -f1)
    end=$(grep -n '^\*\*\* END OF' "$file" | head -1 | cut -d: -f1)
    if [ -n "$start" ] && [ -n "$end" ]; then
        sed -n "$((start+1)),$((end-1))p" "$file" > "$file.clean"
        mv "$file.clean" "$file"
    elif [ -n "$start" ]; then
        tail -n +"$((start+1))" "$file" > "$file.clean"
        mv "$file.clean" "$file"
    fi
}

echo "=== Fetching English Literature ==="

echo "Allegory:"
fetch "https://gutenberg.net.au/ebooks/0100011.txt" "$CORPUS_DIR/allegory/orwell_animal_farm.txt"
fetch "https://www.gutenberg.org/cache/epub/829/pg829.txt" "$CORPUS_DIR/allegory/swift_gullivers_travels.txt"
fetch "https://www.gutenberg.org/cache/epub/11339/pg11339.txt" "$CORPUS_DIR/allegory/aesop_fables.txt"

echo "Argumentative:"
fetch "https://www.gutenberg.org/cache/epub/34901/pg34901.txt" "$CORPUS_DIR/argumentative/mill_on_liberty.txt"
fetch "https://www.gutenberg.org/cache/epub/31270/pg31270.txt" "$CORPUS_DIR/argumentative/paine_rights_of_man.txt"
fetch "https://www.gutenberg.org/cache/epub/3420/pg3420.txt" "$CORPUS_DIR/argumentative/wollstonecraft_vindication.txt"

echo "Narrative:"
fetch "https://www.gutenberg.org/cache/epub/1342/pg1342.txt" "$CORPUS_DIR/narrative/austen_pride_and_prejudice.txt"
fetch "https://www.gutenberg.org/cache/epub/219/pg219.txt" "$CORPUS_DIR/narrative/conrad_heart_of_darkness.txt"
fetch "https://www.gutenberg.org/cache/epub/63107/pg63107.txt" "$CORPUS_DIR/narrative/woolf_mrs_dalloway.txt"

echo "Poetry:"
fetch "https://www.gutenberg.org/cache/epub/1934/pg1934.txt" "$CORPUS_DIR/poetry/blake_songs.txt"
fetch "https://www.gutenberg.org/cache/epub/1322/pg1322.txt" "$CORPUS_DIR/poetry/whitman_leaves_of_grass.txt"
fetch "https://www.gutenberg.org/cache/epub/12242/pg12242.txt" "$CORPUS_DIR/poetry/dickinson_poems.txt"

echo "Drama:"
fetch "https://www.gutenberg.org/cache/epub/1524/pg1524.txt" "$CORPUS_DIR/drama/shakespeare_hamlet.txt"
fetch "https://www.gutenberg.org/cache/epub/844/pg844.txt" "$CORPUS_DIR/drama/wilde_earnest.txt"
fetch "https://www.gutenberg.org/cache/epub/3825/pg3825.txt" "$CORPUS_DIR/drama/shaw_pygmalion.txt"

echo "Conversational (dialogue-heavy novels):"
fetch "https://www.gutenberg.org/cache/epub/1661/pg1661.txt" "$CORPUS_DIR/conversational/doyle_sherlock_holmes.txt"
fetch "https://www.gutenberg.org/cache/epub/11/pg11.txt" "$CORPUS_DIR/conversational/carroll_alice_wonderland.txt"

echo ""
echo "=== Fetching German Literature ==="

echo "Narrative:"
fetch "https://www.gutenberg.org/cache/epub/22367/pg22367.txt" "$CORPUS_DIR/narrative/kafka_verwandlung_de.txt"
fetch "https://www.gutenberg.org/cache/epub/2407/pg2407.txt" "$CORPUS_DIR/narrative/goethe_werther_de.txt"

echo "Argumentative:"
fetch "https://www.gutenberg.org/cache/epub/30821/pg30821.txt" "$CORPUS_DIR/argumentative/kant_aufklaerung_de.txt"
fetch "https://www.gutenberg.org/cache/epub/7207/pg7207.txt" "$CORPUS_DIR/argumentative/nietzsche_menschliches_de.txt"

echo "Drama:"
fetch "https://www.gutenberg.org/cache/epub/6782/pg6782.txt" "$CORPUS_DIR/drama/schiller_raeuber_de.txt"

echo ""
echo "=== Fetching French Literature ==="

echo "Narrative:"
fetch "https://www.gutenberg.org/cache/epub/14155/pg14155.txt" "$CORPUS_DIR/narrative/flaubert_bovary_fr.txt"
fetch "https://www.gutenberg.org/cache/epub/1237/pg1237.txt" "$CORPUS_DIR/narrative/balzac_vendetta_fr.txt"

echo "Poetry:"
fetch "https://www.gutenberg.org/cache/epub/6099/pg6099.txt" "$CORPUS_DIR/poetry/baudelaire_fleurs_du_mal_fr.txt"

echo "Drama:"
fetch "https://www.gutenberg.org/cache/epub/5765/pg5765.txt" "$CORPUS_DIR/drama/moliere_misanthrope_fr.txt"

echo ""
echo "=== Fetching Spanish Literature ==="

echo "Narrative:"
fetch "https://www.gutenberg.org/cache/epub/61089/pg61089.txt" "$CORPUS_DIR/narrative/cervantes_novelas_es.txt"

echo "Poetry:"
fetch "https://www.gutenberg.org/cache/epub/16098/pg16098.txt" "$CORPUS_DIR/poetry/espronceda_poesias_es.txt"

echo ""
echo "=== Fetching Esperanto ==="
fetch "https://www.gutenberg.org/cache/epub/34129/pg34129.txt" "$CORPUS_DIR/factual/zamenhof_fundamento_eo.txt"
fetch "https://www.gutenberg.org/cache/epub/7787/pg7787.txt" "$CORPUS_DIR/factual/esperanto_reader_eo.txt"

echo ""
echo "=== Fetching Factual/Expository ==="
# Hume's Enquiry (philosophy but also expository style)
fetch "https://www.gutenberg.org/cache/epub/9662/pg9662.txt" "$CORPUS_DIR/factual/hume_enquiry.txt"

echo ""
echo "=== Stripping Gutenberg headers ==="
for f in $(find "$CORPUS_DIR" -name "*.txt"); do
    strip_gutenberg "$f"
    size=$(wc -c < "$f")
    echo "  $(basename "$f"): ${size} bytes"
done

echo ""
echo "=== Summary ==="
total=$(find "$CORPUS_DIR" -name "*.txt" | wc -l)
total_bytes=$(find "$CORPUS_DIR" -name "*.txt" -exec cat {} \; | wc -c)
echo "Total files: $total"
echo "Total size: $((total_bytes / 1024)) KB"
echo ""
echo "By category:"
for cat in conversational argumentative narrative allegory poetry drama factual; do
    count=$(find "$CORPUS_DIR/$cat" -name "*.txt" 2>/dev/null | wc -l)
    echo "  $cat: $count files"
done
