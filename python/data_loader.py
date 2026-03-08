"""
Data Loader for Phase 1 Training

Loads text data, runs it through the base model to get token embeddings,
and saves them as numpy arrays for the Julia training loop.

For Phase 1 (distillation), we use a small text corpus — the reconstruction
objective doesn't need labeled data, just raw text processed through the base model.
"""

import json
import numpy as np
from pathlib import Path
from base_model import BaseModel


DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "embeddings_cache"


def prepare_corpus(texts: list[str], bm: BaseModel, cache_dir: Path = CACHE_DIR,
                   chunk_size: int = 128) -> list[dict]:
    """
    Process a list of texts into chunked embeddings.

    Each chunk becomes a training example:
      - embeddings: (d_model, T) float32 array
      - tokens: list of token strings

    Saves to cache_dir for reuse.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, text in enumerate(texts):
        embeddings, tokens = bm.get_embeddings(text)
        d_model, T = embeddings.shape

        # Chunk long sequences
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            if end - start < 4:
                continue

            chunk_embeddings = embeddings[:, start:end]
            chunk_tokens = tokens[start:end]

            chunk_id = f"doc{i:04d}_chunk{start:04d}"
            np.save(cache_dir / f"{chunk_id}_embeddings.npy", chunk_embeddings)

            entry = {
                "id": chunk_id,
                "doc_idx": i,
                "start": start,
                "num_tokens": end - start,
                "tokens": chunk_tokens,
                "embeddings_file": f"{chunk_id}_embeddings.npy"
            }
            manifest.append(entry)

    # Save manifest
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(manifest)} chunks from {len(texts)} documents")
    print(f"Cached to {cache_dir}")
    return manifest


def load_chunk(chunk_entry: dict, cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, list[str]]:
    """Load a single chunk from cache."""
    embeddings = np.load(cache_dir / chunk_entry["embeddings_file"])
    tokens = chunk_entry["tokens"]
    return embeddings, tokens


def sample_training_texts() -> list[str]:
    """
    Training corpus with diverse text types for learning sign-structured representations.
    Categories:
      - Factual / rigid designation
      - Arguments and logical reasoning
      - Descriptions / indexical signs
      - Dialogue / speech acts
      - Mathematical and formal reasoning
      - Historical and narrative
      - Scientific explanation
      - Philosophy and abstract reasoning
      - Legal and normative
      - Metaphor and figurative language
    """
    return [
        # === Factual / rigid designation ===
        "Water is H2O. This is a chemical fact that does not depend on context. "
        "The molecular composition of water was discovered through careful scientific investigation. "
        "Hydrogen and oxygen combine in a fixed ratio to form water molecules.",

        "The Earth orbits the Sun at an average distance of approximately 93 million miles. "
        "This heliocentric model replaced the earlier geocentric view. "
        "Copernicus, Kepler, and Galileo contributed to this understanding.",

        "Gold has the atomic number 79 and the chemical symbol Au. "
        "Its density is 19.3 grams per cubic centimeter. "
        "These properties hold regardless of where or when the gold is found. "
        "A gold atom in ancient Egypt was identical to one mined today.",

        "The speed of light in a vacuum is exactly 299,792,458 meters per second. "
        "This is not an empirical approximation but a defined constant. "
        "Since 1983, the meter itself has been defined in terms of this speed.",

        "DNA consists of four nucleotide bases: adenine, thymine, guanine, and cytosine. "
        "The double helix structure was described by Watson and Crick in 1953. "
        "Base pairing follows strict rules: adenine pairs with thymine, guanine with cytosine.",

        "Mount Everest is the tallest mountain above sea level, at 8,849 meters. "
        "However, measured from base to peak, Mauna Kea in Hawaii is taller. "
        "The distinction depends on the reference frame chosen for measurement.",

        # === Arguments and logical reasoning ===
        "All mammals breathe air. Whales are mammals. Therefore, whales breathe air. "
        "This is a valid syllogism: the conclusion follows necessarily from the premises. "
        "The validity of the argument does not depend on whether the premises are true.",

        "If it rains, the ground gets wet. The ground is wet. "
        "Can we conclude that it rained? No — this is the fallacy of affirming the consequent. "
        "The ground could be wet for other reasons, such as a sprinkler.",

        "No reptiles have fur. All dogs have fur. Therefore, no dogs are reptiles. "
        "This syllogism is valid and its premises are true, making it sound. "
        "A sound argument guarantees the truth of its conclusion.",

        "Either the butler did it or the gardener did it. "
        "The gardener has an alibi for the time of the crime. "
        "Therefore, the butler did it. This is a valid disjunctive syllogism, "
        "but only if we accept that no other suspects exist.",

        "Premise one: if taxes increase, consumer spending decreases. "
        "Premise two: if consumer spending decreases, economic growth slows. "
        "Conclusion: if taxes increase, economic growth slows. "
        "This chain of conditional reasoning is called hypothetical syllogism.",

        "Some arguments are valid but unsound. Consider: all fish can fly. "
        "Salmon are fish. Therefore, salmon can fly. The logic is impeccable, "
        "but the first premise is false, so the argument is unsound.",

        "The argument from analogy: the brain is like a computer. "
        "Computers process information. Therefore, the brain processes information. "
        "Analogical arguments are never deductively valid, but they can be persuasive "
        "when the analogy is strong and relevant.",

        # === Descriptive / indexical ===
        "The cat is sitting on the mat in front of the fireplace. "
        "Its fur is orange with white patches. The fire crackles behind it. "
        "This particular cat has lived in this house for seven years.",

        "Look at that mountain over there. The one with snow on top. "
        "It was named after the explorer who first mapped this region. "
        "The local name for it translates roughly as 'sleeping giant'.",

        "The old clock on the mantelpiece reads quarter past three. "
        "Its pendulum swings with mechanical precision. "
        "The brass housing has developed a green patina over decades of use. "
        "Each tick marks the passage of another moment.",

        "Here is the scene: a narrow street in an old city. "
        "Cobblestones glisten after rain. A baker pulls bread from an oven. "
        "The smell of fresh bread mixes with the damp stone air. "
        "Two children run past, laughing, splashing through puddles.",

        "The patient presents with a fever of 38.9 degrees Celsius. "
        "Heart rate is elevated at 102 beats per minute. "
        "The left pupil is slightly dilated compared to the right. "
        "These signs indicate a possible neurological involvement.",

        "The footprint in the mud was approximately 28 centimeters long. "
        "The tread pattern matched a common hiking boot. "
        "Depth of impression suggested a person weighing roughly 80 kilograms. "
        "The print pointed northeast, toward the abandoned cabin.",

        # === Dialogue / speech acts ===
        "Can you pass me the salt? I didn't mean that as a question about your ability. "
        "It was a request. The literal meaning and the intended meaning diverge. "
        "Understanding this requires pragmatic competence beyond mere word recognition.",

        "I promise to finish the report by Friday. "
        "A promise is not a description of the future — it is a commitment. "
        "By saying these words, I have changed my social obligations. "
        "This is what Austin meant by a performative utterance.",

        "Do you know what time it is? Yes, I do. "
        "The response is technically correct but pragmatically uncooperative. "
        "Grice's maxim of quantity requires giving as much information as needed. "
        "The expected answer includes the actual time.",

        "I hereby declare this meeting adjourned. "
        "These words do not describe reality — they change it. "
        "Before the utterance, the meeting was in session; after it, it is not. "
        "The speaker must have the authority to make this declaration.",

        "Nice weather we're having, said the man standing in the rain. "
        "The statement is ironic: its literal content contradicts the obvious situation. "
        "Understanding irony requires recognizing the speaker's intention to convey the opposite. "
        "Sarcasm adds a layer of social evaluation to irony.",

        "She said she would think about it, which everyone understood as a polite refusal. "
        "The literal meaning preserves the possibility of agreement. "
        "But the conventional implicature in this context is negative. "
        "Indirectness serves the social function of saving face.",

        # === Mathematical and formal reasoning ===
        "The square root of two is irrational. This was proved by contradiction. "
        "Assume it is rational, expressible as a fraction in lowest terms. "
        "Then both numerator and denominator must be even, contradicting the assumption. "
        "Therefore, the square root of two cannot be rational.",

        "In Euclidean geometry, the angles of a triangle sum to 180 degrees. "
        "On a sphere, this is no longer true — angles sum to more than 180. "
        "The parallel postulate distinguishes Euclidean from non-Euclidean geometry. "
        "Changing one axiom changes the entire system of truths.",

        "Zero is neither positive nor negative. It is the additive identity: "
        "any number plus zero equals itself. Division by zero is undefined "
        "because no number multiplied by zero yields a nonzero result. "
        "This is not a gap in mathematics but a consequence of the axioms.",

        "The set of natural numbers is infinite. So is the set of even numbers. "
        "Surprisingly, these two infinite sets have the same cardinality. "
        "Each natural number n maps to exactly one even number 2n, and vice versa. "
        "Cantor showed that some infinities are larger than others.",

        # === Historical and narrative ===
        "In 1969, Neil Armstrong became the first human to walk on the Moon. "
        "His words, 'That's one small step for man, one giant leap for mankind,' "
        "were broadcast to an estimated 600 million viewers. "
        "The event marked the culmination of the Space Race between the US and USSR.",

        "The printing press, invented by Gutenberg around 1440, transformed European society. "
        "Before it, books were copied by hand, making them scarce and expensive. "
        "Mass production of texts accelerated the spread of literacy and ideas. "
        "The Reformation, the Scientific Revolution, and the Enlightenment all followed.",

        "The city of Pompeii was buried under volcanic ash in 79 AD. "
        "The eruption of Mount Vesuvius preserved the city in remarkable detail. "
        "Archaeologists have uncovered streets, buildings, frescoes, and human remains. "
        "The preserved graffiti offers a window into daily Roman life.",

        "During the industrial revolution, child labor was commonplace. "
        "Children as young as five worked in factories and mines. "
        "Reformers argued that this practice was morally wrong and economically unnecessary. "
        "Legislation gradually restricted and eventually banned child labor in most countries.",

        # === Scientific explanation ===
        "Evolution by natural selection works through three mechanisms. "
        "First, variation: individuals within a population differ in traits. "
        "Second, inheritance: traits are passed from parent to offspring. "
        "Third, selection: some traits improve survival and reproduction. "
        "Over generations, advantageous traits become more common in the population.",

        "Antibiotics kill bacteria by disrupting essential cellular processes. "
        "Penicillin, for example, prevents bacteria from building cell walls. "
        "However, bacteria can develop resistance through random mutation. "
        "Overuse of antibiotics accelerates this process, creating resistant strains.",

        "Plate tectonics explains the movement of Earth's lithospheric plates. "
        "Where plates converge, mountains form or one plate subducts beneath another. "
        "Where they diverge, new crust forms at mid-ocean ridges. "
        "Earthquakes and volcanic activity concentrate along plate boundaries.",

        "Photosynthesis converts carbon dioxide and water into glucose and oxygen. "
        "The light reactions capture solar energy in the thylakoid membranes. "
        "The Calvin cycle uses this energy to fix carbon into organic molecules. "
        "This process is the foundation of nearly all food chains on Earth.",

        "The greenhouse effect is a natural phenomenon that warms the Earth. "
        "Without it, the average surface temperature would be about minus 18 degrees Celsius. "
        "Human activities have increased atmospheric greenhouse gas concentrations. "
        "This enhanced greenhouse effect is the primary driver of current climate change.",

        # === Philosophy and abstract reasoning ===
        "Descartes argued that the only thing we cannot doubt is our own thinking. "
        "Even if an evil demon deceives us about everything, the act of doubting proves existence. "
        "Cogito ergo sum: I think, therefore I am. "
        "This became the foundation of modern rationalist philosophy.",

        "The ship of Theseus poses a puzzle about identity. "
        "If every plank of a ship is gradually replaced, is it still the same ship? "
        "If the old planks are reassembled into a ship, which one is the original? "
        "The problem reveals that our concept of identity is more complex than it seems.",

        "Hume argued that we never observe causation directly. "
        "We see one event followed by another and infer a causal connection. "
        "But constant conjunction does not prove necessary connection. "
        "The sun has risen every day so far, but this does not guarantee it will rise tomorrow.",

        "Justice, according to Rawls, is what rational agents would choose behind a veil of ignorance. "
        "If you did not know your position in society, what rules would you want? "
        "You would likely choose rules that protect the worst-off members. "
        "This thought experiment grounds principles of fairness without appeals to tradition.",

        "The trolley problem presents a moral dilemma: a runaway trolley will kill five people. "
        "You can divert it to a side track where it will kill one person. "
        "Most people say diverting is permissible. But would you push someone onto the tracks "
        "to stop the trolley? Most say no, even though the outcome is the same.",

        "Wittgenstein argued that the meaning of a word is its use in language. "
        "There is no private inner meaning that words refer to. "
        "Understanding a word means knowing how to use it correctly in various contexts. "
        "Language is a social practice, not a system of labels for mental objects.",

        # === Legal and normative ===
        "A contract requires offer, acceptance, consideration, and mutual assent. "
        "If any of these elements is missing, the contract is void. "
        "Consideration means each party must give something of value. "
        "A promise to give a gift, without anything in return, is generally unenforceable.",

        "The presumption of innocence means the prosecution bears the burden of proof. "
        "The accused need not prove their innocence. "
        "This principle protects individuals from state power. "
        "It reflects the judgment that convicting an innocent person is worse than acquitting a guilty one.",

        "Property rights are not natural facts but social conventions enforced by law. "
        "What counts as property has changed dramatically over time. "
        "Ideas, frequencies, and genetic sequences can now be owned. "
        "The boundaries of ownership are constantly being renegotiated.",

        # === Metaphor and figurative language ===
        "Time is money, we say, and we mean it more literally than we realize. "
        "We spend time, save time, waste time, and invest time. "
        "This metaphor shapes how we think about an abstract concept. "
        "Lakoff and Johnson argued that all abstract thought is metaphorical.",

        "The news hit her like a ton of bricks. "
        "She stood frozen, the words echoing in her mind. "
        "The world seemed to tilt slightly, as if the floor had shifted beneath her feet. "
        "Figurative language conveys emotional experience more effectively than literal description.",

        "He was a lion on the battlefield, fearless and commanding. "
        "His troops followed him into the breach without hesitation. "
        "The metaphor does not claim he was literally a lion. "
        "It maps specific qualities — courage, power — from one domain to another.",

        "The economy is overheating, analysts warn. Interest rates need to cool things down. "
        "The temperature metaphor for economic activity is pervasive. "
        "We speak of hot markets, frozen assets, and warm leads. "
        "These are not mere decorations but conceptual frameworks that guide policy.",

        # === Technical and procedural ===
        "To change a flat tire, first engage the parking brake and place wheel chocks. "
        "Loosen the lug nuts before raising the vehicle with the jack. "
        "Remove the flat tire and mount the spare. "
        "Tighten the lug nuts in a star pattern to ensure even pressure.",

        "The recipe calls for 200 grams of flour, two eggs, and a pinch of salt. "
        "Mix the dry ingredients first, then add the wet ingredients gradually. "
        "Knead the dough for ten minutes until it becomes smooth and elastic. "
        "Let it rise in a warm place for one hour before shaping.",

        # === Definitions and classifications ===
        "A mammal is a warm-blooded vertebrate that nurses its young with milk. "
        "Most mammals give live birth, but monotremes like the platypus lay eggs. "
        "The class Mammalia includes approximately 6,400 known species. "
        "They occupy every major habitat from deep ocean to high atmosphere.",

        "Verbs can be classified as transitive or intransitive. "
        "A transitive verb takes a direct object: she kicked the ball. "
        "An intransitive verb does not: the baby slept. "
        "Some verbs can be used both ways: she sings a song, she sings beautifully.",

        # === Counterfactual and hypothetical ===
        "If Napoleon had won at Waterloo, the map of Europe would look different today. "
        "Counterfactual reasoning is essential for understanding causation. "
        "We assess what caused an event by imagining what would have happened otherwise. "
        "This requires a mental model of how the world works.",

        "Imagine a world where gravity were twice as strong. "
        "Trees would be shorter, animals would be smaller, and mountains would be lower. "
        "The atmosphere would be thinner, and flight would be much more difficult. "
        "Thought experiments like this help us understand the consequences of physical laws.",
    ]


def prepare_training_data():
    """Full data preparation pipeline."""
    print("Loading base model...")
    bm = BaseModel()
    config = bm.get_config()
    print(f"Model: {config['model_name']}, d_model={config['d_model']}")

    print("\nPreparing corpus...")
    texts = sample_training_texts()
    manifest = prepare_corpus(texts, bm, chunk_size=64)

    # Print summary
    total_tokens = sum(e["num_tokens"] for e in manifest)
    print(f"\nSummary:")
    print(f"  Documents: {len(texts)}")
    print(f"  Chunks: {len(manifest)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  d_model: {config['d_model']}")

    return manifest


if __name__ == "__main__":
    prepare_training_data()
