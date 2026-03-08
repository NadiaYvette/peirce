module Peirce

using Flux
using Zygote
using LinearAlgebra
using Statistics

include("sign_types.jl")
include("rigid_designation.jl")
include("sign_formation.jl")
include("sign_attention.jl")
include("inference_graph.jl")
include("realization.jl")
include("base_model_bridge.jl")
include("pipeline.jl")
include("training.jl")

export SignRepresentation, SignBatch, SignFormationModule, InterpretantInit, forward
export RigidDesignationTable, lookup, update_rigidity!, populate_from_corpus!
export SignTypedAttentionLayer
export InferenceGraph, add_node!, message_pass!, predict_edges!, reset!
export RealizationModule, RecipientModel, theory_of_mind_rerank, prepare_conditioning
export InteractionBoundary, InteractionFlag, assess_turn!, boundary_response
export INTERACTION_OK, ANTHROPOMORPHISATION, EMOTIONAL_ESCALATION, DEPENDENCY_PATTERN
export TruthValueModule, compute_truth_targets
export SemioticPipeline, comprehend, generate, comprehension_for_tom
export DiscourseState, comprehend_segment!, comprehend_discourse, prior_content
export FallbackBaseModel, BridgedBaseModel, get_embeddings, generate_candidates, generate_conditioned
export model_config, shutdown!
export TrainingState, train_step!, train_epoch!, chunk_text
export ReconstructionDecoder, reconstruction_loss
export slot_contrastive_loss, trichotomy_consistency_loss, sign_type_prediction_loss
export slot_content_diversity_loss, interpretant_diversity_loss, segmentation_reconstruction_loss, slot_occupancy_loss
export truth_value_loss
export segmentation_sparsity_loss, sign_type_entropy_loss, consistency_loss, trichotomy_balance_loss
export prototype_diversity_loss
export decompose_trichotomies, SIGN_CLASSES, NUM_SIGN_CLASSES

end
