# Base Model Bridge
#
# Interfaces with the Python base model (Qwen2.5-0.5B) via a subprocess bridge.
# Provides token embeddings for sign formation and conditioned decoding for realization.
#
# BridgedBaseModel spawns python/bridge_server.py and communicates via JSON over stdio.
# FallbackBaseModel generates random embeddings for structural testing without Python.

import Base64: base64decode, base64encode
import JSON3

# --- Fallback mode (no Python) ---

"""
    FallbackBaseModel(d_model, vocab)

Dummy base model that generates random embeddings. Used when Python is unavailable.
"""
struct FallbackBaseModel
    d_model::Int
    vocab::Dict{String, Int}
end

function FallbackBaseModel(d_model::Int)
    FallbackBaseModel(d_model, Dict{String, Int}())
end

function get_embeddings(model::FallbackBaseModel, text::String)
    tokens = split(text)
    token_strings = String.(tokens)
    T = length(token_strings)
    embeddings = randn(Float32, model.d_model, T) .* Float32(0.02)
    (embeddings, token_strings)
end

function generate_candidates(model::FallbackBaseModel, prompt::String;
                              num_candidates::Int=4, max_tokens::Int=128)
    ["[fallback candidate $i for: $prompt]" for i in 1:num_candidates]
end

function model_config(model::FallbackBaseModel)
    Dict(
        "d_model" => model.d_model,
        "vocab_size" => 0,
        "num_layers" => 0,
        "num_heads" => 0,
        "model_name" => "fallback"
    )
end

# --- Bridged mode (subprocess) ---

"""
    BridgedBaseModel

Wraps the Python base model via a subprocess running python/bridge_server.py.
Communicates via JSON over stdin/stdout with base64-encoded float32 arrays.
"""
mutable struct BridgedBaseModel
    process::Base.Process
    stdin::IO
    stdout::IO
    d_model::Int
end

function decode_float32_array(b64::String, shape::Vector{Int})
    bytes = base64decode(b64)
    arr = reinterpret(Float32, bytes)
    reshape(arr, shape...)
end

function encode_float32_array(arr::AbstractMatrix{Float32})
    bytes = reinterpret(UInt8, vec(arr))
    base64encode(bytes)
end

function send_request(model::BridgedBaseModel, req::Dict)
    write(model.stdin, JSON3.write(req) * "\n")
    flush(model.stdin)
    line = readline(model.stdout)
    JSON3.read(line, Dict{String, Any})
end

"""
    BridgedBaseModel(; python_cmd="python3.12", project_dir=nothing)

Start the Python bridge server and return a connected BridgedBaseModel.
"""
function BridgedBaseModel(; python_cmd::String="python3.12",
                            project_dir::String=joinpath(@__DIR__, ".."))
    server_script = joinpath(project_dir, "python", "bridge_server.py")
    isfile(server_script) || error("Bridge server not found: $server_script")

    proc = open(Cmd(`$python_cmd $server_script`; dir=project_dir),
                read=true, write=true)

    # Read the "ready" message
    line = readline(proc)
    resp = JSON3.read(line, Dict{String, Any})
    resp["status"] == "ready" || error("Bridge server failed to start: $line")
    d_model = Int(resp["d_model"])

    BridgedBaseModel(proc, proc, proc, d_model)
end

function get_embeddings(model::BridgedBaseModel, text::String)
    resp = send_request(model, Dict("cmd" => "get_embeddings", "text" => text))
    resp["status"] == "ok" || error("get_embeddings failed: $(get(resp, "message", "unknown"))")
    shape = Int.(resp["shape"])
    embeddings = decode_float32_array(String(resp["embeddings"]), shape)
    tokens = String.(resp["tokens"])
    (embeddings, tokens)
end

function generate_candidates(model::BridgedBaseModel, prompt::String;
                              num_candidates::Int=4, max_tokens::Int=128)
    resp = send_request(model, Dict(
        "cmd" => "generate",
        "prompt" => prompt,
        "num_candidates" => num_candidates,
        "max_tokens" => max_tokens,
    ))
    resp["status"] == "ok" || error("generate failed: $(get(resp, "message", "unknown"))")
    String.(resp["candidates"])
end

function generate_conditioned(model::BridgedBaseModel, prompt::String,
                               conditioning::AbstractMatrix{Float32};
                               proj_matrix::Union{AbstractMatrix{Float32}, Nothing}=nothing,
                               num_candidates::Int=4, max_tokens::Int=128)
    req = Dict(
        "cmd" => "generate_conditioned",
        "prompt" => prompt,
        "conditioning" => encode_float32_array(conditioning),
        "cond_shape" => collect(size(conditioning)),
        "num_candidates" => num_candidates,
        "max_tokens" => max_tokens,
    )
    if proj_matrix !== nothing
        req["proj_matrix"] = encode_float32_array(proj_matrix)
        req["proj_shape"] = collect(size(proj_matrix))
    end
    resp = send_request(model, req)
    resp["status"] == "ok" || error("generate_conditioned failed: $(get(resp, "message", "unknown"))")
    String.(resp["candidates"])
end

function model_config(model::BridgedBaseModel)
    resp = send_request(model, Dict("cmd" => "config"))
    resp["status"] == "ok" || error("config failed: $(get(resp, "message", "unknown"))")
    Dict(
        "d_model" => Int(resp["d_model"]),
        "vocab_size" => Int(resp["vocab_size"]),
        "num_layers" => Int(resp["num_layers"]),
        "num_heads" => Int(resp["num_heads"]),
        "model_name" => String(resp["model_name"]),
    )
end

function shutdown!(model::BridgedBaseModel)
    try
        send_request(model, Dict("cmd" => "shutdown"))
    catch
    end
    close(model.stdin)
    wait(model.process)
end
