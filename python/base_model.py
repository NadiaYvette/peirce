"""
Base Model Interface

Wraps a HuggingFace causal LM (Qwen2.5-0.5B) to provide:
  1. Token embeddings for the sign-formation module
  2. Conditioned decoding for the realization module

The Julia pipeline calls this via PythonCall.jl.
For standalone testing, this module can be used directly from Python.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).parent.parent / "models" / "qwen2.5-0.5b"


class BaseModel:
    def __init__(self, model_dir: str | Path = MODEL_DIR, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        self.d_model = self.model.config.hidden_size

    def get_embeddings(self, text: str) -> tuple[np.ndarray, list[str]]:
        """
        Tokenize text and return token embeddings from the model's embedding layer.

        Returns:
            embeddings: (d_model, T) float32 numpy array
            tokens: list of token strings, length T
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        token_ids = inputs["input_ids"][0]

        with torch.no_grad():
            # Get embeddings from the model's embedding layer
            embeddings = self.model.model.embed_tokens(token_ids)  # (T, d_model)

        # Convert token IDs back to strings for rigid designation lookup
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return embeddings.T.numpy().astype(np.float32), tokens

    def _project_conditioning_to_embeds(
        self, conditioning: np.ndarray,
        proj_matrix: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Project semiotic conditioning signal into token embedding space as soft prompts.

        conditioning: (d_cond, M) numpy array from the semiotic pipeline
        proj_matrix: (d_cond, d_model) projection matrix used to compress embeddings.
                     If provided, used directly to map d_cond back to d_model space
                     (row-vector convention). Keeps conditioning in the original subspace.
        Returns: (M, d_model) tensor of soft prompt embeddings
        """
        d_cond, M = conditioning.shape
        cond_t = torch.tensor(conditioning.T, dtype=torch.float32, device=self.device)  # (M, d_cond)

        if d_cond != self.d_model:
            with torch.no_grad():
                if proj_matrix is not None:
                    # proj_matrix is (d_cond, d_model) e.g. (128, 896)
                    # Forward: proj_matrix @ x maps d_model → d_cond
                    # Backward: x @ proj_matrix maps (M, d_cond) → (M, d_model)
                    pm = torch.tensor(proj_matrix, dtype=torch.float32, device=self.device)  # (d_cond, d_model)
                    soft_prompts = cond_t @ pm  # (M, d_cond) @ (d_cond, d_model) = (M, d_model)
                else:
                    # Fallback: random projection (not recommended)
                    if not hasattr(self, "_cond_proj") or self._cond_proj.in_features != d_cond:
                        self._cond_proj = torch.nn.Linear(d_cond, self.d_model, bias=False).to(self.device)
                        torch.nn.init.xavier_uniform_(self._cond_proj.weight)
                        self._cond_proj.eval()
                    soft_prompts = self._cond_proj(cond_t)
        else:
            soft_prompts = cond_t

        # Scale to match embedding norm distribution
        with torch.no_grad():
            sample_ids = torch.arange(100, 110, device=self.device)
            real_embeds = self.model.model.embed_tokens(sample_ids)
            target_norm = real_embeds.norm(dim=-1).mean()
            current_norm = soft_prompts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            soft_prompts = soft_prompts * (target_norm / current_norm)

        return soft_prompts

    def generate_conditioned(
        self,
        prompt_text: str,
        conditioning: np.ndarray | None = None,
        proj_matrix: np.ndarray | None = None,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        num_return_sequences: int = 4,
    ) -> list[str]:
        """
        Generate candidate responses, optionally conditioned on semiotic signals.

        When conditioning is provided, it is projected into the embedding space and
        prepended as soft prompts before the actual text tokens. This biases generation
        toward the semiotic context without modifying model weights.

        Args:
            prompt_text: input text
            conditioning: (d_cond, M) semiotic conditioning signal, or None
            max_new_tokens: maximum tokens to generate
            num_beams: beam width
            num_return_sequences: number of candidates to return

        Returns:
            list of candidate response strings
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        if conditioning is not None:
            # Soft-prompt injection: prepend conditioning embeddings
            soft_prompts = self._project_conditioning_to_embeds(conditioning, proj_matrix)  # (M, d_model)

            with torch.no_grad():
                # Get token embeddings for the actual prompt
                token_embeds = self.model.model.embed_tokens(inputs["input_ids"])  # (1, T, d_model)
                # Prepend soft prompts
                combined = torch.cat([
                    soft_prompts.unsqueeze(0),  # (1, M, d_model)
                    token_embeds,               # (1, T, d_model)
                ], dim=1)  # (1, M+T, d_model)

                # Create attention mask for the combined sequence
                M = soft_prompts.shape[0]
                T = inputs["input_ids"].shape[1]
                attention_mask = torch.ones(1, M + T, dtype=torch.long, device=self.device)

                outputs = self.model.generate(
                    inputs_embeds=combined,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    do_sample=False,
                    early_stopping=True,
                )

            # When using inputs_embeds, generate() returns only new token IDs
            # (no input token IDs to strip, since we passed embeddings not IDs)
            prompt_len = 0
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    do_sample=False,
                    early_stopping=True,
                )
            prompt_len = inputs["input_ids"].shape[1]

        candidates = []
        for seq in outputs:
            text = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            candidates.append(text)

        return candidates

    def get_config(self) -> dict:
        """Return model configuration relevant to the semiotic pipeline."""
        return {
            "d_model": self.d_model,
            "vocab_size": self.model.config.vocab_size,
            "num_layers": self.model.config.num_hidden_layers,
            "num_heads": self.model.config.num_attention_heads,
            "model_name": "Qwen2.5-0.5B",
        }


def test_base_model():
    """Quick smoke test."""
    print(f"Loading model from {MODEL_DIR}...")
    bm = BaseModel()
    config = bm.get_config()
    print(f"Config: {json.dumps(config, indent=2)}")

    text = "The cat sat on the mat."
    embeddings, tokens = bm.get_embeddings(text)
    print(f"Input: {text}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Embeddings shape: {embeddings.shape}")  # (d_model, T)

    candidates = bm.generate_conditioned(
        "The meaning of life is",
        max_new_tokens=32,
        num_beams=4,
        num_return_sequences=4,
    )
    print(f"\nGeneration candidates:")
    for i, c in enumerate(candidates):
        print(f"  {i+1}: {c}")


if __name__ == "__main__":
    test_base_model()
