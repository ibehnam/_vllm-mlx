# SPDX-License-Identifier: Apache-2.0
"""
Logprobs extraction and formatting utilities.

Provides efficient top-k extraction from full vocabulary log-probability
arrays (mx.array) and conversion to OpenAI API format.
"""

from typing import Any

import mlx.core as mx


def extract_top_logprobs(
    logprobs: mx.array,
    token_id: int,
    top_k: int,
    tokenizer: Any,
) -> dict:
    """
    Extract top-k log probabilities from a full vocabulary logprobs array.

    Uses mx.argpartition for O(n) top-k extraction (no full sort needed).

    Args:
        logprobs: Full vocabulary log-probabilities, shape [vocab_size].
        token_id: The actually sampled token ID.
        top_k: Number of top tokens to include.
        tokenizer: Tokenizer for decoding token IDs to strings.

    Returns:
        Dict with keys: token_id, token, logprob, top_logprobs (list of dicts).
    """
    sampled_logprob = logprobs[token_id].item()
    sampled_token_str = tokenizer.decode([token_id])

    top_entries = []
    if top_k > 0:
        vocab_size = logprobs.shape[0]
        k = min(top_k, vocab_size)

        top_indices = mx.argpartition(-logprobs, kth=k - 1)[:k]
        top_lps = logprobs[top_indices]

        top_ids = top_indices.tolist()
        top_lp_vals = top_lps.tolist()

        # Sort by logprob descending for consistent output
        paired = sorted(zip(top_ids, top_lp_vals), key=lambda x: -x[1])

        for tid, lp in paired:
            tok_str = tokenizer.decode([tid])
            top_entries.append({
                "token_id": tid,
                "token": tok_str,
                "logprob": lp,
            })

        # Ensure sampled token is in top_logprobs
        sampled_in_top = any(e["token_id"] == token_id for e in top_entries)
        if not sampled_in_top:
            top_entries.append({
                "token_id": token_id,
                "token": sampled_token_str,
                "logprob": sampled_logprob,
            })

    return {
        "token_id": token_id,
        "token": sampled_token_str,
        "logprob": sampled_logprob,
        "top_logprobs": top_entries,
    }


def format_logprobs_for_api(token_logprobs: list[dict]) -> dict:
    """
    Convert internal token logprobs list to OpenAI ChoiceLogprobs format.

    Args:
        token_logprobs: List of dicts from extract_top_logprobs().

    Returns:
        Dict matching OpenAI ChoiceLogprobs schema:
        {"content": [{"token", "logprob", "bytes", "top_logprobs": [...]}]}
    """
    content = []
    for entry in token_logprobs:
        token_str = entry["token"]
        token_bytes = list(token_str.encode("utf-8"))

        top_logprobs = []
        for top in entry.get("top_logprobs", []):
            top_str = top["token"]
            top_logprobs.append({
                "token": top_str,
                "logprob": top["logprob"],
                "bytes": list(top_str.encode("utf-8")),
            })

        content.append({
            "token": token_str,
            "logprob": entry["logprob"],
            "bytes": token_bytes,
            "top_logprobs": top_logprobs,
        })

    return {"content": content}
