"""Memory (STUB) — two-phase recall over the shared latent (verge-engineering.md §2, §4).

Open component: **FAISS** HNSW index for the cheap approximate neighborhood, then a
sparse (sparsemax-style) reranker for the precise top-k. Writes use a margin-δ policy
(store a latent only if it is ≥δ from existing neighbors — avoids flooding memory with
near-duplicates). M2 gate: retrieval *measurably* improves L3 pass@1 over a no-retrieval
ablation (otherwise the retrieval design is wrong — fix or cut).
"""
from verge.memory.store import LatentMemory

__all__ = ["LatentMemory"]
