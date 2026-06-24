"""Latent memory (REAL) — two-phase recall + margin-δ writes (verge-engineering.md §2, §4).

Memory is the two-phase recall over the shared latent:
  1. a cheap **approximate neighborhood** — FAISS HNSW when `verge[memory]` is installed,
     else an exact numpy cosine scan (same interface, identical results at small scale);
  2. a precise **sparse reranker** (sparsemax-style) over that neighborhood for the top-k.

Writes use a **margin-δ policy**: store a latent only if it is ≥δ (cosine distance) from
its nearest existing neighbor — so memory does not flood with near-duplicates.

M2 gate: retrieval must *measurably* improve L3 pass@1 over a no-retrieval ablation
(`verge.eval`); if it doesn't, the retrieval design is wrong — fix or cut.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from verge.latent import Latent


def sparsemax(z: np.ndarray) -> np.ndarray:
    """Sparsemax (Martins & Astudillo 2016): a sparse alternative to softmax — exact
    zeros for low-scoring items, so the reranker commits to a precise top set."""
    z = np.asarray(z, dtype=np.float64)
    zs = np.sort(z)[::-1]
    cumsum = np.cumsum(zs)
    k = np.arange(1, len(z) + 1)
    cond = 1 + k * zs > cumsum
    k_z = k[cond][-1] if np.any(cond) else 1
    tau = (cumsum[k_z - 1] - 1) / k_z
    return np.maximum(z - tau, 0.0)


@dataclass
class LatentMemory:
    """Two-phase recall: (HNSW | exact) neighborhood → sparsemax reranker top-k."""

    margin_delta: float = 0.1   # write only if ≥δ cosine distance from nearest neighbor
    k: int = 8
    hnsw_neighborhood: int = 64  # first-phase candidate count
    use_faiss: bool = True       # falls back to exact numpy if faiss absent
    _latents: list = field(default_factory=list, repr=False)
    _mat: np.ndarray | None = field(default=None, repr=False)
    _index: object = field(default=None, repr=False)

    # --- writes (margin-δ) ---------------------------------------------------
    def write(self, latent: Latent) -> bool:
        """Store iff ≥δ from the nearest existing neighbor. Returns whether it was kept."""
        if self._latents:
            nearest = 1.0 - self._max_cosine(latent.z)
            if nearest < self.margin_delta:
                return False
        self._latents.append(latent)
        self._append_to_index(latent.z)
        return True

    def write_many(self, latents: list[Latent]) -> int:
        return sum(int(self.write(l)) for l in latents)

    # --- two-phase retrieval -------------------------------------------------
    def retrieve(self, query: Latent, k: int | None = None) -> list[Latent]:
        k = k or self.k
        if not self._latents:
            return []
        cand_idx = self._neighborhood(query.z, min(self.hnsw_neighborhood, len(self._latents)))
        # phase 2: sparsemax reranker over the candidate cosines (precise top-k)
        sims = self._mat[cand_idx] @ query.z.astype(np.float64)
        weights = sparsemax(sims)
        order = np.argsort(weights)[::-1]
        picked = [cand_idx[i] for i in order if weights[i] > 0][:k]
        if len(picked) < k:  # sparsemax may zero out everything but the top — backfill
            picked += [cand_idx[i] for i in order if cand_idx[i] not in picked][: k - len(picked)]
        return [self._latents[i] for i in picked]

    # --- index internals (FAISS or exact) ------------------------------------
    def _faiss(self):
        if not self.use_faiss:
            return None
        try:
            import faiss  # noqa
            return faiss
        except Exception:
            return None

    def _append_to_index(self, z: np.ndarray):
        z = z.astype(np.float64)[None, :]
        self._mat = z if self._mat is None else np.vstack([self._mat, z])
        faiss = self._faiss()
        if faiss is not None:
            if self._index is None:
                self._index = faiss.IndexHNSWFlat(z.shape[1], 32)
                self._index.metric_type = faiss.METRIC_INNER_PRODUCT
            self._index.add(z.astype(np.float32))

    def _neighborhood(self, z: np.ndarray, n: int) -> list[int]:
        faiss = self._faiss()
        if faiss is not None and self._index is not None:
            _d, idx = self._index.search(z.astype(np.float32)[None, :], n)
            return [int(i) for i in idx[0] if i != -1]
        sims = self._mat @ z.astype(np.float64)        # exact fallback
        return list(np.argsort(sims)[::-1][:n])

    def _max_cosine(self, z: np.ndarray) -> float:
        return float(np.max(self._mat @ z.astype(np.float64)))

    def health(self) -> dict:
        return {"built": True, "size": len(self._latents),
                "index": "FAISS-HNSW" if self._faiss() else "exact-numpy(fallback)",
                "reranker": "sparsemax", "write_policy": f"margin-δ={self.margin_delta}"}
