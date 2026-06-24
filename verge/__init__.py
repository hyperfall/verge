"""VERGE — a layered, verifier-bounded reasoning architecture.

This package realizes the repo layout in `verge-engineering.md` §9. L3 is fully
implemented (by orchestrating the measured `m1-verified-reasoner/` code behind the
new interfaces); L1/L2/L4/L5/L6/L7 are typed, compiling stubs that each name their
open component. The shared-latent contract (`verge.latent`), the workspace bus
(`verge.workspace`), Ring 0 (`verge.ring0`), and the eval harness (`verge.eval`)
are real and tested on CPU with no GPU.

The discipline is the spec's: the verifier is a *bounded amplifier* (spec §2), the
verifier is the only trusted asset and is never learned (Ring 0), and every layer
ships behind the eval harness with a pre-registered slope+CI.
"""

from verge.latent import LATENT_DIM, Latent, LayerService

__all__ = ["LATENT_DIM", "Latent", "LayerService", "__version__"]

__version__ = "0.1.0"
