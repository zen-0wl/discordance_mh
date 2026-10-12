from __future__ import annotations

import torch
import torch.nn as nn


EPS = 1e-8


# ============================================================
# Basic density-state utilities

def symmetrize_density(rho: torch.Tensor) -> torch.Tensor:
    """Enforce numerical symmetry/Hermiticity for real density matrices."""
    return 0.5 * (rho + rho.transpose(-1, -2))


def normalize_state_vectors(
    x: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Normalize vectors along the final feature dimension."""
    norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / norms.clamp_min(eps)


# ============================================================
# Mixed density-state construction
# ============================================================

def build_mixed_density(
    sequence: torch.Tensor,
    attention_logits: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Construct an attention-weighted mixed density state.

        rho = sum_k alpha_k |psi_k><psi_k|

    sequence:
        [B, T, R]
    attention_logits:
        [B, T]
    mask:
        [B, T], True for valid positions.
    """
    states = normalize_state_vectors(sequence, eps)

    masked_logits = attention_logits.masked_fill(~mask, -1e9)
    alpha = torch.softmax(masked_logits, dim=-1)

    alpha = alpha * mask.to(alpha.dtype)
    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(eps)

    outer = states.unsqueeze(-1) * states.unsqueeze(-2)

    rho = torch.sum(
        alpha.unsqueeze(-1).unsqueeze(-1) * outer,
        dim=1,
    )

    rho = symmetrize_density(rho)

    trace = torch.diagonal(
        rho,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1, keepdim=True)

    rho = rho / trace.unsqueeze(-1).clamp_min(eps)
    return rho


# ============================================================
# Quantum-inspired descriptors
# ============================================================

def density_eigenvalues(rho: torch.Tensor) -> torch.Tensor:
    """Return non-negative eigenvalues of a symmetric density matrix."""
    rho = symmetrize_density(rho)
    eigenvalues = torch.linalg.eigvalsh(rho)
    return torch.clamp(eigenvalues, min=0.0)


def von_neumann_entropy(
    rho: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """S(rho) = -Tr(rho log rho)."""
    eigenvalues = density_eigenvalues(rho)
    safe = eigenvalues.clamp_min(eps)
    return -torch.sum(
        eigenvalues * torch.log(safe),
        dim=-1,
    )


def purity(rho: torch.Tensor) -> torch.Tensor:
    """P(rho) = Tr(rho^2)."""
    rho = symmetrize_density(rho)
    rho_squared = rho @ rho
    return torch.diagonal(
        rho_squared,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)


def trace_distance(
    rho_a: torch.Tensor,
    rho_b: torch.Tensor,
) -> torch.Tensor:
    """D(rho_a,rho_b) = 1/2 ||rho_a-rho_b||_1."""
    delta = symmetrize_density(rho_a - rho_b)
    eigenvalues = torch.linalg.eigvalsh(delta)
    return 0.5 * torch.sum(torch.abs(eigenvalues), dim=-1)


def fidelity(
    rho_a: torch.Tensor,
    rho_b: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Uhlmann fidelity:

        F(rho_a,rho_b)
          = [Tr sqrt(sqrt(rho_a) rho_b sqrt(rho_a))]^2
    """
    rho_a = symmetrize_density(rho_a)
    rho_b = symmetrize_density(rho_b)

    eig_a, vec_a = torch.linalg.eigh(rho_a)
    eig_a = torch.clamp(eig_a, min=0.0)

    sqrt_a = (
        vec_a
        @ torch.diag_embed(
            torch.sqrt(eig_a.clamp_min(eps))
        )
        @ vec_a.transpose(-1, -2)
    )

    middle = sqrt_a @ rho_b @ sqrt_a
    middle = symmetrize_density(middle)

    eig_middle = torch.linalg.eigvalsh(middle)
    eig_middle = torch.clamp(eig_middle, min=0.0)

    root_fidelity = torch.sum(
        torch.sqrt(eig_middle.clamp_min(eps)),
        dim=-1,
    )

    return torch.clamp(
        root_fidelity ** 2,
        min=0.0,
        max=1.0,
    )


def sadi(
    rho_audio: torch.Tensor,
    rho_text: torch.Tensor,
) -> torch.Tensor:
    """Semantic-Affective Discordance Index = 1 - fidelity."""
    return 1.0 - fidelity(rho_audio, rho_text)


# ============================================================
# Speech-conditioned Kraus evolution

class SpeechConditionedKraus(nn.Module):
    """
    Generate a trace-preserving Kraus channel for the text state,
    conditioned on the speech density state.

    Input:
        rho_text  [B, R, R]
        rho_audio [B, R, R]

    Output:
        rho_text_post [B, R, R]
        kraus_ops     [B, K, R, R]

    The channel is normalized so that:

        sum_k K_k^T K_k = I

    and is applied to the text state as:

        Phi_A(rho_T) = sum_k K_k rho_T K_k^T

    A residual identity mixture controlled by `strength` keeps the
    evolution stable while retaining the speech-conditioned channel.
    """

    def __init__(
        self,
        rank: int = 16,
        num_kraus: int = 4,
        hidden_dim: int = 256,
        strength: float = 0.25,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be positive.")
        if num_kraus <= 0:
            raise ValueError("num_kraus must be positive.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1].")

        self.rank = rank
        self.num_kraus = num_kraus
        self.strength = strength

        state_dim = rank * rank

        self.conditioner = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_kraus * state_dim),
        )

    def _trace_preserving_kraus(
        self,
        raw_ops: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize raw operators so sum_k K_k^T K_k = I.

        raw_ops:
            [B, K, R, R]
        """
        batch_size = raw_ops.size(0)
        device = raw_ops.device
        dtype = raw_ops.dtype

        gram = torch.einsum(
            "bkri,bkrj->bij",
            raw_ops,
            raw_ops,
        )

        eye = torch.eye(
            self.rank,
            device=device,
            dtype=dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        gram = symmetrize_density(gram)
        gram = gram + 1e-5 * eye

        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        inv_sqrt = (
            eigenvectors
            @ torch.diag_embed(
                torch.rsqrt(eigenvalues.clamp_min(1e-8))
            )
            @ eigenvectors.transpose(-1, -2)
        )

        kraus_ops = torch.matmul(
            raw_ops,
            inv_sqrt.unsqueeze(1),
        )

        return kraus_ops

    def forward(
        self,
        rho_text: torch.Tensor,
        rho_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if rho_text.ndim != 3 or rho_audio.ndim != 3:
            raise ValueError(
                "rho_text and rho_audio must have shape [B, R, R]."
            )

        if rho_text.shape != rho_audio.shape:
            raise ValueError(
                "rho_text and rho_audio must have identical shapes. "
                f"Got {tuple(rho_text.shape)} and {tuple(rho_audio.shape)}."
            )

        if rho_text.size(-1) != self.rank:
            raise ValueError(
                f"Expected rank {self.rank}, got {rho_text.size(-1)}."
            )

        audio_condition = rho_audio.reshape(
            rho_audio.size(0),
            -1,
        )

        raw = self.conditioner(audio_condition)
        raw_ops = raw.view(
            rho_audio.size(0),
            self.num_kraus,
            self.rank,
            self.rank,
        )

        kraus_ops = self._trace_preserving_kraus(raw_ops)

        evolved_text = torch.zeros_like(rho_text)

        for k in range(self.num_kraus):
            K = kraus_ops[:, k]
            evolved_text = evolved_text + (
                K @ rho_text @ K.transpose(-1, -2)
            )

        evolved_text = symmetrize_density(evolved_text)

        # Stable residual channel:
        # Phi_strength(rho) = (1-s)rho + s Phi(rho)
        rho_text_post = (
            (1.0 - self.strength) * rho_text
            + self.strength * evolved_text
        )

        rho_text_post = symmetrize_density(rho_text_post)

        trace = torch.diagonal(
            rho_text_post,
            dim1=-2,
            dim2=-1,
        ).sum(dim=-1, keepdim=True)

        rho_text_post = (
            rho_text_post
            / trace.unsqueeze(-1).clamp_min(EPS)
        )

        return rho_text_post, kraus_ops


def apply_kraus_channel(
    rho_text: torch.Tensor,
    rho_audio: torch.Tensor,
    kraus_channel: SpeechConditionedKraus,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper for speech-conditioned text evolution."""
    return kraus_channel(rho_text, rho_audio)


# ============================================================
# Descriptor extraction

def compute_v5_quantum_descriptors(
    rho_audio: torch.Tensor,
    rho_text_pre: torch.Tensor,
    rho_text_post: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute V5 descriptors.

    V5 reports the pre-evolution text state as well as the
    speech-conditioned post-evolution text state. Discordance
    descriptors used downstream are computed against the updated
    text state.
    """
    return {
        "audio_entropy": von_neumann_entropy(rho_audio),
        "text_entropy_pre": von_neumann_entropy(rho_text_pre),
        "text_entropy": von_neumann_entropy(rho_text_post),
        "audio_purity": purity(rho_audio),
        "text_purity_pre": purity(rho_text_pre),
        "text_purity": purity(rho_text_post),
        "fidelity": fidelity(rho_audio, rho_text_post),
        "trace_distance": trace_distance(
            rho_audio,
            rho_text_post,
        ),
        "sadi": sadi(
            rho_audio,
            rho_text_post,
        ),
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    rank = 16
    batch_size = 2

    audio = torch.randn(batch_size, rank, rank)
    text = torch.randn(batch_size, rank, rank)

    audio = build_mixed_density(
        audio,
        torch.randn(batch_size, rank),
        torch.ones(batch_size, rank, dtype=torch.bool),
    )
    text = build_mixed_density(
        text,
        torch.randn(batch_size, rank),
        torch.ones(batch_size, rank, dtype=torch.bool),
    )

    channel = SpeechConditionedKraus(
        rank=rank,
        num_kraus=4,
        strength=0.25,
    )

    text_post, kraus = channel(text, audio)

    audio_trace = torch.diagonal(audio, dim1=-2, dim2=-1).sum(-1)
    text_trace = torch.diagonal(text, dim1=-2, dim2=-1).sum(-1)
    post_trace = torch.diagonal(text_post, dim1=-2, dim2=-1).sum(-1)

    print("V5 quantum test")
    print("-" * 60)
    print("rho_audio:", audio.shape)
    print("rho_text_pre:", text.shape)
    print("rho_text_post:", text_post.shape)
    print("kraus_ops:", kraus.shape)
    print("audio traces:", audio_trace)
    print("text traces:", text_trace)
    print("post-text traces:", post_trace)
    print("-" * 60)
    print("V5 quantum test: PASS")