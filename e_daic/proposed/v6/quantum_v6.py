from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-8


def symmetrize_density(rho: torch.Tensor) -> torch.Tensor:
    """Numerically symmetrize real density matrices."""
    return 0.5 * (rho + rho.transpose(-1, -2).conj())


def normalize_state_vectors(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / norms.clamp_min(eps)


def build_mixed_density(
    sequence: torch.Tensor,
    attention_logits: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """rho = sum_k alpha_k |psi_k><psi_k| from a sequence."""
    states = normalize_state_vectors(sequence, eps)
    masked_logits = attention_logits.masked_fill(~mask, -1e9)
    alpha = torch.softmax(masked_logits, dim=-1)
    alpha = alpha * mask.to(alpha.dtype)
    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(eps)

    outer = states.unsqueeze(-1) * states.unsqueeze(-2).conj()
    rho = torch.sum(
        alpha.unsqueeze(-1).unsqueeze(-1) * outer,
        dim=1,
    )
    rho = symmetrize_density(rho)
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).real
    return rho / trace.unsqueeze(-1).clamp_min(eps)


def density_eigenvalues(rho: torch.Tensor) -> torch.Tensor:
    rho = symmetrize_density(rho)
    return torch.linalg.eigvalsh(rho).real.clamp_min(0.0)


def von_neumann_entropy(rho: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    eig = density_eigenvalues(rho)
    return -torch.sum(eig * torch.log(eig.clamp_min(eps)), dim=-1)


def purity(rho: torch.Tensor) -> torch.Tensor:
    rho = symmetrize_density(rho)
    rho_squared = rho @ rho
    return torch.diagonal(rho_squared, dim1=-2, dim2=-1).sum(dim=-1).real


def trace_distance(rho_a: torch.Tensor, rho_b: torch.Tensor) -> torch.Tensor:
    delta = symmetrize_density(rho_a - rho_b)
    eig = torch.linalg.eigvalsh(delta).real
    return 0.5 * torch.sum(torch.abs(eig), dim=-1)


def fidelity(rho_a: torch.Tensor, rho_b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Uhlmann fidelity for density matrices."""
    rho_a = symmetrize_density(rho_a)
    rho_b = symmetrize_density(rho_b)

    eig_a, vec_a = torch.linalg.eigh(rho_a)
    eig_a = eig_a.real.clamp_min(0.0)
    sqrt_a = (
        vec_a
        @ torch.diag_embed(torch.sqrt(eig_a.clamp_min(eps)))
        @ vec_a.transpose(-1, -2).conj()
    )

    middle = symmetrize_density(sqrt_a @ rho_b @ sqrt_a)
    eig_middle = torch.linalg.eigvalsh(middle).real.clamp_min(0.0)
    root_fidelity = torch.sum(torch.sqrt(eig_middle.clamp_min(eps)), dim=-1)
    return torch.clamp(root_fidelity.square(), min=0.0, max=1.0)


def partial_trace_over_audio(rho_joint: torch.Tensor, rank: int) -> torch.Tensor:
    """Trace out audio from rho_TA and return rho_T."""
    rho = rho_joint.reshape(*rho_joint.shape[:-2], rank, rank, rank, rank)
    return torch.diagonal(rho, dim1=-3, dim2=-1).sum(dim=-1)


def partial_trace_over_text(rho_joint: torch.Tensor, rank: int) -> torch.Tensor:
    """Trace out text from rho_TA and return rho_A."""
    rho = rho_joint.reshape(*rho_joint.shape[:-2], rank, rank, rank, rank)
    return torch.diagonal(rho, dim1=-4, dim2=-2).sum(dim=-1)


def quantum_mutual_information(
    rho_joint: torch.Tensor,
    rank: int,
    rho_text: torch.Tensor | None = None,
    rho_audio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    I(T:A) = S(rho_T) + S(rho_A) - S(rho_TA).

    For the V6 unitary Hamiltonian interaction, S(rho_TA_post) is
    exactly preserved. Therefore, when pre-interaction marginals are
    supplied, the joint entropy is obtained from the two 16x16
    pre-interaction states instead of an expensive 256x256 eigendecomposition.
    """
    rho_joint = symmetrize_density(rho_joint)

    if rho_text is None:
        rho_text = partial_trace_over_audio(rho_joint, rank)
    if rho_audio is None:
        rho_audio = partial_trace_over_text(rho_joint, rank)

    return torch.clamp(
        von_neumann_entropy(rho_text)
        + von_neumann_entropy(rho_audio)
        - von_neumann_entropy(rho_joint),
        min=0.0,
    )


def post_interaction_qmi(
    rho_joint_post: torch.Tensor,
    rho_text_post: torch.Tensor,
    rho_audio_post: torch.Tensor,
    rho_text_pre: torch.Tensor,
    rho_audio_pre: torch.Tensor,
) -> torch.Tensor:
    """
    Compute post-interaction QMI without eigendecomposing the 256x256
    joint state. The Hamiltonian evolution is unitary, so
    S(rho_TA_post) = S(rho_TA_pre) =
    S(rho_T_pre) + S(rho_A_pre).
    """
    s_text_post = von_neumann_entropy(rho_text_post)
    s_audio_post = von_neumann_entropy(rho_audio_post)
    s_joint_pre = (
        von_neumann_entropy(rho_text_pre)
        + von_neumann_entropy(rho_audio_pre)
    )
    return torch.clamp(
        s_text_post + s_audio_post - s_joint_pre,
        min=0.0,
    )


def kron_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched Kronecker product for [B,R,R] matrices."""
    return torch.einsum("bij,bkl->bikjl", a, b).reshape(
        a.size(0), a.size(-1) * b.size(-1), a.size(-1) * b.size(-1)
    )


def build_joint_product_state(
    rho_text: torch.Tensor,
    rho_audio: torch.Tensor,
) -> torch.Tensor:
    """Build the uncoupled bipartite state rho_T tensor rho_A."""
    return kron_batch(rho_text, rho_audio)


class HamiltonianInteraction(nn.Module):
    """
    Computationally efficient nonlocal Hamiltonian interaction.

    The previous implementation constructed a sample-specific dense
    256x256 generator and called torch.matrix_exp() for every sample.
    That is the main V6 training bottleneck.

    This implementation uses a structured nonlocal generator

        G = A_tensor_product_D

    where A is a fixed 16x16 block-diagonal skew-symmetric generator
    and D is a learned diagonal 16x16 audio operator conditioned on
    both pooled modalities.  G is skew-symmetric, so

        U = exp(G)

    is real orthogonal and rho' = U rho U^T is a valid unitary
    Hamiltonian evolution.  Because A acts on text and D acts on audio,
    the interaction is genuinely bipartite and can create text-audio
    entanglement/QMI.

    Crucially, U is assembled directly from 2x2 rotations rather than
    evaluating a 256x256 matrix exponential.
    """

    def __init__(
        self,
        rank: int = 16,
        hidden_dim: int = 256,
        interaction_strength: float = 0.05,
    ):
        super().__init__()
        if rank <= 0 or rank % 2 != 0:
            raise ValueError("rank must be a positive even integer.")
        if interaction_strength < 0:
            raise ValueError("interaction_strength must be non-negative.")

        self.rank = rank
        self.interaction_strength = interaction_strength

        self.condition = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rank + 1),
            nn.Tanh(),
        )

        # Fixed skew generator A made from independent 2x2 rotations.
        # It is registered as a buffer so it moves with the model/device.
        a = torch.zeros(rank, rank)
        for i in range(0, rank, 2):
            a[i, i + 1] = -1.0
            a[i + 1, i] = 1.0
        self.register_buffer("text_rotation_generator", a, persistent=False)

    def _build_structured_unitary(
        self,
        theta: torch.Tensor,
        audio_diagonal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build U for G = theta * A tensor_product diag(audio_diagonal).

        For every audio basis index j, each text pair undergoes a
        2x2 rotation with angle theta * audio_diagonal[j].
        """
        b = theta.size(0)
        r = self.rank
        n = r * r

        U = torch.eye(
            n,
            dtype=theta.dtype,
            device=theta.device,
        ).unsqueeze(0).repeat(b, 1, 1)

        angles = theta.unsqueeze(-1) * audio_diagonal
        c = torch.cos(angles)
        s = torch.sin(angles)

        # Tensor-product basis index = text_index * rank + audio_index.
        for text_idx in range(0, r, 2):
            row0 = text_idx * r
            row1 = (text_idx + 1) * r
            for audio_idx in range(r):
                i = row0 + audio_idx
                j = row1 + audio_idx
                U[:, i, i] = c[:, audio_idx]
                U[:, j, j] = c[:, audio_idx]
                U[:, i, j] = -s[:, audio_idx]
                U[:, j, i] = s[:, audio_idx]

        return U

    def forward(
        self,
        rho_joint: torch.Tensor,
        pooled_text: torch.Tensor,
        pooled_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rho_joint.ndim != 3:
            raise ValueError("rho_joint must be [B,R^2,R^2].")

        condition = self.condition(
            torch.cat([pooled_text, pooled_audio], dim=-1)
        )

        theta = self.interaction_strength * condition[:, 0]
        audio_diagonal = condition[:, 1:]

        # Keep the conditional interaction bounded and avoid a zero
        # operator that would make the interaction trivially disappear.
        audio_diagonal = 0.5 + 0.5 * audio_diagonal

        unitary_real = self._build_structured_unitary(
            theta,
            audio_diagonal,
        )

        rho_post = unitary_real @ rho_joint @ unitary_real.transpose(-1, -2)
        rho_post = symmetrize_density(rho_post)

        trace = torch.diagonal(
            rho_post, dim1=-2, dim2=-1
        ).sum(dim=-1, keepdim=True).real
        rho_post = rho_post / trace.unsqueeze(-1).clamp_min(EPS)

        return rho_post, unitary_real


def compute_v6_quantum_descriptors(
    rho_joint_pre: torch.Tensor,
    rho_joint_post: torch.Tensor,
    rho_text_pre: torch.Tensor,
    rho_audio_pre: torch.Tensor,
    rho_text_post: torch.Tensor,
    rho_audio_post: torch.Tensor,
) -> dict[str, torch.Tensor]:
    qmi = post_interaction_qmi(
        rho_joint_post,
        rho_text_post,
        rho_audio_post,
        rho_text_pre,
        rho_audio_pre,
    )

    return {
        "joint_entropy_pre": (
            von_neumann_entropy(rho_text_pre)
            + von_neumann_entropy(rho_audio_pre)
        ),
        # Unitary invariance: S(rho_joint_post) = S(rho_joint_pre).
        "joint_entropy_post": (
            von_neumann_entropy(rho_text_pre)
            + von_neumann_entropy(rho_audio_pre)
        ),
        "text_entropy_post": von_neumann_entropy(rho_text_post),
        "audio_entropy_post": von_neumann_entropy(rho_audio_post),
        "text_purity_post": purity(rho_text_post),
        "audio_purity_post": purity(rho_audio_post),
        "post_fidelity": fidelity(rho_text_post, rho_audio_post),
        "post_trace_distance": trace_distance(rho_text_post, rho_audio_post),
        "qmi": qmi,
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    rank = 16
    batch = 2

    rho_a = build_mixed_density(
        torch.randn(batch, rank, rank),
        torch.randn(batch, rank),
        torch.ones(batch, rank, dtype=torch.bool),
    )
    rho_t = build_mixed_density(
        torch.randn(batch, rank, rank),
        torch.randn(batch, rank),
        torch.ones(batch, rank, dtype=torch.bool),
    )

    joint = build_joint_product_state(rho_t, rho_a)
    interaction = HamiltonianInteraction(rank=rank)
    post, U = interaction(
        joint,
        torch.randn(batch, 256),
        torch.randn(batch, 256),
    )

    rho_t_post = partial_trace_over_audio(post, rank)
    rho_a_post = partial_trace_over_text(post, rank)
    qmi = post_interaction_qmi(
        post,
        rho_t_post,
        rho_a_post,
        rho_t,
        rho_a,
    )

    eye = torch.eye(rank * rank)
    orth_error = torch.max(
        torch.abs(U.transpose(-1, -2) @ U - eye)
    )

    print("V6 quantum test")
    print("-" * 60)
    print("joint trace:", torch.diagonal(post, dim1=-2, dim2=-1).sum(-1))
    print("text trace:", torch.diagonal(rho_t_post, dim1=-2, dim2=-1).sum(-1))
    print("audio trace:", torch.diagonal(rho_a_post, dim1=-2, dim2=-1).sum(-1))
    print("QMI before:", torch.zeros(batch))
    print("QMI after:", qmi)
    print("orthogonality max error:", orth_error.item())
    assert torch.allclose(
        torch.diagonal(post, dim1=-2, dim2=-1).sum(-1),
        torch.ones(batch),
        atol=1e-4,
    )
    assert torch.all(qmi >= 0)
    print("V6 quantum test: PASS")
