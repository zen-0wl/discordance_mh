from __future__ import annotations

import torch
import torch.nn as nn

from src.proposed.v6.quantum_v6 import (
    HamiltonianInteraction,
    build_joint_product_state,
    build_mixed_density,
    fidelity,
    partial_trace_over_audio,
    partial_trace_over_text,
    post_interaction_qmi,
    purity,
    trace_distance,
    von_neumann_entropy,
)


class ProposedModelV6(nn.Module):
    """
    V6 Bipartite Hamiltonian Interaction Model.

    Pipeline:
        768-D HuBERT/DeBERTa sequences
            ↓
        256-D modality projections
            ↓
        Bidirectional cross-attention
            ↓
        Attention-weighted mixed density states (rank=16)
            ↓
        Bipartite product state rho_T ⊗ rho_A
            ↓
        Structured nonlocal Hamiltonian interaction
            ↓
        Post-interaction reduced states
            ↓
        Fidelity discordance + trace distance + QMI
            ↓
        Reliability-aware gating
            ↓
        Multimodal fusion
            ↓
        Binary classifier

    The quantum implementation is deliberately structured so training
    does not perform a dense 256x256 matrix exponential or a 256x256
    eigendecomposition for every sample.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        rank: int = 16,
        num_heads: int = 4,
        num_classes: int = 2,
        interaction_strength: float = 0.05,
        projection_dropout: float = 0.30,
        fusion_dropout_1: float = 0.40,
        fusion_dropout_2: float = 0.30,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )

        self.audio_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.text_to_audio_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.audio_pool = nn.Linear(hidden_dim, 1)
        self.text_pool = nn.Linear(hidden_dim, 1)

        self.audio_state_projection = nn.Linear(hidden_dim, rank)
        self.text_state_projection = nn.Linear(hidden_dim, rank)
        self.audio_state_attention = nn.Linear(hidden_dim, 1)
        self.text_state_attention = nn.Linear(hidden_dim, 1)

        self.hamiltonian = HamiltonianInteraction(
            rank=rank,
            hidden_dim=hidden_dim,
            interaction_strength=interaction_strength,
        )

        self.post_audio_state_projection = nn.Sequential(
            nn.Linear(rank * rank, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.post_text_state_projection = nn.Sequential(
            nn.Linear(rank * rank, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        # Reliability features remain tied to V6's quantum descriptors.
        self.reliability_gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
        )

        fusion_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(fusion_dropout_1),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout_2),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def attention_pooling(
        features: torch.Tensor,
        mask: torch.Tensor,
        pool_layer: nn.Module,
    ) -> torch.Tensor:
        logits = pool_layer(features).squeeze(-1)
        logits = logits.masked_fill(
            ~mask,
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights.unsqueeze(-1) * features, dim=1)

    def build_modality_density(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        state_projection: nn.Module,
        state_attention: nn.Module,
    ) -> torch.Tensor:
        state_vectors = state_projection(features)
        attention_logits = state_attention(features).squeeze(-1)
        return build_mixed_density(
            state_vectors,
            attention_logits,
            mask,
        )

    def forward(
        self,
        audio_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        if audio_embedding.ndim != 3 or text_embedding.ndim != 3:
            raise ValueError("Audio and text inputs must be [B,T,768].")
        if audio_embedding.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected audio feature dimension {self.input_dim}."
            )
        if text_embedding.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected text feature dimension {self.input_dim}."
            )

        if audio_mask is None:
            audio_mask = torch.ones(
                audio_embedding.shape[:2],
                dtype=torch.bool,
                device=audio_embedding.device,
            )
        if text_mask is None:
            text_mask = torch.ones(
                text_embedding.shape[:2],
                dtype=torch.bool,
                device=text_embedding.device,
            )

        audio = self.audio_projection(audio_embedding)
        text = self.text_projection(text_embedding)

        audio_attended, _ = self.audio_to_text_attention(
            query=audio,
            key=text,
            value=text,
            key_padding_mask=~text_mask,
            need_weights=False,
        )

        text_attended, _ = self.text_to_audio_attention(
            query=text,
            key=audio,
            value=audio,
            key_padding_mask=~audio_mask,
            need_weights=False,
        )

        pooled_audio = self.attention_pooling(
            audio_attended,
            audio_mask,
            self.audio_pool,
        )
        pooled_text = self.attention_pooling(
            text_attended,
            text_mask,
            self.text_pool,
        )

        rho_audio_pre = self.build_modality_density(
            audio_attended,
            audio_mask,
            self.audio_state_projection,
            self.audio_state_attention,
        )
        rho_text_pre = self.build_modality_density(
            text_attended,
            text_mask,
            self.text_state_projection,
            self.text_state_attention,
        )

        rho_joint_pre = build_joint_product_state(
            rho_text_pre,
            rho_audio_pre,
        )

        rho_joint_post, interaction_unitary = self.hamiltonian(
            rho_joint_pre,
            pooled_text,
            pooled_audio,
        )

        rho_text_post = partial_trace_over_audio(
            rho_joint_post,
            self.rank,
        )
        rho_audio_post = partial_trace_over_text(
            rho_joint_post,
            self.rank,
        )

        rho_text_post = 0.5 * (
            rho_text_post + rho_text_post.transpose(-1, -2)
        )
        rho_audio_post = 0.5 * (
            rho_audio_post + rho_audio_post.transpose(-1, -2)
        )

        text_entropy = von_neumann_entropy(rho_text_post)
        audio_entropy = von_neumann_entropy(rho_audio_post)
        text_purity = purity(rho_text_post)
        audio_purity = purity(rho_audio_post)

        post_fidelity = fidelity(
            rho_text_post,
            rho_audio_post,
        )
        post_trace_distance = trace_distance(
            rho_text_post,
            rho_audio_post,
        )

        # IMPORTANT: QMI uses only 16x16 eigendecompositions.
        # The joint entropy is obtained by unitary entropy invariance.
        qmi = post_interaction_qmi(
            rho_joint_post,
            rho_text_post,
            rho_audio_post,
            rho_text_pre,
            rho_audio_pre,
        )

        sadi = 1.0 - post_fidelity

        reliability_features = torch.stack(
            [
                audio_entropy,
                text_entropy,
                post_trace_distance,
                qmi,
            ],
            dim=1,
        )

        gates = self.reliability_gate(reliability_features)

        audio_gate = 0.5 + 0.5 * gates[:, 0].unsqueeze(1)
        text_gate = 0.5 + 0.5 * gates[:, 1].unsqueeze(1)
        difference_gate = 0.5 + 0.5 * gates[:, 2].unsqueeze(1)
        interaction_gate = 0.5 + 0.5 * gates[:, 3].unsqueeze(1)

        post_audio_state = self.post_audio_state_projection(
            rho_audio_post.reshape(rho_audio_post.size(0), -1)
        )
        post_text_state = self.post_text_state_projection(
            rho_text_post.reshape(rho_text_post.size(0), -1)
        )

        pooled_audio = pooled_audio + post_audio_state
        pooled_text = pooled_text + post_text_state

        difference = torch.abs(pooled_audio - pooled_text)
        interaction = pooled_audio * pooled_text

        pooled_audio = pooled_audio * audio_gate
        pooled_text = pooled_text * text_gate
        difference = difference * difference_gate
        interaction = interaction * interaction_gate

        fused = torch.cat(
            [
                pooled_audio,
                pooled_text,
                difference,
                interaction,
            ],
            dim=1,
        )

        fused = self.fusion(fused)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "audio_entropy": audio_entropy,
            "text_entropy": text_entropy,
            "audio_purity": audio_purity,
            "text_purity": text_purity,
            "fidelity": post_fidelity,
            "trace_distance": post_trace_distance,
            "sadi": sadi,
            "qmi": qmi,
            "joint_entropy_pre": (
                von_neumann_entropy(rho_text_pre)
                + von_neumann_entropy(rho_audio_pre)
            ),
            "joint_entropy_post": (
                von_neumann_entropy(rho_text_pre)
                + von_neumann_entropy(rho_audio_pre)
            ),
            "rho_audio_pre": rho_audio_pre,
            "rho_text_pre": rho_text_pre,
            "rho_joint_pre": rho_joint_pre,
            "rho_joint_post": rho_joint_post,
            "rho_audio_post": rho_audio_post,
            "rho_text_post": rho_text_post,
            "interaction_unitary": interaction_unitary,
            "gates": gates,
            "reliability_features": reliability_features,
            "pooled_audio": pooled_audio,
            "pooled_text": pooled_text,
            "difference": difference,
            "interaction": interaction,
            "fused": fused,
        }


if __name__ == "__main__":
    torch.manual_seed(42)

    model = ProposedModelV6(
        input_dim=768,
        hidden_dim=256,
        rank=16,
        num_heads=4,
    )

    audio = torch.randn(2, 80, 768)
    text = torch.randn(2, 20, 768)
    audio_mask = torch.ones(2, 80, dtype=torch.bool)
    text_mask = torch.ones(2, 20, dtype=torch.bool)

    output = model(
        audio,
        text,
        audio_mask,
        text_mask,
    )

    print("V6 model test")
    print("-" * 60)
    print("logits:", output["logits"].shape)
    print("rho_joint_pre:", output["rho_joint_pre"].shape)
    print("rho_joint_post:", output["rho_joint_post"].shape)
    print("rho_audio_post:", output["rho_audio_post"].shape)
    print("rho_text_post:", output["rho_text_post"].shape)
    print("QMI:", output["qmi"])
    print("fidelity:", output["fidelity"])
    print("trace distance:", output["trace_distance"])
    print(
        "joint traces:",
        torch.diagonal(
            output["rho_joint_post"],
            dim1=-2,
            dim2=-1,
        ).sum(-1),
    )
    print(
        "audio traces:",
        torch.diagonal(
            output["rho_audio_post"],
            dim1=-2,
            dim2=-1,
        ).sum(-1),
    )
    print(
        "text traces:",
        torch.diagonal(
            output["rho_text_post"],
            dim1=-2,
            dim2=-1,
        ).sum(-1),
    )
    print("-" * 60)
    print("V6 forward test: PASS")
