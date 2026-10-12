from __future__ import annotations

import torch
import torch.nn as nn

from src.proposed.v5.quantum_v5 import (
    SpeechConditionedKraus,
    build_mixed_density,
    fidelity,
    purity,
    sadi,
    trace_distance,
    von_neumann_entropy,
)


class ProposedModelV5(nn.Module):
    """
    Corrected V5 Speech-Conditioned Kraus Evolution Model.

    Pipeline:
        768-D HuBERT/DeBERTa sequences
            ↓
        256-D modality projections
            ↓
        Bidirectional cross-attention
            ↓
        Attention-weighted mixed density states (rank=16)
            ↓
        Speech-conditioned Kraus evolution
            ↓
        Updated text density state
            ↓
        Quantum-inspired discordance descriptors
            ↓
        Reliability-aware gating
            ↓
        Multimodal fusion
            ↓
        Binary classifier

    V5 preserves the corrected V4 representation pipeline and adds
    speech-conditioned Kraus evolution to the text density state.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        rank: int = 16,
        num_heads: int = 4,
        num_kraus: int = 4,
        num_classes: int = 2,
        kraus_strength: float = 0.25,
        projection_dropout: float = 0.30,
        fusion_dropout_1: float = 0.40,
        fusion_dropout_2: float = 0.30,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_heads = num_heads
        self.num_kraus = num_kraus
        self.num_classes = num_classes

        # ============================================================
        # Modality projections
        # ============================================================

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

        # ============================================================
        # Bidirectional cross-modal attention
        # ============================================================

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

        # ============================================================
        # Conventional attention pooling
        # ============================================================

        self.audio_pool = nn.Linear(hidden_dim, 1)
        self.text_pool = nn.Linear(hidden_dim, 1)

        # ============================================================
        # Quantum-inspired state projections
        # ============================================================

        self.audio_state_projection = nn.Linear(
            hidden_dim,
            rank,
        )

        self.text_state_projection = nn.Linear(
            hidden_dim,
            rank,
        )

        self.audio_state_attention = nn.Linear(
            hidden_dim,
            1,
        )

        self.text_state_attention = nn.Linear(
            hidden_dim,
            1,
        )

        # ============================================================
        # V5 speech-conditioned Kraus channel
        # ============================================================

        self.kraus_channel = SpeechConditionedKraus(
            rank=rank,
            num_kraus=num_kraus,
            hidden_dim=hidden_dim,
            strength=kraus_strength,
        )

        # The evolved text density state is flattened to R^2 and
        # projected back to the common 256-D representation so that
        # the updated quantum-inspired state directly affects the
        # downstream classifier while preserving V4's fusion width.
        self.post_text_state_projection = nn.Sequential(
            nn.Linear(rank * rank, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        # ============================================================
        # Reliability-aware gating
        # ============================================================

        self.reliability_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
        )

        # ============================================================
        # Multimodal fusion
        #
        # Four 256-D components:
        #   audio
        #   updated text
        #   |audio - text|
        #   audio * text
        #
        # => 1024-D
        # ============================================================

        fusion_dim = hidden_dim * 4

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_dim,
                512,
            ),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(fusion_dropout_1),
            nn.Linear(
                512,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout_2),
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    # ============================================================
    # Masked attention pooling
    # ============================================================

    @staticmethod
    def attention_pooling(
        features: torch.Tensor,
        mask: torch.Tensor,
        pool_layer: nn.Module,
    ) -> torch.Tensor:
        """
        Attention-weighted pooling over a padded sequence.

        features:
            [B, T, H]

        mask:
            [B, T], True = valid timestep.
        """
        logits = pool_layer(features).squeeze(-1)

        logits = logits.masked_fill(
            ~mask,
            torch.finfo(logits.dtype).min,
        )

        weights = torch.softmax(
            logits,
            dim=1,
        ).unsqueeze(-1)

        return torch.sum(
            weights * features,
            dim=1,
        )

    # ============================================================
    # Mixed density state
    # ============================================================

    def build_modality_density(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        state_projection: nn.Module,
        state_attention: nn.Module,
    ) -> torch.Tensor:
        """Convert a contextualized sequence into a rank-r mixed state."""

        state_vectors = state_projection(features)

        attention_logits = state_attention(
            features
        ).squeeze(-1)

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
    ):
        """
        Args:
            audio_embedding:
                [B, T_audio, 768]

            text_embedding:
                [B, T_text, 768]

            audio_mask:
                [B, T_audio]

            text_mask:
                [B, T_text]

        Returns:
            Dictionary containing logits, V5 descriptors, density
            states, Kraus operators, gates, and fusion representations.
        """

        if audio_embedding.ndim != 3:
            raise ValueError(
                "audio_embedding must have shape [B, T_audio, 768]. "
                f"Got {tuple(audio_embedding.shape)}"
            )

        if text_embedding.ndim != 3:
            raise ValueError(
                "text_embedding must have shape [B, T_text, 768]. "
                f"Got {tuple(text_embedding.shape)}"
            )

        if audio_embedding.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected audio feature dimension {self.input_dim}, "
                f"got {audio_embedding.size(-1)}."
            )

        if text_embedding.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected text feature dimension {self.input_dim}, "
                f"got {text_embedding.size(-1)}."
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

        # ------------------------------------------------------------
        # Project modalities
        # ------------------------------------------------------------

        audio = self.audio_projection(audio_embedding)
        text = self.text_projection(text_embedding)

        # ------------------------------------------------------------
        # Bidirectional cross-attention
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Conventional pooling
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Corrected mixed density states
        # ------------------------------------------------------------

        rho_audio = self.build_modality_density(
            features=audio_attended,
            mask=audio_mask,
            state_projection=self.audio_state_projection,
            state_attention=self.audio_state_attention,
        )

        rho_text_pre = self.build_modality_density(
            features=text_attended,
            mask=text_mask,
            state_projection=self.text_state_projection,
            state_attention=self.text_state_attention,
        )

        # ------------------------------------------------------------
        # V5: speech-conditioned Kraus evolution
        #
        # Speech state conditions the channel acting on the text state.
        # ------------------------------------------------------------

        rho_text_post, kraus_ops = self.kraus_channel(
            rho_text=rho_text_pre,
            rho_audio=rho_audio,
        )

        # ------------------------------------------------------------
        # V5 descriptors
        # ------------------------------------------------------------

        audio_entropy = von_neumann_entropy(rho_audio)
        text_entropy_pre = von_neumann_entropy(rho_text_pre)
        text_entropy = von_neumann_entropy(rho_text_post)

        audio_purity = purity(rho_audio)
        text_purity_pre = purity(rho_text_pre)
        text_purity = purity(rho_text_post)

        discordance = trace_distance(
            rho_audio,
            rho_text_post,
        )

        fidelity_score = fidelity(
            rho_audio,
            rho_text_post,
        )

        sadi_score = sadi(
            rho_audio,
            rho_text_post,
        )

        # ------------------------------------------------------------
        # Reliability-aware gating
        # ------------------------------------------------------------

        reliability_features = torch.stack(
            [
                audio_entropy,
                text_entropy,
                discordance,
            ],
            dim=1,
        )

        gates = self.reliability_gate(
            reliability_features
        )

        audio_gate = (
            0.5
            + 0.5 * gates[:, 0].unsqueeze(1)
        )

        text_gate = (
            0.5
            + 0.5 * gates[:, 1].unsqueeze(1)
        )

        difference_gate = (
            0.5
            + 0.5 * gates[:, 2].unsqueeze(1)
        )

        interaction_gate = (
            0.5
            + 0.5 * gates[:, 3].unsqueeze(1)
        )

        # ------------------------------------------------------------
        # Updated text state -> downstream representation
        # ------------------------------------------------------------

        post_text_state = self.post_text_state_projection(
            rho_text_post.reshape(
                rho_text_post.size(0),
                -1,
            )
        )

        # The evolved text state refines the conventional pooled text
        # representation. This makes the V5 evolution directly
        # downstream of the classifier rather than diagnostic-only.
        pooled_text = pooled_text + post_text_state

        # ------------------------------------------------------------
        # Cross-modal interaction features
        # ------------------------------------------------------------

        difference = torch.abs(
            pooled_audio - pooled_text
        )

        interaction = (
            pooled_audio * pooled_text
        )

        # ------------------------------------------------------------
        # Apply reliability gates
        # ------------------------------------------------------------

        pooled_audio = pooled_audio * audio_gate
        pooled_text = pooled_text * text_gate
        difference = difference * difference_gate
        interaction = interaction * interaction_gate

        # ------------------------------------------------------------
        # Fusion
        # ------------------------------------------------------------

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

            # Pre/post text states
            "rho_audio": rho_audio,
            "rho_text_pre": rho_text_pre,
            "rho_text_post": rho_text_post,
            "kraus_ops": kraus_ops,

            # Quantum-inspired descriptors
            "audio_entropy": audio_entropy,
            "text_entropy_pre": text_entropy_pre,
            "text_entropy": text_entropy,
            "audio_purity": audio_purity,
            "text_purity_pre": text_purity_pre,
            "text_purity": text_purity,
            "trace_distance": discordance,
            "fidelity": fidelity_score,
            "sadi": sadi_score,

            # Reliability
            "gates": gates,
            "reliability_features": reliability_features,

            # Representations
            "pooled_audio": pooled_audio,
            "pooled_text": pooled_text,
            "post_text_state": post_text_state,
            "difference": difference,
            "interaction": interaction,
            "fused": fused,
        }


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    audio_length = 80
    text_length = 20

    model = ProposedModelV5(
        input_dim=768,
        hidden_dim=256,
        rank=16,
        num_heads=4,
        num_kraus=4,
        kraus_strength=0.25,
    )

    audio = torch.randn(
        batch_size,
        audio_length,
        768,
    )

    text = torch.randn(
        batch_size,
        text_length,
        768,
    )

    audio_mask = torch.ones(
        batch_size,
        audio_length,
        dtype=torch.bool,
    )

    text_mask = torch.ones(
        batch_size,
        text_length,
        dtype=torch.bool,
    )

    output = model(
        audio_embedding=audio,
        text_embedding=text,
        audio_mask=audio_mask,
        text_mask=text_mask,
    )

    print("V5 model test")
    print("-" * 60)
    print("logits:", output["logits"].shape)
    print("rho_audio:", output["rho_audio"].shape)
    print("rho_text_pre:", output["rho_text_pre"].shape)
    print("rho_text_post:", output["rho_text_post"].shape)
    print("kraus_ops:", output["kraus_ops"].shape)
    print("audio entropy:", output["audio_entropy"].shape)
    print("text entropy:", output["text_entropy"].shape)
    print("trace distance:", output["trace_distance"].shape)
    print("fidelity:", output["fidelity"].shape)
    print("SADI:", output["sadi"].shape)
    print("gates:", output["gates"].shape)

    audio_trace = torch.diagonal(
        output["rho_audio"],
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    text_pre_trace = torch.diagonal(
        output["rho_text_pre"],
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    text_post_trace = torch.diagonal(
        output["rho_text_post"],
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    print("audio traces:", audio_trace)
    print("pre-text traces:", text_pre_trace)
    print("post-text traces:", text_post_trace)
    print("-" * 60)
    print("V5 forward test: PASS")