import torch
import torch.nn as nn

from src.proposed.v4.quantum_v4 import (
    build_mixed_density,
    von_neumann_entropy,
    purity,
    trace_distance,
    fidelity,
    sadi,
)


class ProposedModelV4(nn.Module):
    """
    Corrected V4 Semantic-Affective Discordance Model.

    Pipeline:
        768-D HuBERT/DeBERTa sequences
            ↓
        256-D modality projections
            ↓
        Bidirectional cross-attention
            ↓
        Attention-weighted mixed density states (rank=16)
            ↓
        Quantum-inspired discordance descriptors
            ↓
        Reliability-aware gating
            ↓
        Multimodal fusion
            ↓
        Binary classifier

    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        rank: int = 16,
        num_heads: int = 4,
        num_classes: int = 2,
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

        # Audio queries Text
        self.audio_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Text queries Audio
        self.text_to_audio_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # ============================================================
        # Sequence-level attention pooling
        #
        # Used for conventional multimodal fusion.
        # These pooled vectors are NOT used to construct the
        # quantum-inspired density states.
        # ============================================================

        self.audio_pool = nn.Linear(hidden_dim, 1)
        self.text_pool = nn.Linear(hidden_dim, 1)

        # ============================================================
        # Quantum-inspired state projections
        #
        # Each contextualised frame/token is projected:
        #
        #   R^256 -> R^16
        #
        # The resulting sequence is converted into a mixed density
        # state by quantum_v4.build_mixed_density().
        # ============================================================

        self.audio_state_projection = nn.Linear(
            hidden_dim,
            rank,
        )

        self.text_state_projection = nn.Linear(
            hidden_dim,
            rank,
        )

        # Attention logits used to form the convex mixture:
        #
        #   rho = sum_k alpha_k |psi_k><psi_k|
        #
        self.audio_state_attention = nn.Linear(
            hidden_dim,
            1,
        )

        self.text_state_attention = nn.Linear(
            hidden_dim,
            1,
        )

        # ============================================================
        # Reliability-aware gating
        #
        # Preserves the original V4 gate:
        #
        #   [audio entropy,
        #    text entropy,
        #    trace distance]
        #
        # -> four gates corresponding to:
        #      audio
        #      text
        #      difference
        #      interaction
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
        #
        #   audio
        #   text
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

        # Binary classifier

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    # Masked attention pooling

    @staticmethod
    def attention_pooling(
        features: torch.Tensor,
        mask: torch.Tensor,
        pool_layer: nn.Module,
    ) -> torch.Tensor:
        """
        Attention-weighted pooling over a padded sequence.

        Args:
            features:
                [B, T, H]

            mask:
                [B, T]
                True  = valid timestep
                False = padding

            pool_layer:
                Linear(H -> 1)

        Returns:
            pooled:
                [B, H]
        """

        # [B, T, 1]
        logits = pool_layer(features)

        # [B, T]
        logits = logits.squeeze(-1)

        # Never allow padded positions to receive attention.
        logits = logits.masked_fill(
            ~mask,
            torch.finfo(logits.dtype).min,
        )

        # [B, T]
        weights = torch.softmax(
            logits,
            dim=1,
        )

        # [B, T, 1]
        weights = weights.unsqueeze(-1)

        # [B, H]
        pooled = torch.sum(
            weights * features,
            dim=1,
        )

        return pooled

    # Build a mixed density state from contextualised sequence

    def build_modality_density(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        state_projection: nn.Module,
        state_attention: nn.Module,
    ) -> torch.Tensor:
        """
        Convert a contextualised sequence into a rank-r mixed density
        state.

        Args:
            features:
                [B, T, H]

            mask:
                [B, T]

        Returns:
            rho:
                [B, r, r]

        Construction:

            z_k -> P z_k
                -> normalized psi_k

            alpha_k = softmax(attention_k)

            rho = sum_k alpha_k psi_k psi_k^T
        """

        # [B, T, r]
        state_vectors = state_projection(features)

        # [B, T]
        attention_logits = state_attention(
            features
        ).squeeze(-1)

        rho = build_mixed_density(
            state_vectors,
            attention_logits,
            mask,
        )

        return rho

    # Forward

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
                True for valid frames.

            text_mask:
                [B, T_text]
                True for valid tokens.

        Returns:
            Dictionary containing:

                logits
                audio_entropy
                text_entropy
                audio_purity
                text_purity
                fidelity
                sadi
                trace_distance
                gates
                rho_audio
                rho_text
                pooled_audio
                pooled_text
        """

        # -------------------------------------------------------------
        # Validate dimensions
        # -------------------------------------------------------------

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

        batch_size = audio_embedding.size(0)

        # -------------------------------------------------------------
        # Create masks if caller does not provide them.
        #
        # Normally train_v4.py will provide masks because sequences
        # are padded by the DataLoader.
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Project both modalities into common 256-D space
        # -------------------------------------------------------------

        audio = self.audio_projection(
            audio_embedding
        )

        text = self.text_projection(
            text_embedding
        )

        # -------------------------------------------------------------
        # Bidirectional cross-attention
        #
        # Audio attends to text.
        # Text attends to audio.
        #
        # key_padding_mask expects:
        #     True  = padding
        #     False = valid
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Conventional attention pooling
        #
        # These pooled vectors are ONLY used for the classifier/fusion.
        # They are NOT converted directly into density states.
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Construct corrected mixed density states
        #
        # IMPORTANT:
        # We use the complete contextualised sequence here.
        #
        # NOT:
        #
        #     rho = vv^T
        #
        # from one pooled vector.
        #
        # Instead:
        #
        #     rho = sum_k alpha_k psi_k psi_k^T
        #
        # producing a genuine mixed state.
        # -------------------------------------------------------------

        rho_audio = self.build_modality_density(
            features=audio_attended,
            mask=audio_mask,
            state_projection=self.audio_state_projection,
            state_attention=self.audio_state_attention,
        )

        rho_text = self.build_modality_density(
            features=text_attended,
            mask=text_mask,
            state_projection=self.text_state_projection,
            state_attention=self.text_state_attention,
        )

        # -------------------------------------------------------------
        # Quantum-inspired descriptors
        # -------------------------------------------------------------

        audio_entropy = von_neumann_entropy(
            rho_audio
        )

        text_entropy = von_neumann_entropy(
            rho_text
        )

        audio_purity = purity(
            rho_audio
        )

        text_purity = purity(
            rho_text
        )

        discordance = trace_distance(
            rho_audio,
            rho_text,
        )

        fidelity_score = fidelity(
            rho_audio,
            rho_text,
        )

        sadi_score = sadi(
            rho_audio,
            rho_text,
        )

        # -------------------------------------------------------------
        # Reliability-aware gating
        #
        # Preserve the original V4 gate:
        #
        #     [S_audio, S_text, D]
        #
        # -> 4 gates
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Convert sigmoid gates from [0, 1] into [0.5, 1.0].
        #
        # This follows the original V4 behaviour:
        #
        #     x' = x * (0.5 + 0.5g)
        #
        # so the reliability mechanism cannot completely suppress
        # a modality/component.
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Cross-modal interaction features
        # -------------------------------------------------------------

        difference = torch.abs(
            pooled_audio - pooled_text
        )

        interaction = (
            pooled_audio * pooled_text
        )

        # -------------------------------------------------------------
        # Apply reliability gates
        # -------------------------------------------------------------

        pooled_audio = (
            pooled_audio * audio_gate
        )

        pooled_text = (
            pooled_text * text_gate
        )

        difference = (
            difference * difference_gate
        )

        interaction = (
            interaction * interaction_gate
        )

        # -------------------------------------------------------------
        # Fusion
        #
        # [audio | text | difference | interaction]
        #
        # 4 x 256 = 1024
        # -------------------------------------------------------------

        fused = torch.cat(
            [
                pooled_audio,
                pooled_text,
                difference,
                interaction,
            ],
            dim=1,
        )

        fused = self.fusion(
            fused
        )

        # Classification

        logits = self.classifier(
            fused
        )

        # Return everything needed by training/evaluation/analysis.

        return {
            "logits": logits,

            # Quantum-inspired descriptors
            "audio_entropy": audio_entropy,
            "text_entropy": text_entropy,
            "audio_purity": audio_purity,
            "text_purity": text_purity,
            "trace_distance": discordance,
            "fidelity": fidelity_score,
            "sadi": sadi_score,

            # Reliability
            "gates": gates,
            "reliability_features": reliability_features,

            # States
            "rho_audio": rho_audio,
            "rho_text": rho_text,

            # Conventional representations
            "pooled_audio": pooled_audio,
            "pooled_text": pooled_text,
            "difference": difference,
            "interaction": interaction,
            "fused": fused,
        }


# Simple model construction test

if __name__ == "__main__":
    """
    Lightweight architecture test.

    This does NOT train the model.
    It only verifies tensor dimensions and forward propagation.
    """

    torch.manual_seed(42)

    batch_size = 2
    audio_length = 80
    text_length = 20

    model = ProposedModelV4(
        input_dim=768,
        hidden_dim=256,
        rank=16,
        num_heads=4,
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

    print("V4 model test")
    print("-" * 60)

    print(
        "logits:",
        output["logits"].shape,
    )

    print(
        "rho_audio:",
        output["rho_audio"].shape,
    )

    print(
        "rho_text:",
        output["rho_text"].shape,
    )

    print(
        "audio entropy:",
        output["audio_entropy"].shape,
    )

    print(
        "text entropy:",
        output["text_entropy"].shape,
    )

    print(
        "trace distance:",
        output["trace_distance"].shape,
    )

    print(
        "fidelity:",
        output["fidelity"].shape,
    )

    print(
        "SADI:",
        output["sadi"].shape,
    )

    print(
        "gates:",
        output["gates"].shape,
    )

    # Density-state sanity checks

    audio_trace = torch.diagonal(
        output["rho_audio"],
        dim1=-2,
        dim2=-1,
    ).sum(-1)

    text_trace = torch.diagonal(
        output["rho_text"],
        dim1=-2,
        dim2=-1,
    ).sum(-1)

    print(
        "audio traces:",
        audio_trace.detach().cpu(),
    )

    print(
        "text traces:",
        text_trace.detach().cpu(),
    )

    assert torch.allclose(
        audio_trace,
        torch.ones_like(audio_trace),
        atol=1e-4,
    )

    assert torch.allclose(
        text_trace,
        torch.ones_like(text_trace),
        atol=1e-4,
    )

    print("-" * 60)
    print("V4 forward test: PASS")