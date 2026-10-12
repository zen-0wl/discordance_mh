from __future__ import annotations

import math

import torch
import torch.nn as nn


class B3VariationalQuantumClassifier(nn.Module):
    """
    B3: Variational Quantum Classifier (VQC) baseline.

    Uses masked-mean HuBERT + DeBERTa features, a linear angle encoder,
    a 6-qubit / 2-layer differentiable statevector circuit, Pauli-Z
    expectations, and a linear readout.

    This baseline intentionally contains NO discordance framework:
    no density-state discordance, fidelity, trace distance, entropy,
    purity, SADI, reliability gating, Kraus evolution, Hamiltonian
    text-speech interaction, or QMI.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_qubits: int = 6,
        num_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.feature_dim = input_dim * 2

        self.angle_encoder = nn.Linear(
            self.feature_dim,
            num_qubits,
        )

        self.theta = nn.Parameter(
            torch.empty(num_layers, num_qubits, 2)
        )
        nn.init.uniform_(self.theta, -math.pi, math.pi)

        self.readout = nn.Linear(num_qubits, 1)

        # Cached CNOT index permutations.
        self._cnot_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    @staticmethod
    def masked_mean(
        sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(sequence.dtype)
        summed = (sequence * weights).sum(dim=1)
        count = weights.sum(dim=1).clamp_min(1.0)
        return summed / count

    def _single_qubit_matrix(
        self,
        axis: str,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        """Return [2,2] or [B,2,2] differentiable rotation matrices."""
        half = angle / 2.0
        c = torch.cos(half)
        s = torch.sin(half)
        zero = torch.zeros_like(c)

        if axis == "x":
            row0 = torch.stack(
                [c, torch.complex(zero, -s)],
                dim=-1,
            )
            row1 = torch.stack(
                [torch.complex(zero, -s), c],
                dim=-1,
            )
            return torch.stack([row0, row1], dim=-2)

        if axis == "z":
            row0 = torch.stack(
                [
                    torch.exp(torch.complex(zero, -half)),
                    torch.complex(zero, zero),
                ],
                dim=-1,
            )
            row1 = torch.stack(
                [
                    torch.complex(zero, zero),
                    torch.exp(torch.complex(zero, half)),
                ],
                dim=-1,
            )
            return torch.stack([row0, row1], dim=-2)

        raise ValueError(f"Unknown rotation axis: {axis}")

    def _apply_single_qubit_rotation(
        self,
        state: torch.Tensor,
        gate: torch.Tensor,
        qubit: int,
    ) -> torch.Tensor:
        """
        Apply a 2x2 gate directly to one qubit.

        This avoids constructing a dense 64x64 Kronecker matrix.
        state: [B, 2^n]
        gate:  [2,2] or [B,2,2]
        """
        batch_size = state.size(0)
        left_dim = 2 ** qubit
        right_dim = 2 ** (self.num_qubits - qubit - 1)

        reshaped = state.reshape(
            batch_size,
            left_dim,
            2,
            right_dim,
        )

        if gate.ndim == 2:
            transformed = torch.einsum(
                "ij,bxjr->bxir",
                gate,
                reshaped,
            )
        elif gate.ndim == 3:
            transformed = torch.einsum(
                "bij,bxjr->bxir",
                gate,
                reshaped,
            )
        else:
            raise ValueError(
                f"Gate must have rank 2 or 3, got {gate.ndim}."
            )

        return transformed.reshape(
            batch_size,
            2 ** self.num_qubits,
        )

    def _cnot_permutation(
        self,
        control: int,
        target: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (control, target, str(device))

        if key in self._cnot_cache:
            return self._cnot_cache[key]

        dim = 2 ** self.num_qubits
        indices = torch.arange(dim, device=device)

        control_shift = self.num_qubits - 1 - control
        target_shift = self.num_qubits - 1 - target

        control_on = ((indices >> control_shift) & 1).bool()
        flipped = indices ^ (1 << target_shift)

        permutation = torch.where(
            control_on,
            flipped,
            indices,
        )

        self._cnot_cache[key] = permutation
        return permutation

    def _apply_cnot(
        self,
        state: torch.Tensor,
        control: int,
        target: int,
    ) -> torch.Tensor:
        """Apply CNOT as a basis-state permutation."""
        permutation = self._cnot_permutation(
            control,
            target,
            state.device,
        )
        return state.index_select(-1, permutation)

    def _z_expectations(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate <Z_q> for every qubit."""
        probabilities = torch.abs(state) ** 2
        dim = 2 ** self.num_qubits
        basis = torch.arange(dim, device=state.device)

        values = []

        for qubit in range(self.num_qubits):
            shift = self.num_qubits - 1 - qubit
            signs = 1.0 - 2.0 * (
                ((basis >> shift) & 1).to(
                    probabilities.real.dtype
                )
            )

            values.append(
                torch.sum(
                    probabilities.real * signs,
                    dim=-1,
                )
            )

        return torch.stack(values, dim=-1)

    def quantum_circuit(
        self,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable VQC state evolution.

        The circuit operates directly on the 2^n statevector and
        never constructs full 2^n x 2^n gate matrices.
        """
        batch_size = angles.size(0)
        dim = 2 ** self.num_qubits

        state = torch.zeros(
            batch_size,
            dim,
            dtype=torch.complex64,
            device=angles.device,
        )
        state[:, 0] = 1.0 + 0j

        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                rx = self._single_qubit_matrix(
                    "x",
                    angles[:, qubit]
                    + self.theta[layer, qubit, 0],
                )

                state = self._apply_single_qubit_rotation(
                    state,
                    rx,
                    qubit,
                )

                rz = self._single_qubit_matrix(
                    "z",
                    self.theta[layer, qubit, 1],
                )

                state = self._apply_single_qubit_rotation(
                    state,
                    rz,
                    qubit,
                )

            # Ring entanglement.
            for qubit in range(self.num_qubits - 1):
                state = self._apply_cnot(
                    state,
                    qubit,
                    qubit + 1,
                )

            state = self._apply_cnot(
                state,
                self.num_qubits - 1,
                0,
            )

        return self._z_expectations(state)

    def forward(
        self,
        audio: torch.Tensor,
        text: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        if audio.ndim != 3 or text.ndim != 3:
            raise ValueError(
                "audio and text must have shape [B, T, 768]."
            )

        if audio.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected audio dimension {self.input_dim}, "
                f"got {audio.size(-1)}"
            )

        if text.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected text dimension {self.input_dim}, "
                f"got {text.size(-1)}"
            )

        if audio_mask is None:
            audio_mask = torch.ones(
                audio.shape[:2],
                dtype=torch.bool,
                device=audio.device,
            )

        if text_mask is None:
            text_mask = torch.ones(
                text.shape[:2],
                dtype=torch.bool,
                device=text.device,
            )

        audio_pooled = self.masked_mean(
            audio,
            audio_mask,
        )

        text_pooled = self.masked_mean(
            text,
            text_mask,
        )

        conventional_features = torch.cat(
            [
                audio_pooled,
                text_pooled,
            ],
            dim=-1,
        )

        angles = math.pi * torch.tanh(
            self.angle_encoder(
                conventional_features
            )
        )

        z_expectations = self.quantum_circuit(
            angles
        )

        logits = self.readout(
            z_expectations
        ).squeeze(-1)

        return {
            "logits": logits,
            "z_expectations": z_expectations,
            "angles": angles,
            "audio_pooled": audio_pooled,
            "text_pooled": text_pooled,
        }


if __name__ == "__main__":
    torch.manual_seed(42)

    model = B3VariationalQuantumClassifier(
        input_dim=768,
        num_qubits=6,
        num_layers=2,
    )

    audio = torch.randn(2, 40, 768)
    text = torch.randn(2, 20, 768)

    audio_mask = torch.ones(
        2,
        40,
        dtype=torch.bool,
    )

    text_mask = torch.ones(
        2,
        20,
        dtype=torch.bool,
    )

    output = model(
        audio,
        text,
        audio_mask,
        text_mask,
    )

    print("B3 Variational Quantum Classifier test")
    print("-" * 60)
    print("logits:", output["logits"].shape)
    print("Z expectations:", output["z_expectations"].shape)
    print("angles:", output["angles"].shape)
    print("-" * 60)
    print("B3 forward test: PASS")
