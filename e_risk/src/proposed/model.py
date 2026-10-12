import torch
import torch.nn as nn
import torch.nn.functional as F
from src.proposed.quantum_metrics import (von_neumann_entropy, trace_distance, fidelity, purity)

class AttentionPooling(nn.Module):

    def __init__(

        self,

        embedding_dim=768,

        hidden_dim=256,

    ):

        super().__init__()

        self.score = nn.Sequential(

            nn.Linear(

                embedding_dim,

                hidden_dim,

            ),

            nn.Tanh(),

            nn.Linear(

                hidden_dim,

                1,

            ),

        )

    def forward(

        self,

        embeddings,

        attention_mask,

    ):

        scores = self.score(

            embeddings

        )

        scores = scores.masked_fill(

            attention_mask.unsqueeze(-1) == 0,

            -1e9,

        )

        weights = torch.softmax(

            scores,

            dim=1,

        )

        pooled = torch.sum(

            weights * embeddings,

            dim=1,

        )

        return pooled
    
class MeanPooling(nn.Module):

    def forward(

        self,

        embeddings,

        attention_mask,

    ):

        mask = attention_mask.unsqueeze(-1).float()

        summed = torch.sum(

            embeddings * mask,

            dim=1,

        )

        counts = torch.clamp(

            mask.sum(dim=1),

            min=1e-9,

        )

        return summed / counts

class Projection(nn.Module):

    def __init__(

        self,

        input_dim=768,

        hidden_dim=256,

    ):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(

                input_dim,

                hidden_dim,

            ),

            nn.LayerNorm(

                hidden_dim,

            ),

            nn.GELU(),

            nn.Dropout(

                0.2,

            ),

        )

    def forward(

        self,

        x,

    ):

        return self.layers(

            x,

        )

class StateNormalization(nn.Module):

    def forward(

        self,

        x,

    ):

        return F.normalize(

            x,

            p=2,

            dim=-1,

        )

class DensityState(nn.Module):

    def forward(

        self,

        state,

    ):

        state = F.normalize(

            state,

            p=2,

            dim=-1,

        )

        density = torch.einsum(

            "bi,bj->bij",

            state,

            state,

        )

        return density

class ReliabilityGate(nn.Module):

    def __init__(

        self,

        hidden_dim=256,

    ):

        super().__init__()

        self.gate = nn.Sequential(

            nn.Linear(

                hidden_dim + 4,

                hidden_dim,

            ),

            nn.GELU(),

            nn.Linear(

                hidden_dim,

                hidden_dim,

            ),

            nn.Sigmoid(),

        )

    def forward(

        self,

        features,

        entropy,

        trace,

        fidelity,

        purity,

    ):

        metrics = torch.stack(

            [

                entropy,

                trace,

                fidelity,

                purity,

            ],

            dim=1,

        )

        gate_input = torch.cat(

            [

                features,

                metrics,

            ],

            dim=1,

        )

        weights = self.gate(

            gate_input,

        )

        refined = features * (1 + weights) # i.e features + weights * features

        return refined, weights

class Classifier(nn.Module):

    def __init__(

        self,

        hidden_dim=256,

        dropout=0.3,

    ):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(

                hidden_dim * 3 +4,

                hidden_dim,

            ),

            nn.LayerNorm(

                hidden_dim,

            ),

            nn.GELU(),

            nn.Dropout(

                dropout,

            ),

            nn.Linear(

                hidden_dim,

                2,

            ),

        )

    def forward(

        self,

        x,

    ):

        return self.layers(

            x,
        )

class ProposedV4(nn.Module):

    def __init__(

        self,

        embedding_dim=768,

        hidden_dim=256,

        dropout=0.3,

    ):

        super().__init__()

        # -----------------------------------------
        # pooling
        # -----------------------------------------

        self.attention_pool = AttentionPooling(

            embedding_dim,

            hidden_dim,

        )

        self.mean_pool = MeanPooling()

        # -----------------------------------------
        # projections
        # -----------------------------------------

        self.salient_projection = Projection(

            embedding_dim,

            hidden_dim,

        )

        self.context_projection = Projection(

            embedding_dim,

            hidden_dim,

        )

        self.normalize = StateNormalization()

        # -----------------------------------------
        # density states
        # -----------------------------------------

        self.density = DensityState()

        # -----------------------------------------
        # reliability
        # -----------------------------------------

        self.reliability = ReliabilityGate(

            hidden_dim,

        )

        # -----------------------------------------
        # classifier
        # -----------------------------------------

        self.classifier = Classifier(

            hidden_dim,

            dropout,

        )
        
    def forward(

        self,

        embeddings,

        attention_mask,

    ):

        # -------------------------------------------------
        # semantic views
        # -------------------------------------------------

        salient = self.attention_pool(

            embeddings,

            attention_mask,

        )

        context = self.mean_pool(

            embeddings,

            attention_mask,

        )

        # -------------------------------------------------
        # projection
        # -------------------------------------------------

        salient = self.salient_projection(

            salient,

        )

        context = self.context_projection(

            context,

        )

        # -------------------------------------------------
        # state normalization
        # -------------------------------------------------

        salient = self.normalize(

            salient,

        )

        context = self.normalize(

            context,

        )

        # -------------------------------------------------
        # density matrices
        # -------------------------------------------------

        rho_salient = self.density(

            salient,

        )

        rho_context = self.density(

            context,

        )

        # -------------------------------------------------
        # quantum metrics
        # -------------------------------------------------

        entropy_salient = von_neumann_entropy(

            rho_salient,

        )

        entropy_context = von_neumann_entropy(

            rho_context,

        )

        entropy_gap = torch.abs(

            entropy_salient -

            entropy_context

        )

        trace = trace_distance(

            rho_salient,

            rho_context,

        )

        fidelity_score = fidelity(

            rho_salient,

            rho_context,

        )

        purity_salient = purity(

            rho_salient,

        )

        purity_context = purity(

            rho_context,

        )

        purity_score = (

            purity_salient +

            purity_context

        ) / 2

        # -------------------------------------------------
        # reliability gate
        # -------------------------------------------------

        refined, gate_weights = self.reliability(

            salient,

            entropy_gap,

            trace,

            fidelity_score,

            purity_score,

        )
        
        metrics = torch.stack(

            [

                entropy_gap,

                trace,

                fidelity_score,

                purity_score,

            ],

            dim=1,

        )

        # -------------------------------------------------
        # feature fusion
        # -------------------------------------------------
        
        fused = torch.cat(

            [

                salient,

                context,

                refined,

                metrics,

            ],

            dim=1,

        )

        # -------------------------------------------------
        # classifier
        # -------------------------------------------------

        logits = self.classifier(

            fused,

        )

        # -------------------------------------------------
        # diagnostics
        # -------------------------------------------------

        diagnostics = {

            "trace_distance": trace,

            "entropy_gap": entropy_gap,

            "fidelity": fidelity_score,

            "purity": purity_score,

            "gate_weights": gate_weights,

            "salient_state": salient,

            "context_state": context,
            
            "metrics": metrics,

        }

        return logits, diagnostics