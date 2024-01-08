from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from position_debiased_estimator import PositionDebiasedEstimator


class FactorizedEstimator(PositionDebiasedEstimator):
    """Causal Debiased estimator. This computes logits for each task based on
    user, item and position and all features. Then it combines these four using
    "Mixture-Of-Logits", i.e. weights computed by a gating arch.
    """
    def __init__(
        self,
        num_tasks: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int, 
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        cross_features_size: int,
        user_value_weights: List[float]
    ) -> None:
        """
        initialize the MultiTaskEstimator
        """
        super(FactorizedEstimator, self).__init__(
            num_tasks=num_tasks,
            user_id_hash_size=user_id_hash_size,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size, 
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            cross_features_size=cross_features_size,
            user_value_weights=user_value_weights
        )
        # Initialize the archs for user and item features.
        self.user_feature_arch = nn.Linear(
            in_features=2*user_id_embedding_dim, 
            out_features=num_tasks
        )
        self.item_feature_arch = nn.Linear(
            in_features=2*item_id_embedding_dim, 
            out_features=num_tasks
        )
        self.gating_arch = nn.Linear(
            in_features=2 * user_id_embedding_dim + 2 * item_id_embedding_dim + self.cross_feature_proc_dim,  # noqa
            out_features=4
        )

    def process_features(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        cross_features: torch.Tensor,  # [B, IC]
        position: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Features for user_item, user, item, and position."""

        # Embedding lookup for user and item ids
        user_id_embedding = self.user_id_embedding_arch(user_id)
        item_id_embedding = self.item_id_embedding_arch(item_id)
        user_features_transformed = self.user_features_layer(user_features)
        item_features_transformed = self.item_features_layer(item_features)
        cross_features_transformed = self.cross_features_layer(cross_features)
        position_embedding = self.position_embedding_arch(position)

        # All features that are needed for the arch computing user only logits
        user_arch_input = torch.cat(
            [
                user_id_embedding,
                user_features_transformed,
            ],
            dim=1
        )

        # All features that are needed for the arch computing item only logits
        item_arch_input = torch.cat(
            [
                item_id_embedding,
                item_features_transformed,
            ],
            dim=1
        )

        # All features that are needed for the arch computing user_item_logits
        user_item_arch_input = torch.cat(
            [
                user_id_embedding,
                user_features_transformed,
                item_id_embedding,
                item_features_transformed,
                cross_features_transformed
            ],
            dim=1
        )

        # Compute input to position shallow tower
        position_arch_input = self.position_embedding_arch(position_embedding)

        return (
            user_item_arch_input,
            position_arch_input,
            user_arch_input,
            item_arch_input
        )

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        cross_features: torch.Tensor,  # [B, IC]
        position: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        (
            user_item_arch_input,
            position_arch_input,
            user_arch_input,
            item_arch_input
        ) = self.process_features(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )

        ui_logits = self.task_arch(user_item_arch_input)  # [B, T]
        position_logits = self.position_based_estimator(
            position_arch_input
        )  # [B, T]
        user_logits = self.user_feature_arch(user_arch_input)  # [B, T]
        item_logits = self.item_feature_arch(item_arch_input)  # [B, T]
        gating_weights = self.gating_arch(user_item_arch_input)  # [B, 4]
        # Compute softmax of gating weights
        gating_weights = F.softmax(gating_weights, dim=1)

        # Combine the logits using the gating weights
        stacked_embeddings = torch.stack(
            [ui_logits, position_logits, user_logits, item_logits], 
            dim=2
        )  # [B, T, 4]
        gating_weights = gating_weights.unsqueeze(2)  # [B, 4, 1]
        final_logits = torch.bmm(stacked_embeddings, gating_weights).squeeze(2)
        return final_logits
