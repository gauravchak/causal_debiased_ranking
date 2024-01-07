from typing import List, Tuple
import torch
import torch.nn as nn

from multi_task_estimator import MultiTaskEstimator


class PositionDebiasedEstimator(MultiTaskEstimator):
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
        super(PositionDebiasedEstimator, self).__init__(
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
        position_feature_dim: int = 32
        position_hash_size: int = 100
        self.position_embedding_arch = nn.Embedding(
            position_hash_size, position_feature_dim
        )
        self.position_based_estimator = nn.Linear(
            in_features=position_feature_dim, 
            out_features=num_tasks
        )

    def process_features(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        cross_features: torch.Tensor,  # [B, IC]
        position: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined_features = super().process_features(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )
        position = self.position_embedding(position)
        return combined_features, position

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        cross_features: torch.Tensor,  # [B, IC]
        position: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Forward pass of the model
        """
        combined_features, position_feature = self.process_features(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )
        # Compute per-task scores/logits
        ui_logits = self.task_arch(combined_features)  # [B, T]

        position_based_logits = self.position_based_estimator(position_feature)  # [B, T]
        return position_based_logits + ui_logits
