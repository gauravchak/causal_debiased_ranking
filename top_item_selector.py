from multi_task_estimator import MultiTaskEstimator
from typing import List
import torch


class TopItemSelector:
    """A wrapper around MultiTaskEstimator that selects the top item"""
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
        self.estimator = MultiTaskEstimator(
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

    def forward(
        self,
        user_id: int,
        user_features: torch.Tensor,  # [IU]
        item_ids: List[int],  # len B
        item_features: List[torch.Tensor],  # len B x [II]
        cross_features: List[torch.Tensor],  # len B x [IC]
        user_value_weights: List[float],  # len T
    ) -> int:
        """
        Selects top item from a list of items
        
        params:
            user_id: the user id
            user_features: [IU], the user features 
            item_ids: len B, the item ids
            item_features: len B x [II], the item features
            cross_features: len B x [IC], the cross features
            user_value_weights: len T, weights such that a weighted sum of
                user_labels is a good approximation of incremental value
                to the business and users.
        returns:
            the top item id
        """
        B = len(item_ids)
        # create a tensor [B] by duplicating user_id B times
        user_id = torch.tensor([user_id] * B)
        # create a tensor [B, IU] by duplicating user_features B times
        user_features = torch.tensor([user_features] * B)
        # create a tensor [B, II] by stacking
        item_features = torch.stack(item_features)  # [B, II]
        # create a tensor [B, IC] by stacking
        cross_features = torch.stack(cross_features)  # [B, IC]

        estimates = self.estimator(
            user_id=user_id,
            user_features=user_features,
            item_ids=item_ids,
            item_features=item_features,
            cross_features=cross_features,
            user_value_weights=user_value_weights
        )  # [B, T]

        # multiply estimates with a tensor of user_value_weights to get a 
        # combined score. [B, T] * [T] -> [B]
        combined_score = torch.matmul(
            estimates, torch.tensor(user_value_weights)
        )

        # get the index of item with the max combined score
        top_item_index = torch.argmax(combined_score)
        return item_ids[top_item_index]
