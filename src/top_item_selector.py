from typing import List

import torch
import torch.nn as nn

from causal_debiased_ranking.src.multi_task_estimator import MultiTaskEstimator


class TopItemSelector(nn.Module):
    """A wrapper around MultiTaskEstimator that selects the top item"""
    def __init__(
        self,
        user_value_weights: List[float],
        estimator: MultiTaskEstimator,
    ) -> None:
        """
        params:
            user_value_weights: len T, weights such that a weighted sum of
                user_labels is a good approximation of incremental value
                to the business and users.
            estimator: To compute the estimates per user and item pair
        """
        super(TopItemSelector, self).__init__()
        self.estimator = estimator
        self.user_value_weights = user_value_weights

    def forward(
        self,
        user_id: int,
        user_features: torch.Tensor,  # [IU]
        item_ids: List[int],  # len B
        item_features: List[torch.Tensor],  # len B x [II]
        cross_features: List[torch.Tensor],  # len B x [IC]
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
        user_features = torch.stack([user_features] * B, dim=0)
        item_id = torch.tensor(item_ids)  # [B]
        # create a tensor [B, II] by stacking
        item_features = torch.stack(item_features)  # [B, II]
        # create a tensor [B, IC] by stacking
        cross_features = torch.stack(cross_features)  # [B, IC]
        inference_position = torch.zeros(B, dtype=torch.int32)  # [B]

        estimates = self.estimator(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=inference_position,
        )  # [B, T]

        # multiply estimates with a tensor of user_value_weights to get a 
        # combined score. [B, T] * [T] -> [B]
        combined_score = torch.matmul(
            estimates, torch.tensor(self.user_value_weights)
        )

        # get the index of item with the max combined score
        top_item_index = torch.argmax(combined_score)
        return item_ids[top_item_index]
