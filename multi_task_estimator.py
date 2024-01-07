"""
This is a specific instance of a final ranker in a recommender system.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskEstimator(nn.Module):
    """A core component of multi-task ranking systems where 
    we compute estimates of the getting those binary feedback
    labels from the user."""
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
        user_value_weights: List[float],
    ) -> None:
        """
        params:
            num_tasks (T): The tasks to compute estimates of
            user_id_hash_size: the size of the embedding table for users
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            item_features_size: (II) input feature size for items
            cross_features_size: (IC) size of cross features
            user_value_weights: T dimensional weights, such that a linear
            combination of point-wise immediate rewards is the best predictor
            of long term user satisfaction.
        """
        super(MultiTaskEstimator, self).__init__()
        self.user_value_weights = torch.tensor(user_value_weights)  # noqa TODO add device input.

        # Embedding layers for user and item ids
        self.user_embedding = nn.Embedding(
            user_id_hash_size, user_id_embedding_dim)
        self.item_embedding = nn.Embedding(
            item_id_hash_size, item_id_embedding_dim)

        # Linear projection layer for user features
        self.user_features_layer = nn.Linear(
            in_features=user_features_size, 
            out_features=user_id_embedding_dim)  # noqa

        # Linear projection layer for user features
        self.item_features_layer = nn.Linear(
            in_features=item_features_size, 
            out_features=item_id_embedding_dim)  # noqa

        # Linear projection layer for cross features
        self.cross_features_layer = nn.Linear(
            in_features=cross_features_size, 
            out_features=128)

        # Linear layer for final prediction
        self.task_arch = nn.Linear(2 * user_id_embedding_dim + 2 * item_id_embedding_dim + 128, num_tasks)  # noqa

    def process_features(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        cross_features: torch.Tensor,  # [B, IC]
        position: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Process features. Separated from forward function so that we can change 
        handling of forward and train_forward in future without duplicating 
        feature processing.
        """

        # Embedding lookup for user and item ids
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)

        # Linear transformation for user features
        user_features_transformed = self.user_features_layer(user_features)

        # Linear transformation for item features
        item_features_transformed = self.item_features_layer(item_features)

        # Linear transformation for user features
        cross_features_transformed = self.cross_features_layer(cross_features)

        # Concatenate user embedding, user features, and item embedding
        combined_features = torch.cat(
            [
                user_embedding,
                user_features_transformed, 
                item_embedding,
                item_features_transformed,
                cross_features_transformed
            ],
            dim=1
        )

        return combined_features

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        cross_features: torch.Tensor,  # [B, IC]
        position: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        combined_features = self.process_features(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )
        # Compute per-task scores/logits
        ui_logits = self.task_arch(combined_features)  # [B, T]

        return ui_logits

    def train_forward(
        self,
        user_id,
        user_features,
        item_id,
        item_features,  # [B, II]
        cross_features,  # [B, IC]
        position,  # [B]
        labels
    ) -> float:
        """Compute the loss during training"""
        # Get task logits using forward method
        ui_logits = self.forward(
            user_id=user_id, 
            user_features=user_features, 
            item_id=item_id, 
            item_features=item_features, 
            cross_features=cross_features,
            position=position,
        )

        # Compute binary cross-entropy loss
        cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=ui_logits, target=labels.float(), reduction='sum'
        )

        return cross_entropy_loss

# Example usage:
# Replace the placeholder values with your actual data dimensions
num_tasks = 3
user_id_hash_size = 100
user_id_embedding_dim = 50
user_features_size = 10
item_id_hash_size = 200
item_id_embedding_dim = 30
item_features_size = 10
cross_features_size = 10

# unused in the above implementation
user_value_weights = [0.5, 0.3, 0.2]


# Instantiate the MultiTaskEstimator
model = MultiTaskEstimator(
    num_tasks, user_id_hash_size, user_id_embedding_dim,
    user_features_size, item_id_hash_size, item_id_embedding_dim,
    item_features_size, cross_features_size,
    user_value_weights
)

# Example input data
user_id = torch.tensor([1, 2, 3])
user_features = torch.randn(3, user_features_size)
item_id = torch.tensor([4, 5, 6])
item_features = torch.randn(3, item_features_size)
cross_features = torch.randn(3, cross_features_size)
position = torch.tensor([1, 2, 3])
labels = torch.tensor([0, 1, 2])

# Example train_forward pass
loss = model.train_forward(
    user_id, user_features,
    item_id, item_features,
    cross_features, position,
    labels
)
print("Training Loss:", loss.item())

# Example forward pass
output = model(
    user_id, user_features,
    item_id, item_features, 
    cross_features
)
print("Forward Pass Output:", output)
