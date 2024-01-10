import unittest
import torch

from causal_debiased_ranking.src.multi_task_estimator import MultiTaskEstimator
from causal_debiased_ranking.src.top_item_selector import TopItemSelector


class TestMultiTaskEstimator(unittest.TestCase):
    def test_multi_task_estimator(self):
        num_tasks = 3
        user_id_hash_size = 100
        user_id_embedding_dim = 50
        user_features_size = 10
        item_id_hash_size = 200
        item_id_embedding_dim = 30
        item_features_size = 10
        cross_features_size = 10
        batch_size = 3

        # unused in the baseline MultiTaskEstimator implementation
        user_value_weights = [0.5, 0.3, 0.2]
        assert len(user_value_weights) == num_tasks

        # Instantiate the MultiTaskEstimator
        estimator: MultiTaskEstimator = MultiTaskEstimator(
            num_tasks, user_id_hash_size, user_id_embedding_dim,
            user_features_size, item_id_hash_size, item_id_embedding_dim,
            item_features_size, cross_features_size,
            user_value_weights
        )

        # Example input data
        model_user_id = torch.tensor([1, 2, 3])
        model_user_features = torch.randn(batch_size, user_features_size)
        model_item_id = torch.tensor([4, 5, 6])
        model_item_features = torch.randn(batch_size, item_features_size)
        model_cross_features = torch.randn(batch_size, cross_features_size)
        model_position = torch.tensor([1, 2, 3], dtype=torch.int32)
        model_labels = torch.randint(2, size=(batch_size, num_tasks))

        # Example train_forward pass
        model_loss = estimator.train_forward(
            model_user_id, model_user_features,
            model_item_id, model_item_features,
            model_cross_features, model_position,
            model_labels
        )
        self.assertIsInstance(model_loss, torch.Tensor)
        self.assertGreaterEqual(model_loss.item(), 0)

        # Example input data
        inference_batch_size = 3
        user_id: int = 1  
        user_features = torch.randn(user_features_size)  # [IU]
        item_ids = [4, 5, 6]  # len B
        item_features = [torch.randn(item_features_size)] * inference_batch_size  # len B x [II]
        cross_features = [torch.randn(cross_features_size)] * inference_batch_size  # len B x [IC]

        # Instantiate the TopItemSelector
        selector: TopItemSelector = TopItemSelector(
            user_value_weights, estimator
        )

        # Example forward pass
        top_item = selector(
            user_id, user_features,
            item_ids, item_features,
            cross_features, 
        )
        self.assertTrue(top_item in item_ids)


if __name__ == '__main__':
    unittest.main()
