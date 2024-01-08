import torch.nn.functional as F

from causal_debiased_ranking.src.factorized_estimator import FactorizedEstimator  # noqa


class AnchoredFactorizedEstimator(FactorizedEstimator):
    """
    No difference in the forward pass, but the loss function is different.
    During training, we also compute cross entropy loss from the logits from
    user, item and position archs. This helps anchor them independently.
    """

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
        """
        Compute combined, and component logits and compute cross entropy loss
        for each.
        """
        # Get task logits using forward method
        (
            final_logits,
            _,  # ui_logits
            position_logits,
            user_logits,
            item_logits
        ) = self.compute_logits(
            user_id=user_id,
            user_features=user_features,
            item_id=item_id,
            item_features=item_features,
            cross_features=cross_features,
            position=position,
        )

        # Compute binary cross-entropy losses
        final_ce_loss = F.binary_cross_entropy_with_logits(
            input=final_logits, target=labels.float(), reduction='sum'
        )
        position_ce_loss = F.binary_cross_entropy_with_logits(
            input=position_logits, target=labels.float(), reduction='sum'
        )
        user_ce_loss = F.binary_cross_entropy_with_logits(
            input=user_logits, target=labels.float(), reduction='sum'
        )
        item_ce_loss = F.binary_cross_entropy_with_logits(
            input=item_logits, target=labels.float(), reduction='sum'
        )
        # We are not computing the loss from ui_logits, because if the
        # other 3 components are anchored, ui_logits will be anchored too.
        return final_ce_loss + position_ce_loss + user_ce_loss + item_ce_loss
