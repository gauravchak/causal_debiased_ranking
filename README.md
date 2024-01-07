# Causal / Debiased Ranking

We will show how to factorize ranking in a way that factorizes to remove the influence of power users and items. Using causal recommendation will help you improve the experience of users early on in their adoption curve without compromising the experience of your power users. Same on the item side.

## multi_task_estimator

This is what you normally find in the (final) ranking of a recommender system, a neural network that takes user and item features and produces estimates of multiple labels and trains them by binary cross entropy loss.

## position_debiased_estimator

This extends multi_task_estimator. In addition to per-task logits computed from user + item features, this also computes logits using the position feature. The final logit is the sum of both. One of the earliest papers to talk about this is the [Youtube WatchNext paper](https://daiwk.github.io/assets/youtube-multitask.pdf) but you can also derive this from maximizing Mutual Information. Importance of PMI in personalization was described in [this paper](https://dl.acm.org/doi/pdf/10.1145/3523227.3546753).

## factorized_estimator.py

This extends multi_task_estimator. In addition to per-task logits computed from user + item features, this also computes logits from user, item and position. The final logit is mixture of these four logits. [Reference for Mixture-Of-Logits](https://arxiv.org/abs/2306.04039)

## anchored_factorized_estimator.py

This extends factorized_estimator.py. During training, in addition to passing gradient to the mixed logits, this also trains the user, item and position logits separately.

## top_item_selctor.py

This shows how to use the estimators in ranking. It takes a set of items for a single user, creates batches of user and item features, computes estimates using the estimator and then combines by value_weights to select top item.

## References

1. Position bias
   1. [Shallow tower - Youtube WatchNext paper](https://daiwk.github.io/assets/youtube-multitask.pdf)
   2. [post on recent advances](https://www.linkedin.com/feed/update/urn:li:activity:7140837803960975360?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7140837803960975360%29)
2. popularity bias
   1. [Don't recommend the obvious](https://dl.acm.org/doi/pdf/10.1145/3523227.3546753)
3. Mixture of logits
   1. [Reference for Mixture-Of-Logits in Revisiting Neural Accelerators](https://arxiv.org/abs/2306.04039)
