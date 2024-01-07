# causal_debiased_ranking
We will show how to factorize ranking in a way that factorizes to remove the influence of power users and items

# multi_task_estimator
This is what you normally find in the (final) ranking of a recommender system, a neural network that takes user and item features and produces estimates of multiple labels and trains them by binary cross entropy loss.

# position_debiased_estimator
This extends multi_task_estimator. In addition to per-task logits computed from user + item features, this also computes logits using the position feature. The final logit is the sum of both.

# factorized_estimator.py
This extends multi_task_estimator. In addition to per-task logits computed from user + item features, this also computes logits from user, item and position. The final logit is mixture of these four logits. [Reference for Mixture-Of-Logits](https://arxiv.org/abs/2306.04039)

# anchored_factorized_estimator.py
This extends factorized_estimator.py. During training, in addition to passing gradient to the mixed logits, this also trains the user, item and position logits separately.