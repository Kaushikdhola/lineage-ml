# Atlas roadmap

Coverage is decided up front, so the atlas grows by plan rather than by whim. Target for v1:
**~130 formulas** in two tiers.

- **Computed**: implemented in `src/core`, verified against the oracle, interactive.
- **Catalogued**: formula, notation, citation and lineage edges, rendered in the atlas.
  Promoted to computed over time.

`✓` = in the atlas today. `C` = planned as computed. `·` = planned as catalogued.
Add formulas with `/formula`; nothing merges unless `npm run ci` passes.

## Foundations and probability (26)

| Status | Formula                             | Status | Formula                     |
| ------ | ----------------------------------- | ------ | --------------------------- |
| ✓      | Dot product                         | C      | Cross-entropy               |
| ✓      | Cosine similarity                   | C      | KL divergence               |
| ✓      | Softmax (hub)                       | C      | Mutual information          |
| ✓      | Softmax with temperature            | ·      | Bayes' rule                 |
| C      | Logistic sigmoid                    | C      | Log-likelihood / MLE        |
| C      | Log-sum-exp                         | ·      | MAP estimation              |
| C      | Shannon entropy                     | C      | Gaussian density            |
| C      | Linear regression (normal equation) | ·      | Multivariate Gaussian       |
| C      | Ridge regression                    | ·      | Bias–variance decomposition |
| C      | Logistic regression loss            | ·      | Jensen's inequality         |
| ·      | Hinge loss (SVM)                    | C      | Monte Carlo estimator       |
| C      | Importance sampling                 | C      | Reparameterization trick    |
| ·      | Gumbel-softmax                      | C      | Standard error of the mean  |

## Optimization and training (18)

C Gradient descent · C SGD · C Momentum · C Nesterov momentum · C AdaGrad · C RMSProp · C Adam ·
C AdamW · C Chain rule / backpropagation · C Weight decay · · Warmup + cosine schedule ·
C Xavier/Glorot initialization · C He initialization · C Batch normalization · C Layer normalization ·
C RMSNorm · · Dropout · C Label smoothing

## Architecture (14)

C Perceptron · C Dense layer · C ReLU · C GELU · · SwiGLU · · Residual connection · C 2-D convolution ·
· Max pooling · C RNN cell · C LSTM gates · · GRU · · Embedding lookup · C Sinusoidal positional encoding ·
C Rotary position embedding (RoPE)

## Attention and transformers (18)

✓ Additive attention · ✓ Multiplicative attention · ✓ Attention scores · ✓ Scaled scores ·
✓ Scaled dot-product attention · ✓ Causal attention · ✓ Multi-head attention · C Transformer FFN ·
· Pre-norm transformer block · · KV cache · · Grouped-query attention · · Multi-query attention ·
· ALiBi · · Linear attention · C Online softmax (FlashAttention) · · Mixture-of-experts gating ·
C LoRA · · Compute-optimal scaling law

## Generative models (13)

· Autoencoder objective · C VAE evidence lower bound · · GAN minimax · · Wasserstein GAN ·
C Change of variables (normalizing flows) · C Diffusion forward process · C DDPM simplified loss ·
· Score function · · Denoising score matching · · Classifier-free guidance · · Flow matching ·
C Next-token likelihood · C Perplexity

## Agents and RL (25)

✓ Discounted return · ✓ Bellman equation · ✓ Bellman optimality · ✓ Q-learning ·
✓ Policy gradient (REINFORCE) · ✓ PPO clipped objective · ✓ RLHF reward with KL penalty · ✓ DPO ·
C TD(0) · C SARSA · C TD error · C n-step return · C Generalized advantage estimation ·
· Actor-critic · C Advantage function · C DQN loss with target network · · Double DQN ·
C Entropy regularization · · TRPO trust region · C GRPO group-relative advantage ·
C Reward model (Bradley–Terry) loss · C UCB1 · · Thompson sampling · · PUCT (MCTS) · C ε-greedy

## Evals and statistics (18)

✓ Benchmark accuracy · ✓ Wilson interval · ✓ Exact McNemar · ✓ pass@k · ✓ pass^k · ✓ Bradley–Terry ·
✓ Elo · C Bootstrap confidence interval · C Paired bootstrap · C Holm correction · C Cohen's kappa ·
C Expected calibration error · C Brier score · C F1 score · C AUC-ROC · · BLEU ·
C Paired sample size (Connor) · C Minimum detectable effect

## Chains to tell first

1. **From dot product to attention** (done)
2. **From Bellman to DPO**: how agents learn, and how LLMs are aligned (done)
3. **From accuracy to pass^k**: how we know a model or agent is any good (done)
4. **From gradient descent to AdamW**
5. **From autoencoder to diffusion**
6. **Softmax lens**: one function in attention, policies, preferences and leaderboards

## Token X-ray: deferred from the 2026-11-09 red-team review

- **Optimizer-input readout.** The current readout sees gradients during backward, before clipping and
  the optimizer, so bugs in the optimizer step (the DeepSpeed ZeRO-1/2 CPU-offload gradient copy) are
  outside its view; that plate is an emulation. Read the gradient the optimizer actually consumes, through
  `optimizer.register_step_pre_hook` and, for DeepSpeed, its fp32 partition buffers, and test it with a
  planted copy-skip.
- **A peaked probe.** The probe is untrained and near-uniform, so state-dependent numbers (label
  smoothing's cosine, DPO margins, GRPO ratios and clipping, leak sizes) describe it, not a trained model.
  Add a probe with peaked predictions (a few pre-training steps, or a scaled LM head) and rerun the
  smoothing, DPO and GRPO checks on it.
