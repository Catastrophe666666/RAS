# STM K Selection

Semantic coherence is better when it is higher, meaning less negative here. Exclusivity is better when higher.

- Best balanced K by normalized coherence and exclusivity: K=20
- Best coherence-weighted K: K=15
- Recommendation for the main reported model: K=15, because it is the elbow where exclusivity improves substantially over K=10, while gains after K=15 are small and coherence continues to deteriorate.
- Use K=10 as a simpler robustness check if the priority is maximum semantic coherence and fewer topics.

## Scores

| K | semantic_coherence_mean | exclusivity_mean | coherence_norm | exclusivity_norm | balanced_score | coherence_weighted_score |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | -40.9221 | 9.556 | 1.0 | 0.0 | 0.5 | 0.6 |
| 15 | -43.2929 | 9.7186 | 0.6738 | 0.7291 | 0.7015 | 0.696 |
| 20 | -44.4446 | 9.7585 | 0.5154 | 0.9079 | 0.7116 | 0.6724 |
| 25 | -48.1907 | 9.7616 | 0.0 | 0.9217 | 0.4608 | 0.3687 |
| 30 | -47.9512 | 9.7791 | 0.0329 | 1.0 | 0.5165 | 0.4198 |