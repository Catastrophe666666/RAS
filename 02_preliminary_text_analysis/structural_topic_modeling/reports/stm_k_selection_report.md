# STM K Selection

Semantic coherence is better when it is higher, meaning less negative here. Exclusivity is better when higher.

- Best balanced K by normalized coherence and exclusivity: K=15
- Best coherence-weighted K: K=10
- Recommendation for the main reported model: K=15, because it is the elbow where exclusivity improves substantially over K=10, while gains after K=15 are small and coherence continues to deteriorate.
- Use K=10 as a simpler robustness check if the priority is maximum semantic coherence and fewer topics.

## Scores

| K | semantic_coherence_mean | exclusivity_mean | coherence_norm | exclusivity_norm | balanced_score | coherence_weighted_score |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | -35.7352 | 9.5448 | 1.0 | 0.0 | 0.5 | 0.6 |
| 15 | -41.4993 | 9.7303 | 0.4471 | 0.759 | 0.603 | 0.5718 |
| 20 | -43.2925 | 9.7546 | 0.2751 | 0.8587 | 0.5669 | 0.5085 |
| 25 | -44.6336 | 9.7876 | 0.1464 | 0.9936 | 0.57 | 0.4853 |
| 30 | -46.16 | 9.7892 | 0.0 | 1.0 | 0.5 | 0.4 |