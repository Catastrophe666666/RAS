# Bag-of-Words Topic Model Comparison

## Models Tested
- K values: 8, 10, 12, 15, 20, 25, 30
- NMF on TF-IDF matrix
- LDA on count matrix

## Diagnostics Summary
- LDA K=8: diversity=0.975, exclusivity=0.987, concentration=0.628, near-empty=0
- LDA K=10: diversity=0.970, exclusivity=0.985, concentration=0.614, near-empty=0
- LDA K=12: diversity=0.933, exclusivity=0.967, concentration=0.592, near-empty=0
- LDA K=15: diversity=0.853, exclusivity=0.924, concentration=0.564, near-empty=0
- LDA K=20: diversity=0.790, exclusivity=0.896, concentration=0.543, near-empty=0
- LDA K=25: diversity=0.784, exclusivity=0.903, concentration=0.533, near-empty=0
- LDA K=30: diversity=0.780, exclusivity=0.899, concentration=0.527, near-empty=0
- NMF K=8: diversity=0.975, exclusivity=0.987, concentration=0.594, near-empty=0
- NMF K=10: diversity=0.960, exclusivity=0.979, concentration=0.575, near-empty=0
- NMF K=12: diversity=0.942, exclusivity=0.969, concentration=0.558, near-empty=0
- NMF K=15: diversity=0.967, exclusivity=0.983, concentration=0.538, near-empty=0
- NMF K=20: diversity=0.950, exclusivity=0.975, concentration=0.528, near-empty=0
- NMF K=25: diversity=0.956, exclusivity=0.977, concentration=0.516, near-empty=0
- NMF K=30: diversity=0.957, exclusivity=0.977, concentration=0.508, near-empty=0

## Candidate Models for Human Interpretation
- NMF K=8
- LDA K=8
- LDA K=10

These are diagnostic suggestions only; final labels should be assigned in the human-review template.
