# Noise Filtering Test Results

**Overall Score**: 98.67%

## Individual Results

- Query: What is the average delivery time by country?
  Initial Docs: 15, Final Docs: 0
  Rejection Rate: 100.0%
  Dedup Rate: 0%
  Filtering Score: 100.0%

- Query: What factors explain the high return rate in China?
  Initial Docs: 15, Final Docs: 0
  Rejection Rate: 100.0%
  Dedup Rate: 0%
  Filtering Score: 100.0%

- Query: How has delivery time evolved in Spain over time?
  Initial Docs: 15, Final Docs: 0
  Rejection Rate: 100.0%
  Dedup Rate: 0%
  Filtering Score: 100.0%

- Query: Which countries have the lowest average user age and why?
  Initial Docs: 15, Final Docs: 0
  Rejection Rate: 100.0%
  Dedup Rate: 0%
  Filtering Score: 100.0%

- Query: What is the correlation between distance and delivery speed?
  Initial Docs: 15, Final Docs: 1
  Rejection Rate: 93.33%
  Dedup Rate: 0%
  Filtering Score: 93.33%

## Synthetic Data Test Results

**Overall Score**: 93.33%

### Individual Results

- Query: What is the delivery time in Spain?
  Initial Docs: 15, Final Docs: 1
  Rejection Rate: 93.33%
  Dedup Rate: 0%
  Filtering Score: 93.33%

- Query: What is the return rate in China?
  Initial Docs: 15, Final Docs: 1
  Rejection Rate: 93.33%
  Dedup Rate: 0%
  Filtering Score: 93.33%

- Query: What is the user age in the US?
  Initial Docs: 15, Final Docs: 1
  Rejection Rate: 93.33%
  Dedup Rate: 0%
  Filtering Score: 93.33%

## Filter by Similarity Results

- Threshold: 0.5
  Initial Docs: 5, Filtered Docs: 5
  Rejection Rate: 0.0%

- Threshold: 0.7
  Initial Docs: 5, Filtered Docs: 5
  Rejection Rate: 0.0%

- Threshold: 0.9
  Initial Docs: 5, Filtered Docs: 1
  Rejection Rate: 80.0%

## Deduplicate Docs Results

- Threshold: 0.8
  Initial Docs: 5, Deduped Docs: 3
  Dedup Rate: 40.0%

- Threshold: 0.9
  Initial Docs: 5, Deduped Docs: 3
  Dedup Rate: 40.0%

- Threshold: 0.95
  Initial Docs: 5, Deduped Docs: 3
  Dedup Rate: 40.0%

