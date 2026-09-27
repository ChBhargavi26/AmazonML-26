# Business Entity Resolution

## Overview

This project performs business entity resolution across three independent sources.
Source 1 is the reference source, and the pipeline identifies matching records from
Source 2 and Source 3.

## Pipeline

1. Data cleaning and normalization
2. Candidate generation / blocking
3. Feature generation
4. Random Forest model training
5. Model scoring
6. Test prediction
7. Submission generation

## Source Files

Training data:

- `data/raw/train/train_source1.tsv`
- `data/raw/train/train_source2.tsv`
- `data/raw/train/train_source3.tsv`
- `data/raw/train/train_ground_truth.tsv`

Test data:

- `data/raw/test/test_source1.tsv`
- `data/raw/test/test_source2.tsv`
- `data/raw/test/test_source3.tsv`

## Source Code

- `src/candidate_generation_fast.py` - candidate generation
- `src/feature_generation.py` - feature construction
- `src/model_training.py` - Random Forest training
- `src/model_scoring.py` - model scoring
- `src/fast_test_prediction.py` - test prediction and submission generation

## Model

The matching model is a Random Forest classifier using:

- name similarity
- name token similarity
- address similarity
- address token similarity
- exact name match
- exact address match
- country match

The trained model uses 200 Random Forest trees and a matching threshold of 0.85.

## Outputs

The final output files are:

- `outputs/matching_results.tsv`
- `outputs/candidate_pairs.tsv`

`matching_results.tsv` contains one row for every Source 1 test entity.

`candidate_pairs.tsv` contains the candidate entity IDs considered by the
blocking stage.

## Submission Format

Both output files are tab-separated.

`matching_results.tsv` columns:

- `source1_entity_id`
- `matched_entity_ids`

`candidate_pairs.tsv` columns:

- `source1_entity_id`
- `candidate_entity_ids`

Empty lists are used when no match/candidate exists.

## Reproducibility

Install the required Python dependencies and run the scripts from the project root.
The pipeline uses only the provided training and test data.

## Final Test Output

The generated test set contains 1,732,544 Source 1 entities.
The final matching output contains one row for each of these entities.
