# Business Entity Resolution — Methodology

## 1. Methodology Used

The project implements a machine-learning-based entity resolution pipeline for
matching business records across three independent data sources.

Source 1 is treated as the reference source. Source 2 and Source 3 contain records
that may correspond to Source 1 entities.

The pipeline consists of:

1. Data cleaning and normalization
2. Candidate generation / blocking
3. Feature engineering
4. Random Forest model training
5. Candidate scoring
6. Test prediction
7. Submission generation

## 2. Data Cleaning and Normalization

Business names and addresses are normalized before matching.

The normalization process:

- converts text to lowercase
- removes URLs
- removes punctuation and non-alphanumeric characters
- normalizes repeated whitespace
- tokenizes normalized text
- removes selected common stopwords from token-based processing

Country values are normalized to lowercase strings.

The pipeline treats country as a string attribute rather than restricting the
pipeline to a fixed set of countries.

## 3. Candidate Generation / Blocking

Candidate generation is used to reduce the number of Source 2 and Source 3
records that need to be considered for each Source 1 record.

The production candidate-generation pipeline uses:

- normalized business-name token blocking
- exact normalized-name matching
- exact normalized-address matching

The fast candidate-generation implementation processes Source 1 in chunks and
uses the existing blocking database so that the complete candidate set does not
need to be held in memory at once.

## 4. Feature Engineering

Seven features are generated for each candidate pair:

1. name_ratio
2. name_token_ratio
3. address_ratio
4. address_token_ratio
5. name_exact
6. address_exact
7. country_match

String similarity features are based on normalized text similarity.
Token similarity compares the sets of normalized tokens.

Exact-match features are binary indicators.

Country match is also represented as a binary feature.

## 5. Model Architecture

A Random Forest classifier is used for binary candidate-pair classification.

The training configuration includes:

- 200 decision trees
- random state = 42
- balanced class weights
- square-root feature selection
- parallel training

The model predicts the probability that a candidate pair represents the
same real-world business entity.

A probability threshold of 0.85 is used by the model-scoring pipeline for
selecting predicted matches.

## 6. Training

Training labels are generated from the supplied ground-truth file.

The ground-truth file maps each Source 1 entity to its matching Source 2
and/or Source 3 entity IDs.

Candidate feature rows are labelled positive when the candidate entity appears
in the corresponding ground-truth match list and negative otherwise.

The training data is split into training and validation subsets using a
stratified split.

## 7. Test Prediction

The test set contains Source 1, Source 2, and Source 3 records without labels.

The test prediction pipeline generates candidate pairs and produces the final
matching output for every Source 1 test entity.

Every Source 1 entity is retained in the final output. Entities for which no
match is predicted receive an empty matched_entity_ids field.

## 8. Output Files

The final submission contains:

### matching_results.tsv

Columns:

- source1_entity_id
- matched_entity_ids

Each Source 1 test entity appears exactly once.

### candidate_pairs.tsv

Columns:

- source1_entity_id
- candidate_entity_ids

The candidate file represents the candidate set associated with the matching
pipeline.

## 9. Validation

The generated outputs were checked for:

- correct Source 1 row count
- duplicate Source 1 IDs
- invalid Source 2 / Source 3 ID prefixes
- duplicate IDs within individual match lists
- consistency of Source 1 IDs between matching and candidate outputs

The final test output contains 1,732,544 Source 1 entities.

## 10. Reproducibility

The project source code is contained in the src/ directory.

Required dependencies are listed in requirements.txt.

The pipeline uses the supplied training and test data and does not require
external business-identity data.

## 11. Limitations

The final quick test prediction used for the generated submission prioritizes
execution speed and uses exact normalized business-name and address blocking.

This fast prediction path is distinct from the trained Random Forest scoring
pipeline. Therefore, the generated submission should be interpreted as the
output of the speed-oriented test prediction path rather than as a claim that
the final submission was produced by the Random Forest scorer.
