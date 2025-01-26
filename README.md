# Credit Card Dataset Sampling Techniques Comparison

## Project Overview
This project explores the impact of different sampling techniques on machine learning model performance using a Credit Card dataset.

## Sampling Techniques
- **Simple Random Sampling**
- **Systematic Sampling**
- **Stratified Sampling**
- **Cross-Validation Sampling**
- **Bootstrap Sampling**

## Machine Learning Models
'M1': LogisticRegression
'M2': RandomForestClassifier
'M3': SVC
'M4': KNeighborsClassifier
'M5': DecisionTreeClassifier
## Performance Results

### Accuracy Matrix
# Model Sampling Techniques Comparison

## Performance Matrix

| Model | Simple Random | Systematic | Stratified | Cross-Validation | Bootstrap | Best Sampling Technique | Best Sampling and Model |
|-------|---------------|------------|------------|-----------------|-----------|------------------------|------------------------|
| M1 | 0.826 | 0.783 | 0.913 | 0.899 | 0.783 | Stratified | Stratified (M1) |
| M2 | 0.978 | 0.978 | 0.957 | 0.995 | 0.935 | Cross-Validation | Cross-Validation (M2) |
| M3 | 0.630 | 0.696 | 0.609 | 0.694 | 0.478 | Systematic | Systematic (M3) |
| M4 | 0.717 | 0.674 | 0.609 | 0.831 | 0.609 | Cross-Validation | Cross-Validation (M4) |
| M5 | 0.804 | 0.826 | 0.891 | 0.973 | 0.913 | Cross-Validation | Cross-Validation (M5) |

## Key Observations
- M2 shows highest performance across most sampling techniques
- Cross-Validation emerges as the best sampling technique for multiple models
- Performance varies significantly across different sampling methods

## Methodology
- 5 different sampling techniques applied
- Performance measured using accuracy metric
- Multiple machine learning models evaluated

## Requirements
- Python
- scikit-learn
- pandas
- numpy


## Installation
```bash
pip install -r requirements.txt
```

## License
MIT License

## Contributors
[Kartik Sidana]

