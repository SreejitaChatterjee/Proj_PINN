# Generalized Attack Detector

## Performance
- **Recall**: 81.8% on unseen attack magnitudes (0.25x to 4.0x)
- **FPR**: 10.7%
- **Generalizes**: Yes - tested on attacks NOT seen during training

## Method
- **Algorithm**: Isolation Forest (n=200, contamination=0.07)
- **Features**: Multi-scale statistics at 6 time windows [5, 10, 25, 50, 100, 200]
- **Training**: Unsupervised (normal data only)

## Files
- `isolation_forest.pkl` - Trained IsolationForest model
- `scaler.pkl` - StandardScaler for feature normalization
- `config.json` - Configuration parameters
