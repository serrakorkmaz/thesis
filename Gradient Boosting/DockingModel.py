import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def prepare_fingerprints(data):
    mols = [Chem.MolFromSmiles(smiles) for smiles in data['smiles']]
    valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
    valid_mols = [mols[i] for i in valid_indices]
    fingerprints = []
    for mol in valid_mols:
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024), arr)
        fingerprints.append(arr)
    return np.array(fingerprints), valid_indices

def run_gb_gridsearch(train_fps, train_values):
    params = {
        'n_estimators': [10, 50, 100, 150, 200, 250, 500],
        'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
        'subsample': [0.4, 0.7, 0.9, 1.0],
        'min_samples_split': [2, 3, 5, 7],
        'min_samples_leaf': [1, 3, 5],
        'max_depth': [2, 3, 4],
        'max_features': [None, 'sqrt']
    }
    gb = GradientBoostingRegressor()
    grid_search = RandomizedSearchCV(
        gb, params, n_iter=30, refit=True, scoring='neg_mean_squared_error', verbose=3, random_state=42
    )
    grid_search.fit(train_fps, train_values)
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def save_predictions(test_data, test_preds, valid_indices):
    test_data = test_data.iloc[valid_indices].copy()
    test_data['predicted_value'] = test_preds
    test_data.to_csv('test_predictions_braf.csv', index=False)

# Load datasets
train = pd.read_csv('train_data_braf.csv', sep=',')
test = pd.read_csv('test_data_braf.csv', sep=',')

# Prepare fingerprints
train_fps, valid_train_indices = prepare_fingerprints(train)
test_fps, valid_test_indices = prepare_fingerprints(test)

# Align 'value' column
train_values = train.iloc[valid_train_indices]['value'].tolist()
test_values = test.iloc[valid_test_indices]['value'].tolist()

# Train model
model = run_gb_gridsearch(train_fps, train_values)

# Evaluate on test set
test_preds = model.predict(test_fps)
mse = mean_squared_error(test_values, test_preds)
r2 = r2_score(test_values, test_preds)

print(f"Test Metrics: MSE = {mse:.4f}, R2 = {r2:.4f}")

# Save model and predictions
joblib.dump(model, 'gradient_boosting_regressor_braf.pkl')
save_predictions(test, test_preds, valid_test_indices)
