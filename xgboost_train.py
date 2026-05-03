import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle

# 1. Setup Synthetic Clinical Data (9 Features)
np.random.seed(42)
samples = 5000
data = {
    'z_fev1': np.random.uniform(-4, 2, samples), 'z_fvc': np.random.uniform(-4, 2, samples),
    'z_ratio': np.random.uniform(-4, 2, samples), 'z_fef': np.random.uniform(-4, 2, samples),
    'z_tlc': np.random.uniform(-4, 2, samples), 'z_rv': np.random.uniform(-4, 2, samples),
    'z_frc': np.random.uniform(-4, 2, samples), 'z_erv': np.random.uniform(-4, 2, samples),
    'z_ic': np.random.uniform(-4, 2, samples)
}
df = pd.DataFrame(data)

# 2. Assign Medical Labels (Target)
def get_label(r):
    if r['z_tlc'] < -1.645: return 0 # Restrictive
    if r['z_ratio'] < -1.645: return 1 # Obstructive
    if r['z_fef'] < -1.645: return 2 # SAD
    return 3 # Normal

df['label'] = df.apply(get_label, axis=1)

# 3. Train XGBoost
X = df.drop('label', axis=1)
y = df['label']
model = xgb.XGBClassifier()
model.fit(X, y)

# 4. Save the "Brain"
with open('pulmo_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Success! 'pulmo_model.pkl' created. Your AI is trained.")