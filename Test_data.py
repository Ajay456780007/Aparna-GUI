import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("Dataset/Lung_Cancer_Dataset/lung_cancer_examples.csv")

# ==========================
# SEPARATE FEATURES & LABEL
# ==========================

features = df.drop(columns=["Result"])
labels = df["Result"]

# ==========================
# LABEL ENCODING
# ==========================

object_cols = features.select_dtypes(exclude=["number"]).columns

for col in object_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])

# ==========================
# STANDARDIZATION
# ==========================

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

features_scaled = pd.DataFrame(
    features_scaled,
    columns=features.columns
)

# Add label back
features_scaled["Result"] = labels.values

# ==========================
# CREATE OUTPUT DIRECTORY
# ==========================

save_path = "Test_data/Lung_Cancer"
os.makedirs(save_path, exist_ok=True)

# ==========================
# CLASS-WISE SAMPLING
# ==========================

normal_df = features_scaled[features_scaled["Result"] == 0]
abnormal_df = features_scaled[features_scaled["Result"] == 1]

normal_samples = normal_df.sample(n=25, random_state=42)
abnormal_samples = abnormal_df.sample(n=25, random_state=42)

sample_num = 1

# ==========================
# SAVE NORMAL SAMPLES
# ==========================

for _, row in normal_samples.iterrows():

    sample_df = row.to_frame().T

    # Remove label before saving
    sample_df = sample_df.drop(columns=["Result"])

    sample_df.to_csv(
        os.path.join(save_path, f"sample{sample_num}.csv"),
        index=False
    )

    sample_num += 1

# ==========================
# SAVE ABNORMAL SAMPLES
# ==========================

for _, row in abnormal_samples.iterrows():

    sample_df = row.to_frame().T

    # Remove label before saving
    sample_df = sample_df.drop(columns=["Result"])

    sample_df.to_csv(
        os.path.join(save_path, f"sample{sample_num}.csv"),
        index=False
    )

    sample_num += 1

print(f"Saved {sample_num - 1} files in '{save_path}'")