import os
import joblib

# Path to your dataset (adjust if different)
dataset_path = "datasets/plantVillage"  

# Get all folder names
classes = sorted(os.listdir(dataset_path))

# Create mapping: {"class_name": index}
label_dict = {cls_name: idx for idx, cls_name in enumerate(classes)}

# Save the mapping
joblib.dump(label_dict, "models/disease_labels.pkl")

print("Labels saved successfully:", label_dict)
