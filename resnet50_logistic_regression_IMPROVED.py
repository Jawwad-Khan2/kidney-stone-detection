from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import pickle
import os

# -------------------------
# Config
# -------------------------
DATA_DIR = Path(r"C:\Users\Cp9-30\Saad\kidney_dataset_split")
BATCH_SIZE = 16
DEVICE = "cpu"

# -------------------------
# Preprocessing
# -------------------------
weights = ResNet50_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# -------------------------
# Dataset
# -------------------------
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Classes:", train_ds.classes)
print("Class mapping:", train_ds.class_to_idx)

# -------------------------
# ResNet50 Feature Extractor
# -------------------------
resnet50 = models.resnet50(weights=weights)

# Remove final classification layer
feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

# -------------------------
# Extract Features
# -------------------------
def extract_features(loader, split_name):
    features_list = []
    labels_list = []

    print(f"\nExtracting {split_name} features...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(DEVICE)

            features = feature_extractor(images)
            features = features.view(features.size(0), -1)

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

            if batch_idx % 50 == 0:
                print(f"{split_name}: processed {batch_idx} batches")

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)

    return X, y

X_train, y_train = extract_features(train_loader, "train")
X_val, y_val = extract_features(val_loader, "validation")
X_test, y_test = extract_features(test_loader, "test")

print("\nFeature shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# -------------------------
# Train Logistic Regression
# -------------------------
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs"
)

print("\nTraining Logistic Regression...")
clf.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
def evaluate(name, X, y):
    preds = clf.predict(X)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)

    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n=== {name} RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(f"TP={tp}  FP={fp}")
    print(f"FN={fn}  TN={tn}")

evaluate("VALIDATION", X_val, y_val)
evaluate("TEST", X_test, y_test)

# =========================================
# IMPROVED SAVE WITH VALIDATION
# =========================================

print("\n" + "="*60)
print("SAVING MODELS WITH VALIDATION")
print("="*60)

# Function to validate saved model
def validate_model_file(filepath, file_format="joblib"):
    """Verify that a saved model can be loaded correctly"""
    try:
        if file_format == "joblib":
            loaded = joblib.load(filepath)
        elif file_format == "pickle":
            with open(filepath, 'rb') as f:
                loaded = pickle.load(f)
        
        # Test it works
        test_pred = loaded.predict(X_test[:1])
        
        print(f"  ✓ {filepath} - Valid! ({os.path.getsize(filepath)} bytes)")
        return True
    except Exception as e:
        print(f"  ✗ {filepath} - FAILED: {e}")
        return False

# --------- SAVE WITH JOBLIB (PRIMARY) ---------
joblib_path = "logistic_regression_classifier.pkl"
print(f"\n1. Saving with joblib (PRIMARY METHOD)...")
try:
    joblib.dump(clf, joblib_path, compress=3)
    is_valid = validate_model_file(joblib_path, "joblib")
    if is_valid:
        print("   ✓ Joblib save successful!")
    else:
        print("   ⚠ Joblib save failed validation")
except Exception as e:
    print(f"   ✗ Joblib save error: {e}")

# --------- BACKUP: SAVE WITH PICKLE ---------
pickle_path = "logistic_regression_classifier_backup.pkl"
print(f"\n2. Saving with pickle (BACKUP METHOD)...")
try:
    with open(pickle_path, 'wb') as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    is_valid = validate_model_file(pickle_path, "pickle")
    if is_valid:
        print("   ✓ Pickle backup successful!")
except Exception as e:
    print(f"   ✗ Pickle save error: {e}")

# --------- SAVE METADATA ---------
print(f"\n3. Saving metadata...")
try:
    metadata = {
        "class_to_idx": train_ds.class_to_idx,
        "idx_to_class": {v: k for k, v in train_ds.class_to_idx.items()},
        "feature_size": X_train.shape[1],
        "n_classes": len(train_ds.classes),
        "classes": train_ds.classes,
    }
    torch.save(metadata, "resnet50_feature_metadata.pt")
    print(f"   ✓ Metadata saved ({os.path.getsize('resnet50_feature_metadata.pt')} bytes)")
except Exception as e:
    print(f"   ✗ Metadata save error: {e}")

# =========================================
# FINAL VERIFICATION
# =========================================
print("\n" + "="*60)
print("FINAL VERIFICATION")
print("="*60)

files_to_check = [
    (joblib_path, "joblib"),
    (pickle_path, "pickle"),
    ("resnet50_feature_metadata.pt", "torch")
]

all_valid = True
for filepath, ftype in files_to_check:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {filepath} ({size} bytes)")
        
        # Extra validation
        try:
            if ftype == "joblib":
                joblib.load(filepath)
            elif ftype == "pickle":
                with open(filepath, 'rb') as f:
                    pickle.load(f)
            elif ftype == "torch":
                torch.load(filepath)
            print(f"  └─ Load test: PASSED")
        except Exception as e:
            print(f"  └─ Load test: FAILED ({e})")
            all_valid = False
    else:
        print(f"✗ {filepath} - NOT FOUND")
        all_valid = False

print("\n" + "="*60)
if all_valid:
    print("✓ ALL FILES SAVED AND VALIDATED SUCCESSFULLY!")
    print("\nYou can now upload to Hugging Face Spaces:")
    print("  - logistic_regression_classifier.pkl")
    print("  - resnet50_feature_metadata.pt")
    print("  - app.py")
    print("  - requirements.txt")
else:
    print("⚠ SOME FILES FAILED VALIDATION - CHECK ERRORS ABOVE")
print("="*60)
