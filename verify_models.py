"""
Quick verification script to test if your model files are valid
Run this BEFORE uploading to Hugging Face Spaces
"""

import os
import sys

def check_file_exists(filename):
    """Check if file exists and show size"""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"  ✓ {filename} ({size} bytes)")
        return True
    else:
        print(f"  ✗ {filename} NOT FOUND")
        return False

def test_joblib_load(filename):
    """Test loading a joblib file"""
    try:
        import joblib
        model = joblib.load(filename)
        print(f"  ✓ Joblib load: SUCCESS")
        print(f"    - Model type: {type(model).__name__}")
        if hasattr(model, 'classes_'):
            print(f"    - Classes: {model.classes_}")
        if hasattr(model, 'n_features_in_'):
            print(f"    - Features: {model.n_features_in_}")
        return True
    except Exception as e:
        print(f"  ✗ Joblib load: FAILED")
        print(f"    Error: {e}")
        return False

def test_torch_load(filename):
    """Test loading a torch file"""
    try:
        import torch
        data = torch.load(filename)
        print(f"  ✓ Torch load: SUCCESS")
        print(f"    - Keys: {list(data.keys()) if isinstance(data, dict) else 'tensor'}")
        return True
    except Exception as e:
        print(f"  ✗ Torch load: FAILED")
        print(f"    Error: {e}")
        return False

def test_pickle_load(filename):
    """Test loading a pickle file"""
    try:
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✓ Pickle load: SUCCESS")
        print(f"    - Type: {type(data).__name__}")
        return True
    except Exception as e:
        print(f"  ✗ Pickle load: FAILED")
        print(f"    Error: {e}")
        return False

print("="*70)
print("MODEL FILE VERIFICATION SCRIPT")
print("="*70)

all_ok = True

# ========== CHECK LOGISTIC REGRESSION MODEL ==========
print("\n1. Checking logistic_regression_classifier.pkl...")
print("-" * 70)

if check_file_exists("logistic_regression_classifier.pkl"):
    test_joblib_load("logistic_regression_classifier.pkl")
else:
    all_ok = False
    # Try backup
    if check_file_exists("logistic_regression_classifier_backup.pkl"):
        print("  → Using backup .pkl file instead")
        test_pickle_load("logistic_regression_classifier_backup.pkl")

# ========== CHECK METADATA ==========
print("\n2. Checking resnet50_feature_metadata.pt...")
print("-" * 70)

if check_file_exists("resnet50_feature_metadata.pt"):
    test_torch_load("resnet50_feature_metadata.pt")
else:
    all_ok = False

# ========== CHECK APP FILES ==========
print("\n3. Checking app files...")
print("-" * 70)

app_files = ["app.py", "requirements.txt", "README.md"]
for f in app_files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"  ✓ {f} ({size} bytes)")
    else:
        print(f"  ⚠ {f} NOT FOUND (optional)")

# ========== FINAL STATUS ==========
print("\n" + "="*70)
if all_ok:
    print("✓ ALL REQUIRED FILES ARE VALID!")
    print("\nYou can now safely upload to Hugging Face Spaces:")
    print("  - logistic_regression_classifier.pkl")
    print("  - resnet50_feature_metadata.pt")
    print("  - app.py")
    print("  - requirements.txt")
    print("  - README.md")
else:
    print("⚠ SOME FILES ARE MISSING OR CORRUPTED")
    print("\nTo fix:")
    print("  1. Run: python resnet50_logistic_regression_IMPROVED.py")
    print("  2. Re-run this verification script")
    print("  3. Check the logs for any errors")

print("="*70)

sys.exit(0 if all_ok else 1)
