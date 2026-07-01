"""
VS Code & Python Setup Verification Script
Run this after setting up VS Code to verify everything is installed correctly
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_python():
    """Check Python installation"""
    print_header("1. Python Installation")
    
    try:
        version = sys.version_info
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} installed")
        print(f"  Location: {sys.executable}")
        return True
    except Exception as e:
        print(f"✗ Python check failed: {e}")
        return False

def check_venv():
    """Check if running in virtual environment"""
    print_header("2. Virtual Environment")
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print(f"✓ Virtual environment is ACTIVE")
        print(f"  Location: {sys.prefix}")
        return True
    else:
        print(f"⚠ Virtual environment NOT detected")
        print(f"  IMPORTANT: Activate your virtual environment before continuing")
        print(f"  Windows: .\\venv\\Scripts\\Activate.ps1")
        print(f"  Mac/Linux: source venv/bin/activate")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:<25} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name:<25} NOT INSTALLED")
        return False

def check_packages():
    """Check all required packages"""
    print_header("3. Required Packages")
    
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("Jupyter", "jupyter"),
        ("Gradio", "gradio"),
        ("Pillow", "PIL"),
        ("joblib", "joblib"),
    ]
    
    results = []
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))
    
    return all(results)

def check_gpu():
    """Check GPU availability"""
    print_header("4. GPU/CUDA Support")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ GPU is AVAILABLE")
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Device: {torch.cuda.current_device()}")
            return True
        else:
            print(f"⚠ GPU NOT available (using CPU is fine for training)")
            print(f"  CPU will work but training will be slower")
            return False
    except Exception as e:
        print(f"⚠ GPU check failed: {e}")
        return False

def check_paths():
    """Check project structure"""
    print_header("5. Project Structure")
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    print(f"\nChecking for expected files:")
    
    expected_files = [
        "resnet50_logistic_regression_IMPROVED.py",
        "app.py",
        "requirements.txt",
        "verify_models.py",
    ]
    
    found_files = []
    for file in expected_files:
        file_path = current_dir / file
        if file_path.exists():
            print(f"✓ {file}")
            found_files.append(True)
        else:
            print(f"✗ {file} NOT FOUND")
            found_files.append(False)
    
    return any(found_files)  # At least one file should be present

def check_vscode_extensions():
    """Check if VS Code extensions are likely installed"""
    print_header("6. VS Code Extensions")
    
    print("⚠ Note: This script can't directly check VS Code extensions")
    print("  (They're installed in VS Code, not Python)")
    print("\nMust-have extensions (check in VS Code):")
    print("  ☐ Python (Microsoft)")
    print("  ☐ Pylance (Microsoft)")
    print("  ☐ Jupyter (Microsoft)")
    print("\nTo verify:")
    print("  1. Press Ctrl+Shift+X in VS Code")
    print("  2. Search for each extension name")
    print("  3. Should say 'Installed' next to them")

def main():
    print("\n" + "="*70)
    print("  VS CODE & PYTHON SETUP VERIFICATION")
    print("="*70)
    
    # Run all checks
    checks = {
        "Python": check_python(),
        "Virtual Env": check_venv(),
        "Packages": check_packages(),
        "GPU/CUDA": check_gpu(),
        "Project Files": check_paths(),
    }
    
    check_vscode_extensions()
    
    # Summary
    print_header("SUMMARY")
    
    critical_checks = ["Python", "Virtual Env", "Packages"]
    critical_passed = all(checks[k] for k in critical_checks)
    
    if critical_passed:
        print("✓ SETUP LOOKS GOOD!")
        print("\nYou're ready to train your model:")
        print("  python resnet50_logistic_regression_IMPROVED.py")
    else:
        print("⚠ Some issues detected. Please fix:")
        for check_name, passed in checks.items():
            if not passed and check_name in critical_checks:
                print(f"  - {check_name}")
    
    print("\n" + "="*70)
    print("For detailed setup help, see: VSCODE_SETUP_COMPLETE_GUIDE.md")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()