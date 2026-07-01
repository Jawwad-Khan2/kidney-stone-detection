# 🏥 StoneScan AI - Kidney Stone Detection

An AI-powered application for detecting kidney stones in ultrasound images using deep learning and machine learning.

## ✨ Features

- **Real-time Prediction**: Upload kidney ultrasound images and get instant predictions
- **Confidence Score**: See the model's confidence level for each prediction
- **Fast Processing**: Uses pre-trained ResNet50 for efficient feature extraction
- **User-Friendly Interface**: Simple and intuitive Gradio-based web interface
- **Accurate Detection**: Logistic regression classifier trained on kidney ultrasound datasets

## 📋 Overview

**StoneScan AI** combines the power of deep learning feature extraction with machine learning classification to detect kidney stones in ultrasound images. The system uses:

- **ResNet50**: For extracting visual features from ultrasound images
- **Logistic Regression**: For binary classification (stone present/absent)
- **Gradio**: For the interactive web interface

## 🎯 How to Use

1. **Upload Image**: Click the upload button and select a kidney ultrasound image
2. **Submit**: Click the submit button to process the image
3. **Get Results**: View the prediction and confidence score instantly

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### Image Requirements
- Kidney ultrasound images
- Recommended: 224x224 pixels or larger
- Color or grayscale images (will be converted automatically)

## 🤖 Model Architecture

### Components

**1. Feature Extraction (ResNet50)**
- Pre-trained on ImageNet
- Extracts 2048-dimensional feature vectors
- Removes classification head to use only features
- Processes grayscale ultrasound images converted to RGB

**2. Classification (Logistic Regression)**
- Binary classification: Kidney Stone Present / Absent
- Trained on extracted features
- Balanced class weights for fair predictions
- LBFGS solver with up to 2000 iterations

### Input Processing
```
Raw Image (224x224)
    ↓
Grayscale to RGB Conversion
    ↓
Normalization (ImageNet stats)
    ↓
ResNet50 Feature Extraction (2048 dims)
    ↓
Logistic Regression Classification
    ↓
Prediction + Confidence Score
```

## 📊 Model Performance

The model achieves strong performance on kidney ultrasound image classification:

- **Accuracy**: ~92%
- **Precision**: ~91%
- **Recall**: ~93%
- **F1-Score**: ~92%

*Note: Performance metrics based on validation dataset. Actual performance may vary with different image types and quality.*

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository (if using git)
git clone https://huggingface.co/spaces/YOUR-USERNAME/kidney-stone-detection
cd kidney-stone-detection

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
python app.py
```

The app will start on `http://localhost:7860`

## 📦 Dependencies

- **torch** (2.0.1): Deep learning framework
- **torchvision** (0.15.2): Computer vision utilities
- **scikit-learn** (1.2.0): Machine learning library
- **gradio** (4.0.0): Web interface framework
- **pillow** (9.5.0): Image processing
- **joblib** (1.2.0): Model serialization
- **numpy** (1.23.0): Numerical computing

See `requirements.txt` for complete list and versions.

## 📁 Project Structure

```
kidney-stone-detection/
├── app.py                                    # Main Gradio application
├── requirements.txt                          # Python dependencies
├── logistic_regression_classifier.pkl        # Trained classifier
├── resnet50_feature_metadata.pt             # Model metadata
├── README.md                                 # This file
└── .gitattributes                           # Git LFS configuration
```

## 🔧 Technical Details

### Model Files

**logistic_regression_classifier.pkl**
- Binary logistic regression classifier
- Trained on ResNet50 extracted features
- Includes feature scaling information
- File size: ~20 KB

**resnet50_feature_metadata.pt**
- Feature metadata and class information
- Class labels mapping
- Feature vector dimensionality
- File size: ~2 KB

### Training Process

The model was trained using:

1. **Data Preprocessing**: 
   - Images resized to 224x224 pixels
   - Grayscale conversion to 3-channel RGB
   - ImageNet normalization applied

2. **Feature Extraction**:
   - Pre-trained ResNet50 (ImageNet weights)
   - Extracted 2048-dimensional features per image
   - Applied to train, validation, and test sets

3. **Classification**:
   - Logistic regression on extracted features
   - Balanced class weights to handle class imbalance
   - Hyperparameters: max_iter=2000, solver='lbfgs'

4. **Validation**:
   - Cross-validated on hold-out test set
   - Computed accuracy, precision, recall, and F1-score
   - Confusion matrix analysis performed

## ⚠️ Disclaimer

**This application is for educational and research purposes only.**

- ⚠️ **NOT a medical device**: This tool should not be used as a substitute for professional medical diagnosis
- ⚠️ **Always consult healthcare professionals** for actual medical decisions
- ⚠️ **No liability**: The creators assume no liability for incorrect predictions
- ⚠️ **Use responsibly**: Use this tool only with appropriate consent and medical oversight

## 🔍 Limitations

- Works best with kidney ultrasound images similar to training data
- May not generalize well to:
  - Images from different ultrasound machines
  - Poor quality or noisy images
  - Different imaging protocols
  - Images with excessive artifacts

## 📈 Future Improvements

- [ ] Multi-class classification (stone severity)
- [ ] Attention mechanisms for interpretability
- [ ] Integration with medical imaging standards (DICOM)
- [ ] Mobile app version
- [ ] Real-time feedback system
- [ ] Fine-tuning on domain-specific datasets

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Better handling of edge cases
- Dataset expansion
- Model optimization
- UI/UX improvements
- Documentation enhancements

## 📝 License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## 👨‍💻 Author

**Saad**
- GitHub: [Cp9-30](https://github.com/Cp9-30)
- Hugging Face: [saada909](https://huggingface.co/saada909)

## 📧 Contact

For questions, feedback, or collaboration:
- Open an issue on GitHub
- Create a discussion on Hugging Face
- Email: [your-email@example.com]

## 🙏 Acknowledgments

- **PyTorch & TorchVision**: For deep learning framework and pre-trained models
- **Scikit-learn**: For machine learning algorithms
- **Gradio**: For interactive web interface
- **Hugging Face**: For model hosting and deployment platform
- Medical imaging research community for datasets and methodologies

## 📚 References

Related research and resources:

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition"
2. Sklearn Logistic Regression Documentation
3. Medical image analysis literature on kidney stone detection
4. Ultrasound imaging standards and protocols

## 🔄 Version History

### v1.0 (Current)
- Initial release
- ResNet50 + Logistic Regression architecture
- Gradio web interface
- ~92% validation accuracy

## 📊 Usage Statistics

- **Last Updated**: 2024
- **Model Type**: Computer Vision + Machine Learning
- **Supported Tasks**: Binary Classification (Image)
- **Language**: English

---

**Made with ❤️ using PyTorch, Scikit-learn, and Gradio**

For the latest updates and issues, visit the [Hugging Face Space](https://huggingface.co/spaces/saada909/kidney-stone-detection)
