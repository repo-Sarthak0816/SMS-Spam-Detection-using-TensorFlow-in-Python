# SMS Spam Detection using TensorFlow in Python

A machine learning project that classifies SMS messages as spam or ham (non-spam) using multiple deep learning approaches with TensorFlow and Keras.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements and compares different machine learning models for SMS spam detection, including traditional machine learning and deep learning approaches. The goal is to accurately classify text messages as either spam or legitimate (ham) messages.

## ✨ Features

- **Multiple Model Architectures**: Implements 4 different models for comparison
- **Deep Learning Models**: Uses TensorFlow/Keras for neural network implementations
- **Transfer Learning**: Leverages Universal Sentence Encoder for pre-trained embeddings
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, and F1-score metrics
- **Data Visualization**: Exploratory data analysis with visualizations
- **Text Preprocessing**: Automatic text vectorization and tokenization

## 🤖 Models Implemented

1. **Multinomial Naive Bayes (Baseline)**
   - Traditional machine learning approach
   - Uses TF-IDF vectorization
   - Fast and interpretable

2. **Custom TextVectorization + Embedding Model**
   - Custom word embeddings
   - Global Average Pooling
   - Dense layers with ReLU activation

3. **Bidirectional LSTM Model**
   - Bidirectional LSTM layers for sequence understanding
   - Dropout regularization
   - Best performance among custom models

4. **Universal Sentence Encoder (USE) Transfer Learning Model**
   - Pre-trained sentence embeddings from TensorFlow Hub
   - Transfer learning approach
   - Highest accuracy achieved

## 📊 Dataset

The project uses a standard SMS spam collection dataset (`spam.csv`) with:
- **Labels**: 'ham' (legitimate) and 'spam'
- **Features**: SMS message text
- **Encoding**: Latin-1 encoding

The dataset is split into:
- Training set: 80%
- Test set: 20%

## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow Hub

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SMS-Spam-Detection-using-TensorFlow-in-Python.git
cd SMS-Spam-Detection-using-TensorFlow-in-Python
```

2. Install the required packages:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn tensorflow-hub
```

Or using requirements.txt (create one with the dependencies):
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook SMS_Spam_Detection_using_TensorFlow_in_Python.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the dataset
   - Perform exploratory data analysis
   - Train all models
   - Evaluate and compare model performance

3. Make predictions:
```python
# Example: Using the best model (USE Transfer Learning Model)
prediction = model_3.predict(["Your message here"])
print("Spam" if prediction[0] > 0.5 else "Ham")
```

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **MultinomialNB Model** | 96.32% | 100.00% | 72.48% | 84.05% |
| **Custom-Vec-Embedding Model** | 25.74% | 3.56% | 17.45% | 5.91% |
| **Bidirectional-LSTM Model** | 97.94% | 98.46% | 85.91% | 91.76% |
| **USE-Transfer Learning Model** | **98.30%** | **94.52%** | **92.62%** | **93.56%** |

**Best Model**: Universal Sentence Encoder (USE) Transfer Learning Model achieves the highest overall performance with 98.30% accuracy and balanced precision-recall scores.

## 📁 Project Structure

```
SMS-Spam-Detection-using-TensorFlow-in-Python/
│
├── SMS_Spam_Detection_using_TensorFlow_in_Python.ipynb  # Main notebook
├── spam.csv                                              # Dataset
└── README.md                                             # Project documentation
```

## 🛠️ Technologies Used

- **TensorFlow** - Deep learning framework
- **Keras** - High-level neural networks API
- **TensorFlow Hub** - Pre-trained models and embeddings
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Traditional ML algorithms and utilities

## 🔮 Future Improvements

- [ ] Add model deployment capabilities (Flask/FastAPI)
- [ ] Implement real-time prediction API
- [ ] Add more advanced preprocessing (lemmatization, stemming)
- [ ] Experiment with transformer models (BERT, RoBERTa)
- [ ] Add model explainability features
- [ ] Create a web interface for spam detection
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add support for other languages

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Sarthak Kasaudhan

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Universal Sentence Encoder creators at Google
- The open-source community for the SMS spam dataset

---

⭐ If you find this project helpful, please consider giving it a star!
