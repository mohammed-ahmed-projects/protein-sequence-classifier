# Protein Sequence Classifier

## Overview
This repository provides a deep learning framework for **protein sequence classification**, distinguishing between enzymes and non-enzymes.  
It combines **data preprocessing, model training, evaluation, and interactive visualization**, making it suitable for both research and educational use.

## Features
- **Comprehensive Data Processing**: Protein sequence cleaning, one-hot encoding, and dataset preparation.  
- **Deep Learning Models**: Implements both Convolutional Neural Networks (**CNN**) and Bidirectional Long Short-Term Memory (**LSTM**) models.  
- **Biochemical Insights**: Basic amino acid composition analysis and sequence statistics.  
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, confusion matrix, and ROC curves.  
- **Visualization**: Clear plots to show training dynamics, dataset distributions, and model predictions.  
- **Interactive Web Demo**: Built with Gradio for live testing and exploration.  

## Model Architectures

### CNN
- One-hot encoding (20-dimensional vectors)  
- 1D Convolution (128 filters)  
- Adaptive max pooling  
- Fully connected classifier  

### BiLSTM
- Bidirectional LSTM (64 hidden units)  
- 64-dimensional embedding  
- Fully connected classifier  

## Training Results
Detailed training notebooks, model performance plots, and dataset preparation scripts are included in the repository under:  
- **Main directory**: evaluation metrics and visualizations (e.g., PNG, CSV files)  
- **Main directory**: trained CNN and LSTM weights saved as `.pth` files  

## Technologies
- **Python**, **PyTorch**, **Scikit-learn**, **Pandas**, **Matplotlib**, **Gradio**  

## Installation & Usage

Clone the repository and install dependencies:
```
git clone https://github.com/mohammed-ahmed-projects/protein-sequence-classifier.git
cd protein-sequence-classifier
pip install -r requirements.txt
```

## Train Models
```
python protein_classificatio_1.py
```

## Launch Demo
```
python app.py
```

## Citation

If you use this repository in your research, please cite it:

### APA:
Abdelmagid, M. (2025). Protein Sequence Classifier. GitHub. Retrieved from https://github.com/mohammed-ahmed-projects/protein-sequence-classifier

### BibTeX:
```
@misc{abdelmagid2025_protein_classifier,
  author       = {Mohammed Ahmed Abdelmagid},
  title        = {Protein Sequence Classifier},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/mohammed-ahmed-projects/protein-sequence-classifier}}
}
```

## Contact

For questions, collaborations, or suggestions:

- Email: mohammed.ahmed.projects@gmail.com
- Skype / Microsoft Teams: live:.cid.13bbe26ff8b2c5d2
- Hugging Face: Profile
- GitHub Issues: Open a ticket directly in this repository
