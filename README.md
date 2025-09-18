# Protein Sequence Classification Project

## Overview
This project classifies protein sequences as enzymes or non-enzymes using deep learning models. It features a complete pipeline from data preprocessing to model training and performance evaluation.

## Key Features
- **Comprehensive Data Processing**: Handles protein sequences, performs one-hot encoding, and creates a clean dataset.
- **Advanced Deep Learning Models**: Utilizes both a Convolutional Neural Network (**CNN**) and a Bidirectional Long Short-Term Memory (**LSTM**) model for accurate classification.
- **Detailed Evaluation**: Generates metrics like accuracy, precision, recall, and F1-score, along with a confusion matrix and ROC curves.
- **Data Visualization**: Creates professional plots to visualize model performance and data characteristics.
- **Interactive Web Demo**: A user-friendly interface built with Gradio for a live demonstration.

## Technologies
- **Python**
- **PyTorch**
- **Scikit-learn**
- **Pandas**
- **Matplotlib**
- **Gradio**

## Project Files
- `protein_classificato_1.py`: The main script for the complete project pipeline.
- `huggingface_demo.py`: The script for the interactive Gradio demo.
- `requirements.txt`: Lists all necessary Python libraries.
- `models/`: Folder containing the trained CNN and LSTM models.
- `results/`: Folder containing performance metrics and visualizations.

## Live Demo
You can try the project yourself here:
[Hugging Face Space Live Demo](https://huggingface.co/spaces/your-username/your-project-name)
*(Link will be added after finalizing the demo)*

## Usage
1. Clone this repository.
2. Install the required libraries: `pip install -r requirements.txt`.
3. Run the main script to train the models and generate results: `python protein_classificato_1.py`.
4. Run the demo script to launch the interactive interface: `python huggingface_demo.py`.

## Citation

If you find this repository useful in your research or projects, please cite it as:

**APA:**
Ahmed, M. (2025). *Protein Sequence Classifier*. GitHub. Retrieved from https://github.com/mohammed-ahmed-projects/protein-sequence-classifier  

**BibTeX:**
```
@misc{ahmed2025_protein_classifier,
  author       = {Mohammed Ahmed},
  title        = {Protein Sequence Classifier},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/mohammed-ahmed-projects/protein-sequence-classifier}}
}
```

## Contact
For any questions or support, please open an issue on this repository.
