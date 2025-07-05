# GenAI Project – Feedback Classification using BERT

This project leverages the power of **transformer-based models**, specifically **BERT (Bidirectional Encoder Representations from Transformers)**, to classify student feedback into four categories: **Academics**, **Facilities**, **Administration**, and **Student Life**.

## 📌 Project Objective

The goal is to build a robust multi-class text classification model that can understand and categorize qualitative feedback from students, helping colleges make data-driven improvements.

## 📂 Dataset

The dataset includes two files:

- `college_feedback_train.csv` – Labeled training data with feedback and categories.
- `college_feedback_test.csv` – Unlabeled test data for model inference.

Each feedback is mapped to one of the following labels:

| Label | Category         |
|-------|------------------|
| 0     | Academics        |
| 1     | Facilities       |
| 2     | Administration   |
| 3     | Student Life     |

## 🔍 Key Features

- **Text Preprocessing**: Feedback data is cleaned and tokenized using Hugging Face's `BertTokenizer`.
- **Model**: Pretrained `bert-base-uncased` model from Hugging Face is fine-tuned for classification using PyTorch.
- **Training**: The model is trained with cross-entropy loss and AdamW optimizer. Evaluation metrics include accuracy and classification report.
- **Prediction**: The trained model is used to predict the categories for test feedback, with outputs mapped back to human-readable labels.

## 🛠️ Tools and Libraries

- Python
- Pandas
- NumPy
- scikit-learn
- PyTorch
- Transformers (Hugging Face)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ac265640/GenAi-Project.git
   cd GenAi-Project
2. Install dependencies:
   pip install -r requirements.txt

3. Run the notebook:
   Open genai_proj.ipynb in Jupyter Notebook or VSCode and execute the cells sequentially.


📈 Results

Achieved strong classification performance on the training and validation sets.
The notebook includes visualizations of training loss and accuracy for model evaluation.

🧠 Model Interpretability

Although BERT is a black-box model, future work can include:

Attention visualization
SHAP values for interpretability
📌 Future Improvements

Hyperparameter tuning
Experimenting with other transformer models like RoBERTa or DistilBERT
Adding UI for feedback submission and instant classification

🙌 Acknowledgements

Hugging Face for BERT and transformer tools
