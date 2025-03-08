# NLtoSQL Transformer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-latest-yellow)

## 🚀 Overview

NLtoSQL Transformer is a machine learning solution that translates natural language questions into SQL queries using fine-tuned T5 transformer models. The system handles both SQL generation from natural language and SQL query correction, making database interaction more accessible to non-technical users.

## ✨ Features

- **Natural Language to SQL Generation**: Convert plain English questions to executable SQL queries
- **SQL Query Correction**: Automatically detect and fix errors in problematic SQL queries
- **Multi-task Learning**: Single model trained on both generation and correction tasks
- **T5 Transformer Architecture**: Leverages state-of-the-art language model capabilities

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nltosql-transformer.git
cd nltosql-transformer

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas
transformers
torch
datasets
accelerate
psycopg2
scikit-learn
```

## 📊 Dataset

The model is trained on two types of data:
1. **Generation Task**: Natural language questions paired with correct SQL queries
2. **Correction Task**: Incorrect SQL queries paired with their corrected versions

## 💻 Usage

### Training the Model

```python
# Load and preprocess data
train_generate_task = pd.read_json("train_generate_task.json")
train_query_correction_task = pd.read_json("train_query_correction_task.json")

# Rename and preprocess columns
train_generate_task.rename(columns={'NL': 'natural_language', 'Query': 'sql_query'}, inplace=True)
train_query_correction_task.rename(columns={'IncorrectQuery': 'incorrect_query', 'CorrectQuery': 'correct_query'}, inplace=True)

# Convert to datasets and tokenize
train_dataset_gen = convert_to_dataset(train_generate_task, "generate")
train_dataset_corr = convert_to_dataset(train_query_correction_task, "correct")
train_dataset = concatenate_datasets([train_dataset_gen, train_dataset_corr])

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)
trainer.train()
```

### Inference

```python
# Load the trained model
tokenizer = T5Tokenizer.from_pretrained("./t5_nl_to_sql")
model = T5ForConditionalGeneration.from_pretrained("./t5_nl_to_sql")

# Convert natural language to SQL
def nl_to_sql(question):
    input_text = "Convert the following to SQL: " + question
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
sql_query = nl_to_sql("Find all customers who made purchases over $100 last month")
print(sql_query)
```

## 📈 Performance

The model achieves:
- **Generation Task**: XX% accuracy on validation set
- **Correction Task**: XX% accuracy on validation set
- **Combined Tasks**: XX% overall accuracy

## 🔍 Project Structure

```
nltosql-transformer/
├── data/
│   ├── train_generate_task.json
│   └── train_query_correction_task.json
├── src/
│   ├── data_processing.py
│   ├── model.py
│   └── train.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── saved_models/
│   └── t5_nl_to_sql/
├── requirements.txt
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


```

