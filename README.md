# NLtoSQL Transformer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-latest-yellow)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B-blue)

## 🚀 Overview

NLtoSQL Transformer is a machine learning solution that translates natural language questions into SQL queries using fine-tuned T5 transformer models. The system handles both SQL generation from natural language and SQL query correction, making database interaction more accessible to non-technical users.

## ✨ Features

- **Natural Language to SQL Generation**: Convert plain English questions to executable SQL queries
- **SQL Query Correction**: Automatically detect and fix errors in problematic SQL queries
- **Multi-task Learning**: Single model trained on both generation and correction tasks
- **T5 Transformer Architecture**: Leverages state-of-the-art language model capabilities
- **PostgreSQL Integration**: Direct connectivity to PostgreSQL databases for query execution

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

### Database Integration

```python
import psycopg2

# Connect to PostgreSQL database
def execute_query(sql_query):
    try:
        conn = psycopg2.connect(
            dbname="your_database",
            user="postgres",
            password="your_password",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        return f"Error: {str(e)}"

# Example of end-to-end usage
user_question = "Show me all customers from New York"
sql_query = nl_to_sql(user_question)
results = execute_query(sql_query)
```

## 🔧 PostgreSQL Troubleshooting

### Common PostgreSQL Connection Issues

#### 1. "Password Authentication Failed" Error

If you encounter `FATAL: password authentication failed for user 'postgres'`:

- Verify you're using the correct password
- Reset the password if forgotten:
  ```sql
  ALTER USER postgres WITH PASSWORD 'newpassword';
  ```
- Check `pg_hba.conf` file for authentication settings (typically located in `/etc/postgresql/<version>/main/` on Linux or `C:\Program Files\PostgreSQL\<version>\data\` on Windows)

#### 2. "psql is not recognized" Error (Windows)

If you see `'psql' is not recognized as an internal or external command`:

- Verify PostgreSQL is installed:
  ```
  where psql
  ```
- Add PostgreSQL bin directory to PATH:
  1. Copy the bin folder path (e.g., `C:\Program Files\PostgreSQL\15\bin`)
  2. Add to system environment variables:
     - Win + R → type `sysdm.cpl` → Advanced → Environment Variables
     - Under System variables, edit Path and add the PostgreSQL bin path
  3. Restart Command Prompt and try again
- Alternatively, use the full path to psql:
  ```
  "C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres
  ```

#### 3. Check If PostgreSQL Service Is Running

- On Windows:
  - Press Win + R, type `services.msc`
  - Find PostgreSQL service, ensure it's running
  - If not, right-click and select "Start"

- On Linux/macOS:
  ```bash
  sudo systemctl status postgresql
  sudo systemctl start postgresql  # to start if not running
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
│   ├── train.py
│   └── database.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── saved_models/
│   └── t5_nl_to_sql/
├── requirements.txt
└── README.md
```

## 🔮 Future Work

- Add support for more complex SQL constructs (e.g., window functions, CTEs)
- Implement database schema awareness for more accurate query generation
- Create a web interface for easy interaction with the model
- Extend to multiple database dialects (MySQL, PostgreSQL, etc.)
- Implement automatic PostgreSQL setup and connection validation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this code in your research, please cite:

```
@software{nltosql_transformer,
  author = {Your Name},
  title = {NLtoSQL Transformer},
  year = {2025},
  url = {https://github.com/yourusername/nltosql-transformer}
}
```

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/nltosql-transformer](https://github.com/yourusername/nltosql-transformer)
