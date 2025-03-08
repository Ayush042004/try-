<div align="center">

# 🔮 NLtoSQL Transformer

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-latest-yellow)](https://huggingface.co/docs/transformers/index)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B-blue)](https://www.postgresql.org/)

**Translate natural language to SQL with the power of T5 transformers**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Troubleshooting](#postgresql-troubleshooting) • [Contributing](#contributing) • [License](#license)

</div>

---

## 📋 Overview

**NLtoSQL Transformer** bridges the gap between natural language and database queries, enabling non-technical users to interact with databases using plain English. Built on state-of-the-art T5 transformer models, this system not only generates accurate SQL queries from natural language but also corrects problematic SQL statements, making database interaction more accessible and efficient.

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=NLtoSQL+Transformer+Architecture" alt="NLtoSQL Architecture" width="600"/>
</p>

## ✨ Features

<table>
  <tr>
    <td width="50%">
      <h3>🔄 Natural Language to SQL</h3>
      Convert plain English questions to optimized SQL queries
    </td>
    <td width="50%">
      <h3>🛠️ SQL Query Correction</h3>
      Automatically detect and fix errors in problematic SQL queries
    </td>
  </tr>
  <tr>
    <td>
      <h3>🧠 Multi-task Learning</h3>
      Single model trained on both generation and correction tasks
    </td>
    <td>
      <h3>🚀 T5 Architecture</h3>
      Leverages state-of-the-art language model capabilities
    </td>
  </tr>
  <tr>
    <td>
      <h3>🗃️ PostgreSQL Integration</h3>
      Direct connectivity to PostgreSQL databases
    </td>
    <td>
      <h3>📊 Performance Metrics</h3>
      Comprehensive evaluation on diverse query types
    </td>
  </tr>
</table>

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nltosql-transformer.git
cd nltosql-transformer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.3.0
transformers>=4.18.0
torch>=1.10.0
datasets>=2.0.0
accelerate>=0.5.0
psycopg2-binary>=2.9.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

## 📊 Dataset

The model is trained on two complementary datasets:

<div align="center">
  <table>
    <tr>
      <th>Generation Task</th>
      <th>Correction Task</th>
    </tr>
    <tr>
      <td>
        Natural language questions paired with correct SQL queries<br>
        <code>{"natural_language": "Show all employees in IT dept", "sql_query": "SELECT * FROM employees WHERE department = 'IT'"}</code>
      </td>
      <td>
        Incorrect SQL queries paired with their corrected versions<br>
        <code>{"incorrect_query": "SELECT FROM employees WHERE dept = IT", "correct_query": "SELECT * FROM employees WHERE department = 'IT'"}</code>
      </td>
    </tr>
  </table>
</div>

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

<details>
<summary><b>Authentication Issues</b></summary>

### "Password Authentication Failed" Error

If you encounter `FATAL: password authentication failed for user 'postgres'`:

1. **Verify Password**: Ensure you're using the correct password
2. **Reset Password** (if forgotten):
   ```sql
   ALTER USER postgres WITH PASSWORD 'newpassword';
   ```
3. **Check `pg_hba.conf`** for authentication settings:
   - Linux/macOS: `/etc/postgresql/<version>/main/`
   - Windows: `C:\Program Files\PostgreSQL\<version>\data\`

</details>

<details>
<summary><b>Command Not Found (Windows)</b></summary>

### "psql is not recognized" Error

If you see `'psql' is not recognized as an internal or external command`:

1. **Verify Installation**:
   ```
   where psql
   ```

2. **Add to PATH**:
   - Copy the bin folder path (e.g., `C:\Program Files\PostgreSQL\15\bin`)
   - Add to system environment variables:
     - Win + R → type `sysdm.cpl` → Advanced → Environment Variables
     - Under System variables, edit Path and add the PostgreSQL bin path
   - Restart Command Prompt

3. **Use Full Path**:
   ```
   "C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres
   ```

</details>

<details>
<summary><b>Service Not Running</b></summary>

### Check PostgreSQL Service Status

#### Windows:
- Press Win + R, type `services.msc`
- Find PostgreSQL service, ensure it's running
- If not, right-click and select "Start"

#### Linux/macOS:
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql  # to start if not running
```

</details>

## 📈 Performance

<div align="center">
  <table>
    <tr>
      <th>Task</th>
      <th>Accuracy</th>
      <th>F1 Score</th>
    </tr>
    <tr>
      <td>Generation Task</td>
      <td>XX%</td>
      <td>XX%</td>
    </tr>
    <tr>
      <td>Correction Task</td>
      <td>XX%</td>
      <td>XX%</td>
    </tr>
    <tr>
      <td>Combined Tasks</td>
      <td>XX%</td>
      <td>XX%</td>
    </tr>
  </table>
</div>

## 🔍 Project Structure

```
nltosql-transformer/
├── data/                      # Data files
│   ├── train_generate_task.json
│   └── train_query_correction_task.json
├── src/                       # Source code
│   ├── data_processing.py     # Data preprocessing utilities
│   ├── model.py               # T5 model definition and customization
│   ├── train.py               # Training pipeline
│   └── database.py            # PostgreSQL connection and query execution
├── notebooks/                 # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── saved_models/              # Trained model checkpoints
│   └── t5_nl_to_sql/
├── tests/                     # Unit and integration tests
│   ├── test_model.py
│   └── test_database.py
├── docs/                      # Documentation
│   └── api_reference.md
├── requirements.txt           # Dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
```

## 🔮 Future Work

- [ ] Support for complex SQL constructs (window functions, CTEs)
- [ ] Database schema awareness for context-specific query generation
- [ ] Web interface for interactive query building
- [ ] Extension to multiple database dialects (MySQL, MongoDB, etc.)
- [ ] Automatic schema detection and database setup
- [ ] Few-shot learning for custom domain adaptation

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork** the repository
2. **Create** your feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add some amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Submit** a Pull Request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{nltosql_transformer,
  author = {Your Name},
  title = {NLtoSQL Transformer},
  year = {2025},
  url = {https://github.com/yourusername/nltosql-transformer}
}
```

## 📧 Contact

<div align="center">
  <a href="mailto:your.email@example.com">
    <img src="https://img.shields.io/badge/Email-your.email%40example.com-red?style=for-the-badge&logo=gmail" alt="Email"/>
  </a>
  <a href="https://github.com/yourusername">
    <img src="https://img.shields.io/badge/GitHub-yourusername-black?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <a href="https://linkedin.com/in/yourusername">
    <img src="https://img.shields.io/badge/LinkedIn-yourusername-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/>
  </a>
</div>

<p align="center">⭐ Star this repository if you find it useful! ⭐</p>
