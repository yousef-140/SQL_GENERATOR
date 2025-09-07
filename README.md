# Text-to-SQL with Fine-Tuned Phi-2

This project demonstrates how to **fine-tune a causal language model (Phi-2)** on the [Spider dataset](https://huggingface.co/datasets/xlangai/spider) and then use it to convert **natural language questions** into **SQL queries**, execute them on a MySQL database, and return the results.

---

## Project Structure

- **`train_sql_model.py`**  
  Script for fine-tuning `microsoft/phi-2` on SQL generation using LoRA.

- **`using_model.py`**  
  Script for loading the fine-tuned model, generating SQL from user questions, and executing the queries on a MySQL database.

---

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/text-to-sql.git
cd text-to-sql
```

### 2. Install Dependencies
```bash
pip install torch transformers datasets peft bitsandbytes accelerate
pip install sqlalchemy pymysql pandas
```

### 3. MySQL Setup
Make sure you have a running MySQL database and update your connection details in `using_model.py`:

```python
username = "your_mysql_username"
password = "your_mysql_password"
host = "localhost"
port = 3306
database = "employees"
```

---

## Fine-Tuning
Run the training script to fine-tune Phi-2 on SQL generation:

```bash
python train_sql_model.py
```

This will save the fine-tuned model under:

```bash
./phi2-sql-lora
```

---

## Using the Model
Once the model is trained (or downloaded), you can run:

```bash
python using_model.py
```

Example:

```
User Question: Show me 5 employees with their name and salary

Generated SQL:
SELECT name, salary FROM employees LIMIT 5;

Final Result:
      name   salary
0   Alice    5000
1   Bob      4200
2   Carol    6100
```

---

## Example Workflow
User asks a question in plain English:
```
"Show me the employees in the Sales department"
```

Model generates SQL:
```sql
SELECT name, department FROM employees WHERE department = 'Sales';
```

Query is executed on MySQL and results are returned.

---

## Notes
- The fine-tuned model is causal (text-generation), not seq2seq.  
- LoRA (peft) is used for efficient fine-tuning.  
- The Spider dataset is used for training (questions + SQL queries).  
- You can replace MySQL with any SQL-compatible database by adjusting the SQLAlchemy connection string.  

---

##📜 License
This project is open-source under the MIT License.
