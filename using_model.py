from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sqlalchemy import create_engine, text
import pandas as pd

#  Load Fine-Tuned Model 
model_path = model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

sql_generator = pipeline(
    "text-generation",   # causal models = text-generation
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150
)

# Connect to MySQL 
#put you Variables here
username = user
password = password
host = host
port = 3306
database = database

engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")

#Helper Functions
def question_to_sql(question: str) -> str:
    """Use fine-tuned model to convert question into SQL query"""
    response = sql_generator(question)
    sql_query = response[0]["generated_text"].strip()
    return sql_query

def execute_sql(query: str):
    """Run SQL query on MySQL and return results"""
    try:
        with engine.connect() as conn:
            result = pd.read_sql_query(text(query), conn)
        return result
    except Exception as e:
        return f"SQL Execution Error: {str(e)}"

# Full Pipeline
def pipeline_run(user_question: str):
    sql_query = question_to_sql(user_question)
    print(f"\n Generated SQL: {sql_query}\n")
    result = execute_sql(sql_query)
    return result

# Example
user_question = "Show me 5 employees with their name and salary"
output = pipeline_run(user_question)
print("\nFinal Result:")
print(output)
