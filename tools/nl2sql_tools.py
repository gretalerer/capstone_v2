import pandas as pd
from google.cloud import bigquery
from langchain.llms import OpenAI  # or whatever LLM you're using
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Set up Google BigQuery (ensure GOOGLE_APPLICATION_CREDENTIALS is set in .env)

client = bigquery.Client()
project_id = client.project  

# Initialize clients
client = bigquery.Client()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_large = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
db = SQLDatabase.from_uri(f"bigquery://{project_id}")

def write_query(question: str) -> str:
    prompt = f"""
    You are an AI that generates SQL queries for a BigQuery database.

    **Database dialect:** {db.dialect}
    **Schema:** {db.get_table_info()}
    
    **User question:** "{question}"
    
    **Rules:**
    - Write a valid BigQuery SQL query that answers the question
    - Return only the SQL query, no explanations
    - Do not assume unknown columns
    - Sort intelligently based on question intent:
      * DESC for "highest/most/top"
      * ASC for "lowest/least/bottom"
      * By date for trends
      * By metric for averages/totals
    - Limit results only when:
      * Question specifies a number (e.g., "top 5")
      * Question asks about extremes
      * Data would be too large
    
    Example: SELECT column FROM table ORDER BY metric DESC;
    """
    raw_result = llm.invoke(prompt)

    sql_query = raw_result.content.strip().replace("```sql", "").replace("```", "").strip()
    return sql_query

def execute_query(sql_query: str) -> str:
    try:
        query_job = client.query(sql_query)
        results_df = query_job.result().to_dataframe()
        
        # If results are too large, truncate to first 100 rows
        if len(results_df) > 100:
            print("[EXECUTE] Results too large, truncating to first 100 rows")
            results_df = results_df.head(100)
            
        # Convert to markdown with limited precision for numeric columns
        pd.set_option('display.precision', 2)
        results_markdown = results_df.to_markdown()
        
        # If markdown is still too large, truncate it
        if len(results_markdown) > 10000:
            print("[EXECUTE] Markdown too large, truncating to first 10000 characters")
            results_markdown = results_markdown[:10000] + "\n... (results truncated)"
            
        return results_markdown
    except Exception as e:
        return f"❌ Error: {str(e)}"

def summarize_result(sql_query: str, results: str) -> str:
    if "❌ Error" in results:
        return "I couldn't process the query due to an error."

    prompt = f"""
    You are a data analyst. Based on the following SQL query and its output, write a straightforward natural language answer that directly communicates the query result.

    **SQL Query:** 
    ```sql
    {sql_query}
    ```

    **Results:**
    {results}

    **Rules:**
    - Write in a conversational style.
    - Do not give an opinion or interpretation.

    Example:
    China was the country with the highest number of orders, with a total of 42,355 orders.
    """
    response = llm_large.invoke(prompt)
    return response.content.strip()



