import pandas as pd
import numpy as np
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def setup_database(df: pd.DataFrame, table_name: str) -> SQLDatabase:
    """
    Create a temporary SQLite database from a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to be converted to SQL database
        table_name (str): Name of the table to create in the database

    Returns:
        SQLDatabase: LangChain SQLDatabase object
    """
    engine = create_engine("sqlite:///temp.db")
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    return SQLDatabase(engine=engine)

def get_answer(db: SQLDatabase, question: str, table_name: str) -> str:
    """
    Generate an answer to a question about the data using LangChain and Groq.

    Args:
        db (SQLDatabase): Database containing the data
        question (str): User's question
        table_name (str): Name of the table to query

    Returns:
        str: Generated answer to the question
    """
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    
    contextualized_question = f"Using the {table_name} table, {question}"
    response = agent_executor.invoke({"input": contextualized_question})
    return response.get("output", "Sorry, I couldn't find an answer to your question.")

def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response, handling various formats.

    Args:
        response (str): Response from the LLM

    Returns:
        str: Extracted Python code
    """
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0].strip()
    elif "```" in response:
        code = response.split("```")[1].strip()
    else:
        code = response.strip()
    return code

def clean_generated_code(code: str) -> str:
    """
    Clean and validate the generated code.

    Args:
        code (str): Generated Python code

    Returns:
        str: Cleaned Python code
    """
    code = code.replace('fig.show()', '')
    lines = code.split('\n')
    cleaned_lines = [
        line for line in lines 
        if not line.strip().startswith('import') 
        and not line.strip().startswith('from')
    ]
    return '\n'.join(cleaned_lines)

def generate_chart_code(query: str, df: pd.DataFrame) -> str:
    """
    Generate Python code for creating a chart based on the user's query.

    Args:
        query (str): User's query for the chart
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        str: Generated Python code for the chart
    """
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    prompt = (
        "Generate only executable Python code for a plotly chart. "
        f"Task: {query}\n"
        f"Available columns: {list(df.columns)}\n"
        "Requirements:\n"
        "1. Only use plotly.express (px) or plotly.graph_objects (go)\n"
        "2. Name the figure variable 'fig'\n"
        "3. Don't include imports or fig.show()\n"
        "4. Use 'df' as the DataFrame name\n"
        "Example format:\n"
        "filtered_df = df.sort_values('column', ascending=False).head(10)\n"
        "fig = px.bar(filtered_df, x='column1', y='column2', title='Chart Title')"
    )
    response = llm.invoke(prompt)
    code = extract_code_from_response(response.content)
    return clean_generated_code(code)

def execute_generated_code(code: str, df: pd.DataFrame):
    """
    Execute the generated Python code in the current context.

    Args:
        code (str): Generated Python code
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        go.Figure: Generated Plotly figure
    """
    try:
        namespace = {
            'df': df,
            'px': px,
            'go': go,
            'pd': pd,
            'np': np
        }
        exec(code.strip(), namespace)
        if 'fig' in namespace:
            return namespace['fig']
        else:
            raise Exception("No figure was created (missing 'fig' variable)")
    except Exception as e:
        raise Exception(f"Code execution failed:\n{str(e)}\nGenerated code:\n{code}")

def generate_prompt(query: str, columns: list) -> str:
    """
    Generate a structured prompt based on user query.

    Args:
        query (str): User's query
        columns (list): List of available columns in the data

    Returns:
        str: Generated prompt
    """
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    system_prompt = (
        f"Based on the following user query and available columns {columns}, "
        "generate a clear, specific prompt that would help get better results. "
        "Consider data analysis best practices and focus on actionable insights."
    )
    
    response = llm.invoke(f"{system_prompt}\nUser query: {query}")
    return response.content

def handle_reuse_prompt(prompt_type: str, prompt_text: str):
    """
    Handle the reuse of a previously generated prompt.

    Args:
        prompt_type (str): Type of the prompt (Q&A or Chart)
        prompt_text (str): Text of the prompt
    """
    if prompt_type == "Q&A":
        st.session_state.reuse_qa = prompt_text
        st.session_state.active_tab = "Q&A"
    else:  # Chart
        st.session_state.reuse_chart = prompt_text
        st.session_state.active_tab = "Charts"

def handle_feedback(feedback_type: str, key: str, text: str, feedback_value: str):
    """
    Handle user feedback for Q&A or chart.

    Args:
        feedback_type (str): Type of feedback (qa or chart)
        key (str): Unique key for the feedback
        text (str): Text of the question or chart query
        feedback_value (str): Feedback value (helpful, not_helpful, good, needs_improvement)
    """
    if feedback_type == "qa":
        if key not in st.session_state.qa_feedback:
            st.session_state.qa_feedback[key] = feedback_value
            st.session_state.feedback_history.append(f"Q&A Feedback: {feedback_value} for question '{text}'")
    else:  # chart
        if key not in st.session_state.chart_feedback:
            st.session_state.chart_feedback[key] = feedback_value
            st.session_state.feedback_history.append(f"Chart Feedback: {feedback_value} for query '{text}'")
