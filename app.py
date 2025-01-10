import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from utils import setup_database,get_answer,extract_code_from_response,clean_generated_code,generate_chart_code,execute_generated_code,generate_prompt,handle_reuse_prompt,handle_feedback 

load_dotenv()

def main():
    """
    Main application function that sets up the Streamlit interface and handles all user interactions.
    """
    # Add new session states
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = None
    # Initialize additional session states
    if 'qa_feedback' not in st.session_state:
        st.session_state.qa_feedback = {}
    if 'chart_feedback' not in st.session_state:
        st.session_state.chart_feedback = {}
    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []
    if 'data_preview' not in st.session_state:
        st.session_state.data_preview = None
    if 'qa_answer' not in st.session_state:
        st.session_state.qa_answer = None
    if 'chart' not in st.session_state:
        st.session_state.chart = None
    if 'prompt_generator' not in st.session_state:
        st.session_state.prompt_generator = None
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []

    st.title("🤖AI-Powered CSV/XLS App")

    # Sidebar with file upload and history
    with st.sidebar:
        st.header("Upload Data")
        uploaded_files = st.file_uploader("Upload CSV/XLS files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)
        
        if uploaded_files:
            file_data = {}
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension in ['xls', 'xlsx']:
                    excel_file = pd.ExcelFile(uploaded_file)
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                        file_data[f"{uploaded_file.name} - {sheet_name}"] = df
                else:
                    df = pd.read_csv(uploaded_file)
                    file_data[uploaded_file.name] = df
            
            st.header("Select File")
            selected_file = st.selectbox("Choose a file/sheet to explore", list(file_data.keys()))

        # Combined History section
        st.header("History & Feedback")
        
        # Create tabs for better organization within sidebar
        history_tabs = st.tabs(["Prompts", "Feedback"])
        
        with history_tabs[0]:  # Prompts tab
            if st.session_state.prompt_history:
                # Reverse the prompt history to show newest first
                for i, (prompt_type, prompt_text) in enumerate(reversed(st.session_state.prompt_history)):
                    with st.expander(f"{prompt_type}: {prompt_text[:50]}..."):
                        st.write(prompt_text)
                        if st.button("Reuse", key=f"reuse_prompt_{i}", on_click=handle_reuse_prompt, 
                                   args=(prompt_type, prompt_text)):
                            pass  # The callback will handle the action
            else:
                st.info("No prompt history yet")
        
        with history_tabs[1]:  # Feedback tab
            if st.session_state.feedback_history:
                # Reverse the feedback history to show newest first
                for i, feedback in enumerate(reversed(st.session_state.feedback_history)):
                    st.write(f"{len(st.session_state.feedback_history)-i}. {feedback}")
            else:
                st.info("No feedback history yet")

    if 'uploaded_files' in locals() and uploaded_files and selected_file:
        df = file_data[selected_file]
        st.write(f"Displaying data from: **{selected_file}**")
        
        data_tab, qa_tab, chart_tab, prompt_tab = st.tabs(["Data View", "Q&A", "Charts", "Prompt Generator"])
        
        with data_tab:
            st.header("Data Preview")
            with st.form("data_form"):
                num_rows = st.slider("Number of rows to display", min_value=1, max_value=len(df), value=5)
                submit_data = st.form_submit_button("Update Preview")
            if submit_data:
                st.session_state.data_preview = df.head(num_rows)
            
            # Display persistent data preview
            if st.session_state.data_preview is not None:
                st.dataframe(st.session_state.data_preview)
        
        table_name = "current_data"
        db = setup_database(df, table_name)
        
        with qa_tab:
            st.header("Ask Questions About Your Data")
            with st.form("qa_form"):
                # Pre-fill with reused prompt if exists
                initial_q = st.session_state.get('reuse_qa', '')
                question = st.text_input("Enter your question about the data:", value=initial_q)
                submit_question = st.form_submit_button("Get Answer")
            
            if submit_question and question:
                # Add to prompt history
                if question not in [x[1] for x in st.session_state.prompt_history]:
                    st.session_state.prompt_history.append(("Q&A", question))
                
                with st.spinner("Finding the answer..."):
                    try:
                        st.session_state.qa_answer = get_answer(db, question, table_name)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                
                # Clear reused prompt
                if 'reuse_qa' in st.session_state:
                    del st.session_state.reuse_qa
                        
            # Display persistent QA answer with feedback
            if st.session_state.qa_answer is not None:
                st.write("Answer:", st.session_state.qa_answer)
                
                # Create columns for feedback buttons
                col1, col2, col3 = st.columns([1, 1, 3])
                
                # Generate unique key for current Q&A
                qa_key = f"qa_{hash(question)}_{hash(str(st.session_state.qa_answer))}"
                
                if qa_key not in st.session_state.qa_feedback:
                    with col1:
                        if st.button("👍 Helpful", key=f"helpful_{qa_key}", 
                                   on_click=handle_feedback,
                                   args=("qa", qa_key, question, "helpful")):
                            pass

                    with col2:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{qa_key}", 
                                   on_click=handle_feedback,
                                   args=("qa", qa_key, question, "not_helpful")):
                            pass
                else:
                    st.info(f"You rated this answer as: {st.session_state.qa_feedback[qa_key]}")

        with chart_tab:
            st.header("Create Charts Based on Your Data")
            with st.form("chart_form"):
                # Pre-fill with reused prompt if exists
                initial_c = st.session_state.get('reuse_chart', '')
                chart_query = st.text_input("Enter task for chart:", value=initial_c)
                submit_chart = st.form_submit_button("Generate Chart")

            if submit_chart and chart_query:
                # Add to prompt history
                if chart_query not in [x[1] for x in st.session_state.prompt_history]:
                    st.session_state.prompt_history.append(("Chart", chart_query))
                
                with st.spinner("Generating chart..."):
                    try:
                        chart_code = generate_chart_code(chart_query, df)
                        st.session_state.chart = execute_generated_code(chart_code, df)
                    except Exception as e:
                        st.error(f"An error occurred while generating the chart: {str(e)}")
                
                # Clear reused prompt
                if 'reuse_chart' in st.session_state:
                    del st.session_state.reuse_chart
                        
            # Display persistent chart with feedback
            if st.session_state.chart is not None:
                st.plotly_chart(st.session_state.chart)
                
                # Create columns for feedback buttons
                col1, col2, col3 = st.columns([1, 1, 3])
                
                # Generate unique key for current chart
                chart_key = f"chart_{hash(chart_query)}"
                
                if chart_key not in st.session_state.chart_feedback:
                    with col1:
                        if st.button("👍 Good Chart", key=f"good_{chart_key}", 
                                   on_click=handle_feedback,
                                   args=("chart", chart_key, chart_query, "good")):
                            pass
                    
                    with col2:
                        if st.button("👎 Needs Improvement", key=f"bad_{chart_key}", 
                                   on_click=handle_feedback,
                                   args=("chart", chart_key, chart_query, "needs_improvement")):
                            pass
                else:
                    st.info(f"You rated this chart as: {st.session_state.chart_feedback[chart_key]}")

        with prompt_tab:
            st.header("Prompt Generator")
            with st.form("prompt_form"):
                user_query = st.text_input("Enter your query:")
                submit_prompt = st.form_submit_button("Generate Better Prompt")

            if submit_prompt and user_query:
                with st.spinner("Generating enhanced prompt..."):
                    try:
                        enhanced_prompt = generate_prompt(user_query, list(df.columns))
                        st.session_state.prompt_generator = enhanced_prompt
                    except Exception as e:
                        st.error(f"An error occurred while generating the prompt: {str(e)}")

            if st.session_state.prompt_generator:
                st.write("Enhanced Prompt:")
                st.info(st.session_state.prompt_generator)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use in Q&A", on_click=handle_reuse_prompt, 
                               args=("Q&A", st.session_state.prompt_generator)):
                        pass
                
                with col2:
                    if st.button("Use in Charts", on_click=handle_reuse_prompt, 
                               args=("Chart", st.session_state.prompt_generator)):
                        pass

if __name__ == "__main__":
    main()