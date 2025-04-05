import streamlit as st
from hybrid_rag import app, write_query, execute_query, summarize_result
import pandas as pd

st.set_page_config(
    page_title="RAG Pipeline Visualization",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Pipeline Visualization")

# Input question
question = st.text_input("Enter your question:", "How many orders were there in January 2024?")

if st.button("Run Pipeline"):
    # Run the pipeline
    initial_state = {"question": question, "iteration": 0}
    result = app.invoke(initial_state)
    
    # Display initial question and answer
    st.header("Initial Question & Answer")
    st.markdown(f"**Question:** {result['question']}")
    st.markdown(f"**Answer:** {result['generation']}")
    
    # Create tabs for subquestions if they exist
    if "subquestions" in result and result["subquestions"]:
        st.header("Subquestions Analysis")
        tabs = st.tabs([f"Subquestion {i+1}" for i in range(len(result["subquestions"]))])
        
        for i, tab in enumerate(tabs):
            with tab:
                st.markdown(f"**Subquestion {i+1}:** {result['subquestions'][i]}")
                
                # Get the corresponding SQL query and result
                sql_query = write_query(result["subquestions"][i])
                sql_result = execute_query(sql_query)
                summary = summarize_result(sql_query, sql_result)
                
                st.markdown("**SQL Query:**")
                st.code(sql_query, language="sql")
                
                st.markdown("**Results:**")
                st.markdown(summary)
                
                # Display the raw results if they're in a table format
                try:
                    df = pd.read_html(sql_result)[0]
                    st.dataframe(df)
                except:
                    st.text(sql_result)

# Add some styling
st.markdown("""
<style>
    .stTextInput>div>div>input {
        font-size: 1.2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True) 