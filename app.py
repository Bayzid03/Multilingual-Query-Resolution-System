import streamlit as st
from pipeline import search_pipeline

st.set_page_config(page_title="🛒 Multilingual Grocery Search 🇧🇩", page_icon="🍀", layout="centered")

st.title("🛒 Multilingual Grocery Search 🇧🇩")
st.write("Enter your query in Bangla, English or mixed:")

query = st.text_input("🔍 Your Query", "")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            result = search_pipeline(query)

        st.subheader("🔎 Initial Retrieved Products")
        for doc in result["initial_context"]:
            st.markdown(f" **{doc.metadata['product_name']}** — *{doc.metadata['category']}*")

        if result["llm_output"]:
            st.subheader("🤖 LLM Correction & Reasoning")
            st.json(result["llm_output"])

        st.subheader("Top 3 Final Products")
        for i, doc in enumerate(result["final_results"], 1):
            st.markdown(f"{i}. **{doc.metadata['product_name']}** — *{doc.metadata['category']}*")
