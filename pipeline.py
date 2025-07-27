import os
import json
import logging
import pandas as pd
from io import StringIO
from typing import Dict
from dotenv import load_dotenv; load_dotenv()
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.embeddings import FastEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configure logging for console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load Dataset

csv_data = """product_id,product_name,category,tags
1,Fresh Premium Atta 5kg,Staples,"atta, ময়দা, গম, wheat"
2,Arong Full Cream Milk 1L,Dairy,"milk, দুধ, ফুল ক্রীম দুধ"
3,Molla Salt 1kg,Staples,"salt, লবণ, namok"
4,Kellogg's Corn Flakes 475g,Breakfast,"কর্নফ্লেক্স, সিরিয়াল, makkai"
5,Olympic Energy Plus Biscuit 1kg,Snacks,"বিস্কুট, কুকিজ, biscuit"
6,Meridian Milk Chocolate,Chocolates,"চকলেট, চকো, meridian"
7,Pran Banana Chips,Snacks,"কলা চিপস, banana wafers, chips"
8,Radhuni Morich Powder,Spices,"মরিচ গুঁড়া, মসলা, spice, red chili"
9,Fresh Dhonepata Bunch,Vegetables,"ধনে পাতা, coriander, cilantro"
10,Fresh Pudina Patta Bunch,Vegetables,"পুদিনা পাতা, mint, pudina"
11,Ispahani Mirzapore Tea 500g,Beverages,"চা, tea, ispahani"
12,Nescafe Classic Coffee 100g,Beverages,"কফি, coffee, nescafe"
13,Deshi Piaj 1kg,Vegetables,"পেয়াজ, onion"
14,Deshi Tomato 1kg,Vegetables,"টমেটো, tomato"
15,RC Cola 750ml,Beverages,"কোলা, rc cola, soft drink"
16,Maggi 2-Minute Noodles Masala,Snacks,"ম্যাগি, নুডলস, instant noodles"
17,Arong Cheese Slices 100g,Dairy,"চিজ, cheese slice"
18,Pran Cheese Spread 180g,Dairy,"চিজ স্প্রেড, creamy cheese"
19,Lemon 4pcs,Vegetables,"লেবু, lemon"
20,Rupchanda Soyabean Oil 1L,Staples,"তেল, cooking oil, রূপচাঁদা"
21,Miniket Rice 1kg,Staples,"চাল, rice, মিনিকেট"
22,Mr. Twist Masala,Snacks,"মিস্টার টুইস্ট, snacks, chips"
"""
df = pd.read_csv(StringIO(csv_data))

# Embedding and Vector Store Setup

documents = [
    Document(
        page_content=f"{row['product_name']}. Category: {row['category']}. Tags: {row['tags']}",
        metadata={
            "product_id": row['product_id'],
            "product_name": row['product_name'],
            "category": row['category']
        }
    ) for _, row in df.iterrows()
]

embedding_model = FastEmbeddings(model_name="BAAI/bge-m3")
vector_store = Chroma.from_documents(documents, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# LLM and Prompt Setup

# Get API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.critical("GROQ_API_KEY environment variable not set")
    raise ValueError("GROQ_API_KEY environment variable not set")
logger.info("Successfully loaded GROQ API key")

llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

prompt_template = """
You are a world-class search query interpretation engine for a Bangladeshi grocery delivery service like Chaldal or Khaas Food.
Your primary goal is to understand the user's intent, even if their query is misspelled, in Bangla, English, or mixed language.
Analyze the user's `RAW QUERY` and the `CONTEXT` of semantically similar products retrieved from our catalog.
Return a single JSON:
- "corrected_query"
- "identified_product"
- "confidence" ("High", "Medium", "Low")
- "reasoning"
If no good match, confidence="Low" & identified_product=null.

---
CONTEXT:
{context}

RAW QUERY:
{query}

---
JSON OUTPUT:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    """Format document objects into LLM-friendly context string.

    Serves two key purposes:
    1. Data Formatting:
       - Converts list of document objects into formatted strings
       - Each product appears on a new line
       - Format: "- [Product Name] | [Full Product Details]"
    
    2. Context Preparation:
       - Creates structured context for LLM processing
       - Enhances readability for RAG (Retrieval Augmented Generation)
       - Enables better query interpretation

    Args:
        docs (List[Document]): List of document objects containing product information

    Returns:
        str: Formatted string with each product on a new line
    """
    return "\n".join([
        f"- {d.metadata['product_name']} | {d.page_content}"
        for d in docs
    ])

rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Main Function
def search_pipeline(query: str) -> dict:
    """Run full pipeline and return results.
    
    Args:
        query (str): User query in English, Bangla, or mixed language
        
    Returns:
        dict: Contains initial_context, llm_output, and final_results
    """
    initial_context = retriever.get_relevant_documents(query)

    llm_output_str = rag_chain.invoke(query)
    try:
        llm_output = json.loads(llm_output_str)
        corrected_query = llm_output.get("corrected_query", query)
        logger.info(f"Successfully processed query: {query} -> {corrected_query}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing LLM output: {e}\nRaw output: {llm_output_str}")
        llm_output = None
        corrected_query = query
    except Exception as e:
        logger.exception(f"Unexpected error while processing query: {query}")
        llm_output = None
        corrected_query = query
    
    final_results = vector_store.similarity_search(corrected_query, k=3)

    return {
        "initial_context": initial_context,
        "llm_output": llm_output,
        "final_results": final_results
    }

