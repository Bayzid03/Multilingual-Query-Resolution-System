import os
import json
import logging
import pandas as pd
from io import StringIO
from typing import Dict
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
