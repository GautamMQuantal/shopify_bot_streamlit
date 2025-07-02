import streamlit as st
import requests
import json
import pandas as pd
from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import re
from fuzzywuzzy import fuzz, process
import numpy as np

# Load environment variables
load_dotenv()

# ================================
# CONFIGURATION
# ================================

class Config:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        self.model = "gpt-4o-mini"
        self.max_tokens = 1500
        self.temperature = 0.7

        # Shopify Storefront settings
        self.shopify_store_url = os.getenv('SHOPIFY_STORE_URL')
        self.shopify_storefront_token = os.getenv('SHOPIFY_STOREFRONT_ACCESS_TOKEN')

# ================================
# CSV PRODUCT DATABASE
# ================================

class ProductDatabase:
    def __init__(self, csv_file_path: str = None):
        self.df = None
        self.csv_file_path = csv_file_path
        self.load_data()
        
    def load_data(self):
        """Load product data from CSV file"""
        try:
            if self.csv_file_path and os.path.exists(self.csv_file_path):
                self.df = pd.read_csv(self.csv_file_path)
                # Clean and standardize data
                self.df = self.df.fillna('')
                # Ensure required columns exist
                required_cols = ['SKU', 'Shopify Title', 'Price', 'Cost']
                missing_cols = [col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    st.warning(f"Missing columns in CSV: {missing_cols}")
                
                st.success(f"✅ Loaded {len(self.df)} products from CSV database")
            else:
                st.warning("📄 CSV file not found. Using API-only mode.")
                self.df = pd.DataFrame()  # Empty dataframe
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            self.df = pd.DataFrame()
    
    def search_products(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Search products using fuzzy matching"""
        if self.df.empty:
            return pd.DataFrame()
            
        query = query.lower().strip()
        if not query:
            return self.df.head(limit)
        
        # Create searchable text by combining relevant columns
        search_columns = ['Shopify Title', 'SKU', 'Categories', 'Tags', 'Vendor']
        available_columns = [col for col in search_columns if col in self.df.columns]
        
        if not available_columns:
            return pd.DataFrame()
            
        # Combine searchable text
        self.df['search_text'] = self.df[available_columns].fillna('').apply(
            lambda x: ' '.join(x.astype(str)).lower(), axis=1
        )
        
        # Fuzzy matching
        matches = []
        for idx, row in self.df.iterrows():
            score = fuzz.partial_ratio(query, row['search_text'])
            if score > 30:  # Threshold for relevance
                matches.append((idx, score))
        
        # Sort by relevance score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches
        result_indices = [match[0] for match in matches[:limit]]
        result_df = self.df.loc[result_indices].copy()
        result_df['relevance_score'] = [match[1] for match in matches[:limit]]
        
        return result_df.drop('search_text', axis=1, errors='ignore')
    
    def get_product_by_sku(self, sku: str) -> pd.Series:
        """Get specific product by SKU"""
        if self.df.empty:
            return pd.Series()
            
        matches = self.df[self.df['SKU'].str.upper() == sku.upper()]
        if not matches.empty:
            return matches.iloc[0]
        return pd.Series()
    
    def get_all_products(self, limit: int = 100) -> pd.DataFrame:
        """Get all products with limit"""
        if self.df.empty:
            return pd.DataFrame()
        return self.df.head(limit)
    
    def get_product_stats(self) -> dict:
        """Get database statistics"""
        if self.df.empty:
            return {"total_products": 0}
            
        stats = {
            "total_products": len(self.df),
            "unique_vendors": self.df['Vendor'].nunique() if 'Vendor' in self.df.columns else 0,
            "unique_categories": self.df['Categories'].nunique() if 'Categories' in self.df.columns else 0,
            "price_range": {
                "min": float(pd.to_numeric(self.df['Price'], errors='coerce').min()) if 'Price' in self.df.columns else 0,
                "max": float(pd.to_numeric(self.df['Price'], errors='coerce').max()) if 'Price' in self.df.columns else 0
            } if 'Price' in self.df.columns else {"min": 0, "max": 0}
        }
        return stats

# ================================
# IMPROVED QUERY ANALYZER
# ================================

class QueryAnalyzer:
    def __init__(self):
        self.intent_patterns = {
            'list_all': [
                r'(?:show|list|display)\s+(?:all|every)\s+(?:products?|items?)',
                r'(?:what|which)\s+(?:products?|items?)\s+(?:are\s+)?(?:available|do\s+you\s+have)',
                r'(?:all|every)\s+(?:available|current)\s+(?:products?|items?)',
                r'(?:complete\s+)?(?:product\s+)?(?:catalog|inventory|list)'
            ],
            'search_specific': [
                r'(?:find|search|look\s+for|show\s+me)\s+(.+)',
                r'(?:do\s+you\s+have|got\s+any)\s+(.+)',
                r'(?:about|details?\s+(?:about|on|for))\s+(.+)',
                r'(?:tell\s+me\s+about|info\s+(?:about|on))\s+(.+)'
            ],
            'price_inquiry': [
                r'(?:cost|price|how\s+much)\s+(?:is|are|for|of)\s*(.+)?',
                r'(?:what\s+(?:is\s+the\s+)?(?:cost|price))\s+(?:of|for)\s+(.+)',
                r'(.+)\s+(?:cost|price|pricing)'
            ],
            'comparison': [
                r'(?:compare|difference\s+between)\s+(.+)',
                r'(.+)\s+(?:vs|versus|compared\s+to)\s+(.+)',
                r'(?:which\s+is\s+better)\s+(.+)'
            ],
            'specifications': [
                r'(?:specs?|specifications?|dimensions?|size)\s+(?:of|for)\s+(.+)',
                r'(.+)\s+(?:specs?|specifications?|dimensions?|measurements?)',
                r'(?:technical\s+details?)\s+(?:of|for)\s+(.+)'
            ]
        }
    
    def analyze_query(self, query: str) -> dict:
        """Analyze user query with improved intent detection"""
        query_lower = query.lower().strip()
        
        analysis = {
            'intent': 'general_search',
            'search_terms': '',
            'confidence': 0.5,
            'entities': []
        }
        
        # Check for specific intents
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    analysis['intent'] = intent
                    analysis['confidence'] = 0.8
                    
                    # Extract search terms from capture groups
                    if match.groups():
                        terms = ' '.join([group for group in match.groups() if group]).strip()
                        analysis['search_terms'] = self._clean_search_terms(terms)
                    else:
                        analysis['search_terms'] = self._clean_search_terms(query_lower)
                    break
            if analysis['confidence'] > 0.5:
                break
        
        # If no specific intent found, extract general search terms
        if analysis['confidence'] <= 0.5:
            analysis['search_terms'] = self._clean_search_terms(query_lower)
        
        return analysis
    
    def _clean_search_terms(self, terms: str) -> str:
        """Clean and optimize search terms"""
        # Remove common stop words that don't help with product search
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'can', 'could',
            'should', 'would', 'will', 'shall', 'may', 'might', 'must', 'ought',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'get', 'got', 'give', 'take', 'go', 'come',
            'tell', 'show', 'find', 'look', 'see', 'know', 'think', 'say', 'ask',
            'me', 'you', 'i', 'we', 'they', 'them', 'us', 'him', 'her', 'it'
        }
        
        words = terms.split()
        cleaned_words = [word for word in words if word.lower() not in stop_words and len(word) > 1]
        
        return ' '.join(cleaned_words) if cleaned_words else terms

# ================================
# ENHANCED DYNAMIC CHATBOT
# ================================

class EnhancedShopifyBot:
    def __init__(self, config: Config, product_db: ProductDatabase, shopify_api=None):
        self.config = config
        self.product_db = product_db
        self.shopify_api = shopify_api
        self.analyzer = QueryAnalyzer()
        self.conversation_context = []
        
    def _create_system_prompt(self, product_data: str) -> str:
        stats = self.product_db.get_product_stats()
        return f"""You are an intelligent product assistant for a protective cases and equipment store. You have access to a comprehensive product database with {stats['total_products']} products.

**YOUR CAPABILITIES:**
- Search and find products using keywords, SKUs, or descriptions
- Provide detailed product information including pricing, specifications, and availability
- Calculate markup and margin for business inquiries
- Compare products and make recommendations
- Handle follow-up questions contextually

**CONVERSATION RULES:**
- If more than 5 products are found, ask the user to specify features like color, size, or other specifications
- Always provide SKU references for specific product details
"""
    
    def _format_products_for_ai(self, df: pd.DataFrame, query_intent: str) -> str:
        if df.empty:
            return "No products found matching the search criteria."
        
        formatted_data = f"FOUND {len(df)} RELEVANT PRODUCTS:\n\n"
        
        if len(df) > 5:
            # Ask user for clarification if more than 5 products are found
            formatted_data += "I found several matching products. Could you specify the color, size, or other features you're looking for?\n"
        
        for idx, row in df.head(5).iterrows():  # Show up to 5 products
            formatted_data += f"=== {row.get('Shopify Title', 'Unknown Product')} ===\n"
            formatted_data += f"SKU: {row.get('SKU', 'N/A')}\n"
            formatted_data += f"Price: ${row.get('Price', 'N/A')}\n"
            formatted_data += f"Vendor: {row.get('Vendor', 'N/A')}\n"
            formatted_data += f"Category: {row.get('Categories', 'N/A')}\n"
            formatted_data += "\n"
        
        return formatted_data

    def process_query(self, user_input: str) -> str:
        try:
            if not self.config.openai_client:
                return "❌ OpenAI API key is required. Please check your environment variables."
            
            # Analyze the query
            analysis = self.analyzer.analyze_query(user_input)
            
            # Search products based on the intent
            if analysis['intent'] == 'list_all':
                products_df = self.product_db.get_all_products(limit=200)
            else:
                products_df = self.product_db.search_products(analysis['search_terms'], limit=50)
            
            # Format product data for AI
            product_context = self._format_products_for_ai(products_df, analysis['intent'])
            
            # Build conversation messages
            messages = [{"role": "system", "content": self._create_system_prompt(product_context)}]
            messages.extend(self.conversation_context[-6:])  # Last 3 exchanges
            
            # Add current user message
            messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            response = self.config.openai_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            ai_response = response.choices[0].message.content
            
            # Update conversation context
            self.conversation_context.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": ai_response}
            ])
            
            # Keep context manageable
            if len(self.conversation_context) > 12:  # 6 exchanges
                self.conversation_context = self.conversation_context[-12:]
            
            return ai_response
        
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            return error_msg


# ================================
# STREAMLIT APP
# ================================

def main():
    st.set_page_config(
        page_title="Enhanced Shopify Assistant",
        page_icon="🛍️",
        layout="wide"
    )
    
    st.title("🛍️ Enhanced Shopify Assistant")
    st.markdown("*Intelligent product advisor with comprehensive CSV database*")
    
    # Sidebar for file upload and configuration
    st.sidebar.header("📁 Configuration")
    
    # CSV file upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Product CSV", 
        type=['csv'],
        help="Upload your cleaned product CSV file for comprehensive search"
    )
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    st.session_state['debug_mode'] = debug_mode
    
    # Initialize configuration
    config = Config()
    
    # Check OpenAI API key
    if not config.openai_api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return
    
    # Initialize product database
    csv_path = None
    if uploaded_file is not None:
        # Save uploaded file temporarily
        csv_path = f"temp_{uploaded_file.name}"
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Initialize product database
    if 'product_db' not in st.session_state or csv_path:
        st.session_state.product_db = ProductDatabase(csv_path)
    
    product_db = st.session_state.product_db
    
    # Display database stats
    if not product_db.df.empty:
        stats = product_db.get_product_stats()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", stats['total_products'])
        with col2:
            st.metric("Vendors", stats.get('unique_vendors', 0))
        with col3:
            st.metric("Categories", stats.get('unique_categories', 0))
        with col4:
            price_range = stats.get('price_range', {})
            if price_range.get('max', 0) > 0:
                st.metric("Price Range", f"${price_range.get('min', 0):.0f} - ${price_range.get('max', 0):.0f}")
            else:
                st.metric("Price Range", "N/A")
    else:
        st.warning("📄 No product database loaded. Please upload a CSV file.")
        st.info("""
        **CSV Format Requirements:**
        - Required columns: `SKU`, `Shopify Title`, `Price`, `Cost`
        - Optional columns: `Categories`, `Vendor`, `Status`, `Dimensions`, etc.
        - Ensure headers match exactly as shown above
        """)
        return
    
    # Initialize bot
    if 'enhanced_bot' not in st.session_state:
        st.session_state.enhanced_bot = EnhancedShopifyBot(config, product_db)
    
    bot = st.session_state.enhanced_bot
    
    # Chat interface
    st.subheader("💬 Enhanced Product Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = f"""👋 Hello! I'm your enhanced product assistant with access to {stats['total_products']} products in our database. 

I can help you:
- Find specific products by name, SKU, or keywords
- Show all available products with summaries
- Calculate pricing, margins, and markups
- Compare products and provide recommendations
- Answer detailed questions about specifications

Try asking: "Show me all available products" or search for something specific!"""
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about products..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching database..."):
                response = bot.process_query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar controls
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        if 'enhanced_bot' in st.session_state:
            st.session_state.enhanced_bot.conversation_context = []
        st.rerun()
    
    # Example queries
    st.sidebar.subheader("💡 Example Queries")
    example_queries = [
        "Show me all available products",
        "Find Pelican cases",
        "What's the price of foam inserts?",
        "Show me products from B&W",
        "Calculate margin for SKU PC-1150",
        "Compare different tool cases",
        "What are the dimensions of waterproof cases?"
    ]
    
    for query in example_queries:
        if st.sidebar.button(query, key=f"example_{hash(query)}"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching database..."):
                    response = bot.process_query(query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if __name__ == "__main__":
    main()