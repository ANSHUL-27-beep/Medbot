import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import wikipedia
import pickle
import json
from datetime import datetime
import time
import requests
import httpx
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure Wikipedia
wikipedia.set_lang("en")

# Load environment variables
load_dotenv()

class MedicalBot:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.documents = []
        self.qa_chain = None
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=3000,  # Reduced from 5000 for faster processing
            ngram_range=(1, 1),  # Simplified to single words
            min_df=2,  # Increased to reduce vocabulary size
            max_df=0.95  # Increased to remove very common terms
        )
        self.query_history = []
        self.history_file = "query_history.json"
        self.wiki_cache = {}  # Cache for Wikipedia results
        self.load_history()
        self.setup()

    def load_history(self):
        """Load query history from file if it exists"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.query_history = json.load(f)
                print(f"Loaded {len(self.query_history)} previous queries from history")
            except Exception as e:
                print(f"Error loading history: {e}")
                self.query_history = []
        else:
            self.query_history = []
            print("No query history found, starting fresh")

    def save_history(self):
        """Save query history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.query_history, f, indent=2)
            print(f"Saved {len(self.query_history)} queries to history")
        except Exception as e:
            print(f"Error saving history: {e}")

    def add_to_history(self, question: str, answer: str, source: str):
        """Add a query and its response to the history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.query_history.append({
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "source": source
        })
        self.save_history()

    def get_recent_history(self, limit: int = 5) -> List[dict]:
        """Get the most recent queries from history"""
        return self.query_history[-limit:] if self.query_history else []

    def find_similar_previous_queries(self, question: str, limit: int = 3) -> List[dict]:
        """Find similar previous queries using TF-IDF and cosine similarity"""
        if not self.query_history:
            return []
        
        # Create TF-IDF vectors for all previous questions
        previous_questions = [item["question"] for item in self.query_history]
        
        # Use a simpler approach for small datasets
        if len(previous_questions) < 10:
            # Simple keyword matching for small datasets
            keywords = question.lower().split()
            similar_queries = []
            
            for i, prev_q in enumerate(previous_questions):
                prev_keywords = prev_q.lower().split()
                common_keywords = set(keywords) & set(prev_keywords)
                if len(common_keywords) >= 2:  # At least 2 common keywords
                    similar_queries.append(self.query_history[i])
            
            return similar_queries[:limit]
        
        # For larger datasets, use TF-IDF
        try:
            question_vectors = self.vectorizer.fit_transform(previous_questions)
            current_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(current_vector, question_vectors).flatten()
            top_indices = np.argsort(similarities)[-limit:][::-1]
            
            similar_queries = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Only include if similarity is above threshold
                    similar_queries.append(self.query_history[idx])
            
            return similar_queries
        except Exception as e:
            print(f"Error finding similar queries: {str(e)}")
            return []
        
    def setup(self):
        """Initialize the bot by loading and processing the PDF"""
        # Check if cached data exists
        cache_file = "medical_bot_cache.pkl"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.clean_chunks = cached_data['clean_chunks']
                    self.tfidf_matrix = cached_data['tfidf_matrix']
                    self.vectorizer = cached_data['vectorizer']
                    print("Loaded cached data successfully")
                    self._setup_llm()
                    return
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load()
        
        # Split documents into chunks with smaller size and less overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Reduced from 2000
            chunk_overlap=200,  # Reduced from 400
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.splits = text_splitter.split_documents(self.documents)
        
        # Clean and prepare chunks for vectorization
        self.clean_chunks = []
        for chunk in self.splits:
            chunk_text = chunk.page_content.lower()
            if any(skip in chunk_text for skip in ['copyright', 'gale encyclopedia', 'staff', 'editor']):
                continue
            cleaned_text = re.sub(r'\s+', ' ', chunk_text).strip()
            if cleaned_text and len(cleaned_text) > 50:
                self.clean_chunks.append(cleaned_text)
        
        # Create TF-IDF vectors for all chunks
        self.tfidf_matrix = self.vectorizer.fit_transform(self.clean_chunks)
        
        # Cache the processed data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'clean_chunks': self.clean_chunks,
                    'tfidf_matrix': self.tfidf_matrix,
                    'vectorizer': self.vectorizer
                }, f)
            print("Cached data saved successfully")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
        self._setup_llm()

    def _setup_llm(self):
        """Set up the language model and prompts"""
        # Get API key from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Create custom HTTP clients without proxies
        http_client = httpx.Client(
            timeout=30.0,  # Reduced timeout
            follow_redirects=True
        )
        http_async_client = httpx.AsyncClient(
            timeout=30.0,  # Reduced timeout
            follow_redirects=True
        )
        
        # Initialize Groq LLM with optimized parameters
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.5,  # Increased for faster responses
            max_tokens=1024,  # Reduced for faster responses
            max_retries=2,  # Reduced retries
            request_timeout=30,  # Reduced timeout
            http_client=http_client,
            http_async_client=http_async_client
        )
        
        # Create prompt template for PDF content
        pdf_prompt_template = """You are a friendly and knowledgeable medical assistant having a conversation with a patient. 
        Provide a clear, concise, and detailed answer to the question using the provided context.
        
        Guidelines:
        1. Start with a direct answer to the question in 1-2 sentences.
        2. Then provide 2-3 key details or explanations that support your answer.
        3. Use simple language and avoid medical jargon unless necessary.
        4. If you use medical terms, explain them briefly.
        5. Keep your response focused and to the point.
        6. If you don't have information about the topic, say "I don't have information about that."

        Context: {context}

        Question: {question}

        Answer:"""
        
        # Create prompt template for Wikipedia content
        wiki_prompt_template = """You are a friendly and knowledgeable medical assistant having a conversation with a patient. 
        Provide a clear, concise, and detailed answer to the question using the provided Wikipedia content.
        
        Guidelines:
        1. Start with a direct answer to the question in 1-2 sentences.
        2. Then provide 2-3 key details or explanations that support your answer.
        3. Use simple language and avoid medical jargon unless necessary.
        4. If you use medical terms, explain them briefly.
        5. Keep your response focused and to the point.
        6. If you don't have information about the topic, say "I don't have information about that."

        Wikipedia Content: {context}

        Question: {question}

        Answer:"""
        
        # Create prompt templates
        self.pdf_prompt = PromptTemplate(
            template=pdf_prompt_template,
            input_variables=["context", "question"]
        )
        
        self.wiki_prompt = PromptTemplate(
            template=wiki_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create LLM chain
        self.qa_chain = LLMChain(
            llm=llm,
            prompt=self.pdf_prompt
        )

    def get_wikipedia_content(self, query: str) -> str:
        """Search Wikipedia for relevant medical information"""
        # Check cache first
        if query in self.wiki_cache:
            print(f"Using cached Wikipedia content for: {query}")
            return self.wiki_cache[query]
        
        try:
            query = query.lower().strip()
            
            # Extract key terms from the query
            common_words = ["what", "is", "are", "the", "and", "for", "with", "about", "how", "why", "when", "where", "who", "which", "can", "you", "tell", "me", "more", "about", "explain", "describe", "list", "name", "give", "show", "provide", "information", "on", "regarding", "concerning", "related", "to", "same", "it", "this", "that", "these", "those", "they", "them", "their", "its"]
            key_terms = [word for word in query.split() if word not in common_words and len(word) > 2]
            
            # If we have key terms, use them for searching
            if key_terms:
                search_term = " ".join(key_terms)
            else:
                search_term = query
            
            # Try direct search first (faster)
            try:
                page = wikipedia.page(search_term, auto_suggest=False)
                content = page.summary + "\n\n" + page.content[:500]
                self.wiki_cache[query] = content
                print(f"Found direct Wikipedia match for: {search_term}")
                return content
            except:
                # If direct search fails, try with search
                search_queries = [
                    search_term,
                    f"{search_term} (medical)",
                    f"{search_term} disease",
                    f"{search_term} condition",
                    f"{search_term} health"
                ]
                
                all_results = []
                for search_query in search_queries:
                    try:
                        results = wikipedia.search(search_query, results=2)  # Reduced from 3
                        all_results.extend(results)
                    except:
                        continue
                
                all_results = list(dict.fromkeys(all_results))
                
                if not all_results:
                    self.wiki_cache[query] = ""
                    return ""
                
                medical_categories = [
                    'health', 'medical', 'medicine', 'disease', 'condition',
                    'symptom', 'treatment', 'diagnosis', 'pathology', 'physiology',
                    'anatomy', 'surgery', 'therapy', 'pharmacology', 'epidemiology',
                    'infectious disease', 'chronic condition', 'mental health',
                    'public health', 'clinical medicine'
                ]
                
                best_content = ""
                best_score = 0
                
                for result in all_results:
                    try:
                        page = wikipedia.page(result, auto_suggest=False)
                        category_score = sum(
                            1 for category in page.categories
                            if any(term in category.lower() for term in medical_categories)
                        )
                        title_score = 1 if search_term in page.title.lower() else 0
                        summary_score = 1 if search_term in page.summary.lower() else 0
                        total_score = category_score + title_score + summary_score
                        
                        if total_score > best_score:
                            content = page.summary + "\n\n"
                            sections = []
                            for section in page.sections:
                                if search_term in section.lower():
                                    try:
                                        section_content = page.section(section)
                                        if section_content:
                                            sections.append(f"{section}:\n{section_content}")
                                    except:
                                        continue
                            
                            content += "\n\n".join(sections[:1])  # Reduced from 2
                            main_content = page.content[:500]  # Reduced from 1000
                            if main_content:
                                content += f"\n\nAdditional Information:\n{main_content}"
                            
                            best_content = content
                            best_score = total_score
                            
                    except wikipedia.exceptions.DisambiguationError as e:
                        try:
                            options = e.options
                            for option in options:
                                if any(term in option.lower() for term in medical_categories):
                                    try:
                                        page = wikipedia.page(option, auto_suggest=False)
                                        if any(term in category.lower() for term in medical_categories 
                                              for category in page.categories):
                                            content = page.summary + "\n\n" + page.content[:500]  # Reduced from 1000
                                            self.wiki_cache[query] = content
                                            return content
                                    except:
                                        continue
                        except:
                            continue
                    except:
                        continue
                
                self.wiki_cache[query] = best_content
                return best_content
                
        except:
            self.wiki_cache[query] = ""
            return ""

    def find_relevant_context(self, question: str) -> str:
        """Find relevant context using TF-IDF and cosine similarity"""
        try:
            # Clean the question
            cleaned_question = re.sub(r'\s+', ' ', question.lower()).strip()
            
            # Extract key terms from the question (remove common words)
            common_words = ["what", "is", "are", "the", "and", "for", "with", "about", "how", "why", "when", "where", "who", "which", "can", "you", "tell", "me", "more", "about", "explain", "describe", "list", "name", "give", "show", "provide", "information", "on", "regarding", "concerning", "related", "to", "same", "it", "this", "that", "these", "those", "they", "them", "their", "its"]
            question_terms = [word for word in cleaned_question.split() if word not in common_words and len(word) > 2]
            
            # If we have key terms, use them to find relevant chunks
            if question_terms:
                # Create a query that emphasizes the key terms
                enhanced_query = " ".join(question_terms)
                question_vector = self.vectorizer.transform([enhanced_query])
            else:
                # Fall back to the full question
                question_vector = self.vectorizer.transform([cleaned_question])
            
            # Calculate similarities
            similarities = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
            
            # Get top 3 most relevant chunks (increased from 2 for better accuracy)
            top_indices = np.argsort(similarities)[-3:][::-1]
            relevant_chunks = [self.clean_chunks[i] for i in top_indices if similarities[i] > 0.05]
            
            if not relevant_chunks:
                return ""
            
            return "\n\n".join(relevant_chunks)
        except Exception as e:
            print(f"Error in find_relevant_context: {str(e)}")
            return ""

    def _get_pdf_answer(self, question: str) -> str:
        """Get answer from PDF content"""
        try:
            # Find relevant context from PDF
            pdf_context = self.find_relevant_context(question)
            if pdf_context:
                # Create prompt for PDF content
                prompt = PromptTemplate(
                    template="""You are a friendly and knowledgeable medical assistant. 
                    Answer the question directly and concisely using the provided context.
                    
                    Guidelines:
                    1. Give a direct answer in 1-2 sentences.
                    2. Add 2-3 key supporting details.
                    3. Use simple language.
                    4. Explain any medical terms briefly.
                    5. Stay focused and to the point.
                    6. If the context doesn't contain enough information to answer the question, say so.

                    Context: {context}

                    Question: {question}

                    Answer:""",
                    input_variables=["context", "question"]
                )
                
                # Create chain using RunnableSequence with optimized parameters
                chain = (
                    {"context": lambda x: pdf_context, "question": lambda x: x}
                    | prompt
                    | ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.3,  # Reduced for more focused answers
                        max_tokens=512,  # Reduced for faster responses
                        max_retries=1,   # Reduced retries
                        request_timeout=15,  # Reduced timeout
                        http_client=httpx.Client(timeout=15.0, follow_redirects=True),
                        http_async_client=httpx.AsyncClient(timeout=15.0, follow_redirects=True)
                    )
                    | StrOutputParser()
                )
                
                # Get response
                response = chain.invoke(question)
                return response.strip()
            return None
        except Exception as e:
            print(f"Error in _get_pdf_answer: {str(e)}")
            return None

    def _get_wiki_answer(self, question: str) -> str:
        """Get answer from Wikipedia content"""
        try:
            # Get Wikipedia content
            wiki_content = self.get_wikipedia_content(question)
            if wiki_content:
                # Create prompt for Wikipedia content
                prompt = PromptTemplate(
                    template="""You are a friendly and knowledgeable medical assistant. 
                    Answer the question directly and concisely using the provided Wikipedia content.
                    
                    Guidelines:
                    1. Give a direct answer in 1-2 sentences.
                    2. Add 2-3 key supporting details.
                    3. Use simple language.
                    4. Explain any medical terms briefly.
                    5. Stay focused and to the point.
                    6. If the Wikipedia content doesn't contain enough information to answer the question, say so.

                    Wikipedia Content: {context}

                    Question: {question}

                    Answer:""",
                    input_variables=["context", "question"]
                )
                
                # Create chain using RunnableSequence with optimized parameters
                chain = (
                    {"context": lambda x: wiki_content, "question": lambda x: x}
                    | prompt
                    | ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.3,  # Reduced for more focused answers
                        max_tokens=512,  # Reduced for faster responses
                        max_retries=1,   # Reduced retries
                        request_timeout=15,  # Reduced timeout
                        http_client=httpx.Client(timeout=15.0, follow_redirects=True),
                        http_async_client=httpx.AsyncClient(timeout=15.0, follow_redirects=True)
                    )
                    | StrOutputParser()
                )
                
                # Get response
                response = chain.invoke(question)
                return response.strip()
            return None
        except Exception as e:
            print(f"Error in _get_wiki_answer: {str(e)}")
            return None

    def query(self, question: str, context: str = None) -> str:
        """Query the bot with a question and optional context from previous questions"""
        try:
            # Check if this is a follow-up question
            is_follow_up = False
            if context and any(word in question.lower() for word in ["same", "it", "this", "that", "these", "those", "they", "them", "their", "its"]):
                is_follow_up = True
                
                # Extract the main topic from the previous question
                # First, try to find medical terms in the context
                words = context.split()
                medical_terms = []
                
                # Common medical prefixes and suffixes to help identify medical terms
                medical_indicators = ["itis", "oma", "emia", "pathy", "algia", "derma", "itis", "oma", "osis", "pathy", "plasty", "rrhagia", "rrhaphy", "rrhea", "rrhexis", "sclerosis", "stasis", "tomy", "uria"]
                
                for i, word in enumerate(words):
                    # Skip common words
                    if word.lower() in ["what", "is", "are", "the", "and", "for", "with", "about", "how", "why", "when", "where", "who", "which", "can", "you", "tell", "me", "more", "about", "explain", "describe", "list", "name", "give", "show", "provide", "information", "on", "regarding", "concerning", "related", "to"]:
                        continue
                    
                    # Check if word contains medical indicators
                    if any(indicator in word.lower() for indicator in medical_indicators):
                        medical_terms.append((i, word))
                        continue
                    
                    # Check if word is longer than 3 characters (likely a medical term)
                    if len(word) > 3:
                        medical_terms.append((i, word))
                
                # If we found medical terms, use the first one
                if medical_terms:
                    main_topic = medical_terms[0][1]
                else:
                    # Fallback: use the first non-common word
                    for word in words:
                        if word.lower() not in ["what", "is", "are", "the", "and", "for", "with", "about", "how", "why", "when", "where", "who", "which"]:
                            main_topic = word
                            break
                    else:
                        # If all else fails, use the first word
                        main_topic = words[0]
                
                # Combine the context with the current question
                full_question = f"{main_topic} {question}"
            else:
                full_question = question

            # Try to get answer from PDF first
            pdf_answer = self._get_pdf_answer(full_question)
            if pdf_answer:
                self.add_to_history(question, pdf_answer, "PDF")
                return pdf_answer

            # If no PDF answer, try Wikipedia
            wiki_answer = self._get_wiki_answer(full_question)
            if wiki_answer:
                self.add_to_history(question, wiki_answer, "Wikipedia")
                return wiki_answer

            error_msg = "I apologize, but I couldn't find specific information about that in my medical knowledge base. Please try rephrasing your question or ask about a different medical topic."
            self.add_to_history(question, error_msg, "None")
            return error_msg

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again."
            print(f"Error in query: {str(e)}")
            self.add_to_history(question, error_msg, "Error")
            return error_msg

    def delete_query(self, index: int) -> bool:
        """Delete a specific query from history by index"""
        try:
            if 0 <= index < len(self.query_history):
                del self.query_history[index]
                self.save_history()
                return True
            return False
        except Exception as e:
            print(f"Error deleting query: {str(e)}")
            return False
    
    def clear_history(self) -> bool:
        """Clear all queries from history"""
        try:
            self.query_history = []
            self.save_history()
            return True
        except Exception as e:
            print(f"Error clearing history: {str(e)}")
            return False