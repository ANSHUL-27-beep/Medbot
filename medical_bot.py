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
    def __init__(self, pdf_paths):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        self.pdf_paths = pdf_paths
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
        tfidf_matrix = self.vectorizer.fit_transform(previous_questions)
        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
        
        # Get indices of top matching queries
        similar_indices = similarities.argsort()[-limit:][::-1]
        similar_queries = []
        for idx in similar_indices:
            similar_queries.append(self.query_history[idx])
        
        return similar_queries

    def setup(self):
        """Initialize the bot by loading and processing the PDFs, only rebuilding cache if PDFs have changed."""
        cache_file = "medical_bot_cache.pkl"
        pdf_timestamps = {pdf: os.path.getmtime(pdf) for pdf in self.pdf_paths}
        cache_valid = False
        # Check if cached data exists and is valid
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    cached_timestamps = cached_data.get('pdf_timestamps', {})
                    if cached_timestamps == pdf_timestamps:
                        self.clean_chunks = cached_data['clean_chunks']
                        self.tfidf_matrix = cached_data['tfidf_matrix']
                        self.vectorizer = cached_data['vectorizer']
                        print("Loaded cached data successfully")
                        self._setup_llm()
                        return
            except Exception as e:
                print(f"Error loading cache: {e}")
        # If cache is not valid, process PDFs
        self.documents = []
        for pdf_path in self.pdf_paths:
            loader = PyPDFLoader(pdf_path)
            self.documents.extend(loader.load())
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
        # Cache the processed data and PDF timestamps
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'clean_chunks': self.clean_chunks,
                    'tfidf_matrix': self.tfidf_matrix,
                    'vectorizer': self.vectorizer,
                    'pdf_timestamps': pdf_timestamps
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
        
        # Define prompt for PDF answer
        pdf_prompt_template = (
            "Use the following medical text to answer the question in a clear, human-like, and concise way. "
            "Do not be too brief or too detailed. Make the answer just long enough for the user to understand everything important, but avoid unnecessary information.\n"
            "Text: {context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        self.pdf_prompt = PromptTemplate(
            template=pdf_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Set up question-answering chain
        self.qa_chain = LLMChain(
            llm=llm,
            prompt=self.pdf_prompt,
            verbose=False
        )

    def _get_pdf_answer(self, question: str) -> str:
        """Get answer from PDF content using TF-IDF to find relevant context."""
        if not self.clean_chunks:
            return ""
        # Transform question into TF-IDF vector
        question_vec = self.vectorizer.transform([question.lower()])
        # Compute cosine similarities
        similarities = cosine_similarity(question_vec, self.tfidf_matrix).flatten()
        # Find index of most relevant chunk
        most_relevant_idx = int(np.argmax(similarities))
        max_similarity = similarities[most_relevant_idx]
        # Check if similarity is above a threshold
        if max_similarity < 0.1:
            return ""
        # Build context from the most relevant chunk
        context = self.clean_chunks[most_relevant_idx]
        # Query the LLM with the relevant context
        try:
            chain = LLMChain(
                llm=self.qa_chain.llm,
                prompt=self.pdf_prompt,
                output_parser=StrOutputParser()
            )
            response = chain.invoke({"context": context, "question": question})
            # If response is a dict, extract the answer string
            if isinstance(response, dict):
                # Try common keys
                answer = response.get('text') or response.get('output') or next(iter(response.values()), "")
            else:
                answer = response
            # Ensure answer is a string
            if not isinstance(answer, str):
                answer = str(answer)
            answer = answer.strip()
            # Return the first 2-3 sentences for a balanced answer
            sentences = re.split(r'(?<=[.!?]) +', answer)
            return ' '.join(sentences[:3]).strip() if answer else ""
        except Exception as e:
            print(f"Error getting PDF answer: {e}")
            return ""

    def _get_wiki_answer(self, question: str) -> str:
        """Get answer from Wikipedia as a fallback, but return a short, clear, and human-like summary (2-3 sentences)."""
        try:
            search_term = question
            # Check cache first
            if search_term in self.wiki_cache:
                content = self.wiki_cache[search_term]
                # Return the first 2-3 sentences for a balanced answer
                sentences = re.split(r'(?<=[.!?]) +', content)
                return ' '.join(sentences[:3]).strip() if content else ""
            # Attempt direct Wikipedia search
            try:
                page = wikipedia.page(search_term)
                content = page.content
                content = re.sub(r'\n', ' ', content)
                self.wiki_cache[search_term] = content
                sentences = re.split(r'(?<=[.!?]) +', content)
                return ' '.join(sentences[:3]).strip() if content else ""
            except wikipedia.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0])
                    content = page.content
                    content = re.sub(r'\n', ' ', content)
                    self.wiki_cache[search_term] = content
                    sentences = re.split(r'(?<=[.!?]) +', content)
                    return ' '.join(sentences[:3]).strip() if content else ""
                except:
                    pass
            except wikipedia.PageError:
                pass
            # If direct search fails, try search suggestions
            try:
                results = wikipedia.search(search_term, results=3)
                if results:
                    for title in results:
                        try:
                            page = wikipedia.page(title)
                            content = page.content
                            content = re.sub(r'\n', ' ', content)
                            self.wiki_cache[search_term] = content
                            sentences = re.split(r'(?<=[.!?]) +', content)
                            return ' '.join(sentences[:3]).strip() if content else ""
                        except:
                            continue
            except:
                pass
            # If nothing found
            self.wiki_cache[search_term] = ""
            return ""
        except Exception as e:
            print(f"Error in Wikipedia fallback: {e}")
            return ""

    def query(self, question: str, context: str = None) -> str:
        """Query the bot with a question and optional context from previous questions"""
        try:
            # Check if this is a follow-up question
            is_follow_up = False
            if context and any(word in question.lower() for word in ["same", "it", "this", "that", "these", "those", "they", "them", "their", "its"]):
                is_follow_up = True
                # Extract the main topic from the previous question
                words = context.split()
                medical_terms = []
                medical_indicators = ["itis", "oma", "emia", "pathy", "algia", "derma", "itis", "oma", "osis", "pathy", "plasty", "rrhagia", "rrhaphy", "rrhea", "rrhexis", "sclerosis", "stasis", "tomy", "uria"]
                for i, word in enumerate(words):
                    if word.lower() in ["what", "is", "are", "the", "and", "for", "with", "about", "how", "why", "when", "where", "who", "which", "can", "you", "tell", "me", "more", "about", "explain", "describe", "list", "name", "give", "show", "provide", "information", "on", "regarding", "concerning", "related", "to"]:
                        continue
                    if any(indicator in word.lower() for indicator in medical_indicators):
                        medical_terms.append((i, word))
                        continue
                    if len(word) > 3:
                        medical_terms.append((i, word))
                if medical_terms:
                    main_topic = medical_terms[0][1]
                else:
                    for word in words:
                        if word.lower() not in ["what", "is", "are", "the", "and", "for", "with", "about", "how", "why", "when", "where", "who", "which"]:
                            main_topic = word
                            break
                    else:
                        main_topic = words[0]
                full_question = f"{main_topic} {question}"
            else:
                full_question = question

            # Try to get answer from PDF first
            pdf_answer = self._get_pdf_answer(full_question)
            # Check for generic/irrelevant PDF answers
            irrelevant_patterns = [
                r"no relevant answer", r"text does not mention", r"not found", r"no information", r"not discussed", r"not available", r"no data", r"no mention"
            ]
            is_irrelevant = False
            if pdf_answer:
                for pat in irrelevant_patterns:
                    if re.search(pat, pdf_answer, re.IGNORECASE):
                        is_irrelevant = True
                        break
            if pdf_answer and not is_irrelevant:
                self.add_to_history(question, pdf_answer, "PDF")
                return pdf_answer

            # If no PDF answer or irrelevant, try Wikipedia
            wiki_answer = self._get_wiki_answer(full_question)
            if wiki_answer:
                self.add_to_history(question, wiki_answer, "Wikipedia")
                return wiki_answer

            # If still no answer, try similar previous queries
            similar = self.find_similar_previous_queries(question)
            for item in similar:
                if item and item.get("answer"):
                    self.add_to_history(question, item["answer"], "History")
                    return item["answer"]

            # If all else fails, return a default message
            default_msg = "I'm sorry, I couldn't find information on that."
            self.add_to_history(question, default_msg, "None")
            return default_msg

        except Exception as e:
            print(f"Error during query: {e}")
            return "An error occurred while processing your question."

    def clear_history(self) -> bool:
        """Clear the query history and save the empty history to file."""
        try:
            self.query_history = []
            self.save_history()
            return True
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False

