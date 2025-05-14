import os
from typing import Dict, List, Tuple
import string
import nltk
from nltk.corpus import stopwords
import math

def load_documents(directory: str) -> Dict[str, str]:
    """
    Load all .txt files from the specified directory.
    
    Args:
        directory (str): Path to the documents directory
        
    Returns:
        Dict[str, str]: Dictionary mapping filenames to their contents
    """
    documents = {}
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Documents directory not found: {directory}")
        
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents[filename] = file.read()
            except Exception as e:
                print(f"Warning: Could not read {filename}: {str(e)}")
                
    return documents

def search_documents(documents: Dict[str, str], query: str, case_sensitive: bool = False) -> Dict[str, List[str]]:
    """
    Search for a query string in the documents.
    
    Args:
        documents (Dict[str, str]): Dictionary of document contents
        query (str): Search query
        case_sensitive (bool): Whether to perform case-sensitive search
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping filenames to lists of matching lines
    """
    if not case_sensitive:
        query = query.lower()
        
    results = {}
    
    for filename, content in documents.items():
        matches = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            search_line = line if case_sensitive else line.lower()
            if query in search_line:
                # Add line number and context
                context = line.strip()
                if len(context) > 100:  # Truncate long lines
                    context = context[:97] + "..."
                matches.append(f"Line {i}: {context}")
                
        if matches:
            results[filename] = matches
            
    return results 

def tokenize(text: str) -> list:
    """
    Convert a text string into a list of lowercase words with punctuation removed and stopwords filtered out.
    Args:
        text (str): The input text string
    Returns:
        list: List of lowercase words with stopwords removed
    """
    # Ensure stopwords are downloaded
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)
    no_punct = text.translate(translator)
    words = no_punct.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words 

def build_term_frequency_index(documents: dict) -> dict:
    """
    Build a term frequency index from a dictionary of documents.
    Returns a dictionary where each word maps to another dictionary
    showing how many times it appears in each file.
    Args:
        documents (dict): Dictionary mapping filenames to file contents
    Returns:
        dict: {word: {filename: count, ...}, ...}
    """
    index = {}
    for filename, content in documents.items():
        words = tokenize(content)
        for word in words:
            if word not in index:
                index[word] = {}
            if filename not in index[word]:
                index[word][filename] = 0
            index[word][filename] += 1
    return index 

def compute_tfidf(term: str, term_freq_index: dict, documents: dict) -> dict:
    """
    Calculate the TF-IDF score for a given term in each document.
    Args:
        term (str): The query term (should be preprocessed/tokenized as in the index)
        term_freq_index (dict): The term frequency index as built by build_term_frequency_index
        documents (dict): The original documents dictionary {filename: content}
    Returns:
        dict: {filename: tf-idf score, ...}
    """
    N = len(documents)
    tfidf_scores = {}
    # Document frequency: number of docs containing the term
    df = len(term_freq_index.get(term, {}))
    if df == 0:
        return {filename: 0.0 for filename in documents}
    idf = math.log(N / df)
    for filename in documents:
        tf = term_freq_index.get(term, {}).get(filename, 0)
        tfidf_scores[filename] = tf * idf
    return tfidf_scores 

def search_tfidf(query: str, term_freq_index: dict, documents: dict) -> list:
    """
    Search for documents relevant to the query using TF-IDF ranking.
    Args:
        query (str): The user query string.
        term_freq_index (dict): The term frequency index.
        documents (dict): The original documents dictionary {filename: content}
    Returns:
        list: List of (filename, score) tuples, sorted by score descending.
    """
    query_terms = tokenize(query)
    doc_scores = {filename: 0.0 for filename in documents}
    for term in query_terms:
        tfidf_scores = compute_tfidf(term, term_freq_index, documents)
        for filename, score in tfidf_scores.items():
            doc_scores[filename] += score
    # Sort by score descending, filter out zero scores
    ranked = [(filename, score) for filename, score in doc_scores.items() if score > 0]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked 