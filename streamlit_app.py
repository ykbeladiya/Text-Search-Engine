import streamlit as st
from utils import load_documents, build_term_frequency_index, search_tfidf
import os
import re

def highlight_keyword(text, keyword):
    # Highlight all case-insensitive matches of the keyword in the text
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f'<mark>{m.group(0)}</mark>', text)

def main():
    st.title('Text Search Engine')
    st.write('Enter a search query to find relevant documents:')
    query = st.text_input('Search Query')
    if query:
        documents_dir = os.path.join(os.path.dirname(__file__), 'documents')
        documents = load_documents(documents_dir)
        tf_index = build_term_frequency_index(documents)
        ranked_results = search_tfidf(query, tf_index, documents)
        if ranked_results:
            st.success(f"Found {len(ranked_results)} matching document(s) for '{query}'")
            for i, (filename, score) in enumerate(ranked_results, 1):
                preview = documents[filename][:100].replace('\n', ' ')
                preview = highlight_keyword(preview, query)
                st.markdown(f"**{i}. {filename}**  ", unsafe_allow_html=True)
                st.markdown(f"Relevance Score: `{score:.4f}`", unsafe_allow_html=True)
                st.markdown(f"Preview: {preview}", unsafe_allow_html=True)
                st.markdown('---')
        else:
            st.warning(f"No relevant documents found for '{query}'")

if __name__ == "__main__":
    main() 