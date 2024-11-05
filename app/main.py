import os
from dotenv import load_dotenv
from vector_search import load_vectordb
from chat_utils import create_chat_chain, chatbot

def setup():
    """
    Initialize environment and load vector database
    """
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    base_dir = os.path.dirname(__file__)
    vectordb_path = os.path.join(base_dir, 'data', 'vectordb')
    vectordb = load_vectordb(vectordb_path)

    chain = create_chat_chain(openai_api_key)

    return vectordb, chain


# main
vectordb, chain = setup()
test_queries = [
    "What are the core courses in the MS in Applied Data Science program?",
    "What are the admission requirements for the program?",
    "Tell me about the capstone project.",
    "What is the tuition cost for the program?",
    "What scholarships are available for the program?",
    "What are the minimum scores for the TOEFL and IELTS English Language Requirement?",
    "Is there an application fee waiver?",
    "What are the deadlines for the in-person program?",
    "How long will it take for me to receive a decision on my application?",
    "Can I set up an advising appointment with the enrollment management team?",
    "Where can I mail my official transcripts?",
    "Does the Master’s in Applied Data Science Online program provide visa sponsorship?",
    "How do I apply to the MBA/MS program?",
    "Is the MS in Applied Data Science program STEM/OPT eligible?",
    "How many courses must you complete to earn UChicago’s Master’s in Applied Data Science?"
]
query = test_queries[2]

print(chatbot(query, vectordb, chain))