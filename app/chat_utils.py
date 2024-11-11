from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector_search import search_similar_chunks, format_chunk_results
import json
from typing import List, Optional, Dict
from dotenv import load_dotenv
import os
from langchain.load import dumps, loads


load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")


def create_chat_chain(openai_api_key: str, model_name: str = "gpt-4o-2024-08-06", temperature: float = 0.0):
    """
    Create a chat chain with OpenAI model
    """
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key
    )

    template = """You are an assistant for University of Chicago's MS in Applied Data Science program.
Use the following context (including metadata about the source and type of content) to answer the question at the end. 
If you cannot find the answer in the context, just say "I'm sorry, I don't have enough information to answer that question."
You can use multiple contexts. Try to be as specific as possible. Including the relevant URL(s) at the end of your response. If multiple URLs are referenced, list all of them.

Context:
{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def subcategory_finder(question: str, model_name: str = "gpt-4o-2024-08-06", temperature: float = 0.0) -> Optional[Dict[str, str]]:
    """
    Identify the relevant subcategory based on a question using the OpenAI chat model.
    
    Args:
        question: The question for which to find a related subcategory.
        model_name: The name of the OpenAI model (default is "gpt-4o-2024-08-06").
        temperature: The temperature setting for the model's response (default is 0.0).
    
    Returns:
        A dictionary containing the relevant subcategory or None if no relevant subcategory is found.
    """
    # Initialize the OpenAI model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key
    )
    
    # Create a simple prompt template with escaped braces
    template = """You are an assistant for the University of Chicago's MS in Applied Data Science program.
Your task is to identify the most relevant subcategory based on the "Question". The subcategory options are provided below. 
If there is a possibility that it could be a candidate, please include it just to be safe. 
Please respond in a dictionary format as shown in the Output Example. Since your response will be used directly as a filter, do not include any comments outside of the dictionary format.
If no relevant subcategory is found, respond with "None".

## Subcategory Options ## 
  - in-person-program
  - course-progressions
  - online-program
  - tuition-fees-aid
  - capstone-projects
  - instructors-staff
  - how-to-apply
  - our-students
  - events-deadlines
  - career-outcomes
  - faqs

## Output Example ##
{{
    "subcategory": ["in-person-program", "course-progressions"]
}}

## Question ##
{question}

## Answer ##
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create a formatted input for the prompt
    formatted_prompt = prompt.format(question=question)
    
    # Use `invoke` method and extract only the content
    response = llm.invoke(formatted_prompt)

    print("Original Response from Subcategory Finder:", response.content)
    
    try:
        # Attempt to parse the response content as JSON
        result = json.loads(response.content)
        
        # Check if the result is a dictionary and has a 'subcategory' key
        if isinstance(result, dict) and 'subcategory' in result:
            return result
        else:
            return None
    except (json.JSONDecodeError, AttributeError):
        return None


def rewrite_queries(question: str, model_name: str = "gpt-4o-2024-08-06", temperature: float = 0.7) -> List[str]:
    """
    Generate multiple rewrites of a user's query using the OpenAI model.

    Args:
        question: The original question to be rewritten.
        model_name: The name of the OpenAI model (default is "gpt-4o-2024-08-06").
        temperature: The temperature setting for the model's response (default is 0.7 for more variability).
    
    Returns:
        A list containing the original question and multiple rewrites.
    """
    # Initialize the OpenAI model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key
    )

    # Create a prompt template to rewrite the query
    template = """You are an assistant for the University of Chicago's MS in Applied Data Science program. Your task is rephrasing queries to improve retrieval performance(RAG).
Your have to generate multiple rephrased versions of the original "Question" to enhance diversity in information retrieval.
Please output the rephrased versions in a JSON list format, with the original question included. 
Since your response will be used directly as a filter, do not include any comments outside of the JSON format.

## Example Output ##
[
    "{{original question}}",
    "Rephrased question 1",
    "Rephrased question 2",
    "Rephrased question 3"
]

## Question ##
{question}

## Answer ##
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format the prompt with the original question
    formatted_prompt = prompt.format(question=question)
    
    # Get the response with `invoke` method
    response = llm.invoke(formatted_prompt)

    print("Original Response from Rewrite Queries:", response.content)
    
    try:
        # Attempt to parse the response content as JSON list
        result = json.loads(response.content)
        
        # Check if result is a list and contains the original question and rewrites
        if isinstance(result, list) and question in result:
            return result
        else:
            # Return original question if parsing fails
            return [question]
    except (json.JSONDecodeError, AttributeError):
        # Return original question if there's an error
        return [question]


def reciprocal_rank_fusion(results: list[list], k=60, top_n=5):
    """
    Reciprocal rank fusion for merging multiple lists of ranked documents.
    Returns only the top_n documents.
    """
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            # Serialize doc to use as a dictionary key
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    # Print scores for debugging purposes
    # print("Document Scores (Before Ranking):")
    # for doc_str, score in fused_scores.items():
    #     doc = loads(doc_str)  # Deserialize doc for readability
    #     print(f"Document: {doc}, Score: {score}")
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return only the top_n documents
    return [doc for doc, score in reranked_results[:top_n]]


class ChatHistory:
    def __init__(self):
        self.history = []

    def add_interaction(self, user_query: str, response: str):
        self.history.append(f"User: {user_query}\nAssistant: {response}")

    def get_history(self) -> str:
        return "\n".join(self.history)


def chatbot(user_query: str, vectordb, chain, chat_history: ChatHistory, routing=False, fusion=False):
    """
    Process user query and generate response using retrieved context with metadata.
    Combines results from both filtered and unfiltered searches if routing is enabled.
    Includes conversation history.
    """
    
    if fusion and routing:
       # Generate rewritten queries for fusion
        rewritten_queries = rewrite_queries(user_query)
        all_results = []
        # Execute search for each rewritten query with k=10
        for query in rewritten_queries:
            chunks_per_query = search_similar_chunks(vectorstore=vectordb, query=query, k=10)
            all_results.append(chunks_per_query)
        # Perform reciprocal rank fusion to get top 5 fused results
        fused_chunks = reciprocal_rank_fusion(all_results, top_n=5)
        # Initialize chunks with top 5 fused results
        chunks = fused_chunks
        
        # Apply routing with subcategory filtering to get additional 5 chunks
        subcategory = subcategory_finder(user_query)
        filtered_chunks = search_similar_chunks(vectorstore=vectordb, query=user_query, k=5, filter_dict=subcategory)
        # Combine both fusion and routing results (5 from fusion + 5 from routing)
        chunks.extend(filtered_chunks)

    elif fusion:
        rewritten_queries = rewrite_queries(user_query)
        all_results = []
        for query in rewritten_queries:
            chunks = search_similar_chunks(vectorstore=vectordb, query=query, k=10)
            all_results.append(chunks)
        chunks = reciprocal_rank_fusion(all_results, top_n=10)

    elif routing:
        subcategory = subcategory_finder(user_query)
        filtered_chunks = search_similar_chunks(vectorstore=vectordb, query=user_query, k=5, filter_dict=subcategory)
        unfiltered_chunks = search_similar_chunks(vectorstore=vectordb, query=user_query, k=5)
        all_chunks = {doc.page_content: doc for doc in (filtered_chunks + unfiltered_chunks)}
        chunks = list(all_chunks.values())

    else:
        chunks = search_similar_chunks(vectorstore=vectordb, query=user_query, k=5)
    
    
    context = format_chunk_results(
        chunks,
        metadata_fields=[
            'page_type',
            'primary_category',
            'subcategory',
            'title',
            'source'
        ],
        include_content=True
    )
    
    # Include history in the context
    history = chat_history.get_history()
    combined_context = f"{history}\n\nContext:\n{context}"

    print('======Contexts======\n', combined_context)

    response = chain.invoke({
        "context": combined_context,
        "question": user_query
    })

    # Add this interaction to the chat history
    chat_history.add_interaction(user_query, response)

    return response