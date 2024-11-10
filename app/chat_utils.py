from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector_search import search_similar_chunks, format_chunk_results
import json
from typing import Optional, Dict
from dotenv import load_dotenv
import os

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


def chatbot(user_query: str, vectordb, chain, routing=False):
    """
    Process user query and generate response using retrieved context with metadata.
    Combines results from both filtered and unfiltered searches if routing is enabled.
    """
    if routing:
        # Find subcategory for filtering
        subcategory = subcategory_finder(user_query)

        # Search with filter
        filtered_chunks = search_similar_chunks(vectorstore=vectordb, query=user_query, k=5, filter_dict=subcategory)

        # Search without filter
        unfiltered_chunks = search_similar_chunks(vectorstore=vectordb, query=user_query, k=5)

        # Combine the results, ensuring no duplicates
        all_chunks = {doc.page_content: doc for doc in (filtered_chunks + unfiltered_chunks)}
        chunks = list(all_chunks.values())
    else:
        # Retrieve and format chunks without routing
        chunks = search_similar_chunks(vectorstore=vectordb, query=user_query)
    
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
    
    print('======Contexts======\n', context)

    # Generate response using the formatted context
    response = chain.invoke({
        "context": context,
        "question": user_query
    })

    return response