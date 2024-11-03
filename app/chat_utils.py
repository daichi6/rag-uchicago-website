from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector_search import search_similar_chunks, format_chunk_results


def create_chat_chain(openai_api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
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
Try to be as specific as possible. Including the relevant URL(s) at the end of your response.

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


def chatbot(user_query: str, vectordb, chain=None):
    """
    Process user query and generate response using retrieved context with metadata
    """
    # Retrieve and format chunks with metadata
    chunks = search_similar_chunks(vectorstore=vectordb, query=user_query, k=3)
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