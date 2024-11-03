from typing import List, Dict, Optional
from datetime import datetime
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urlparse
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")


def fetch_web_content(url: str, verify_ssl: bool = True, timeout: int = 10) -> str:
    """
    Fetch content from a single URL.
    """
    response = requests.get(url, verify=verify_ssl, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_metadata(url: str, html_content: str) -> Dict:
    """
    Extract metadata from HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    metadata = {
        "source": url,
        "domain": urlparse(url).netloc,
        "fetch_date": datetime.now().isoformat(),
        "content_type": "web_page"
    }
    
    # Extract meta tags
    if soup.title:
        metadata["title"] = soup.title.string
        
    for meta in soup.find_all('meta'):
        name = meta.get('name', '').lower()
        property = meta.get('property', '').lower()
        content = meta.get('content')
        
        if name == 'description' or property == 'og:description':
            metadata['description'] = content
        elif name == 'keywords':
            metadata['keywords'] = content
        elif name == 'author':
            metadata['author'] = content
            
    return metadata


def clean_content(html_content: str) -> str:
    """
    Clean and extract main content from HTML.
    """
    return trafilatura.extract(html_content) or ""


def create_document(url: str, verify_ssl: bool = True) -> Document:
    """
    Create a single document from a URL.
    """
    try:
        html_content = fetch_web_content(url, verify_ssl)
        metadata = extract_metadata(url, html_content)
        cleaned_content = clean_content(html_content)
        
        return Document(
            page_content=cleaned_content,
            metadata=metadata
        )
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None


def split_document(doc: Document, 
                  chunk_size: int = 1000, 
                  chunk_overlap: int = 200) -> List[Document]:
    """
    Split a document into chunks.
    """
    if not doc:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents([doc])
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk.page_content)
        })
    
    return chunks


def process_urls(urls: List[str], 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200, 
                verify_ssl: bool = True) -> List[Document]:
    """
    Main function to process multiple URLs into chunked documents.
    """
    all_chunks = []
    
    for url in urls:
        # Create document from URL
        doc = create_document(url, verify_ssl)
        if doc:
            # Split into chunks
            chunks = split_document(doc, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
    
    return all_chunks


def enhance_metadata(documents: List[Document]) -> List[Document]:
    """
    Enhance metadata of documents with path-based classifications.
    Extracts categories from URL paths after '/masters-programs/'.
    
    Args:
        documents: List of Document objects to enhance
    
    Returns:
        List of Documents with enhanced metadata
    """
    base_path = "education/masters-programs/"
    enhanced_docs = []
    
    for doc in documents:
        print("\n" + "="*50)
        print("Processing document...")
        print("Original metadata:", doc.metadata)
        
        # Create a copy of the document to avoid modifying the original
        new_doc = Document(
            page_content=doc.page_content,
            metadata=doc.metadata.copy()
        )
        
        if 'source' in new_doc.metadata:
            url = new_doc.metadata['source']
            
            # Find the part of the path after base_path
            try:
                path_parts = url.split(base_path)[1].strip('/').split('/')
                
                # Add primary category
                if path_parts:
                    new_doc.metadata['primary_category'] = path_parts[0]
                
                # Add subcategory if it exists
                if len(path_parts) > 1:
                    new_doc.metadata['subcategory'] = path_parts[1]
                    
                # Add readable format of the full path
                new_doc.metadata['page_type'] = ' - '.join(
                    part.replace('-', ' ').title() 
                    for part in path_parts
                )
                
            except IndexError:
                # Handle cases where URL doesn't contain the expected base path
                new_doc.metadata['page_type'] = 'Other'
                
        print("Enhanced metadata:", new_doc.metadata)
        print("New fields added:")

        new_fields = set(new_doc.metadata.keys()) - set(doc.metadata.keys())
        for field in new_fields:
            print(f"  - {field}: {new_doc.metadata[field]}")
            
        enhanced_docs.append(new_doc)
    
    return enhanced_docs


def init_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    Initialize OpenAI embeddings model.
    
    Args:
        model: Name of the OpenAI embedding model to use
    
    Returns:
        OpenAIEmbeddings instance configured with the specified model
    """
    return OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model=model
    )


def create_and_save_vectordb(documents: List[Document], 
                           embeddings: OpenAIEmbeddings,
                           save_path: Optional[str] = None) -> FAISS:
    """
    Create FAISS vector database from documents and optionally save it.
    
    Args:
        documents: List of Document objects to create embeddings for
        embeddings: Configured OpenAI embeddings model instance
        save_path: Optional path to save the vector database
    
    Returns:
        FAISS vector store object
    """
    if not documents:
        print("Error: Document list is empty")
        return None
        
    try:
        print("Creating vector store...")
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        print("Vector store created successfully")
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving vector database to {save_path}...")
            vectorstore.save_local(save_path)
            print(f"Vector database saved successfully to {save_path}")
            
        return vectorstore
        
    except Exception as e:
        print(f"Error occurred while creating vector store: {str(e)}")
        return None


def get_subpage_links(url):
    """
    Fetches all subpage URLs from a main page.
    """
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, "html.parser")

    subpage_links_all = [url]

    # Find all links and filter those that match our pattern
    base_url = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"

    for link in soup.find_all("a", href=True):
        href = link['href']
        # Check if href starts with the base URL or matches the exact URL we want to include
        if href.startswith(base_url) or href == url:
            # Append the complete URL if relative or absolute
            full_url = href if href.startswith("http") else requests.compat.urljoin(url, href)
            subpage_links_all.append(full_url)

    subpage_links_all = list(set(subpage_links_all))

    print("Number of URLs", len(subpage_links_all))
    print("All URLs", subpage_links_all)
    
    return subpage_links_all


def main():
    urls = get_subpage_links("https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/in-person-program/")

    # Process URLs
    documents = process_urls(
        urls=urls,
        chunk_size=2000,
        chunk_overlap=200
    )
    
    print(f"\nInitial processing:")
    print(f"Processed {len(documents)} chunks from {len(urls)} URLs")
    
    # Enhance metadata
    enhanced_docs = enhance_metadata(documents)
    print("\nMetadata enhancement complete")

    # Initialize embeddings
    embeddings = init_embeddings()
    
    # Create and save vector database
    base_dir = os.path.dirname(__file__)
    vectordb = create_and_save_vectordb(
        documents=enhanced_docs,
        embeddings=embeddings,
        save_path=os.path.join(base_dir, 'data', 'vectordb')
    )
    print("\nVector database creation complete")
    

    print("\nExample of enhanced documents:")
    for i, doc in enumerate(enhanced_docs[:2]):
        print(f"\nChunk {i+1}:")
        print("Enhanced Metadata:", doc.metadata)
        print("Content preview:", doc.page_content[:200], "...")


if __name__ == "__main__":
    main()