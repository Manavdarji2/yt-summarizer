from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import time

load_dotenv()
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Using a compatible embedding model and adding task_type
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document"
)


class APILimitReachedError(Exception):
    """Exception raised when the API key has reached its limit."""
    pass


def create_vector_db_from_youtube_url(video_url)->FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url, language='hi')
    transcript = loader.load()
    
    # Increased chunk size to 4000 to reduce the number of API calls.
    # For summarization, larger context chunks are often more efficient and effective.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    docs = text_splitter.split_documents(transcript)
    
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    
    # To handle potential 429 rate limits on the free tier, 
    # we process chunks in batches with a small delay.
    db = None
    batch_size = 50 # Smaller batches to be safe on free tier
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        try:
            if db is None:
                db = FAISS.from_texts(batch_texts, embedding=embeddings, metadatas=batch_metadatas)
            else:
                db.add_texts(batch_texts, metadatas=batch_metadatas)
        except ResourceExhausted:
            raise APILimitReachedError("API Key limit reached during vector database creation.")
        
        if i + batch_size < len(texts):
            # Short sleep between batches to avoid hitting RPM limits
            time.sleep(1)
            
    return db



def get_response_from_query(db:FAISS, query, k=3):
    # Search for relevant documents
    docs = db.similarity_search(query, k=k)
    
    docs_page_content = " ".join([d.page_content for d in docs])
    
    # Using gemini-2.0-flash (stable) or gemini-2.5-flash as per environment
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    
    video_summary_prompt = PromptTemplate(
        input_variables=["query", "docs"],
        template="""
            You are an expert YouTube video summarizer. Your task is to read the video transcript and related documents, then generate a clear, accurate, and complete summary.

            Guidelines:
            1. Focus on the main points, arguments, and insights presented in the video.
            2. Organize the summary in a logical flow (beginning, middle, end).
            3. Highlight key takeaways, facts, or steps mentioned in the video.
            4. If the video explains a process, outline it step-by-step.
            5. If the video discusses multiple topics, separate them into sections with headings or bullet points.
            6. Keep the tone neutral, factual, and concise.
            7. Do NOT add any preamble, postamble, or extra commentary outside the summary.

            Input:
            - Query: {query}
            - Transcript & Documents: {docs}

            Output:
            Only return the final structured summary of the video. Format Answer Only
            """
    )
    
    chain = video_summary_prompt | llm
    try:
        response = chain.invoke({"query": query, "docs": docs_page_content})
    except ResourceExhausted:
        raise APILimitReachedError("API Key limit reached during query execution.")
        
    content = response.content.replace("\n", " ")
    return content

