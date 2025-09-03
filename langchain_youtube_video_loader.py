from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.chat_models import init_chat_model
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
load_dotenv()
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def create_vector_db_from_youtube_url(video_url)->FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url, language='hi')
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embedding=embeddings)
    return db



def get_response_from_query(db:FAISS, query, k=3):
    docs=db.similarity_search(query, k=k)
    
    docs_page_content=" ".join([d.page_content for d in docs])
    llm=init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    
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
        Only return the final structured summary of the video.
        """)
    
    
    chain=LLMChain(llm=llm, prompt=video_summary_prompt)
    response=chain.run(query=query, docs=docs_page_content)
    response=response.replace("\n"," ")
    return response