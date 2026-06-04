import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

dotenv.load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("/run/media/biprarshi/COMMON/files/AI/Agentic_AI/LangChain_Course/langchain-course/RAG/CL16.txt")
document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

PineconeVectorStore.from_documents(texts , embeddings , index_name="learning-rag2")

