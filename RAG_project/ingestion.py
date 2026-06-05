import asyncio
import os
import ssl
import dotenv

from typing import Dict , List , Any , Optional

import certifi
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract, TavilyMap, TavilyCrawl

import logger

dotenv.load_dotenv()


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
embeddings_model = HuggingFaceEndpointEmbeddings(
    model="google/embeddinggemma-300m",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

vector_store = PineconeVectorStore(index_name="langchain-docs" , embedding=embeddings_model)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 7 , max_breadth = 20 , max_pages = 1000)
tavily_crawl = TavilyCrawl()

async def index_documents(documents : List[Document] , batch_size : int):
    logger.log_header("VECTOR STORAGE")
    logger.log_info(f"Vector store indexing" , logger.Colors.DARKCYAN)

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    logger.log_info(f"Batch indexing" , logger.Colors.PURPLE)

    async def add_batches(batch : List[Document] , batch_num : int):
        try:
            await vector_store.aadd_documents(batch)
            logger.log_success(f"Added batch #{batch_num}: Processing {len(batch)} chunks.")

        except Exception as e:
            logger.log_error(f"Failed batch #{batch_num}: {e}")
            return False
        return True


    tasks = [add_batches(batch , i+1)  for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks , return_exceptions=True)


async def main():
    logger.log_header("DOCUMENTATION INGESTION PIPELINE")
    logger.log_info(
        "Tavily Crawling - https://python.langchain.com/",
        color=logger.Colors.PURPLE,
    )

    # res = tavily_crawl.invoke({
    #     "url": "https://python.langchain.com/",
    #
    #     # 1. Structural Scoping
    #     "max_depth": 4,              # 4 layers down captures core API references
    #     "max_breadth": 60,           # Allows scanning a broad set of side-panel navigation links
    #     "limit": 600,                # Hard stop cap to prevent infinite page crawl loops
    #
    #     # 2. Strict Boundary Isolation
    #     "allow_external": False,             # NEVER follow external outbound links (e.g., GitHub, Medium)
    #     "select_domains": ["python.langchain.com"], # Lock focus strictly to the Python sub-domain
    #     "categories": ["Documentation"],     # Pre-filter out Careers, Blogs, or About pages
    #
    #     # 3. Content Tuning
    #     "extract_depth": "advanced",         # Extracts hidden tables, code blocks, and dynamic layouts
    #     "instructions": "Extract all technical guides, API references, code snippets, and structural tutorials on LangChain components, expressions (LCEL), and AI agents."
    # })

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 2,
        "limit": 20,
    })

    results_list = res.get("results", [])

    if not results_list:
        logger.log_error("Tavily returned zero base results. Check API credits or limits.")
        return

    all_docs = []
    for result in results_list:
        # Gracefully fall back to 'content' if 'raw_content' is absent or blank
        content_string = result.get("raw_content") or result.get("content") or ""
        if content_string.strip():
            all_docs.append(Document(
                page_content=content_string,
                metadata={"source": result.get("url", "unknown_source")},
            ))

    logger.log_success(f"Successfully extracted {len(all_docs)} base root page documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000 , chunk_overlap=150)
    split_docs = text_splitter.split_documents(all_docs)

    await index_documents(split_docs , batch_size=200)
    logger.log_success(f"Successfully indexed {len(split_docs)} documents")

if __name__ == "__main__":
    asyncio.run(main())