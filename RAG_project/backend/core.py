from mailbox import Message
from typing import Dict, Any

import dotenv
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

dotenv.load_dotenv()

embeddings_model = HuggingFaceEndpointEmbeddings(
    model="google/embeddinggemma-300m",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)

vector_store = PineconeVectorStore(index_name="langchain-docs" , embedding=embeddings_model)

model = init_chat_model("openai/gpt-oss-120b" , model_provider="groq" , temperature = 0.7)

@tool(response_format = "content_and_artifact")
def retrieve_context(query : str):
    """Retrieve relevant documentation to help answer user queries about Lanchain"""
    retrieved_docs = vector_store.as_retriever().invoke(query , k = 2)

    serialized = "\n\n".join(
        f"Source : {doc.metadata.get('Source' , 'Unknown')}\n\nContent : {doc.page_content}" for doc in retrieved_docs
    )

    return serialized , retrieved_docs # content and artifact

def run_llm(query : str) -> Dict[str , Any]:
    """
    Run the RAG pipeline to answer a query from the retrieved documentation
    :param query: The user's question to run the RAG pipeline on
    :return: Dictionary containing:
    answer - The generated answer
    context - List of retrieved documents
    """

    system_prompt =  "You are a helpful AI assistant that answers questions about LangChain documentation. "
    "You have access to a tool that retrieves relevant documentation. "
    "Use the tool to find relevant information before answering questions. "
    "Always cite the sources you use in your answers. "
    "If you cannot find the answer in the retrieved documentation, say so."

    agent = create_agent(model , tools = [retrieve_context] , system_prompt = system_prompt)

    messages = [{'role' : 'user' , 'content' : query}]
    response = agent.invoke({"messages" : messages})
    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message , ToolMessage) and hasattr(message , "artifact"): # if tool call and is artifact
            if isinstance(message.artifact , list):
                context_docs.extend(message.artifact)

    return {
        "answer" : answer,
        "context" : context_docs
    }

if __name__ == "__main__":
    result = run_llm("Explain LCEL")
    print(result)