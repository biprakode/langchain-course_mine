from operator import itemgetter

import dotenv
from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

# Initialize your hosted Chroma cloud client
client = chromadb.CloudClient(
    api_key='ck-Gqr2hAatfLQ4NoNCspKj1apaXt16cJkiC1Pp9mqQ7WNz',
    tenant='d82df22f-ba98-4ac7-96b6-049265bd52b0',
    database='learning_rag'
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatGroq(model="openai/gpt-oss-20b")


file_path = "/run/media/biprarshi/COMMON/files/AI/Agentic_AI/LangChain_Course/langchain-course/RAG/CL16.txt"
loader = TextLoader(file_path)
raw_document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(raw_document)

vector_store = Chroma(
    client=client,
    collection_name="learning-rag",
    embedding_function=embeddings
)

vector_store.add_documents(texts)


retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 6,
                                      "fetch_k": 20,
                                      "lambda_mult": 0.25}
)

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you do not know the answer, or if the context does not contain enough information, "
        "say exactly: 'I cannot find the answer in the provided documents.' "
        "Do not try to make up or hallucinate an answer.\n\n"
        "Context:\n{context}"
    ),
    (
        "human",
        "{question}"
    )
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    # 1. Fetch relevant documents from your cloud database
    docs = retriever.invoke(query)
    context = format_docs(docs)

    # Fixed Bug 3: Changed .format() to .format_messages() so Groq receives List[Messages]
    messages = rag_prompt.format_messages(
        context=context,
        question=query
    )

    # 2. Generate the completion
    response = llm.invoke(messages)

    # 3. Pretty Print Wrapper (Formatted Layout)
    print("\n" + "="*40 + " RAG SYSTEM OUTPUT " + "="*40)
    print(f"User Query: '{query}'\n")
    print(f"Retrieved Documents Count: {len(docs)}")
    print("-"*99)
    print(f"Response:\n{response.content}")
    print("="*99 + "\n")

    return response.content


def retrieval_chain_with_lcel():
    retrieval_chain = (
            RunnablePassthrough.assign(
                context = itemgetter[str]("question") | retriever | format_docs
            )
            | rag_prompt | llm | StrOutputParser)
    return retrieval_chain

questions = [
    "What is Charles Leclerc's date of birth?",
    "In which year did Leclerc win the GP3 Series championship?",
    "Who did Leclerc finish runner-up to in the Formula Renault 2.0 Alps Series?",
    "What team did Leclerc drive for when he won the FIA Formula 2 Championship?",
    "How old was Leclerc when he started competitive kart racing?",
    "Where was Charles Leclerc born and raised?",
    "What was Leclerc's finishing position in his rookie FIA European Formula 3 season?",
    "In which year did Leclerc win the karting junior World Cup?",
    "What was Leclerc's first Formula 1 race victory, and where did it take place?",
    "Which Ferrari car number does Charles Leclerc race with?",
    "Who was Leclerc's teammate at Ferrari during his early years?",
    "What year did Leclerc join the Ferrari Driver Academy?",
]

# for q in questions:
#     retrieval_chain_without_lcel(q)

retrieval_chain = retrieval_chain_with_lcel()
for q in questions:
    print("\n" + "="*40 + " RAG SYSTEM OUTPUT " + "="*40)
    print(f"User Query: '{q}'\n")
    print("-"*99)
    response = retrieval_chain.invoke({"question": q})
    print(f"Response:\n{response.content}")
    print("="*99 + "\n")