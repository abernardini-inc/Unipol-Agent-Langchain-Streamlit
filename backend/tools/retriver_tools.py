from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.xml import UnstructuredXMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

loaders = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.xml': UnstructuredXMLLoader,
    '.csv': CSVLoader,
}

docs = []
folder_path = "store/file"
vectorstore_path = "store/vectorstore"
vectorstore = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100,
    separators=["#", "\n\n", "\n", " ", ""]
)

if not os.path.exists(vectorstore_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            file_extension = os.path.splitext(file_path)[1]

            if file_extension in loaders:
                loader_class = loaders[file_extension]
                loader = loader_class(file_path)
                documents = loader.load()
                docs.extend(text_splitter.split_documents(documents)) 

    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"]))
    vectorstore.save_local(vectorstore_path)
else:
    vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"]), allow_dangerous_deserialization=True)
    
@tool
def get_all_info(query: str) -> str:
    """Utilizza questo tool per rispondere a qualsiasi domanda del cliente legata ai servizi di Unipol. Questo tool ti darà tutte le informazioni necessarie per rispondere alla domanda del cliente."""
    try:
        print("\nAttivazione tool get_all_info\n")
        print(f"Richiesta: {query}")
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=10)
        print(f"Documenti recuperati: {docs_and_scores}")
        return docs_and_scores
    except Exception as e:
        return "Errore nel recupero delle informazioni"

