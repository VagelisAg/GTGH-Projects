import os
import logging
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv




logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# load_dotenv()
# #Azure credentials loaded from .env
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") 
# AZURE_OPENAI_ENDPOINT =os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
# AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
# AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION")




prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant and you have to act like one.
    You must emphasize that answers should be based strictly on
    the provided context from the stories. If the answer cannot be found the model should state that clearly.
    Question: {question}
    Context: {context}
    """
)

class SciFiExplorer:
    def __init__(self,data_directory="data",
                 persist_directory="db/chroma_db",
                 embeddings_model="text-embedding-3-small",
                 chat_model="gpt-4o"):
        self.data_directory=data_directory
        self.persist_directory=persist_directory
        self.embeddings_model=embeddings_model
        self.chat_model=chat_model


        self.embeddings=OpenAIEmbeddings( ##Run locally 
            model=self.embeddings_model,
            openai_api_key="my-secret-key",
            openai_api_base="http://localhost:4000/",
            )


        self.llm = AzureChatOpenAI( #Run on Cloud Azure
            azure_deployment=self.chat_model,
            api_key="1WrA6nOsPNNtKcYjblU0bteFPWouxvxJRWH58dnhdVSjPN6Aj5fZJQQJ99BIACHrzpqXJ3w3AAAAACOGx8Ga",
            azure_endpoint="https://eagat-mfpc8eml-northcentralus.services.ai.azure.com/",
            api_version="2024-12-01-preview",
            )

    #Load all .txts from the data folder into a list of Documents
    def load_documents(self):
        docs = []
        for file in os.listdir(self.data_directory):
                path = os.path.join(self.data_directory, file)
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())   
        return docs
    
    def text_splitter(self,docs):
         splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=150)
         chunks=splitter.split_documents(docs)
         logging.info(f"Splitted into {len(chunks)}")
         return chunks
    #First run → it will build the DB from your .txt chunks and save it in persist_directory.
    #Later runs → even though you’re still calling from_documents(), it will overwrite/rebuild the DB again.
    def vector_DB(self,chunks):
            vector_store=Chroma.from_documents(documents=chunks, #this always rebuild the DB from scrach
                                                embedding=self.embeddings,
                                                persist_directory=self.persist_directory,)
            logging.info("Vector store from document created successfully!")
            s_retriever = vector_store.as_retriever()
            mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
            return(vector_store,s_retriever,mmr_retriever)

    def chain(self,retriever):
            chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser())
            logging.info("LCEL chain is ready")
            return chain


if __name__ == "__main__":
    app = SciFiExplorer()
    #Load docs and split
    docs = app.load_documents()
    logging.info(f"Loaded docs: {len(docs)}")
    chunks = app.text_splitter(docs)
    #print(chunks)
    #Build or load vector DB
    vector_store,s_retriever,mmr_retriever = app.vector_DB(chunks)
    logging.info("Vector DB ready")
    #print(vector_store)
    #Build chain with similarity retriever
    chain_s = app.chain(s_retriever)
    logging.info("Chain with similarity retriever is ready")
    #print(chain_s)
    #Build chain with MMR retriever
    chain_mmr=app.chain(mmr_retriever)
    logging.info("Chain with mmr_retriever is ready")

    #Ask a test question and answer with 2 retrievers

    q=input("Give related question otherwise press exit: ")
    while q!="exit":
        #q = "What are some interesting examples of first contact with alien life?"
        answer_s = chain_s.invoke(q)
        logging.info("The query is received")
        print(f"Q: {q}\nA with similarity search: {answer_s}")

        answer_mmr=chain_mmr.invoke(q)
        logging.info("The query is received")
        print(f"Q: {q}\nA with MMR search: {answer_mmr}")
        q=input("Give another related question otherwise press exit if you want to exit: ")

    print("You exited thank for chatting with me")


    







        
         
         

        