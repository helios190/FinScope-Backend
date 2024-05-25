from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.search.documents import SearchClient
from azure.search.documents.models import IndexAction
from azure.search.documents import IndexDocumentsBatch
from azure.core.credentials import AzureKeyCredential
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ocr_handle.ocr import ocr_image_from_blob
from ocr_handle.storage import upload_file_to_blob, get_blob_client
from langchain.chat_models import ChatOpenAI
from ocr_handle.database import get_db, update_user_data
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import hub
from openai import AzureOpenAI
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
import openai
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
import datetime

load_dotenv()
db = get_db()

app = Flask(__name__)
CORS(app)

# Azure Cognitive Search configuration
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
api_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-02-01",
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

def generate_embeddings(text, model="embedding-mbm"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def upload_documents(documents):
    search_client = SearchClient(
        endpoint=os.getenv('AZURE_SEARCH_ENDPOINT2'),
        index_name='vector',
        credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY_2'))
    )

    search_client.upload_documents(documents=documents)

def get_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    
    document = Document(page_content=text)
    chunks = text_splitter.split_documents([document])
    print(chunks)
    
    documents_to_upload = []
    for idx, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk.page_content)
        documents_to_upload.append({
            "id": str(idx),  # Unique ID for each document
            "content": chunk.page_content,
            "embedding": [float(e) for e in embedding]  # Convert embedding to list of floats
        })
    
    upload_documents(documents_to_upload)
    return documents_to_upload

def search_query(query):
    embedding = generate_embeddings(query)
    search_client = SearchClient(
        endpoint=os.getenv('AZURE_SEARCH_ENDPOINT2'),
        index_name='vector',
        credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY_2'))
    )

    search_results = search_client.search(search_text=query, top=10)


    results = []
    for result in search_results:
        results.append({
            "score": result['@search.score'],
            "content": result['content']
        })
    
    return results


def get_rag_chain(retriever,query_text):
    
    llm = OpenAI()


    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query_text}. \n Information: {retriever}"}
    ]
    
    response = llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

@app.route('/upload_and_search', methods=['GET'])
def upload_and_search():
    try:
        # Upload documents with chunking and vector embeddings
        last_file = db.User_Data.find_one(sort=[('_id', -1)])
        if not last_file or 'file_name' not in last_file:
            return jsonify({'error': 'No file found in the database'}), 400
        file_name = last_file['file_name']
        # blob_client = get_blob_client(file_name)
        # extracted_text = ocr_image_from_blob(blob_client)
        extracted_text = last_file['ocr_result']
        update_user_data(last_file['_id'], extracted_text)
        
        # Chunk the extracted text
        get_vector_store(extracted_text)
        query_text = '''How is the Condition of the Profitability of this Company. You need to Asses it by this
        How much is the Gross Profit of this Company, and How much is the Total Sales. Please Calculate the Gross Profit Margin by Gross profit Divided by Sales
    Secondly. How much is the Operating Profit of this Company. Search the Operating Profit Margin by Dividing Operating Profit by Sales.
    Thirdly. How much is the Net Income. Calculate the Net Profit Margin by Dividing Net Income by Total Sales. Lastly. Calculate the Return On Assets
    By dividing Net Income by Total Assets. Draw a Conclusion on the Company Profitablity Condition. Dont say if you dont find the number. But Please Draw
    a Conclusion. Dont also say that you cannot draw a conclusion based on this number. Just Say what it is.'''
        search_results = search_query(query_text)
        answer =  get_rag_chain(search_results,query_text)

        print(answer)
        
        if not query_text:
            return jsonify({'error': 'No query text provided'}), 400
       


        return jsonify({
            'message': 'OCR extraction, chunking, and search successful',
            'search_results': answer
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)