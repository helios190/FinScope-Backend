from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from azure.search.documents import SearchClient
from azure.search.documents.models import IndexAction
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI, OpenAI
import portofolio_calculation
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from azure.storage.blob import BlobServiceClient
import datetime
import os
from ocr_handle.database import get_db, update_user_data
from ocr_handle.ocr import ocr_image_from_blob
from ocr_handle.storage import upload_file_to_blob, get_blob_client


load_dotenv()



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


db = get_db()

@app.route('/calculates', methods=['POST'])
def calculate():
    global calculation_results
    try:
        # Ambil data JSON dari permintaan
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400
        
        stock_symbols = data.get('stock_symbols')

        # Periksa apakah stock_symbols ada dalam JSON body
        if not stock_symbols:
            return jsonify({"error": "No stock symbols provided"}), 400

        start_date = "2023-01-01"
        end_date = "2024-01-01"

        adj_close_prices = portofolio_calculation.get_adjusted_close_prices(stock_symbols, start_date, end_date)
        if adj_close_prices.empty:
            return jsonify({"error": "Failed to retrieve stock data"}), 500

        adj_close_prices['^IRX'] = adj_close_prices['^IRX'] / 12 / 100
        adj_close_prices.dropna(inplace=True)

        excess = adj_close_prices.subtract(adj_close_prices['^IRX'], axis=0)
        excess.drop('^IRX', axis=1, inplace=True)

        risk_free_rate = adj_close_prices['^IRX'].mean()
        market_return = excess['S&P500'].mean()

        alpha_beta_dict = portofolio_calculation.calculate_alpha_beta(excess)

        min_var_portfolios = portofolio_calculation.calculate_minimum_variance_frontier(adj_close_prices)
        max_sharpe_weights, _ = portofolio_calculation.calculate_max_sharpe_ratio(adj_close_prices, risk_free_rate)

        calculation_results = {
            "min_var_portfolios": min_var_portfolios,
            "max_sharpe_weights": list(max_sharpe_weights),
            'alpha_beta_dict': alpha_beta_dict
        }
        
        min_var_portfolios = calculation_results['min_var_portfolios']
        max_sharpe_weights = calculation_results['max_sharpe_weights']
        alpha_beta_dict = calculation_results['alpha_beta_dict']
        
        load_dotenv()
        os.getenv('OPENAI_API_KEY')

        prompt_content = f''' 
        I have this result of calculation Minimum Variance Portfolios {min_var_portfolios}, max_sharpe_weights of {max_sharpe_weights}, and alpha and beta of {alpha_beta_dict}
        1. In the Alpha and Beta, there's the Stock Name. Please describe the Security Market Line Result (NOT THE DEFINITION) based on the Value of Alpha and Beta. Give a Conclusion
        2. In the Minimum Variance Portfolios, describe the highest to lowest risk to reward based on this. Give a Conclusion
        3. In the Max Sharpe Weights, please describe the amount of percentage this person needs to take on the stocks that they put. Give a Conclusion
        '''

        high_level_behavior = """
        You are an AI for FinScope-AI. This is a section for Portfolio Management of the desired amount of stocks.
        """

        chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        response = chatgpt([
            SystemMessage(content=high_level_behavior),
            AIMessage(content="Hello! I am a Financial Helper from FinScope-AI. Let me help you through your Investment Opportunity."),
            HumanMessage(content=prompt_content),
        ])

        return jsonify(response.content)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ocr_extract', methods=['POST'])
def ocr_extract():
    try:
        last_file = db.User_Data.find_one(sort=[('_id', -1)])
        
        if not last_file or 'file_name' not in last_file:
            return jsonify({'error': 'No file found in the database'}), 400
        
        file_name = last_file['file_name']
        blob_client = get_blob_client(file_name)
        
        extracted_text = ocr_image_from_blob(blob_client)
        if "PermissionDenied" in extracted_text:
            return jsonify({'error': 'Permission Denied to access the blob'}), 403
        
        update_user_data(last_file['_id'], extracted_text)
        
        # Konversi extracted_text ke daftar dokumen
        document = Document(page_content=extracted_text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )

        chunks = text_splitter.split_documents([document])
        
        chunks_content = [chunk.page_content for chunk in chunks]
       
        return jsonify({
            'message': 'OCR extraction successful',
            'extracted_text': extracted_text,
            'chunks': chunks_content
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    try:
        file_url, filename = upload_file_to_blob(file)
        
        # Simpan metadata file di MongoDB
        document = {
            "file_name": filename,
            "file_url": file_url,
            "upload_date": datetime.datetime.utcnow()
        }
        result = db.files.insert_one(document)

        user_data_document = {
            "file_name": filename,
            "upload_date": datetime.datetime.utcnow()
        }
        user_data_result = db.User_Data.insert_one(user_data_document)
        
        return jsonify({
            'message': 'File successfully uploaded',
            'file_url': file_url,
            'document_id': str(result.inserted_id),
            'user_data_id': str(user_data_result.inserted_id)
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# endpoint RAG
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
        blob_client = get_blob_client(file_name)
        extracted_text = ocr_image_from_blob(blob_client)
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

        if not query_text:
            return jsonify({'error': 'No query text provided'}), 400
       


        return jsonify({
            'message': 'OCR extraction, chunking, and search successful',
            'search_results': answer
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)