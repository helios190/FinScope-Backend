from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sys
import portofolio_calculation
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

import datetime
import os
from ocr_handle.database import get_db, update_user_data
from ocr_handle.ocr import ocr_image_from_blob
from ocr_handle.storage import upload_file_to_blob, get_blob_client


load_dotenv()


app = Flask(__name__)
CORS(app)

db = get_db()

@app.route('/calculates', methods=['POST'])
def calculate():
    global calculation_results
    stock_symbols = request.form.getlist('stock_symbols')
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

    # Store the results in the global variable
    calculation_results = {
        "min_var_portfolios": min_var_portfolios,
        "max_sharpe_weights": list(max_sharpe_weights),
        'alpha_beta_dict': alpha_beta_dict
    }
    min_var_portfolios = calculation_results['min_var_portfolios']
    max_sharpe_weights = calculation_results['max_sharpe_weights']
    alpha_beta_dict = calculation_results['alpha_beta_dict']
    
    # Get embedding model
    load_dotenv()
    os.getenv('OPENAI_API_KEY')

    prompt_content = f''' 
    I have this result of calculation Minimum Variance Portofolios {min_var_portfolios}, max_sharpe_weights of {max_sharpe_weights}, and alpha and beta of {alpha_beta_dict}
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


@app.route('/ocr_extract', methods=['POST'])
def ocr_extract():
    try:
        last_file = db.User_Data.find_one(sort=[('_id', -1)])
        
        if not last_file or 'file_name' not in last_file:
            return jsonify({'error': 'No file found in the database'}), 400
        
        file_name = last_file['file_name']
        blob_client = get_blob_client(file_name)
        
        extracted_text = ocr_image_from_blob(blob_client)
        update_user_data(last_file['_id'], extracted_text)
        
        # Di chunk dulu
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        # split documents into text and embeddings

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
        )

        chunks = text_splitter.split_documents(extracted_text)
       
        return jsonify({
            'message': 'OCR extraction successful',
            'extracted_text': extracted_text
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)