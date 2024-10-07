from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
import os
import io
import json
import time
import re
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes
from google.cloud import vision
import requests
from PIL import Image, ImageDraw, ImageFont
import ollama

alpaca_prompt = """    You are an expert in detecting specific information from a block of text.
    The input text is extracted from OCR and may contain various types of information.
    We have a text dataset containing information about various shops. Each entry includes a shop name, address, and telephone number. The information may be split across multiple lines and in various formats.
    We need to detect and exctrat those shop name, address and telephone number in each given image.

### Input Text:
{}

### Output :
{}
"""
# text = """Dialog
#                 Sri Lankawe #1 saha vegavathma jalaya
#                 Anagataya adayi.
#                 S
#                 Ruvan Grocery
#                 No.45, Golahala, Kegalle.
#                 Google
#                 L Image capture: Dec 2021
#                 © 2024 Google
#                 Glow&
#                 Sri Lanka Terms Privacy Report a problem"""


# res = ollama.chat(
#         model='SB_unslothModel',  # Use your SQL query generator model
#         messages=[
#             {'role': 'system', 'content': alpaca_prompt},  # Instruction
#             {'role': 'user', 'content': text}
#         ]
#     )
# print (res['message']['content'])

# Set your Google Cloud Vision API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\DS_projects\Models\ShopBoard_detection_models\SB_gguf_unsloth_model\Google_vision_API.json'

# Get Google Ocr text -------------------------------------------------------------------------------------------------------------------------
def detect_text(path):
    
    try:
        # Initialize the client
        client = vision.ImageAnnotatorClient()

        # Read the image content
        with open(path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Perform text detection (document_text_detection for dense text)
        response = client.document_text_detection(image=image)

        # Split the full text into lines
        if response.full_text_annotation:
            texts = response.full_text_annotation.text.split('\n')
        else:
            texts = []

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        return '\n'.join(texts)
    except Exception as e:
        print(f"Error during text detection: {e}")
        return []
    
# Function to transliterate Sinhala text into English letters using Google Gemini Model
def transliterate_sinhala_with_gemini(text_result, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt[0], text_result])
    return response.text
    # print(response.text)

# Define the prompt for transliteration
prompt = [
    """
    You are an expert in transliterating Sinhala text into English letters.
    The input text is in Sinhala and may contain names, addresses, and other details.

    Some examples given below:

    Example 1:
        - Input:
                ogle
                Sri Lanka
                Telecom
                Mooner
                ඔබ සමඟයි. සැබේ.
                දිදුල
                ස්ටෝර්ස්
                තලල්ල, ගන්දර,
                දුරකථන: 041-3407418
        - Output:
                Ogle
                Sri Lanka
                Telecom
                Mooner
                Oba samangayi
                Sabe
                Didula
                Stores
                Talalla, Gandara,
                Durakathana: 041-3407418


    Example 2:
        - Input:
                bටවල් 45 ක විs දි....අපේ b... .....
                CBL
                Munchee
                රස ලොව රජ කිරුළ
                SUPER
                CREAM
                CRACKER
                Minchre
                283059
                ඇබර්
                රුහුණුසිරි ආපනශාලාව විමුව, කෙවිටිපොළ
                Thilakarathne grocery
        - Output:
                CBL
                Munchee
                Ras lova raja kirula
                Super
                Cream
                Cracker
                Minchre
                283059
                Aber
                Ruhunusiri apanasalawa vimuva, kevitipola
                Thilakarathne grocery


    Example 3:
        - Input:
                Ath
                ශ්‍රී ලංකාවේ #1 සහ
                වේගවත්ම ජාලය
                Dialog
                අනාගතය අදයි.
                S
                රුවන් ග්‍රොසරි
                නො.45, ගොලහල, කෑගල්ල.
                >
                Google
                L lmage capture: Dec 2021
                © 2024 Google
                Glow&
                Sri Lanka Terms Privacy Report a problem
        - output:
                Dialog
                Sri Lankawe #1 saha vegavathma jalaya
                Anagataya adayi.
                S
                Ruvan Grocery
                No.45, Golahala, Kegalle.
                Google
                L Image capture: Dec 2021
                © 2024 Google
                Glow&
                Sri Lanka Terms Privacy Report a problem


    The task is to transliterate the provided Sinhala text into English letters, preserving the pronunciation and meaning as accurately as possible.

    Input: [Input]
    Output: [Output]
    """
]

# def contains_sinhala_phrases(text):
    
#     # List of Sinhala phrases to check
#     phrases = [
#         'ට්‍රේඩ් සෙන්ටර්',
#         'ස්ටෝරිස්',
#         'සුපර්',
#         'අලෙවි හල',
#         'ග්‍රොසරි',
#         'ආපනශාලාව',
#         'හෝටලය',
#         'ස්ටෝර්ස්',
#         'ස්ටොර්ස්',
#         'සුපර් සිටි'
#     ]

#     # Combine phrases into a single regex pattern for efficient searching
#     pattern = '|'.join(map(re.escape, phrases))

#     # Search for the pattern in the given text
#     match = re.search(pattern, text)

#     # Return True if any phrase is found, else False
#     return match is not None



def get_ocr_text(file_path):
    # Azure API configuration
    API_KEY = os.getenv("AZURE_API_KEY")  # Ensure the API key is stored in .env file
    ENDPOINT = os.getenv("AZURE_ENDPOINT")  # Ensure the endpoint is stored in .env file

    if not API_KEY or not ENDPOINT:
        raise ValueError("API Key or Endpoint is missing. Check the .env file.")

    # Create a client
    cv_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

    # Read the image from the file path and call the API
    with open(file_path, 'rb') as image:
        response = cv_client.read_in_stream(image, language='en', raw=True)

    # Extract the operation ID
    operationLocation = response.headers['Operation-Location']
    operation_id = operationLocation.split('/')[-1]

    # Wait for the result
    time.sleep(5)

    # Get the result from the API
    result = cv_client.get_read_result(operation_id)

    # Extract and return the detected text
    final_output = ""
    if result.status == OperationStatusCodes.succeeded:
        read_results = result.analyze_result.read_results
        for analyzed_result in read_results:
            for line in analyzed_result.lines:
                final_output += line.text + "\n"

    return final_output

# def get_shop_details(text):
#     res = ollama.chat(
#         model='SB_unslothModel',  # Use your SQL query generator model
#         messages=[
#             {'role': 'system', 'content': alpaca_prompt},  # Instruction
#             {'role': 'user', 'content': text}
#         ]
#     )
#     return res['message']['content']


# # Streamlit app
# st.title("Shop Details Extractor")

# # Image uploader widget
# uploaded_file = st.file_uploader("Upload an image of the shop board", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Shop Board Image", use_column_width=True)

#     # Save the uploaded image to a temporary path
#     temp_image_path = f"temp_{uploaded_file.name}"
#     image.save(temp_image_path)

#     azure_ocr_text = get_ocr_text(temp_image_path)
#     google_ocr_text = detect_text(temp_image_path)
#     translate_ocr_test = transliterate_sinhala_with_gemini(google_ocr_text, prompt)

#     if contains_sinhala_phrases(google_ocr_text):
#         # Transliterate Sinhala text using Gemini
#         ocr_text = translate_ocr_test
#     else:
#         ocr_text = azure_ocr_text

#     st.write(google_ocr_text)

