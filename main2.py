# streamlit_easyocr_fixed.py
# pip install streamlit easyocr pillow numpy
import os
from dotenv import load_dotenv
import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import io
from typing import Union, List
import cv2
# LangChain and related imports

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
from pydantic import BaseModel, Field
from typing import Dict, TypedDict, List, Union, Annotated, Sequence


def object_creator(text, model):
    print("rECIEVED TEXT", text)
    from pydantic import BaseModel, Field
    
    # ✅ Correct - Use BaseModel for structured output
    class Schema(BaseModel):
        hospital_name: str = Field(description="From the text, find out the name of the hospital")
        date: str = Field(description="From the text, find out the date and time")
        patient_name: str = Field(description="From the text, find out the name of the patient")
        doctor_name: str = Field(description="From the text, find out the name of the doctor")
        doctor_speciality: str = Field(description="From the text, find out the specialty of the doctor")
        assessment: str = Field(description="From the text, find out the assessment")
        diagnosis: str = Field(description="From the text, find out the diagnosis")
        prescription: str = Field(description="From the text, find out the prescription")
        others: str = Field(description="From the text, find out any other information")

    llm_with_schema = model.with_structured_output(Schema)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical transcriptionist. Extract information from medical documents and return structured data. If information is not available, use 'Not specified'."),
        ("human", "Extract medical information from this text: {text}")
    ])
    
    chain = prompt | llm_with_schema
    
    try:
        # ✅ No need to escape - let LangChain handle it
        result = chain.invoke({"text": text})
        print("Result is ", result)
        return result
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return None















# Global reader
reader = easyocr.Reader(['en'], gpu=False)

def read_text_from_image(
    image_input: Union[str, bytes, np.ndarray, Image.Image],
    min_confidence: float = 0.5
) -> List[str]:
    """
    Extract text from an image using EasyOCR.

    Args:
        image_input: File path, image bytes, OpenCV/numpy image, or PIL image.
        min_confidence: Minimum confidence for returned texts.

    Returns:
        List of recognized text strings.
    """
    
    # Load the image based on the input type
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    elif isinstance(image_input, bytes):
        image = np.array(Image.open(io.BytesIO(image_input)))
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise ValueError("Unsupported image input type.")

    # Run OCR
    results = reader.readtext(image)

    # Filter results based on confidence
    texts = [text for _, text, conf in results if conf >= min_confidence]

    return texts





# Streamlit UI
st.title("🔍 OCR App for Medical Reports")
st.write("This OCR extracts hospital_name, date, patient_name, doctor_name, doctor_speciality, assessment, diagnosis, prescription, others from a medical report.")

# API Key Input Section
st.sidebar.header("🔑 API Configuration")
api_key = st.sidebar.text_input(
    "Enter your API Key:", 
    type="password",
    placeholder="gsk_...",
)

# Initialize model only if API key is provided
if api_key:
    try:
        # Update the model with the provided API key
        model = ChatGroq(
            model="llama-3.1-8b-instant", 
            groq_api_key=api_key, 
            temperature=0.7
        )
        st.sidebar.success("✅ API Key configured successfully!")
        
        # File upload section (only show if API key is provided)
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)

            if st.button("Extract Text"):
                with st.spinner("Extracting text..."):
                    # print(uploaded.getvalue())
                    with open("aaa.png", "wb") as f:
                        f.write(uploaded.getvalue())
                    extracted = read_text_from_image("aaa.png")
                
                if extracted:
                    st.success(f"✅ Found {len(extracted)} text blocks:")
                    
                    # Display extracted text blocks
                    with st.expander("📝 View Extracted Text Blocks", expanded=False):
                        for i, text in enumerate(extracted, 1):
                            st.write(f"{i}. {text}")
                    
                    # Process with AI
                    text_to_be_sent = " ".join(extracted)
                    print("EXTRACTED IS \n", type(text_to_be_sent),"\n\n\n\n\n")
                    
                    with st.spinner("🤖 Processing with AI..."):
                        try:
                            final_object = object_creator(text_to_be_sent, model)
                            print("OBJECT IS\n", final_object, "\n\n\n\n\n")
                            print("OBJECT IS\n", type(final_object), "\n\n\n\n\n")
                            
                            # Display structured results
                            st.subheader("📋 Extracted Medical Information")
                            
                            # Display as JSON
                            # if hasattr(final_object, 'dict'):
                            #     st.json(final_object.dict())
                            # else:
                            #     st.write(final_object)
                            st.write(final_object)
                                
                            # Clean up temp file
                            # try:
                            #     os.remove("aaa.png")
                            # except:
                            #     pass
                                
                        except Exception as e:
                            st.error(f"❌ Error processing with AI: {str(e)}")
                            st.info("Please check your API key and try again.")
                
                else:
                    st.warning("⚠️ No text found in the image.")
                    
    except Exception as e:
        st.sidebar.error("❌ Invalid API Key!")
        st.error("Please enter a valid API key to continue.")
        
else:
    # Show instructions when no API key is provided
    st.info("👆 Please enter your API key in the sidebar to get started.")
    st.markdown("""
    
    ### Features:
    - Hospital name extraction
    - Date and time detection
    - Patient information
    - Doctor details and specialty
    - Medical assessment
    - Diagnosis and prescription
    - Additional information
                
    ##### Made with ❤️ by Priyesh
    """)
