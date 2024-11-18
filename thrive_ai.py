import streamlit as st
from PIL import Image
import cohere

# Initialize TrOCR Model and Processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Initialize Cohere API Client
COHERE_API_KEY = "BNZbH4qfEYA9ot4A4ZZcd2rTcgZxhuYEifpo6abh"  # Replace with your Cohere API key
co = cohere.Client(COHERE_API_KEY)

# Title and Description
st.title("Thrive.ai Prototype")
st.write("Enhance your journaling experience with AI-powered insights. Type your entry or upload a picture of your handwritten journal.")

# User Input Section
input_method = st.radio("How would you like to input your journal entry?", ["Type", "Upload Image"])

if input_method == "Type":
    journal_text = st.text_area("Write your journal entry here:")
elif input_method == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image of your journal entry", type=["jpg", "jpeg", "png"])
    journal_text = ""
    if uploaded_image is not None:
        # Perform OCR on the uploaded image using TrOCR
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting text from the image..."):
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            journal_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        st.write("Extracted Text:")
        st.write(journal_text)

# Process the Input if Available
if journal_text:
    st.subheader("AI Insights and Thought-Provoking Questions")

    if st.button("Analyze Journal Entry"):
        # Generate Thought-Provoking Questions using Cohere API
        try:
            response = co.generate(
                model="command-xlarge-nightly",
                prompt=f"Analyze this journal entry and provide three thought-provoking questions and mental health tips. Journal entry: {journal_text}",
                max_tokens=300,
                temperature=0.7,
            )
            st.write("AI-Generated Questions and Insights:")
            st.write(response.generations[0].text)
        except Exception as e:
            st.error(f"An error occurred: {e}")
