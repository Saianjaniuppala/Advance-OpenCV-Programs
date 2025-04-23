from PIL import Image
import pytesseract
from docx import Document
import os

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None

def save_text_to_word(text, output_path):
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = Document()
        doc.add_paragraph(text)
        doc.save(output_path)
        print(f"Text successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving Word document: {e}")
        print(f"Try saving to a different location like: {os.path.expanduser('~/Desktop')}")

def main():
    image_path = r"C:\Users\hi\OneDrive\Documents\VSgima\Chapter C\input_image.png"  # Change to your image path
    output_docx = os.path.expanduser("~/Desktop/output_text.docx")  # Now saves to Desktop
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    extracted_text = extract_text_from_image(image_path)
    
    if extracted_text:
        save_text_to_word(extracted_text, output_docx)
    else:
        print("No text was extracted from the image.")

if __name__ == "__main__":
    main()




