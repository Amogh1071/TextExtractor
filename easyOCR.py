import os
from easyocr import Reader

# Initialize EasyOCR reader for Spanish
reader = Reader(['es'])

# Specify your single image path
image_path = "Buendia-Instruccion_pdf_page_2_png.rf.3dfe8354b705915ab7602b31ca902f74.jpg"  # Replace with your actual image path

# Read text from the image
results = reader.readtext(image_path)

# Confidence threshold
threshold = 0.7

# Filter results based on confidence threshold
filtered_results = [text for _, text, conf in results if conf > threshold]

# Print results
print(f"Results for {os.path.basename(image_path)}:")
print("\n".join(filtered_results))
print("-" * 50)