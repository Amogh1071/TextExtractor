
---

## Overview of the Code

This Python script processes images to detect and extract text, then evaluates the accuracy of the extracted text against ground truth data. Here’s what it does at a high level:

1. **Text Detection**: Uses a YOLO model to identify text blocks in images.
2. **Text Extraction**: Employs Tesseract OCR to extract text from these blocks.
3. **Evaluation**: Compares the extracted text to ground truth text using word-level accuracy and Levenshtein similarity.
4. **Visualization**: Displays an example image with detected text blocks highlighted.

The script is designed to work with historical Spanish documents, which influences some of its design choices.

---

## Step-by-Step Explanation

### 1. Importing Libraries
The script begins by importing necessary libraries:
- `os`: For handling file paths and directories.
- `cv2` (OpenCV): For image loading, processing, and drawing bounding boxes.
- `pytesseract`: For OCR (text extraction from images).
- `numpy`: For numerical operations on arrays (e.g., bounding box coordinates).
- `pandas`: For structuring and saving evaluation results as CSV files.
- `matplotlib.pyplot`: For visualizing images with detected text blocks.
- `Levenshtein`: For calculating Levenshtein distance, a measure of string similarity.

These libraries provide the foundational tools for image processing, OCR, and evaluation.

### 2. Setting Up Paths and Variables
Paths are defined for:
- Input images (`IMAGE_DIR`)
- Cropped text blocks (`EXTRACTED_LAYOUTS_DIR`)
- Extracted text files (`EXTRACTED_TEXTS_DIR`)
- Ground truth text files (`TXT_DIR`)
- Evaluation results (`EVALUATION_CSV_PATH` and `NEW_CSV_PATH`)

A placeholder `yolo_model` (assumed to be a YOLO model from a library like Ultralytics) and a `results` dictionary are also initialized. The dictionary will store extracted and ground truth texts for each image.

### 3. Defining the `clean_historical_text` Function
```python
def clean_historical_text(text):
    if isinstance(text, list):
        text = "\n".join(text)
    return text.lower()  # Simplified example
```
- **Purpose**: This function normalizes text by converting it to lowercase (and could be extended to handle more complex cleaning).
- **Why**: Historical texts may have inconsistent capitalization or special characters (e.g., 'ſ' for 's' in old Spanish). Normalizing ensures fair comparison between extracted and ground truth texts.

### 4. Processing Images
The script loops through image files (`.jpg` or `.png`) in `IMAGE_DIR`:
- **Loading**: Images are loaded with OpenCV and converted from BGR to RGB (since OpenCV uses BGR by default, but other tools like Matplotlib expect RGB).
- **Text Block Detection**: The YOLO model detects text blocks, returning bounding box coordinates (x1, y1, x2, y2). These are sorted by y-coordinate (top-to-bottom) to mimic reading order.

### 5. Extracting Text with Tesseract
For each detected text block:
- **Cropping**: The region is cropped from the image and saved as a PNG file (lossless format for better OCR quality).
- **Preprocessing**: The cropped image is resized (doubled in size) and converted to grayscale to enhance Tesseract’s accuracy.
- **OCR**: Tesseract extracts text with the configuration `--oem 3 --psm 3 -l spa`:
  - `--oem 3`: Default OCR engine mode.
  - `--psm 3`: Assumes a single uniform text block (suitable for document snippets).
  - `-l spa`: Uses the Spanish language model (`spa.traineddata`).

Extracted texts are saved to a single `.txt` file per image and cleaned using `clean_historical_text`.

### 6. Loading Ground Truth
For each image, the corresponding ground truth text file is loaded from `TXT_DIR`, cleaned, and stored in the `results` dictionary alongside the extracted text.

---

## Evaluation Process

### 7. Word-Level Accuracy with Partial Matching
- **How**: For each extracted word, the script finds the closest ground truth word using Levenshtein distance. If the distance is ≤ 2 (allowing minor errors like typos), it’s a match.
- **Calculation**: Accuracy = (number of matched words / total extracted words) × 100.
- **Why**: This measures how well individual words are extracted, tolerating small OCR errors common in historical texts (e.g., 'hola' vs. 'holá').

### 8. Levenshtein Similarity
- **How**: The full extracted text and ground truth text are compared using Levenshtein distance (number of edits needed to transform one into the other). Similarity is calculated as:
  ```python
  lev_similarity = 1 - (lev_distance / max_length)
  ```
  where `max_length` is the length of the longer text.
- **Why**: This gives a holistic view of text similarity, accounting for insertions, deletions, and substitutions across the entire text.

### 9. Determining Correctness
- **Rule**: If word accuracy ≥ 50%, the image is marked "Correct"; otherwise, "Incorrect."
- **Why**: A 50% threshold balances strictness and leniency, suitable for initial evaluation (adjustable based on needs).

### 10. Storing and Saving Results
Evaluation details (status, accuracy, similarity, etc.) are stored in a list, converted to a `pandas` DataFrame, and saved to two CSV files: `evaluation.csv` and `combined_results.csv`.

---

## Visualization

For the first image, the script:
- Runs YOLO to detect text blocks.
- Draws red bounding boxes around them.
- Displays the annotated image using Matplotlib.

This helps verify visually whether the text blocks were detected correctly.

---

## Answering Your Specific Questions

### Why Use `spa.traineddata`?
- **What**: The `-l spa` flag tells Tesseract to use the Spanish language model (`spa.traineddata`).
- **Why**: This script targets Spanish text, likely historical documents. The Spanish model is trained to recognize:
  - Spanish-specific characters (e.g., 'ñ', 'á', 'é', 'í', 'ó', 'ú').
  - Linguistic patterns of Spanish words.
  Using `spa.traineddata` improves OCR accuracy for Spanish compared to the default English model (`eng.traineddata`), especially for accented characters or historical variants.

### How Are We Evaluating the Model?
The evaluation combines two metrics:
1. **Word-Level Accuracy**:
   - **Process**: Compares each extracted word to ground truth words, allowing a Levenshtein distance of ≤ 2 for partial matches.
   - **Purpose**: Assesses precision at the word level, accommodating minor OCR errors (e.g., 'casa' vs. 'caza').
   - **Output**: Percentage of correctly extracted words.

2. **Levenshtein Similarity**:
   - **Process**: Measures the edit distance between the full extracted and ground truth texts, normalized to a similarity score (0 to 1).
   - **Purpose**: Evaluates overall text similarity, capturing broader errors like missing or extra phrases.
   - **Output**: A score where 1 is identical, and 0 is completely dissimilar.

- **Why These Metrics?**: Together, they provide a balanced evaluation:
  - Word accuracy focuses on individual word correctness.
  - Levenshtein similarity assesses the entire text, useful for detecting structural issues.
  - The 50% threshold for "Correct" status is a practical benchmark for usability.

### Why Are We Replacing Some Text (Encoding Issues)?
- **What**: The `clean_historical_text` function (currently just lowercase conversion) is a placeholder for handling encoding or character issues.
- **Why**: Historical Spanish texts may include:
  - Non-standard characters (e.g., 'ſ' [long s] instead of 's').
  - OCR misreads (e.g., 'á' as 'a' or garbled symbols due to encoding mismatches).
  - Inconsistent formatting (e.g., uppercase/lowercase variations).
- **Purpose**: Replacing or normalizing these ensures the extracted text aligns with the ground truth for fair comparison. For example:
  - Converting 'ſ' to 's' avoids false mismatches.
  - Using UTF-8 encoding (as in file operations) prevents garbled output from special characters.
- **Current State**: The function is simplified, but it could be expanded to replace specific characters or fix encoding artifacts based on the dataset.

---

## Summary

This code:
- **Detects** text blocks with YOLO.
- **Extracts** text using Tesseract with `spa.traineddata` for Spanish accuracy.
- **Evaluates** performance with word accuracy and Levenshtein similarity.
- **Handles Text**: Cleans text to address historical or encoding quirks.
- **Visualizes** results for verification.

The use of `spa.traineddata` optimizes for Spanish text, the dual-metric evaluation ensures robustness, and text cleaning mitigates encoding challenges, making this script well-suited for processing and analyzing historical Spanish documents.

### Accuracy

#### The overall accuracy of the model id 81.25%
#### Average Word-Level = 91.00515557 % 
#### Average Levenshtein Similarity = 0.704979

## Overview of the Evaluation Process

The evaluation section of your code measures how well the extracted text matches the ground truth text for each image. It uses two metrics:
1. **Word-Level Accuracy**: Measures the percentage of extracted words that match ground truth words, allowing for small differences (using Levenshtein distance).
2. **Levenshtein Similarity**: Measures the overall similarity between the full extracted text and ground truth text.

The **accuracy** of the model (reported as a percentage at the end) is based on the number of images where the word-level accuracy exceeds a threshold (50%). Let’s focus on how this word-level accuracy is calculated, as it directly influences the overall accuracy.

---

## Step-by-Step Explanation of Accuracy Calculation

### 1. **Extracted and Ground Truth Words**
For each image, the script has:
- `extracted_words`: A list of words from the OCR output, after cleaning with `clean_historical_text`.
- `ground_truth`: A list of words from the ground truth text, also cleaned.

These lists are stored in the `results` dictionary for each image:
```python
extracted_words = data.get('extracted_text', [])
ground_truth = data.get('ground_truth', [])
```

### 2. **Word-Level Accuracy with Partial Matching**
The script calculates word-level accuracy by comparing each extracted word to the ground truth words, allowing for small differences using the Levenshtein distance.


- **Filter Empty Words**:
  - `total_extracted = len([w for w in extracted_words if w])`: Counts non-empty words in `extracted_words`. Empty strings (e.g., from OCR errors) are excluded to avoid skewing the count.
  - **Why**: Ensures only meaningful words contribute to the accuracy calculation.


- **Find Best Match**:
  - `best_match = min(ground_truth, key=lambda gt: levenshtein_distance(word, gt), default="")`: Finds the ground truth word with the smallest Levenshtein distance to the extracted word.
  - **Levenshtein Distance**: Measures the minimum number of single-character edits (insertions, deletions, substitutions) needed to transform one word into another. For example, `levenshtein_distance("cat", "hat") = 1` (one substitution: c → h).
  - **Why**: Allows partial matching, so small OCR errors (e.g., "hola" vs. "holá") are still considered correct.

- **Determine Correctness**:
  - `if best_match and levenshtein_distance(word, best_match) <= 2`: A word is considered correct if its best match in the ground truth has a Levenshtein distance of 2 or less.
  - **Threshold of 2**: Allows for minor errors (e.g., one or two character differences). For example:
    - "hola" vs. "holá" (distance = 1, correct).
    
  - **Why**: Historical texts and OCR often introduce small errors (e.g., accented characters, misread letters). A threshold of 2 balances leniency and accuracy.
  - If correct, `correct_words += 1`; otherwise, the word is added to `missing_words` for reporting.







## Role of Levenshtein Similarity

- **Purpose**: Measures overall text similarity, complementing word-level accuracy by capturing structural differences (e.g., missing or extra phrases).
- **Why Not Used in Accuracy?**: It’s a broader metric, less granular than word-level accuracy, and is included in the evaluation report for context.
