import json
import spacy
import yake
from sentence_transformers import SentenceTransformer
import numpy as np
import re

nlp = spacy.load("en_core_web_sm")

SHORTCUTS = {
    'e.g.', 'i.e.', 'etc.', 'cf.', 'vs.', 'viz.', 'nb.',
    'eg', 'ie', 'etc', 'cf', 'vs', 'viz', 'nb'  # Without punctuation
}

# Load JSON file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Postprocess text to remove shortcuts
def postprocess_text(keywords):
    filtered = []
    for kw in keywords:
        # Split the keyword into words
        words = kw.split()
        # Remove any word that matches a shortcut
        cleaned_words = [word for word in words if word.lower() not in SHORTCUTS]
        # Rejoin if there’s anything left
        if cleaned_words:
            filtered.append(" ".join(cleaned_words))
    return filtered

# Extract text and course_code
def extract_text(data):
    course_data = []
    for course in data:
        course_code = course.get("course_code", "Unknown")  # Default to "Unknown" if not found
        summary = course.get("course_summary", "")
        course_description = course.get("course_desc", "")
        combined_text = summary + "\n" + course_description
        course_data.append({"course_code": course_code, "text": combined_text})
    return course_data

# Transformer-based keyword extraction with phrases
def extract_keywords_transformer(text, num_keywords=20):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc = nlp(text)
    phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
    phrases = list(set([preprocess_text(phrase) for phrase in phrases if phrase.lower() not in nlp.Defaults.stop_words]))
    if not phrases:
        return []
    embeddings = model.encode(phrases)
    text_embedding = model.encode(text)
    similarities = np.inner(embeddings, text_embedding)
    ranked_indices = np.argsort(similarities)[::-1]
    keywords = [phrases[i] for i in ranked_indices[:num_keywords]]
    # Filter out shortcuts
    filtered_keywords = postprocess_text(keywords)
    # Return up to num_keywords after filtering
    return filtered_keywords[:num_keywords]

# Save results to JSON file
def save_to_json(results, output_file="./data/course_keywords.json"):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

def main():
    # Input file path
    file_path = "./data/course_descr/course_info.json"
    data = load_json(file_path)
    
    # Extract course codes and texts
    course_data = extract_text(data)
    
    # List to store results
    results = []
    
    # Process each course
    for i, course in enumerate(course_data):
        course_code = course["course_code"]
        text = course["text"]
        
        print(f"\nExtracting keywords for course {i+1} ({course_code})...\n")
        keywords = extract_keywords_transformer(text)
        print(f"Top Keywords for course {i+1} ({course_code}):")
        print(keywords)
        
        # Add to results as a dictionary
        results.append({
            "course_code": course_code,
            "keywords": keywords
        })
    
    # Save results to JSON file
    save_to_json(results)
    print(f"\nResults saved to 'results.json'")

if __name__ == "__main__":
    main()