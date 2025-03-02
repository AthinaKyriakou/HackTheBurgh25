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

positive_sentiments = "Interesting Exciting Engaging Fascinating Insightful Helpful Useful Valuable Enjoyable Informative Relevant Practical Well-explained Easy-to-understand Clear Comprehensive Thought-provoking Innovative Fun Challenging Rewarding Inspiring Eye-opening Must-learn Beneficial Effective Enlightening Worthwhile Great Awesome Fantastic love like adore"


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
        combined_text = summary + "\n" + course_description + positive_sentiments
        course_data.append({"course_code": course_code, "text": combined_text})
    return course_data

# Transformer-based keyword extraction with phrases
def create_embed_transformer(text, num_keywords=20):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text, convert_to_numpy=True)

# Save results to JSON file
def save_to_json(results, output_file="./data/course_embed.json"):
    # Convert numpy arrays to lists for JSON serialization
    for result in results:
        result["embedding"] = result["embedding"].tolist()  # Convert ndarray to list
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

def main():
    # Input file path
    file_path = "./data/course_info.json"
    data = load_json(file_path)
    
    # Extract course codes and texts
    course_data = extract_text(data)
    
    # List to store results
    results = []
    
    # Process each course
    for i, course in enumerate(course_data):
        course_code = course["course_code"]
        text = course["text"]
        
        print(f"\nExtracting embeddings for course {i+1} ({course_code})...\n")
        embedding = create_embed_transformer(text)
        #print(f"Top Keywords for course {i+1} ({course_code}):")
        #print(keywords)
        
        # Add to results as a dictionary
        results.append({
            "course_code": course_code,
            "embedding": embedding
        })
    
    # Save results to JSON file
    save_to_json(results, "./data/with_sents/course_embed_pos_sent.json")

if __name__ == "__main__":
    main()