import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the JSON file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Compute cosine similarity matrix
def compute_similarity(courses):
    course_codes = [course["course_code"] for course in courses]
    embeddings = np.array([course["embedding"] for course in courses])  # Convert to NumPy array
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Store results in a dictionary
    similarity_results = {}
    for i, course_code in enumerate(course_codes):
        similar_courses = {
            course_codes[j]: similarity_matrix[i][j]
            for j in range(len(course_codes)) if i != j  # Exclude self-similarity
        }
        # Sort similar courses by similarity score in descending order
        similarity_results[course_code] = sorted(similar_courses.items(), key=lambda x: x[1], reverse=True)
    
    return similarity_results

# Save results to JSON
def save_to_json(results, output_file="course_similarity.json"):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

# Main function
def main():
    file_path = "./data/course_descr/course_embed.json"  # Update path if needed
    data = load_json(file_path)

    # Compute similarity
    similarity_results = compute_similarity(data)

    # Save results
    save_to_json(similarity_results)
    print(f"Similarity results saved to 'course_similarity.json'")

if __name__ == "__main__":
    main()
