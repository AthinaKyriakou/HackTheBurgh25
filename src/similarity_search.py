import ollama
from sentence_transformers import SentenceTransformer
import torch
import json
import re

# with open(
#     "/home/gokul/gigs/hacktheburgh/data/course_desc/informatics_course_info.json",
#     "r",
# ) as file:
#     course_info = json.load(file)

# embedder = SentenceTransformer("all-mpnet-base-v2")

# course_summaries = [course["course_summary"] for course in course_info]
# course_desc = [course["course_desc"] for course in course_info]
# course_learning_outcomes = [course["learning_outcomes"] for course in course_info]

# course_summ_embed = embedder.encode(course_summaries, convert_to_tensor=True)
# course_desc_embed = embedder.encode(course_desc, convert_to_tensor=True)
# course_learning_outcomes_embed = embedder.encode(
#     course_learning_outcomes, convert_to_tensor=True
# )

# torch.save(
#     course_summ_embed,
#     "/home/gokul/gigs/hacktheburgh/data/course_desc/course_summ_embed.pt",
# )
# torch.save(
#     course_desc_embed,
#     "/home/gokul/gigs/hacktheburgh/data/course_desc/course_desc_embed.pt",
# )
# torch.save(
#     course_learning_outcomes_embed,
#     "/home/gokul/gigs/hacktheburgh/data/course_desc/course_learning_outcomes_embed.pt",
# )


def extract_course_preferences(sentence):
    prompt = f"""
    Extract preferences of domains/courses from the following sentence:

    "{sentence}"

    Don't provide the python code to do this. Don't infer anything from external knowledge. Directly output a JSON object with two keys: 'pos_domains' and 'neg_domains'.
    'pos_domains' should be a list of domains/courses the user likes.
    'neg_domains' should be a list of domains/courses the user dislikes.
    Make the decision solely based on the input sentence.

    Example Output:
    {{
        "pos_domains": [<list of domains/courses the user likes>],
        "neg_domains": [<list of domains/courses the user dislikes>]
    }}
    """

    response = ollama.chat(
        model="llama3.1", messages=[{"role": "user", "content": prompt}]
    )

    # Extract JSON from response
    match = re.search(r"\{.*\}", response["message"]["content"], re.DOTALL)
    if match:
        json_response = match.group(0)
        try:
            parsed_data = json.loads(json_response)
            return parsed_data
        except json.JSONDecodeError:
            return {"pos_domains": [], "neg_domains": []}

    return {"pos_domains": [], "neg_domains": []}


if __name__ == "__main__":
    course_summ_embed = torch.load(
        "/home/gokul/gigs/hacktheburgh/data/course_desc/course_summ_embed.pt"
    )
    course_desc_embed = torch.load(
        "/home/gokul/gigs/hacktheburgh/data/course_desc/course_desc_embed.pt"
    )
    course_learning_outcomes_embed = torch.load(
        "/home/gokul/gigs/hacktheburgh/data/course_desc/course_learning_outcomes_embed.pt"
    )
    embedder = SentenceTransformer("all-mpnet-base-v2")

    with open(
        "/home/gokul/gigs/hacktheburgh/data/course_desc/informatics_course_info.json",
        "r",
    ) as file:
        course_info = json.load(file)

    test_cases = [
        # "I like machine learning and I don't like reinforcement learning.",
        # "I enjoy deep learning but I dislike statistics.",
        # "My favorite subjects are computer vision and NLP, but I find mathematics boring.",
        # "I don't enjoy ethics in AI, but I like big data analytics.",
        # "I love robotics, however, I am not a fan of Bayesian statistics.",
        # "The subjects I like include data science and AI, while I don't like computational geometry.",
        # "I'm interested in information retrieval but I hate mathematical optimization.",
        # "I find evolutionary algorithms fascinating but not stochastic processes.",
        # "I don't really enjoy data engineering, but I do like knowledge graphs.",
        # "Cybersecurity is something I love, but I can't stand software engineering.",
        "I wanna explore courses in the intersection of machine learning and quantum computing",
    ]

    for test_case in test_cases:
        course_preferences = extract_course_preferences(test_case)

        user_query_pos = course_preferences["pos_domains"]
        user_query_neg = course_preferences["neg_domains"]

        if len(user_query_pos) == 0:
            pos_embedding = torch.zeros(768, device="cuda")
        for idx, query in enumerate(user_query_pos):

            if idx == 0:
                pos_embedding = embedder.encode(query, convert_to_tensor=True)
            else:
                pos_embedding += embedder.encode(query, convert_to_tensor=True)

        if len(user_query_neg) == 0:
            neg_embedding = torch.zeros(768, device="cuda")
        for idx, query in enumerate(user_query_neg):
            if idx == 0:
                neg_embedding = embedder.encode(query, convert_to_tensor=True)
            else:
                neg_embedding += embedder.encode(query, convert_to_tensor=True)

        user_query_embedding = pos_embedding - neg_embedding

        course_summ_similarity = torch.cosine_similarity(
            course_summ_embed, user_query_embedding, dim=1
        )

        course_desc_similarity = torch.cosine_similarity(
            course_desc_embed, user_query_embedding, dim=1
        )

        course_learning_outcomes_similarity = torch.cosine_similarity(
            course_learning_outcomes_embed, user_query_embedding, dim=1
        )

        similarity_scores = (
            course_summ_similarity
            + course_desc_similarity
            + course_learning_outcomes_similarity
        )

        top_5_courses = torch.topk(similarity_scores, 5).indices

        # print(test_case)
        # i = 0
        # for idx in top_5_courses:
        #     print(f"Reco - {i}")
        #     print(course_info[idx]["course_code"])
        #     print(course_info[idx]["course_title"])
        #     print("\n")
        #     i += 1

        selected_courses = [
            {
                "title": course_info[idx]["course_title"],
                "summary": course_info[idx]["course_summary"],
                "description": course_info[idx]["course_desc"],
                "learning_outcomes": course_info[idx]["learning_outcomes"],
                "similarity_score": similarity_scores[idx].item(),
            }
            for idx in top_5_courses
        ]

        user_preferences = (
            f"Based on the user's interests in {', '.join(user_query_pos)}"
        )
        if user_query_neg:
            user_preferences += f" and dislikes in {', '.join(user_query_neg)}"

        explanation_prompt = f"""
        {user_preferences}, the following courses were recommended:
        
        {json.dumps(selected_courses, indent=4)}

        For each course, first display its course code and title, and then explain why it was chosen based on the user's preferences.
        Display all of the courses provide above.
        Do not display similarity scores.
        Do not say anything apart from what is mentioned above.
        """

        explanation_response = ollama.chat(
            model="llama3.1", messages=[{"role": "user", "content": explanation_prompt}]
        )

        print("Recommended Courses Explanation:")
        print(explanation_response["message"]["content"])
