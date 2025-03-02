from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import ollama
import re
import json
import torch
from sentence_transformers import SentenceTransformer

app = FastAPI()

# CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define data models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


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


# Chat endpoint to handle form submissions and feedback
@app.post("/chat")
async def chat(request: ChatRequest):
    # Convert messages to a format compatible with Ollama
    message = request.messages[0].json()
    print(message)
    # message = {"role": message["role"], "content": message["Additional Info"]}

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

    # Convert input to pos and neg domains
    course_preferences = extract_course_preferences(message)

    user_query_pos = course_preferences["pos_domains"]
    user_query_neg = course_preferences["neg_domains"]

    print(user_query_pos)
    print(user_query_neg)

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

    user_preferences = f"Based on the user's interests in {', '.join(user_query_pos)}"
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

    # Call Ollama with streaming enabled (adjust based on actual Ollama API)
    stream = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": explanation_prompt}],
        stream=True,
    )

    # Generator function to stream response chunks
    def generate():
        for chunk in stream:
            # Assuming Ollama returns chunks with 'message' and 'content'
            yield chunk["message"]["content"]

    return StreamingResponse(generate(), media_type="text/plain")
