from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import ollama
import re
import json
import torch
from sentence_transformers import SentenceTransformer
import pickle

app = FastAPI()

ollama_input_model_name = "llama3.1"
ollama_final_model_name = "deepseek-r1:1.5b"

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
    Extract preferences of the user from the following sentence:

    "{sentence}"

    Don't provide the python code to do this. Don't infer anything from external knowledge. Directly output a JSON object with two keys: 'pos_preferences' and 'neg_preferences'.
    'pos_preferences' should be a list of domains/courses the user likes and any other positive preference condition (phrase) the user specifies.
    'neg_preferences' should be a list of domains/courses the user dislikes and any other negative preference condition the user specifies.
    Make the decision solely based on the input sentence.

    Example Output:
    {{
        "pos_preferences": [<list of domains/courses the user likes and any other positive preference condition (phrase) the user specifies>],
        "neg_preferences": [<list of domains/courses the user dislikes and any other negative preference condition the user specifies.>]
    }}
    """

    response = ollama.chat(
        model=ollama_input_model_name, messages=[{"role": "user", "content": prompt}]
    )

    # Extract JSON from response
    match = re.search(r"\{.*\}", response["message"]["content"], re.DOTALL)
    if match:
        json_response = match.group(0)
        try:
            parsed_data = json.loads(json_response)
            return parsed_data
        except json.JSONDecodeError:
            return {"pos_preferences": [], "neg_preferences": []}

    return {"pos_preferences": [], "neg_preferences": []}


def extract_student_info_from_message(message):
    if message.role == "user":
        try:
            return json.loads(message.content)
        except json.JSONDecodeError:
            return {"year": 0, "semester": 0, "major": "", "minor": "", "interests": ""}


# Chat endpoint to handle form submissions and feedback
@app.post("/chat")
async def chat(request: ChatRequest):
    # Convert messages to a format compatible with Ollama
    message = request.messages[0]
    student_info = extract_student_info_from_message(message)

    # check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    course_summ_embed = torch.load(
        "../../data/course_desc/course_summ_embed.pt", map_location=device
    )
    course_desc_embed = torch.load(
        "../../data/course_desc/course_desc_embed.pt", map_location=device
    )
    course_learning_outcomes_embed = torch.load(
        "../../data/course_desc/course_learning_outcomes_embed.pt", map_location=device
    )
    embedder = SentenceTransformer("all-mpnet-base-v2")

    with open(
        "../../data/course_desc/informatics_course_info.json",
        "r",
    ) as file:
        course_info = json.load(file)

    course_id_idx_mapping = {}
    for idx, course in enumerate(course_info):
        course_id_idx_mapping[course["course_code"]] = idx

    # Filtering by year and semester

    # with open("../../data/yearSemester.pkl", "rb") as f:
    #     yearSemester = pickle.load(f)
    # keepCourses = [
    #     course_id_idx_mapping[k]
    #     for k, v in yearSemester.items()
    #     if (0 in v[0] or student_info["year"] in v[0])
    #     and (v[1] == student_info["semester"] or v[1] == 0)
    # ]
    # course_summ_embed = course_summ_embed[keepCourses]
    # course_desc_embed = course_desc_embed[keepCourses]
    # course_learning_outcomes_embed = course_learning_outcomes_embed[keepCourses]

    print(student_info["additionalInfo"])
    # Convert input to pos and neg domains
    course_preferences = extract_course_preferences(student_info["additionalInfo"])

    user_query_pos = course_preferences["pos_preferences"]
    user_query_neg = course_preferences["neg_preferences"]

    print(user_query_pos, user_query_neg)

    if len(user_query_pos) == 0:
        pos_embedding = torch.zeros(768, device=device)
    for idx, query in enumerate(user_query_pos):

        if idx == 0:
            pos_embedding = embedder.encode(
                query, convert_to_tensor=True, device=device
            )
        else:
            pos_embedding += embedder.encode(
                query, convert_to_tensor=True, device=device
            )

    if len(user_query_neg) == 0:
        neg_embedding = torch.zeros(768, device=device)
    for idx, query in enumerate(user_query_neg):
        if idx == 0:
            neg_embedding = embedder.encode(
                query, convert_to_tensor=True, device=device
            )
        else:
            neg_embedding += embedder.encode(
                query, convert_to_tensor=True, device=device
            )

    user_query_embedding = pos_embedding - neg_embedding
    user_query_embedding = user_query_embedding.to(device)

    print(type(course_summ_embed), course_summ_embed.get_device())
    print(type(user_query_embedding), user_query_embedding.get_device())
    course_summ_similarity = torch.cosine_similarity(
        course_summ_embed, user_query_embedding, dim=1
    )
    course_desc_similarity = torch.cosine_similarity(
        course_desc_embed,
        user_query_embedding,
        dim=1,
    )
    course_learning_outcomes_similarity = torch.cosine_similarity(
        course_learning_outcomes_embed,
        user_query_embedding,
        dim=1,
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
            "course_title": course_info[idx]["course_title"],
            "course_code": course_info[idx]["course_code"],
            "course_summary": course_info[idx]["course_summary"],
            "course_description": course_info[idx]["course_desc"],
            "course_learning_outcomes": course_info[idx]["learning_outcomes"],
            "similarity_score": similarity_scores[idx].item(),
        }
        for idx in top_5_courses
    ]

    print("************************")
    print([course["course_title"] for course in selected_courses])

    explanation_prompt = f"""
    You should provide a readable summary of a study plan to a student. The following courses were recommended as part of the study plan:
    
    {json.dumps(selected_courses, indent=4)}

    For each course, display it in the following format:

    Course Title: <course_title>
    Course Code: <course_code>
    Explanation: <explain why the course was recommended based on the similarity score, do not display the similarity score>
    
    Only respond with the top-5 courses that were recommended and the details of which was shared above.
    DO NOT INCLUDE YOUR REASONING IN THE RESPONSE. 
    DO NOT HALLUCINATE AND CREATE NON-EXISTENT COURSES.
    Do not display similarity scores.
    Do not say anything apart from what is mentioned above.
    """

    # Call Ollama with streaming enabled (adjust based on actual Ollama API)
    stream = ollama.chat(
        model=ollama_final_model_name,
        messages=[{"role": "user", "content": explanation_prompt}],
        stream=True,
    )

    # Generator function to stream response chunks
    def generate():
        for chunk in stream:
            # Assuming Ollama returns chunks with 'message' and 'content'
            yield chunk["message"]["content"]

    return StreamingResponse(generate(), media_type="text/plain")
