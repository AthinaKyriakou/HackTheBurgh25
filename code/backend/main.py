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

    top_courses = torch.topk(similarity_scores, 20).indices

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
        for idx in top_courses
    ]

    # Finding all courses combinations with specific total credits
    # Mind that the courses dictionary should contain only courses which respect:
    # most-preferences-fitting not-taken courses not prohibited by taken courses with the key-specified number of credits

    with open("../../data/prerequisites.pkl", "rb") as f:
        prerequisites = pickle.load(f)
    with open("../../data/prohibitions.pkl", "rb") as f:
        prohibitions = pickle.load(f)
    with open("../../data/credits.pkl", "rb") as f:
        creditsDict = pickle.load(f)

    course_idx_id_mapping = {value: key for key, value in course_id_idx_mapping.items()}

    def allowed_plans(creditsN: int, courses: dict(), takenCourses: set() = set(), prohibitions: dict() = prohibitions, prerequisites: dict() = prerequisites):
        """
        INPUTS:
        creditsN: integer of credits required for the study plan
        courses: dictionary with number of credits (integers) as keys and courses selected for similarity and lists of best query-aligning courses with that number of credits
        takenCourses: set of courses already taken by the student
        prohibitions: dictionary with courses IDs (strings) as keys and sets of other courses IDs prohibited by that key-course
        prerequisites: dictionary with courses IDs (strings) as keys and sets of other courses IDs required (mandatory prerequisites) to take that key-course

        OUTPUT:
        set of all allowed study plans (in list format) obtained by combining the input courses and resecting given constraints
        """

        # Find all ways to reach the target credit sum using unique courses
        combinations = set()
        def find_combinations(selected, used_letters, remaining_credits):
            if remaining_credits == 0:
                combinations.add(frozenset(selected))  # Use frozenset to make order irrelevant
                return
            if remaining_credits < 0:
                return

            for credit, subjects in courses.items():
                for choice in subjects:
                    if choice not in used_letters:  # Avoid repeating letters
                        find_combinations(selected + [choice], used_letters | {choice}, remaining_credits - credit)
        find_combinations([], set(), creditsN)


        # Filtering study plans by prerequisites and prohibited
        notAllowed = set()
        for plan in combinations:
            for course in plan & set(prohibitions.keys()):
                if prohibitions[course] in plan:
                    notAllowed = notAllowed | {plan}
                    break
        combinations = combinations - notAllowed
        notAllowed = set()
        for plan in combinations:
            taking = plan | takenCourses
            for course in plan & set(prerequisites.keys()):
                if prerequisites[course] not in taking:
                    notAllowed = notAllowed | {plan}
                    break
        plans = {tuple(plan) for plan in combinations - notAllowed}
        return plans
    
    courses = dict()
    for course in map(lambda x: course_idx_id_mapping[x], top_courses.numpy()):
        if creditsDict[course] in courses.keys():
            courses[creditsDict[course]].append(course_id_idx_mapping[course])
        else:
            courses[creditsDict[course]] = [course_id_idx_mapping[course]]


    
    allowedStudyPlans = allowed_plans(student_info['credits'],courses)
    # print("Allowed study plans", allowedStudyPlans)

    allowedStudyPlansIdx = [{course for course in plan} for plan in allowedStudyPlans]
    plansSumEmbedding = []
    for plan in allowedStudyPlansIdx:
        sumEmbedding = torch.zeros(768, device=device)
        for course in plan:
            sumEmbedding += course_summ_embed[course]
        plansSumEmbedding.append(sumEmbedding)

    # print("Debug ---------")
    # print(len(plansSumEmbedding))    
    # print(range(len(plansSumEmbedding)))

    # print(user_query_embedding.shape)
    # print(plansSumEmbedding[0].shape)


    plansSimilarities = list(map(lambda x: torch.cosine_similarity(x.unsqueeze(0), user_query_embedding.unsqueeze(0)), plansSumEmbedding))
    
    plansRanking = sorted(zip(plansSimilarities, allowedStudyPlans), key=lambda x: x[0], reverse=True)

    studyPlans = [
    {
        "PlanNum": i + 1,
        # "Similarity": plansRanking[i][0],
        "Plan": {f'Course{j+1}': course_idx_id_mapping[plansRanking[i][1][j]] for j in range(len(plansRanking[i][1]))}
    }
    for i in range(len(plansRanking))
]

    print("Study plans ----")
    print(studyPlans)

    explanation_prompt = f"""
    Based on the following Study plans, describe the plans to the user. You are only supposed to display your response, do not respond with anything else. Be helpful. 
    
    {json.dumps(studyPlans[:5], indent=4)}
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
