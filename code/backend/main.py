from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import ollama  # Hypothetical Ollama client (install via pip if available)

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

# Chat endpoint to handle form submissions and feedback
@app.post("/chat")
async def chat(request: ChatRequest):
    # Convert messages to a format compatible with Ollama
    messages = [msg.dict() for msg in request.messages]
    # Call Ollama with streaming enabled (adjust based on actual Ollama API)
    stream = ollama.chat(messages=messages, stream=True)
    
    # Generator function to stream response chunks
    def generate():
        for chunk in stream:
            # Assuming Ollama returns chunks with 'message' and 'content'
            yield chunk['message']['content'] + "\n"
    
    return StreamingResponse(generate(), media_type="text/plain")