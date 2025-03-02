## Running the frontend, run the following in /frontend
npm start

## Running the backend, run the following in /frontend
uvicorn main:app --reload --port 8000

## Running Ollama, run the following
ollama pull llama3.1 & ollama serve