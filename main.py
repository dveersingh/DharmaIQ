from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

class ChatRequest(BaseModel):
    character: str
    user_message: str

class ChatResponse(BaseModel):
    character: str
    response: str

def generate_character_response(character: str, user_message: str) -> str:
    try:
        prompt = f"""You are {character}, a movie character. 
        Respond to the user in 1-2 sentences using {character}'s personality and style.
        Stay completely in character.
        
        User message: {user_message}"""
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 150
            }
        )
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        ai_response = generate_character_response(
            request.character, 
            request.user_message
        )
        return {
            "character": request.character,
            "response": ai_response
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)