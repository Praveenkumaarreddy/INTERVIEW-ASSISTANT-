import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# Allow frontend connection from Vercel or any client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class InterviewRequest(BaseModel):
    user_speech: str
    image_b64: str

# Tailored specifically for English Lecturers
SYSTEM_INSTRUCTION = """
You are a Senior Head of the English Department conducting an interview for an English Lecturer position.

Your goal is to evaluate the candidate's:
1. Spoken English fluency, vocabulary, and professional tone.
2. Pedagogy and classroom management skills for teaching literature and language.
3. Ability to explain complex literary concepts or grammar rules clearly.

Behavior Rules:
- Act like an encouraging yet rigorous academic interviewer.
- Analyze their body language from the webcam frame if available (eye contact, confidence, posture).
- Keep your response under 3 sentences: acknowledge their previous answer briefly, then ask ONE sharp, relevant interview question.
"""

@app.post("/api/interview-step")
async def interview_step(req: InterviewRequest):
    try:
        image_bytes = base64.b64decode(req.image_b64.split(",")[-1]) if req.image_b64 else None
        
        contents = []
        if image_bytes:
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )
            )
        contents.append(req.user_speech)

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.7,
            )
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
