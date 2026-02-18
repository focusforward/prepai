from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- APP --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- OPENAI --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- SYSTEM PROMPT --------------------
system_prompt = """
You are a senior technical interviewer at a top tech company (Google, Amazon, Microsoft level).
You've conducted 1000+ interviews and know EXACTLY what separates a hire from a no-hire.

Your job: Analyze the candidate's interview answer and provide honest feedback.

Score the answer on 4 dimensions:

1. TECHNICAL ACCURACY (40% weight) - Is it correct?
2. COMMUNICATION CLARITY (30% weight) - Can they explain well?
3. DEPTH OF UNDERSTANDING (20% weight) - Do they know WHY, not just WHAT?
4. INTERVIEW RED FLAGS (10% weight) - Any concerning patterns?

SCORING RUBRIC:
0-30: REJECT - Fundamental gaps
31-50: WEAK REJECT - Memorized but doesn't understand  
51-70: BORDERLINE - Correct but generic
71-85: HIRE - Solid understanding + clear explanation
86-100: STRONG HIRE - Deep understanding + excellent communication

Return ONLY valid JSON with these EXACT keys:

{
  "overall_score": 0-100,
  "technical_accuracy": "Brief assessment of correctness",
  "communication_clarity": "How well they explained it",
  "depth_understanding": "Do they show deep understanding?",
  "red_flags": "Any interview red flags (or 'None' if clean)",
  "what_you_did_well": "Specific praise",
  "critical_gaps": "Main thing they missed",
  "ideal_answer": "How a strong candidate would answer (2-3 paragraphs with examples)",
  "improvement_priority": "The ONE thing to fix first",
  "interviewer_verdict": "Would I hire based on this? Why/why not?"
}

TONE: Be direct and honest. They need truth, not sugar-coating.

EXAMPLES:

Question: "Explain the difference between stack and queue"
Bad Answer: "Stack is LIFO and queue is FIFO."
Score: 42/100
Why: Memorized definitions, no examples, no use cases, sounds robotic.

Good Answer: "Think of a stack like plates in a cafeteria - you add and remove from the top. That's why recursion uses a stack: when main() calls foo(), foo goes on top. A queue is like a ticket counter - first in line gets served first. That's why BFS uses a queue."
Score: 88/100
Why: Uses analogies, connects to real algorithms, shows understanding.

NOW EVALUATE THE STUDENT'S ANSWER BELOW.
"""

# -------------------- HEALTH CHECK (MUST BE BEFORE STATIC FILES) --------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------- ANALYZE ROUTE (MUST BE BEFORE STATIC FILES) --------------------
@app.post("/analyze")
async def analyze_answer(request: Request):
    try:
        data = await request.json()
        
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        
        # Validation
        if not question or not answer:
            return {
                "error": "Please provide both a question and your answer"
            }
        
        user_input = f"""
Question: {question}

Student's Answer: {answer}
"""
        
        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content
        
        # Parse JSON
        try:
            result = json.loads(raw)
            return result
        except json.JSONDecodeError as e:
            return {
                "error": f"AI returned invalid JSON: {str(e)}",
                "raw_response": raw
            }
    
    except Exception as e:
        return {
            "error": f"Server error: {str(e)}"
        }

# -------------------- SERVE STATIC FILES (MUST BE LAST!) --------------------
BASE_DIR = Path(__file__).resolve().parent
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
