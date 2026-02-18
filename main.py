from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import os
import random
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

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

# -------------------- LOAD EXCEL DATA --------------------
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "AICTE_Syllabus_of_All_Branches.xlsx"

# Global variable to store topics
TOPICS_DATA = None

def load_topics():
    """Load topics from Excel file into memory"""
    global TOPICS_DATA
    try:
        df = pd.read_excel(EXCEL_PATH)
        # Skip header row if it got read as data
        df = df[df['Branch'] != 'Branch']
        df['Semester'] = pd.to_numeric(df['Semester'], errors='coerce')
        df = df.dropna(subset=['Branch', 'Semester'])
        TOPICS_DATA = df
        print(f"✅ Loaded {len(df)} topics from Excel")
        return True
    except Exception as e:
        print(f"❌ Failed to load Excel: {e}")
        return False

# Load topics on startup
load_topics()

# -------------------- SYSTEM PROMPTS --------------------

# Custom Question Mode (original PrepAI)
custom_prompt = """
You are a senior technical interviewer at a top tech company.

Analyze the candidate's interview answer and provide honest feedback.

Score on 4 dimensions:
1. TECHNICAL ACCURACY (40%)
2. COMMUNICATION CLARITY (30%)
3. DEPTH OF UNDERSTANDING (20%)
4. INTERVIEW RED FLAGS (10%)

SCORING:
0-30: REJECT
31-50: WEAK REJECT
51-70: BORDERLINE
71-85: HIRE
86-100: STRONG HIRE

Return ONLY valid JSON:
{
  "overall_score": 0-100,
  "score_out_of_10": 0-10 (rounded to 1 decimal),
  "technical_accuracy": "Brief assessment",
  "communication_clarity": "How well explained",
  "depth_understanding": "Shows deep understanding?",
  "red_flags": "Any red flags or 'None'",
  "what_you_did_well": "Specific praise",
  "critical_gaps": "Main thing missed",
  "ideal_answer": "How strong candidate would answer (2-3 paragraphs)",
  "improvement_priority": "ONE thing to fix first",
  "interviewer_verdict": "Would I hire? Why/why not?"
}
"""

# Practice Mode (structured syllabus-based)
practice_prompt = """
You are a technical interviewer evaluating a student's answer for a placement interview.

The student is practicing for placements. Give constructive, encouraging feedback.

Score on:
1. CORRECTNESS (40%) - Is the answer technically right?
2. CLARITY (30%) - Can they explain well?
3. COMPLETENESS (20%) - Did they cover key points?
4. CONFIDENCE (10%) - Do they sound interview-ready?

Return ONLY valid JSON:
{
  "overall_score": 0-100,
  "score_out_of_10": 0-10 (rounded to 1 decimal),
  "what_you_got_right": "Praise what's correct",
  "what_you_missed": "What's incomplete or wrong",
  "how_to_improve": "Specific actionable advice",
  "ideal_answer": "Model answer (concise, interview-friendly)",
  "interviewer_perspective": "How an interviewer would see this"
}
"""

# Question generation prompt
question_gen_prompt = """
You are creating a technical interview question for a placement interview.

Generate ONE interview question on the given topic suitable for campus placements.

The question should:
- Be commonly asked in real interviews
- Be answerable in 2-3 minutes
- Test understanding, not just memorization
- Be at medium difficulty level

Return ONLY valid JSON:
{
  "question": "The interview question",
  "hint": "Optional hint if the question is tricky (or empty string)"
}
"""

# -------------------- ENDPOINTS --------------------

@app.get("/health")
def health():
    return {"status": "ok", "topics_loaded": TOPICS_DATA is not None}

@app.get("/get-branches")
def get_branches():
    """Return list of all branches"""
    if TOPICS_DATA is None:
        return {"error": "Topics not loaded"}
    
    branches = sorted(TOPICS_DATA['Branch'].unique().tolist())
    return {"branches": branches}

@app.get("/get-semesters")
def get_semesters():
    """Return list of semesters"""
    if TOPICS_DATA is None:
        return {"error": "Topics not loaded"}
    
    semesters = sorted([int(s) for s in TOPICS_DATA['Semester'].unique()])
    return {"semesters": semesters}

@app.post("/generate-question")
async def generate_question(request: Request):
    """Generate a random question from selected branch and semester"""
    try:
        data = await request.json()
        branch = data.get("branch")
        semester = data.get("semester")
        
        if not branch or not semester:
            return {"error": "Branch and semester required"}
        
        if TOPICS_DATA is None:
            return {"error": "Topics database not loaded"}
        
        # Filter topics by branch and semester
        filtered = TOPICS_DATA[
            (TOPICS_DATA['Branch'] == branch) & 
            (TOPICS_DATA['Semester'] == int(semester))
        ]
        
        if len(filtered) == 0:
            return {"error": f"No topics found for {branch} Semester {semester}"}
        
        # Pick random topic
        random_row = filtered.sample(n=1).iloc[0]
        topic = random_row['Topic']
        subject = random_row['Subject']
        
        # Generate question using AI
        prompt = f"""
Generate a placement interview question on this topic:
Subject: {subject}
Topic: {topic}
Branch: {branch}
Semester: {semester}

The question should be appropriate for a {branch} student and commonly asked in campus placements.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": question_gen_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Slightly creative for question variety
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return {
            "topic": topic,
            "subject": subject,
            "branch": branch,
            "semester": semester,
            "question": result.get("question", ""),
            "hint": result.get("hint", "")
        }
        
    except Exception as e:
        return {"error": f"Failed to generate question: {str(e)}"}

@app.post("/analyze-practice")
async def analyze_practice(request: Request):
    """Evaluate answer in practice mode (simpler, encouraging feedback)"""
    try:
        data = await request.json()
        question = data.get("question")
        answer = data.get("answer")
        topic = data.get("topic", "")
        
        if not question or not answer:
            return {"error": "Question and answer required"}
        
        user_input = f"""
Topic: {topic}
Question: {question}
Student's Answer: {answer}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": practice_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@app.post("/analyze-custom")
async def analyze_custom(request: Request):
    """Evaluate answer in custom mode (detailed interviewer feedback)"""
    try:
        data = await request.json()
        question = data.get("question")
        answer = data.get("answer")
        
        if not question or not answer:
            return {"error": "Question and answer required"}
        
        user_input = f"""
Question: {question}
Student's Answer: {answer}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

# -------------------- SERVE STATIC FILES --------------------
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
