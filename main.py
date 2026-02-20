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
APTITUDE_PATH = BASE_DIR / "aptitude_verbal_questions.xlsx"

# Global variables to store data
TOPICS_DATA = None
APTITUDE_DATA = None

def load_topics():
    """Load topics from Excel file into memory"""
    global TOPICS_DATA
    try:
        df = pd.read_excel(EXCEL_PATH)
        df = df[df['Branch'] != 'Branch']
        df['Semester'] = pd.to_numeric(df['Semester'], errors='coerce')
        df = df.dropna(subset=['Branch', 'Semester'])
        TOPICS_DATA = df
        print(f"✅ Loaded {len(df)} topics from Excel")
        return True
    except Exception as e:
        print(f"❌ Failed to load Excel: {e}")
        return False

def load_aptitude():
    """Load aptitude questions from Excel"""
    global APTITUDE_DATA
    try:
        df = pd.read_excel(APTITUDE_PATH)
        APTITUDE_DATA = df
        print(f"✅ Loaded {len(df)} aptitude questions")
        return True
    except Exception as e:
        print(f"❌ Failed to load aptitude: {e}")
        return False

# ✅ MOBILE FIX: Don't load on startup — load lazily on first request
# load_topics()  # COMMENTED OUT — was causing 15-20s mobile load time

# -------------------- DIAGRAM MAP --------------------
DIAGRAM_MAP = {
    # CSE — matched to exact Excel topic names
    "Binary search trees": "/diagrams/cse/binary_tree.svg",
    "Binary Search Tree": "/diagrams/cse/binary_tree.svg",
    "Binary Tree": "/diagrams/cse/binary_tree.svg",
    "Trees": "/diagrams/cse/binary_tree.svg",
    "Stacks": "/diagrams/cse/stack_queue.svg",
    "Stack": "/diagrams/cse/stack_queue.svg",
    "Queues": "/diagrams/cse/stack_queue.svg",
    "Queue": "/diagrams/cse/stack_queue.svg",
    "Network models (OSI & TCP/IP)": "/diagrams/cse/tcp_handshake.svg",
    "TCP": "/diagrams/cse/tcp_handshake.svg",
    "UDP": "/diagrams/cse/tcp_handshake.svg",
    "Paging": "/diagrams/cse/paging.svg",
    "Virtual memory": "/diagrams/cse/paging.svg",
    "Virtual Memory": "/diagrams/cse/paging.svg",
    "Normalization": "/diagrams/cse/normalization.svg",
    "BCNF": "/diagrams/cse/normalization.svg",
    "Finite automata": "/diagrams/cse/dfa_nfa.svg",
    "DFA": "/diagrams/cse/dfa_nfa.svg",
    "NFA": "/diagrams/cse/dfa_nfa.svg",
    "Sorting algorithms": "/diagrams/cse/sorting_comparison.svg",
    "Sorting": "/diagrams/cse/sorting_comparison.svg",
    "Graph traversal (BFS)": "/diagrams/cse/bfs_dfs.svg",
    "Graph traversal (DFS)": "/diagrams/cse/bfs_dfs.svg",
    "BFS": "/diagrams/cse/bfs_dfs.svg",
    "DFS": "/diagrams/cse/bfs_dfs.svg",
    "Heap": "/diagrams/cse/heap_structure.svg",
    "Network models (OSI & TCP/IP)": "/diagrams/cse/network_layers.svg",
    "OSI": "/diagrams/cse/network_layers.svg",
    "Flip flops": "/diagrams/ece/flip_flop_types.svg",
    "Flip flop": "/diagrams/ece/flip_flop_types.svg",
    "Karnaugh maps": "/diagrams/ece/kmap_example.svg",
    "Logic gates": "/diagrams/ece/logic_gates.svg",
    # ECE
    "Thevenin": "/diagrams/ece/thevenin_norton.svg",
    "Norton": "/diagrams/ece/thevenin_norton.svg",
    "Bode": "/diagrams/ece/bode_plot.svg",
    "Flip-flop": "/diagrams/ece/flip_flop_types.svg",
    "Flip flop": "/diagrams/ece/flip_flop_types.svg",
    "Op-Amp": "/diagrams/ece/opamp_circuits.svg",
    "Op Amp": "/diagrams/ece/opamp_circuits.svg",
    "BJT": "/diagrams/ece/bjt_biasing.svg",
    "FET": "/diagrams/ece/bjt_biasing.svg",
    "Modulation": "/diagrams/ece/am_fm_modulation.svg",
    "Logic Gate": "/diagrams/ece/logic_gates.svg",
    "K-map": "/diagrams/ece/kmap_example.svg",
    "Filter": "/diagrams/ece/filter_types.svg",
    # Mechanical
    "Rankine": "/diagrams/mechanical/rankine_cycle.svg",
    "Carnot": "/diagrams/mechanical/carnot_cycle.svg",
    "Otto": "/diagrams/mechanical/otto_diesel.svg",
    "Diesel": "/diagrams/mechanical/otto_diesel.svg",
    "Bernoulli": "/diagrams/mechanical/bernoulli.svg",
    "Venturi": "/diagrams/mechanical/venturi_meter.svg",
    "Shear Force": "/diagrams/mechanical/sfd_bmd.svg",
    "Bending Moment": "/diagrams/mechanical/sfd_bmd.svg",
    "Gear": "/diagrams/mechanical/gear_train.svg",
    "Stress-Strain": "/diagrams/mechanical/stress_strain.svg",
    # Electrical
    "Transformer": "/diagrams/electrical/transformer.svg",
    "Induction Motor": "/diagrams/electrical/induction_motor.svg",
    "Inverter": "/diagrams/electrical/inverter.svg",
    "Wheatstone": "/diagrams/electrical/wheatstone_bridge.svg",
    "Thyristor": "/diagrams/electrical/thyristor.svg",
    # Civil
    "Truss": "/diagrams/civil/truss_structure.svg",
    "Soil": "/diagrams/civil/soil_classification.svg",
    "Water Treatment": "/diagrams/civil/water_treatment.svg",
    # Chemical
    "Distillation": "/diagrams/chemical/distillation_column.svg",
    "Heat Exchanger": "/diagrams/chemical/heat_exchanger.svg",
    "PFR": "/diagrams/chemical/pfr_cstr.svg",
    "CSTR": "/diagrams/chemical/pfr_cstr.svg",
}

def get_diagram_for_topic(topic: str):
    """Check if a diagram exists for a given topic"""
    topic_lower = topic.lower()
    for key, url in DIAGRAM_MAP.items():
        if key.lower() in topic_lower:
            return url
    return None

# -------------------- SYSTEM PROMPTS --------------------

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
    return {
        "status": "ok",
        "topics_loaded": TOPICS_DATA is not None,
        "aptitude_loaded": APTITUDE_DATA is not None
    }

@app.get("/get-branches")
def get_branches():
    """Return list of all branches — lazy loads Excel on first call"""
    global TOPICS_DATA
    if TOPICS_DATA is None:
        load_topics()
    if TOPICS_DATA is None:
        return {"error": "Failed to load topics"}
    branches = sorted(TOPICS_DATA['Branch'].unique().tolist())
    return {"branches": branches}

@app.get("/get-semesters")
def get_semesters():
    """Return list of semesters"""
    global TOPICS_DATA
    if TOPICS_DATA is None:
        load_topics()
    if TOPICS_DATA is None:
        return {"error": "Failed to load topics"}
    semesters = sorted([int(s) for s in TOPICS_DATA['Semester'].unique()])
    return {"semesters": semesters}

@app.get("/get-topics")
async def get_topics(branch: str, semester: int):
    """Return all topics for a branch/semester for topic selection dropdown"""
    global TOPICS_DATA
    if TOPICS_DATA is None:
        load_topics()
    if TOPICS_DATA is None:
        return {"error": "Failed to load topics"}

    filtered = TOPICS_DATA[
        (TOPICS_DATA['Branch'] == branch) &
        (TOPICS_DATA['Semester'] == semester)
    ]

    topics_list = []
    for _, row in filtered.iterrows():
        topics_list.append({
            "topic": row['Topic'],
            "subject": row['Subject']
        })

    return {"topics": topics_list, "count": len(topics_list)}

@app.post("/generate-question")
async def generate_question(request: Request):
    """Generate a question from selected branch, semester, and optional specific topic"""
    try:
        global TOPICS_DATA
        if TOPICS_DATA is None:
            load_topics()
        if TOPICS_DATA is None:
            return {"error": "Topics database not loaded"}

        data = await request.json()
        branch = data.get("branch")
        semester = data.get("semester")
        specific_topic = data.get("specific_topic", None)  # NEW: optional specific topic

        if not branch or not semester:
            return {"error": "Branch and semester required"}

        filtered = TOPICS_DATA[
            (TOPICS_DATA['Branch'] == branch) &
            (TOPICS_DATA['Semester'] == int(semester))
        ]

        if len(filtered) == 0:
            return {"error": f"No topics found for {branch} Semester {semester}"}

        # Pick topic: specific if selected, else random
        if specific_topic:
            matched = filtered[filtered['Topic'] == specific_topic]
            if len(matched) == 0:
                return {"error": f"Topic '{specific_topic}' not found"}
            random_row = matched.iloc[0]
        else:
            random_row = filtered.sample(n=1).iloc[0]

        topic = random_row['Topic']
        subject = random_row['Subject']

        # Check if a diagram exists for this topic
        diagram_url = get_diagram_for_topic(topic)

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
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return {
            "topic": topic,
            "subject": subject,
            "branch": branch,
            "semester": semester,
            "question": result.get("question", ""),
            "hint": result.get("hint", ""),
            "diagram": diagram_url  # NEW: diagram URL if available
        }

    except Exception as e:
        return {"error": f"Failed to generate question: {str(e)}"}

@app.post("/analyze-practice")
async def analyze_practice(request: Request):
    """Evaluate answer in practice mode"""
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
    """Evaluate answer in custom mode"""
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

# -------------------- APTITUDE ENDPOINTS --------------------

# Hardcoded type mapping — works with ANY category names in your Excel
VERBAL_CATEGORIES = {
    'analogies', 'synonyms', 'antonyms', 'sentence correction',
    'syllogism', 'classification', 'verbal reasoning', 'grammar',
    'vocabulary', 'comprehension', 'reading comprehension',
    'fill in the blanks', 'idioms', 'one word substitution',
    'sentence arrangement', 'para jumbles'
}

@app.get("/get-aptitude-categories")
def get_aptitude_categories():
    """Return aptitude categories grouped by Quantitative and Verbal"""
    global APTITUDE_DATA
    if APTITUDE_DATA is None:
        load_aptitude()
    if APTITUDE_DATA is None:
        return {"error": "Aptitude questions not loaded"}

    all_categories = APTITUDE_DATA["Category"].dropna().unique().tolist()

    quant_categories = []
    verbal_categories = []

    for cat in all_categories:
        if cat.strip().lower() in VERBAL_CATEGORIES:
            verbal_categories.append(cat)
        else:
            quant_categories.append(cat)

    return {
        "quantitative": sorted(quant_categories),
        "verbal": sorted(verbal_categories),
        "all": sorted(all_categories)
    }

@app.post("/generate-aptitude")
async def generate_aptitude(request: Request):
    """Return a random aptitude question, optionally filtered by category"""
    try:
        global APTITUDE_DATA
        if APTITUDE_DATA is None:
            load_aptitude()
        if APTITUDE_DATA is None:
            return {"error": "Excel file not found. Make sure aptitude_verbal_questions.xlsx is in your GitHub repo."}

        # Log actual columns for debugging
        print(f"Aptitude columns: {list(APTITUDE_DATA.columns)}")

        data = await request.json()
        category = data.get("category", None)
        type_filter = data.get("type", None)

        if category:
            filtered = APTITUDE_DATA[APTITUDE_DATA['Category'] == category]
        elif type_filter and type_filter != "all":
            # Use the same VERBAL_CATEGORIES set to split quant vs verbal
            if type_filter == "verbal":
                mask = APTITUDE_DATA['Category'].str.strip().str.lower().isin(VERBAL_CATEGORIES)
            else:  # quantitative
                mask = ~APTITUDE_DATA['Category'].str.strip().str.lower().isin(VERBAL_CATEGORIES)
            filtered = APTITUDE_DATA[mask]
        else:
            filtered = APTITUDE_DATA

        if len(filtered) == 0:
            return {"error": "No questions found for this selection. Check your Excel Category column values."}

        random_q = filtered.sample(n=1).iloc[0]

        # Flexible column name lookup (handles spaces, case differences)
        cols = {c.strip().lower().replace(" ", ""): c for c in APTITUDE_DATA.columns}

        def get_col(name):
            key = name.lower().replace(" ", "")
            actual = cols.get(key)
            if actual is None:
                raise KeyError(f"Column '{name}' not found. Available: {list(APTITUDE_DATA.columns)}")
            return random_q[actual]

        return {
            "category":      str(get_col("Category")),
            "subcategory":   str(get_col("Subcategory")) if "subcategory" in cols else "",
            "question":      str(get_col("Question")),
            "options": {
                "a": str(get_col("OptionA")),
                "b": str(get_col("OptionB")),
                "c": str(get_col("OptionC")),
                "d": str(get_col("OptionD")),
            },
            "correct_answer": str(get_col("Answer")).strip().lower(),
            "explanation":   str(get_col("Explanation")),
        }

    except Exception as e:
        print(f"generate-aptitude error: {e}")
        return {"error": f"Server error: {str(e)}"}

@app.post("/check-aptitude")
async def check_aptitude(request: Request):
    """Check a submitted aptitude answer"""
    data = await request.json()
    user_answer = data.get("user_answer", "").strip().lower()
    correct_answer = data.get("correct_answer", "").strip().lower()
    explanation = data.get("explanation", "")

    is_correct = (user_answer == correct_answer)

    return {
        "is_correct": is_correct,
        "explanation": explanation,
        "message": "Correct! Well done." if is_correct else f"Incorrect. The correct answer is ({correct_answer.upper()})."
    }

# -------------------- DEBUG ENDPOINT --------------------
@app.get("/debug-aptitude")
def debug_aptitude():
    """Debug endpoint — visit this URL to diagnose aptitude issues"""
    global APTITUDE_DATA
    if APTITUDE_DATA is None:
        success = load_aptitude()
        if not success:
            return {
                "status": "error",
                "message": "Failed to load Excel file",
                "path_checked": str(APTITUDE_PATH),
                "file_exists": APTITUDE_PATH.exists()
            }
    return {
        "status": "ok",
        "rows": len(APTITUDE_DATA),
        "columns": list(APTITUDE_DATA.columns),
        "sample_categories": APTITUDE_DATA["Category"].unique().tolist()[:10] if "Category" in APTITUDE_DATA.columns else "Column 'Category' not found",
        "path": str(APTITUDE_PATH),
        "file_exists": APTITUDE_PATH.exists()
    }

# -------------------- SERVE STATIC FILES --------------------
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
