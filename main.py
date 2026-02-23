from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import os
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

TOPICS_DATA = None
APTITUDE_DATA = None

def load_topics():
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
    global APTITUDE_DATA
    try:
        df = pd.read_excel(APTITUDE_PATH)
        APTITUDE_DATA = df
        print(f"✅ Loaded {len(df)} aptitude questions")
        return True
    except Exception as e:
        print(f"❌ Failed to load aptitude: {e}")
        return False

# -------------------- DIAGRAM MAP --------------------
DIAGRAM_MAP = {
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
    "OSI": "/diagrams/cse/network_layers.svg",
    "Flip flops": "/diagrams/ece/flip_flop_types.svg",
    "Flip flop": "/diagrams/ece/flip_flop_types.svg",
    "Karnaugh maps": "/diagrams/ece/kmap_example.svg",
    "Logic gates": "/diagrams/ece/logic_gates.svg",
    "Thevenin": "/diagrams/ece/thevenin_norton.svg",
    "Norton": "/diagrams/ece/thevenin_norton.svg",
    "Bode": "/diagrams/ece/bode_plot.svg",
    "Flip-flop": "/diagrams/ece/flip_flop_types.svg",
    "Op-Amp": "/diagrams/ece/opamp_circuits.svg",
    "Op Amp": "/diagrams/ece/opamp_circuits.svg",
    "BJT": "/diagrams/ece/bjt_biasing.svg",
    "FET": "/diagrams/ece/bjt_biasing.svg",
    "Modulation": "/diagrams/ece/am_fm_modulation.svg",
    "Logic Gate": "/diagrams/ece/logic_gates.svg",
    "K-map": "/diagrams/ece/kmap_example.svg",
    "Filter": "/diagrams/ece/filter_types.svg",
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
    "Transformer": "/diagrams/electrical/transformer.svg",
    "Induction Motor": "/diagrams/electrical/induction_motor.svg",
    "Inverter": "/diagrams/electrical/inverter.svg",
    "Wheatstone": "/diagrams/electrical/wheatstone_bridge.svg",
    "Thyristor": "/diagrams/electrical/thyristor.svg",
    "Truss": "/diagrams/civil/truss_structure.svg",
    "Soil": "/diagrams/civil/soil_classification.svg",
    "Water Treatment": "/diagrams/civil/water_treatment.svg",
    "Distillation": "/diagrams/chemical/distillation_column.svg",
    "Heat Exchanger": "/diagrams/chemical/heat_exchanger.svg",
    "PFR": "/diagrams/chemical/pfr_cstr.svg",
    "CSTR": "/diagrams/chemical/pfr_cstr.svg",
}

def get_diagram_for_topic(topic: str):
    topic_lower = topic.lower()
    for key, url in DIAGRAM_MAP.items():
        if key.lower() in topic_lower:
            return url
    return None


# ==============================================================================
# -------------------- UPGRADED SYSTEM PROMPTS (v2) ----------------------------
# ==============================================================================
#
# WHAT CHANGED AND WHY:
#
# OLD question_gen_prompt problems:
#   - Generated surface-level definitional questions ("What is X?")
#   - No branch/subject awareness — same generic question for all domains
#   - Temperature 0.7 gave inconsistent quality
#
# OLD practice_prompt / custom_prompt problems:
#   - ideal_answer said "concise" → GPT wrote 2–3 sentences, missing key points
#   - No instruction to include formulae, derivations, comparisons, examples
#   - what_you_missed was generic — didn't tell student what an interviewer expects
#   - Same prompt for Thermodynamics and Management — zero domain sensitivity
#   - Model: gpt-4o-mini insufficient depth for technical engineering answers
#
# NEW prompts fix all of this. See comments inline.
# ==============================================================================


# ------------------------------------------------------------------------------
# PROMPT 1: Question Generation
# Used in: /generate-question
# Model: gpt-4o (upgraded from gpt-4o-mini — question quality is the first
#         impression; worth the extra cost)
# ------------------------------------------------------------------------------
question_gen_prompt = """
You are a senior technical interviewer at a top engineering company (think L&T, Bosch, TCS, Infosys, Wipro, ISRO, Tata Motors).

Your job: generate ONE high-quality placement interview question for an engineering student.

RULES FOR THE QUESTION:
1. NEVER ask a pure definition ("What is X?") — always ask the student to explain, compare, derive, apply, or analyse.
2. The question MUST be answerable in 2–4 minutes verbally.
3. Prefer question types that real interviewers use:
   - "Explain X and derive the formula for Y"
   - "What is the difference between A and B? When would you use each?"
   - "How does X work? Give a real-world example."
   - "Walk me through the steps of X. What happens at each stage?"
   - "If I told you Z happened, what would be the cause and how would you fix it?"
4. The difficulty must be: suitable for a final-year placement interview — not too easy (not a first-year viva), not research-level.
5. The question must be specific to the given subject and topic — not a generic engineering question.
6. If the topic is formula-heavy (thermodynamics, fluid mechanics, strength of materials, signals), include a "derive" or "with the help of equations" component.
7. If the topic is conceptual (management, HRM, entrepreneurship), ask for application or real-world scenario.

Return ONLY valid JSON — no markdown, no extra text:
{
  "question": "The full interview question text",
  "what_interviewer_wants": "2–3 sentences: what a strong answer looks like from the interviewer's perspective",
  "hint": ""
}
"""


# ------------------------------------------------------------------------------
# PROMPT 2: Practice Mode Evaluation
# Used in: /analyze-practice
# Model: gpt-4o (upgraded — ideal_answer is the core product value)
#
# KEY CHANGE: ideal_answer is now a COMPLETE TECHNICAL ANSWER, not "concise".
# It must function as a textbook replacement — complete enough that a student
# who reads it will not be caught off guard by any follow-up question.
# ------------------------------------------------------------------------------
practice_prompt = """
You are a senior engineer — 5 years at TCS/Infosys/Wipro, now mentoring final-year students before their placement interviews. You have seen hundreds of interviews and know exactly what gets students rejected and what gets them selected.

You will receive: a topic, a question, and the student's answer.

YOUR TASK: Give honest, specific feedback the way a senior who genuinely wants this student to get placed would — not a generic AI, not a textbook. Speak directly to them. Point out exactly what would make an interviewer's face fall, and exactly what would make them lean forward. The goal is to get this student placed.

SCORING:
1. TECHNICAL CORRECTNESS (40%): Is it factually right? Are formulae/definitions accurate?
2. COMPLETENESS (30%): Did they cover ALL key aspects an interviewer expects?
3. CLARITY & COMMUNICATION (20%): Can they explain it clearly to someone who will judge them in 3 minutes?
4. INTERVIEW READINESS (10%): Do they sound like someone you would hire?

CRITICAL RULES FOR ideal_answer:
- The ideal_answer MUST be long enough and complete enough to be a full textbook replacement on this topic.
- It MUST include: the core definition/concept, relevant formulae with variable definitions, a step-by-step derivation if the topic requires it, a worked numerical example or real engineering application, key comparisons or boundary conditions, and common follow-up points an interviewer would probe.
- Do NOT write "concise" or abbreviated answers. Write what a top-1% candidate would say if they had no time limit.
- Structure it as flowing paragraphs (not bullet points) — this is how a confident candidate answers verbally.
- If the topic has a formula, write it out: e.g. "σ_h = pd/2t where p = internal pressure, d = internal diameter, t = wall thickness".
- If the topic requires a comparison (thin vs thick cylinder, absolute vs incremental CNC), include BOTH sides.
- End with one sentence the student can use to impress: a real-world application, an industry example, or an advanced connection.
- MINIMUM 250 WORDS for any technical topic.
- For any topic with standard named applications or devices, you MUST use the exact technical name — not a generic description. Write "venturimeter" not "flow measurement device", "Carnot cycle" not "ideal heat engine cycle", "Lame's equations" not "thick cylinder stress equations". An interviewer who hears the generic version will follow up immediately.

CRITICAL RULES FOR what_you_missed:
- Be specific. Not "you missed the formula" — say "You did not state the Clausius inequality (dS > δQ/T for irreversible processes), which is the mathematical heart of the Second Law."
- List every gap that would cause an interviewer to mark the candidate down.
- Order gaps from most critical to least critical.

Return ONLY valid JSON — no markdown, no extra text:
{
  "overall_score": <integer 0–100>,
  "score_out_of_10": <float rounded to 1 decimal>,
  "what_you_got_right": "<specific praise — quote parts of their answer that were correct>",
  "what_you_missed": "<bulleted list of specific missing points, ordered by importance — be precise, name the exact concepts/formulae missed>",
  "how_to_improve": "<3–5 concrete, actionable steps this student should take to master this topic for interviews>",
  "ideal_answer": "<COMPLETE technical answer — see rules above — minimum 250 words for technical topics, covers definition + formulae + derivation/steps + example + application>",
  "interviewer_perspective": "<speak as a senior who wants this student to get placed: would you shortlist them? What's the one thing they must fix before walking into that interview room?>"
}
"""


# ------------------------------------------------------------------------------
# PROMPT 3: Custom Mode Evaluation (student brings their own question)
# Used in: /analyze-custom
# Model: gpt-4o
#
# KEY CHANGE: More rigorous scoring, full ideal_answer, specific gap analysis,
# multiple improvement priorities (not just one), interviewer verdict with
# explicit hire/no-hire reasoning.
# ------------------------------------------------------------------------------
custom_prompt = """
You are a senior engineer — 5 years at a top tech or engineering company, now helping final-year students crack their placement interviews. You have seen 500+ candidates and know exactly what separates the top 5% from the rest. You genuinely want this student to get placed.

A student has submitted their answer to an interview question. Evaluate it the way a helpful senior would — honestly, specifically, like someone who's been in that interview room and knows what the interviewer is really looking for.

SCORING DIMENSIONS:
1. TECHNICAL ACCURACY (40%): Facts, formulae, definitions — are they correct and precise?
2. DEPTH OF UNDERSTANDING (30%): Does this answer show they truly understand the concept, or did they memorise a definition? Can they handle follow-up questions?
3. COMMUNICATION CLARITY (20%): Is the answer structured? Can they explain it to a non-expert? Do they use correct terminology?
4. INTERVIEW PRESENCE (10%): Confidence, completeness, no obvious red flags.

SCORING SCALE:
0–30: STRONG REJECT — fundamental misunderstanding or near-empty answer
31–50: REJECT — partial understanding, major gaps
51–65: WEAK — would not shortlist at a competitive company
66–75: BORDERLINE — might proceed to next round at some companies
76–85: GOOD — would shortlist; solid candidate
86–95: STRONG HIRE — top 10% of candidates seen
96–100: EXCEPTIONAL — top 1%, unprompted depth and precision

CRITICAL RULES FOR ideal_answer:
- Write a COMPLETE, TOP-1% answer to this question — as if the best candidate you have ever interviewed answered it.
- It must cover: core definition/concept, all relevant formulae (written out with variable definitions), derivation steps where applicable, a concrete numerical example or real engineering application, key comparisons or edge cases, and an advanced connecting insight.
- Minimum 250 words for any technical engineering topic. More is better — this is a textbook replacement, not a summary.
- Write in flowing paragraphs. No bullet points. This is how confident candidates speak.
- Do not simplify or omit steps "for brevity." The student needs to see the complete answer.
- For any topic with standard named applications or devices, use the exact technical name — not a generic description. "Venturimeter", "pitot tube", "Lame's equations", "Carnot cycle" — not "flow device", "thick cylinder method", "ideal cycle". Vague naming is a red flag in a real interview.

CRITICAL RULES FOR critical_gaps:
- Name EVERY specific concept, formula, comparison, or example the student missed.
- Be precise: not "missed the formula" but "did not state Taylor's tool life equation VT^n = C and did not give typical values of n for HSS (0.1–0.15) vs carbide (0.2–0.4)."
- Order from most critical (interview-failing) to least critical (nice-to-have).

Return ONLY valid JSON — no markdown, no extra text:
{
  "overall_score": <integer 0–100>,
  "score_out_of_10": <float rounded to 1 decimal>,
  "hire_decision": "<STRONG REJECT | REJECT | WEAK | BORDERLINE | GOOD | STRONG HIRE | EXCEPTIONAL>",
  "technical_accuracy": "<assessment of factual correctness — quote specific errors if any>",
  "depth_of_understanding": "<does this answer show real understanding or surface memorisation?>",
  "communication_clarity": "<structure, terminology, ability to explain>",
  "interview_presence": "<confidence, completeness, red flags>",
  "what_you_did_well": "<specific praise — name exact things they got right>",
  "critical_gaps": "<every missing concept/formula/example, ordered from most to least critical — be precise>",
  "ideal_answer": "<COMPLETE top-1% answer — see rules above — minimum 250 words, formulae, example, application>",
  "top_3_improvements": ["<most important thing to fix>", "<second most important>", "<third most important>"],
  "likely_follow_up_question": "<the follow-up question this interviewer would ask next — tests if student truly understands>",
  "interviewer_verdict": "<as a senior who wants this student placed: would you back them for this role? What's the one thing standing between them and the offer?>"
}
"""


# ==============================================================================
# -------------------- ENDPOINTS --------------------
# ==============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "topics_loaded": TOPICS_DATA is not None,
        "aptitude_loaded": APTITUDE_DATA is not None,
        "prompt_version": "v2"
    }

@app.get("/get-branches")
def get_branches():
    global TOPICS_DATA
    if TOPICS_DATA is None:
        load_topics()
    if TOPICS_DATA is None:
        return {"error": "Failed to load topics"}
    branches = sorted(TOPICS_DATA['Branch'].unique().tolist())
    return {"branches": branches}

@app.get("/get-semesters")
def get_semesters():
    global TOPICS_DATA
    if TOPICS_DATA is None:
        load_topics()
    if TOPICS_DATA is None:
        return {"error": "Failed to load topics"}
    semesters = sorted([int(s) for s in TOPICS_DATA['Semester'].unique()])
    return {"semesters": semesters}

@app.get("/get-topics")
async def get_topics(branch: str, semester: int):
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
        topics_list.append({"topic": row['Topic'], "subject": row['Subject']})
    return {"topics": topics_list, "count": len(topics_list)}


@app.post("/generate-question")
async def generate_question(request: Request):
    """
    Generate a question from selected branch, semester, and optional specific topic.
    
    UPGRADE: Now uses gpt-4o (was gpt-4o-mini) and the new question_gen_prompt
    that forces application/derivation questions instead of pure definitions.
    Returns what_interviewer_wants to help the student frame their answer.
    """
    try:
        global TOPICS_DATA
        if TOPICS_DATA is None:
            load_topics()
        if TOPICS_DATA is None:
            return {"error": "Topics database not loaded"}

        data = await request.json()
        branch = data.get("branch")
        semester = data.get("semester")
        specific_topic = data.get("specific_topic", None)

        if not branch or not semester:
            return {"error": "Branch and semester required"}

        filtered = TOPICS_DATA[
            (TOPICS_DATA['Branch'] == branch) &
            (TOPICS_DATA['Semester'] == int(semester))
        ]

        if len(filtered) == 0:
            return {"error": f"No topics found for {branch} Semester {semester}"}

        if specific_topic:
            matched = filtered[filtered['Topic'] == specific_topic]
            if len(matched) == 0:
                return {"error": f"Topic '{specific_topic}' not found"}
            random_row = matched.iloc[0]
        else:
            random_row = filtered.sample(n=1).iloc[0]

        topic = random_row['Topic']
        subject = random_row['Subject']
        diagram_url = get_diagram_for_topic(topic)

        prompt = f"""Generate a placement interview question for:
Subject: {subject}
Topic: {topic}
Branch: {branch}
Semester: {semester}

This student is a final-year {branch} student preparing for campus placements at companies like TCS, Infosys, L&T, Bosch, and core engineering firms.
The question must be specific to {topic} within {subject} — not a generic question."""

        response = client.chat.completions.create(
            model="gpt-4o",          # UPGRADED from gpt-4o-mini
            messages=[
                {"role": "system", "content": question_gen_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,         # LOWERED from 0.7 — more consistent quality
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return {
            "topic": topic,
            "subject": subject,
            "branch": branch,
            "semester": semester,
            "question": result.get("question", ""),
            "what_interviewer_wants": result.get("what_interviewer_wants", ""),
            "hint": result.get("hint", ""),
            "diagram": diagram_url
        }

    except Exception as e:
        return {"error": f"Failed to generate question: {str(e)}"}


@app.post("/analyze-practice")
async def analyze_practice(request: Request):
    """
    Evaluate answer in practice mode.

    UPGRADE: Now uses gpt-4o (was gpt-4o-mini). The new practice_prompt
    produces a complete ideal_answer (250+ words, with formulae and examples)
    instead of a "concise" 2–3 sentence summary that left students unprepared
    for follow-up questions.
    """
    try:
        data = await request.json()
        question = data.get("question")
        answer = data.get("answer")
        topic = data.get("topic", "")
        subject = data.get("subject", "")

        if not question or not answer:
            return {"error": "Question and answer required"}

        user_input = f"""Subject: {subject}
Topic: {topic}
Interview Question: {question}
Student's Answer: {answer}"""

        response = client.chat.completions.create(
            model="gpt-4o",          # UPGRADED from gpt-4o-mini
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
    """
    Evaluate answer in custom mode (student's own question).

    UPGRADE: Now uses gpt-4o. New custom_prompt adds:
    - hire_decision field (clear threshold language)
    - depth_of_understanding dimension (catches surface memorisation)
    - top_3_improvements instead of single improvement_priority
    - likely_follow_up_question (trains students for real interview dynamics)
    - Complete ideal_answer (was capped at 2–3 paragraphs)
    - precise critical_gaps (names exact formulae/concepts missed)
    """
    try:
        data = await request.json()
        question = data.get("question")
        answer = data.get("answer")

        if not question or not answer:
            return {"error": "Question and answer required"}

        user_input = f"""Interview Question: {question}
Student's Answer: {answer}"""

        response = client.chat.completions.create(
            model="gpt-4o",          # UPGRADED from gpt-4o-mini
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

VERBAL_CATEGORIES = {
    'analogies', 'synonyms', 'antonyms', 'sentence correction',
    'syllogism', 'classification', 'verbal reasoning', 'grammar',
    'vocabulary', 'comprehension', 'reading comprehension',
    'fill in the blanks', 'idioms', 'one word substitution',
    'sentence arrangement', 'para jumbles'
}

@app.get("/get-aptitude-categories")
def get_aptitude_categories():
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
    try:
        global APTITUDE_DATA
        if APTITUDE_DATA is None:
            load_aptitude()
        if APTITUDE_DATA is None:
            return {"error": "Excel file not found. Make sure aptitude_verbal_questions.xlsx is in your repo."}

        print(f"Aptitude columns: {list(APTITUDE_DATA.columns)}")

        data = await request.json()
        category = data.get("category", None)
        type_filter = data.get("type", None)

        if category:
            filtered = APTITUDE_DATA[APTITUDE_DATA['Category'] == category]
        elif type_filter and type_filter != "all":
            if type_filter == "verbal":
                mask = APTITUDE_DATA['Category'].str.strip().str.lower().isin(VERBAL_CATEGORIES)
            else:
                mask = ~APTITUDE_DATA['Category'].str.strip().str.lower().isin(VERBAL_CATEGORIES)
            filtered = APTITUDE_DATA[mask]
        else:
            filtered = APTITUDE_DATA

        if len(filtered) == 0:
            return {"error": "No questions found for this selection."}

        random_q = filtered.sample(n=1).iloc[0]
        cols = {c.strip().lower().replace(" ", ""): c for c in APTITUDE_DATA.columns}

        def get_col(name):
            key = name.lower().replace(" ", "")
            actual = cols.get(key)
            if actual is None:
                raise KeyError(f"Column '{name}' not found. Available: {list(APTITUDE_DATA.columns)}")
            return random_q[actual]

        return {
            "category":       str(get_col("Category")),
            "subcategory":    str(get_col("Subcategory")) if "subcategory" in cols else "",
            "question":       str(get_col("Question")),
            "options": {
                "a": str(get_col("OptionA")),
                "b": str(get_col("OptionB")),
                "c": str(get_col("OptionC")),
                "d": str(get_col("OptionD")),
            },
            "correct_answer": str(get_col("Answer")).strip().lower(),
            "explanation":    str(get_col("Explanation")),
        }

    except Exception as e:
        print(f"generate-aptitude error: {e}")
        return {"error": f"Server error: {str(e)}"}

@app.post("/check-aptitude")
async def check_aptitude(request: Request):
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

# -------------------- SEO ENDPOINTS --------------------
@app.get("/sitemap.xml")
def sitemap():
    content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://prepto.in/</loc>
    <priority>1.0</priority>
  </url>
</urlset>"""
    return Response(content=content, media_type="application/xml")

@app.get("/robots.txt")
def robots():
    content = "User-agent: *\nAllow: /\nSitemap: https://prepto.in/sitemap.xml"
    return Response(content=content, media_type="text/plain")

# -------------------- SERVE STATIC FILES --------------------
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
