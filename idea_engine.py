import os, json, time
import requests
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- Shared helper: send a system+user prompt, get clean parsed JSON back ---
def ask_json(system, user):
    r = client.chat.completions.create(
        model="gemini-2.5-flash",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = r.choices[0].message.content
    fence = "`" * 3
    cleaned = raw.strip()
    if cleaned.startswith(fence):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


# --- Embeddings + similarity (concept 5) ---
def embed(text):
    r = client.embeddings.create(model="gemini-embedding-001", input=text)
    return r.data[0].embedding

def similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / ((sum(x*x for x in a)**0.5) * (sum(y*y for y in b)**0.5))


# --- News tool (real headlines) ---
def get_headlines(topic):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "apiKey": os.environ["NEWS_API_KEY"],
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
    }
    data = requests.get(url, params=params).json()
    if data.get("status") != "ok":
        print("News API error:", data)
        return []
    return [article["title"] for article in data["articles"]]


# --- AGENT 1: the problem finder ---
def find_problems(topic):
    headlines_text = "\n".join(get_headlines(topic))
    system = ("You are a sharp hackathon mentor. You read news headlines and extract real problems "
              "solvable with AI. Respond ONLY with a JSON list; each item has keys "
              "'problem' and 'why_ai_helps'.")
    user = f"Topic: {topic}\n\nHeadlines:\n{headlines_text}"
    return ask_json(system, user)


# --- AGENT 2: the idea generator + judge ---
def generate_ideas(problem):
    system = ("You are a hackathon idea generator and judge. Given ONE problem, propose 3 concrete "
              "AI project ideas to solve it. Score each from 1-10 on innovation, feasibility, and "
              "impact. Respond ONLY with a JSON list; each item has keys "
              "'idea', 'innovation', 'feasibility', 'impact', and 'pitch'.")
    user = f"Problem: {problem}"
    return ask_json(system, user)


# --- DIFFERENTIATION ENGINE: knowledge base of overdone projects ---
generic_projects = [
    "A to-do list and task management app",
    "An expense tracker and budgeting app",
    "A weather forecast app",
    "A basic FAQ chatbot",
    "A movie or book recommendation system",
]
generic_embeddings = []
for p in generic_projects:
    generic_embeddings.append(embed(p))
    time.sleep(1)            # gentle on the free-tier rate limit

def novelty_check(idea_text):
    e = embed(idea_text)
    scores = [similarity(e, ge) for ge in generic_embeddings]
    closest = generic_projects[scores.index(max(scores))]
    top = max(scores)
    if top > 0.7:
        return f"TOO GENERIC (close to '{closest}') - push further", round(top, 3)
    return "Fresh - not close to the usual overdone projects", round(top, 3)


# --- MAIN FLOW: Agent 1 -> Agent 2 -> novelty check ---
topic = "cybersecurity"

problems = find_problems(topic)
print(f"=== Problems found for: {topic} ===")
for i, p in enumerate(problems, 1):
    print(f"{i}. {p['problem']}")

chosen = problems[0]["problem"]
print(f"\n>>> Generating ideas for: {chosen}\n")

ideas = generate_ideas(chosen)
print("=== Hackathon ideas (scored + novelty-checked) ===\n")
for i, idea in enumerate(ideas, 1):
    total = idea["innovation"] + idea["feasibility"] + idea["impact"]
    verdict, novelty = novelty_check(idea["idea"])
    time.sleep(1)
    print(f"{i}. {idea['idea']}   (score: {total}/30)")
    print(f"   Novelty: {verdict}  [closeness: {novelty}]")
    print(f"   Pitch: {idea['pitch']}\n")