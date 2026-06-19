import os, json
import requests
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- Shared helper: send a system+user prompt, get clean parsed JSON back ---
# (We call the model twice now, so we wrap the repeated work once.)
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


# --- News tool (unchanged) ---
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


# --- MAIN FLOW: Agent 1 hands off to Agent 2 ---
topic = "cybersecurity"

problems = find_problems(topic)                      # AGENT 1 runs
print(f"=== Problems found for: {topic} ===")
for i, p in enumerate(problems, 1):
    print(f"{i}. {p['problem']}")

chosen = problems[0]["problem"]                       # pick the first problem...
print(f"\n>>> Handing this to Agent 2: {chosen}\n")

ideas = generate_ideas(chosen)                        # AGENT 2 runs (input = Agent 1's output!)
print("=== Hackathon ideas (scored) ===\n")
for i, idea in enumerate(ideas, 1):
    total = idea["innovation"] + idea["feasibility"] + idea["impact"]
    print(f"{i}. {idea['idea']}   (score: {total}/30)")
    print(f"   Innovation {idea['innovation']} | Feasibility {idea['feasibility']} | Impact {idea['impact']}")
    print(f"   Pitch: {idea['pitch']}\n")