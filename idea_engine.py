import os, json
import requests
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# BEAT 1 — SETUP (the phone line)
client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Our "news tool" — now REAL. Fetches live headlines for the given topic.
def get_headlines(topic):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,                              # search the news for our topic
        "apiKey": os.environ["NEWS_API_KEY"],
        "language": "en",
        "sortBy": "publishedAt",                 # newest first
        "pageSize": 5,                           # just 5 headlines
    }
    data = requests.get(url, params=params).json()

    if data.get("status") != "ok":              # NewsAPI tells us if something went wrong
        print("News API error:", data)
        return []

    return [article["title"] for article in data["articles"]]   # pull out just the titles


topic = "cybersecurity"

headlines = get_headlines(topic)                 # fetch REAL news (our tool)
headlines_text = "\n".join(headlines)

# BEAT 2 — PREPARE what we send (headlines + instructions into context)
messages = [
    {"role": "system", "content":
        "You are a sharp hackathon mentor. You read news headlines and extract real problems "
        "solvable with AI. Respond ONLY with a JSON list; each item has keys "
        "'problem' and 'why_ai_helps'."},
    {"role": "user", "content": f"Topic: {topic}\n\nHeadlines:\n{headlines_text}"}
]

# BEAT 3 — THE CALL
r = client.chat.completions.create(
    model="gemini-2.5-flash",
    temperature=0,
    messages=messages,
)

# BEAT 4 — READ the reply
raw = r.choices[0].message.content

# BEAT 5 — USE IT: strip JSON fences, parse, print nicely
fence = "`" * 3
cleaned = raw.strip()
if cleaned.startswith(fence):
    cleaned = cleaned.strip("`").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

problems = json.loads(cleaned)

print(f"=== Hackathon problems for: {topic} ===\n")
for i, p in enumerate(problems, 1):
    print(f"{i}. {p['problem']}")
    print(f"   Why AI helps: {p['why_ai_helps']}\n")