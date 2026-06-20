import os, json, requests
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 1. The REAL tool — your live NewsAPI fetcher
def get_headlines(topic):
    url = "https://newsapi.org/v2/everything"
    params = {"q": topic, "apiKey": os.environ["NEWS_API_KEY"],
              "language": "en", "sortBy": "publishedAt", "pageSize": 5}
    articles = requests.get(url, params=params).json().get("articles", [])
    if not articles:
        return "No headlines found."
    return "\n".join("- " + a["title"] for a in articles)

# 2. The TOOL MENU — describe it so the model knows WHEN to use it
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_headlines",
            "description": "Fetch the latest real news headlines about a topic. Use this when the user asks about current events, recent developments, or what's happening now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to search news for"},
                },
                "required": ["topic"],
            },
        },
    }
]

# 3. Ask + hand over the menu
question = "What's a good team size for a hackathon?"
messages = [
    {"role": "system", "content": "You are a sharp, practical hackathon mentor. Answer questions directly and confidently with specific, useful advice. Only use your news tool when the question is about current events or recent developments."},
    {"role": "user", "content": question}
]
r = client.chat.completions.create(model="gemini-2.5-flash", messages=messages, tools=tools)
msg = r.choices[0].message

# 4. Did the agent DECIDE to fetch news?
if msg.tool_calls:
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    print("Agent decided to fetch news on:", args["topic"])

    # 5. YOUR code runs the real tool
    headlines = get_headlines(args["topic"])
    print("\nHeadlines fetched:\n", headlines)

    # 6. Feed headlines back → grounded answer
    messages.append(msg)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": headlines})
    final = client.chat.completions.create(model="gemini-2.5-flash", messages=messages, tools=tools)
    print("\nAgent's grounded answer:\n", final.choices[0].message.content)
else:
    print("Agent answered directly (no news needed):\n", msg.content)