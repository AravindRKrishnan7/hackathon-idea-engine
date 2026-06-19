import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# MEMORY: this list holds the WHOLE conversation. It grows every turn.
# It starts with the agent's job description (the system prompt).
messages = [
    {"role": "system", "content":
        "You are an energetic hackathon brainstorming buddy. The user pitches project ideas. "
        "You encourage good instincts, ask sharp questions, and push every idea to be more "
        "innovative, feasible, and impactful. Keep replies short - 2 to 4 sentences."}
]

print("Hackathon Buddy is ready! Pitch an idea. (type 'quit' to exit)\n")

# THE AGENT LOOP: read -> respond -> remember -> repeat
while True:
    user_input = input("You: ")                  # wait for you to type
    if user_input.lower() == "quit":             # a way out of the loop
        print("Buddy: Good luck at the hackathon!")
        break

    messages.append({"role": "user", "content": user_input})   # REMEMBER your message

    r = client.chat.completions.create(
        model="gemini-2.5-flash",
        temperature=0.7,                          # some creativity for brainstorming
        messages=messages,                        # send the WHOLE history every time
    )
    reply = r.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})   # REMEMBER the buddy's reply
    print("Buddy:", reply, "\n")