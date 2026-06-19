import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Turn text into an embedding: a list of numbers that captures its MEANING
def embed(text):
    r = client.embeddings.create(model="gemini-embedding-001", input=text)
    return r.data[0].embedding

# BLACK BOX: how close are two vectors? 1 = same meaning, 0 = unrelated.
# (Don't parse the math - just trust the number. That's altitude. ✈️)
def similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = sum(x*x for x in a) ** 0.5
    mag_b = sum(y*y for y in b) ** 0.5
    return dot / (mag_a * mag_b)

s1 = "An AI tutor that gives students personalized feedback"
s2 = "A learning app that adapts to each student's needs"
s3 = "A tool to find which restaurants are open right now"

e1, e2, e3 = embed(s1), embed(s2), embed(s3)

print("An embedding is a list of", len(e1), "numbers\n")
print("s1 vs s2 (both education):   ", round(similarity(e1, e2), 3))
print("s1 vs s3 (education vs food):", round(similarity(e1, e3), 3))
print("s2 vs s3 (education vs food):", round(similarity(e2, e3), 3))