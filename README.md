# Hackathon Idea Engine

An agentic AI system that finds real problems from live news and generates *original*,
hackathon-ready project ideas — then checks each idea for originality so you never pitch
the same tired app everyone else does.

## Why I built this
99% of hackathon and portfolio projects are the same generic ideas (to-do apps, expense
trackers). I wanted a tool that surfaces *real* problems from current events and pushes
toward genuinely novel solutions.

## What it does
1. Problem Finder (Agent 1) — pulls live news headlines for any topic and extracts real,
   AI-solvable problems.
2. Idea Generator + Judge (Agent 2) — turns a problem into 3 concrete project ideas, each
   scored on innovation, feasibility, and impact.
3. Differentiation Engine — embeds each idea and compares it (vector similarity) against a
   database of overdone projects, flagging anything too generic.

## Architecture
topic -> Agent 1 (news -> problems) -> Agent 2 (problem -> scored ideas) -> Novelty check

A multi-agent pipeline where each agent's output feeds the next.

## Tech & concepts
- Agentic AI: multi-agent pipeline with handoffs
- Tool use: live NewsAPI integration
- Structured outputs: reliable JSON parsing
- RAG / embeddings: vector similarity for novelty checking
- LLM: Google Gemini (via OpenAI-compatible API)
- Python

## Run it
Install dependencies:

    pip install openai python-dotenv requests

Create a `.env` file:

    GEMINI_API_KEY=your_key
    NEWS_API_KEY=your_key

Run:

    python3 idea_engine.py

