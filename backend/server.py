from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy import text
from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

eng = sa.create_engine(DB_DSN)

class Q(BaseModel):
    question: str


@app.post("/api/chat")
def answer(q: Q):
    print(f"[INFO] Received question: {q.question}")

    qvec = ollama_embed(EMBED_MODEL, q.question)
    qvec_str = "[" + ",".join(map(str, qvec)) + "]"

    with eng.begin() as cx:
        rows = cx.execute(text("""
            SELECT 
                gd.game_id,
                gd.game_timestamp,
                gd.home_points,
                gd.away_points,
                gd.winning_team_id,
                th.name AS home_team,
                ta.name AS away_team,
                wt.name AS winning_team_name
            FROM game_details gd
            JOIN teams th ON gd.home_team_id = th.team_id
            JOIN teams ta ON gd.away_team_id = ta.team_id
            LEFT JOIN teams wt ON gd.winning_team_id = wt.team_id
            ORDER BY gd.embedding <-> (:vec)::vector
            LIMIT :k
        """), {"vec": qvec_str, "k": 5}).mappings().all()

    ctx = "\n".join([
        f"[Game {r['game_id']}] {r['game_timestamp']}: "
        f"{r['home_team']} ({r['home_points']}) vs {r['away_team']} ({r['away_points']}) | "
        f"Winner: {r['winning_team_name']}"
        for r in rows
    ])

    prompt = f"""
        You are ThunderBot ⚡, an NBA data assistant. 
        Use ONLY the factual context below to answer the question accurately.
        Write your response naturally, like a basketball analyst — short, confident, and conversational. 
        Do NOT add information not supported by the context.

        Context:
        {ctx}

        Question:
        {q.question}

        Answer (natural but factual):
    """

    resp = ollama_generate(LLM_MODEL, prompt).strip()

    return {
        "answer": resp,
        "evidence": [{"table": "game_details", "id": int(r["game_id"])} for r in rows],
    }
