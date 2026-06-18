import os
import json
import sqlalchemy as sa
from sqlalchemy import text
from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate


# --- File paths ---
BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.json"))
TEMPLATE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers_template.json"))


# --- Retrieve top matching games ---
def retrieve_games(cx, qvec, k):
    sql = """
        SELECT 
            gd.game_id,
            gd.season,
            gd.game_timestamp,
            gd.home_points,
            gd.away_points,
            gd.winning_team_id,
            th.name AS home_team,
            ta.name AS away_team,
            wt.name AS winning_team_name,
            1 - (gd.embedding <=> (:q)::vector) AS score
        FROM game_details gd
        JOIN teams th ON gd.home_team_id = th.team_id
        JOIN teams ta ON gd.away_team_id = ta.team_id
        LEFT JOIN teams wt ON gd.winning_team_id = wt.team_id
        ORDER BY gd.embedding <-> (:q)::vector
        LIMIT :k
    """
    return cx.execute(text(sql), {"q": qvec, "k": k}).mappings().all()


# --- Retrieve all player stats for a specific game ---
def retrieve_players(cx, game_id):
    sql = """
        SELECT 
            pbs.game_id,
            pbs.person_id,
            p.first_name,
            p.last_name,
            t.name AS team_name,
            pbs.points,
            pbs.assists,
            (pbs.offensive_reb + pbs.defensive_reb) AS rebounds,
            pbs.fg2_made, pbs.fg2_attempted,
            pbs.fg3_made, pbs.fg3_attempted,
            pbs.ft_made,  pbs.ft_attempted,
            pbs.blocks, pbs.steals, pbs.turnovers
        FROM player_box_scores pbs
        JOIN players p ON p.player_id = pbs.person_id
        JOIN teams t ON pbs.team_id = t.team_id
        WHERE pbs.game_id = :gid
        ORDER BY pbs.points DESC
    """
    return cx.execute(text(sql), {"gid": game_id}).mappings().all()


# --- Build the context ---
def build_context(game, players):
    game_section = (
        f"SELECTED_GAME:\n"
        f"- game_id: {game['game_id']}\n"
        f"- date: {game['game_timestamp']}\n"
        f"- home_team: {game['home_team']} ({game['home_points']})\n"
        f"- away_team: {game['away_team']} ({game['away_points']})\n"
        f"- winning_team_name: {game.get('winning_team_name')}\n"
    )

    if not players:
        return game_section

    players_section = "PLAYERS_IN_SELECTED_GAME:\n" + "\n".join(
        [
            f"- {p['first_name']} {p['last_name']} ({p['team_name']}) | "
            f"points: {p['points']}, assists: {p['assists']}, rebounds: {p['rebounds']}, "
            f"fg2: {p['fg2_made']}/{p['fg2_attempted']}, fg3: {p['fg3_made']}/{p['fg3_attempted']}, "
            f"ft: {p['ft_made']}/{p['ft_attempted']}, steals: {p['steals']}, blocks: {p['blocks']}, turnovers: {p['turnovers']}"
            for p in players
        ]
    )
    return f"{game_section}\n{players_section}"


# --- LLM-only answer generation ---
def answer(question, game, players):
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        answers_template = json.load(f)

    ctx = build_context(game, players)

    reasoning_instruction = """
You are an assistant that answers questions about NBA games and players using ONLY the provided context.

Follow these reasoning rules:
    - Consider only the SELECTED_GAME and its players. Ignore all other games.
    - If the question asks who won or what was the final score, use the 'winning_team_name' and the scores from SELECTED_GAME.
    - If the question asks for a leading scorer or top performer, examine PLAYERS_IN_SELECTED_GAME and compare numbers directly (e.g., 40 > 28 > 25).
    - Always select ONLY the single best answer unless there is a true tie.
    - Never invent data, names, or tables not shown in the context.
    - Always return EXACTLY ONE JSON array, formatted according to the provided template.
    - Do not include explanations or extra text, only valid JSON.
"""

    prompt = f"""{reasoning_instruction}

Question:
{question}

Answer format (use this JSON structure):
{json.dumps(answers_template, indent=2)}

Context:
{ctx}

JSON Answer:
"""

    print("\n==================== QUESTION ====================")
    print(question)
    print("\n--- CONTEXT (first 900 chars) ---")
    print(ctx[:900])
    print("==================================================\n")

    return ollama_generate(LLM_MODEL, prompt)


# --- Main RAG pipeline ---
if __name__ == "__main__":
    eng = sa.create_engine(DB_DSN)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        qs = json.load(f)

    outs = []

    with eng.begin() as cx:
        for q in qs:
            question_text = q["question"]

            # --- Detect year mentioned in question ---
            year_hint = None
            for y in ["2022", "2023", "2024", "2025"]:
                if y in question_text:
                    year_hint = y
                    break
            if year_hint:
                print(f"[INFO] Year hint detected: {year_hint}")

            print(f"\n[INFO] Processing question: {question_text}")
            qvec = ollama_embed(EMBED_MODEL, question_text)

            games = retrieve_games(cx, qvec, k=5)

            # --- Post-filter retrieved games by year, if mentioned ---
            if year_hint:
                filtered = [g for g in games if str(year_hint) in str(g["game_timestamp"])]
                if filtered:
                    print(f"[INFO] Filtered to {len(filtered)} games from {year_hint}")
                    games = filtered

            if not games:
                print("[WARN] No game found for this question.")
                continue

            selected_game = games[0]
            players = retrieve_players(cx, selected_game["game_id"])
            ans = answer(question_text, selected_game, players)

            outs.append({
                "id": q.get("id"),
                "question": q["question"],
                "answer": ans,
                "evidence": [
                    {"table": "game_details", "id": int(selected_game["game_id"])}
                ] + [
                    {"table": "player_box_scores", "id": int(p["game_id"])} for p in players
                ],
            })

    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False, indent=2)

    print(f"\nAnswers written to {ANSWERS_PATH}")
