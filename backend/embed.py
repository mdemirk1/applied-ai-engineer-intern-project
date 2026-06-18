import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from backend.config import DB_DSN, EMBED_MODEL
from backend.utils import ollama_embed

# Tables we want embeddings for
TABLES_TO_EMBED = ["game_details", "player_box_scores"]

BASE_DIR = os.path.dirname(__file__)

def row_to_text(table_name, row):
    """
    Converts a database row into a text string suitable for embeddings.
    Includes team and player names for better semantic understanding.
    """
    if table_name == "game_details":
        ts = pd.to_datetime(row.game_timestamp, utc=True)
        date = ts.strftime('%Y-%m-%d')
        return (
            f"game | season:{row.season} | date:{date} | "
            f"{row.home_team_name} ({row.home_team_id}) vs {row.away_team_name} ({row.away_team_id}) | "
            f"home_points:{row.home_points} | away_points:{row.away_points}"
            f"winning_team_id:{row.winning_team_id}"
        )

    elif table_name == "player_box_scores":
        return (
            f"player_box | game_id:{row.game_id} | "
            f"{row.first_name} {row.last_name} ({row.person_id}) | team_id:{row.team_id} | "
            f"starter:{row.starter} | points:{row.points} | assists:{row.assists} | "
            f"rebounds:{row.offensive_reb + row.defensive_reb} | "
            f"fg2:{row.fg2_made}/{row.fg2_attempted} | fg3:{row.fg3_made}/{row.fg3_attempted} | "
            f"ft:{row.ft_made}/{row.ft_attempted} | steals:{row.steals} | blocks:{row.blocks} | turnovers:{row.turnovers}"
        )

    else:
        # fallback: convert all columns to a string
        return " | ".join(f"{c}:{row[c]}" for c in row.index)

def main():
    print("Starting Embedding Process")
    eng = sa.create_engine(DB_DSN)

    with eng.begin() as cx:
        # Ensure pgvector extension exists
        cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        for table_name in TABLES_TO_EMBED:
            # Ensure embedding column exists
            cx.execute(
                text(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS embedding vector(768);")
            )
            # Create HNSW index for fast similarity search
            cx.execute(
                text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding "
                     f"ON {table_name} USING hnsw (embedding vector_cosine_ops);")
            )

            # Load table rows with joins to get names
            if table_name == "game_details":
                # Join teams to get home and away team names
                df = pd.read_sql("""
                    SELECT gd.*, 
                           th.name AS home_team_name, 
                           ta.name AS away_team_name
                    FROM game_details gd
                    JOIN teams th ON gd.home_team_id = th.team_id
                    JOIN teams ta ON gd.away_team_id = ta.team_id
                """, cx)
            elif table_name == "player_box_scores":
                # Join players to get player names
                df = pd.read_sql("""
                    SELECT pbs.*, p.first_name, p.last_name
                    FROM player_box_scores pbs
                    JOIN players p ON pbs.person_id = p.player_id
                """, cx)
            else:
                df = pd.read_sql(f"SELECT * FROM {table_name}", cx)

            print(f"Embedding {len(df)} rows from table '{table_name}'")

            # Update embeddings row by row
            for _, row in df.iterrows():
                vec = ollama_embed(EMBED_MODEL, row_to_text(table_name, row))

                if table_name == "game_details":
                    cx.execute(
                        text("UPDATE game_details SET embedding = :v WHERE game_id = :id"),
                        {"v": vec, "id": int(row.game_id)},
                    )

                elif table_name == "player_box_scores":
                    cx.execute(
                        text(
                            "UPDATE player_box_scores "
                            "SET embedding = :v "
                            "WHERE game_id = :gid AND person_id = :pid"
                        ),
                        {"v": vec, "gid": int(row.game_id), "pid": int(row.person_id)},
                    )

                else:
                    # fallback: first column as ID
                    cx.execute(
                        text(f"UPDATE {table_name} SET embedding = :v WHERE {df.columns[0]} = :id"),
                        {"v": vec, "id": row[df.columns[0]]},
                    )

    print("Finished Embeddings for all tables.")

if __name__ == "__main__":
    main()
