#  2025 Applied AI Engineer Internship Project

Your work must be your own and original. You may use AI tools to help aid your work if you include a single text file containing an ordered list of any AI prompts, along with the specific model queried (e.g. ChatGPT 5 Thinking) in the `prompts` directory. Do not include the AI's output.

## Assignment: 

The goal of this project is to build an end-to-end RAG pipeline with an interactive chat interface for answering basic NBA stats questions.

1. Load data – Ingest CSVs related to NBA game information from the 2023-24 and 2024-25 seasons into PostgresSQL tables. Note this data is limited to only matchups involving at least one Western Conference team for size considerations.
2. Create embeddings – Generate text embeddings with Ollama [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) and store them alongside the source rows.
3. Retrieve and join – Perform semantic retrieval using the `pgvector` extension to find relevant game summaries, then join the matched embeddings back to the original structured table rows to provide factual context.
4. Answer questions – Use Llama [`llama3.2:3b`](https://ollama.com/library/llama3.2:3b) to produce answers grounded on the retrieved data to the questions under the **Submission Requirements** section. If you find this model too large for your machine, feel free to use a smaller model and note this in your submission.

## Quick Start: Part 1
1) Install [`Docker Desktop`](https://www.docker.com/get-started/) and open it (to ensure the docker daemon is running).
2) Clone this repository.
3) Start services and pull models by running the following commands:
```bash
docker compose up -d db ollama
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.2:3b
docker compose build app
```

Edit these files and run them using the following commands:

1) Ingestion (`backend/ingest.py`) for schema details
```
docker compose run --rm app python -m backend.ingest
```

2) Embedding (`backend/embed.py`) for text serialization strategy. **Note the embedding process can take a long time to complete depending on your machine**
```
docker compose run --rm app python -m backend.embed
```

3) RAG Script (`backend/rag.py`) for retrieval joins, prompt, and answer formatting. This script generates answers to the 10 prompts in Part 1.
```
docker compose run --rm app python -m backend.rag
```

## Quick Start: Part 2

### Run the backend server
```
docker compose run --rm --service-ports app uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

### Installing Prerequisites
Install Node.js (16.x.x), then in a new tab, run the following commands
```
cd /path/to/project/frontend
# Install Angular-Cli
npm install -g @angular/cli@15.1.0 typescript@4.9.4 --force
# Install dependencies
npm install --force
# Start the frontend
npm start
```

The frontend should run on http://localhost:4200/. Visit this address to see the app in your browser.


**Part 2: Frontend Solution**

- Create a chat interface for interacting with the backend retrieval pipeline. Some minimal Angular skeleton code is provided in the [`frontend/src/app`](frontend/src/app) directory, feel free to edit it as you wish.

Submit a video in the [`part2`](part2) folder that demonstrates how your UI functions.
