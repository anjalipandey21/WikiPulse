# WikiPulse

WikiPulse is an agentic web application that converts weekly Wikipedia attention data into commercially meaningful audience segments.

It was built for the **AI Builder Candidate Challenge — Trending Audience Builder** using the public Wikimedia Pageviews API, semantic clustering, a LangGraph-based LLM workflow, and a React dashboard.

## What It Does

WikiPulse:

- Fetches the latest complete seven-day Wikipedia pageview window
- Filters administrative, noisy, and commercially irrelevant articles
- Enriches selected articles with Wikipedia summaries
- Groups related articles using TF-IDF, sentence embeddings, and deterministic clustering
- Generates market-ready audience recommendations with an LLM
- Validates cluster identity, evidence references, and calculated metrics in Python
- Supports automatic analysis and bounded analyst review
- Provides grounded follow-up Q&A for published review evidence

Each generated audience can include:

- Audience name
- Audience description
- Estimated size index
- Buying-power assessment
- Recommended brand categories
- Commercial-confidence score
- Supporting Wikipedia evidence

## Live Link

>https://wikipulse-ai.vercel.app/

Suggested demo flow:

1. Run a Standard Analysis
2. Show the live pipeline progress
3. Review the generated topic clusters
4. Open the Emerging Audience Portfolio
5. Switch to Analyst Review
6. Approve, reject, or request one bounded edit
7. Ask WikiPulse a grounded follow-up question

## Architecture

```text
Wikimedia Pageviews API
        |
        v
Normalization and Noise Filtering
        |
        v
Wikipedia Summary Enrichment
        |
        v
TF-IDF + Sentence Embeddings
        |
        v
Deterministic Topic Clustering
        |
        v
Commercial-Safety Routing
        |
        v
Evidence Preparation
        |
        v
LangGraph Audience Workflow
        |
        +--> Standard Analysis
        |
        +--> Analyst Review
        |
        v
FastAPI API
        |
        v
React Dashboard
```

The application keeps responsibilities separated across:

- Data-access services
- Filtering and clustering
- LLM generation and validation
- Review-state management
- API routes
- Frontend presentation

## Tech Stack

### Backend

- Python
- FastAPI
- LangGraph
- OpenAI Responses API
- Pydantic
- HTTPX
- scikit-learn
- Sentence Transformers
- Uvicorn

### Frontend

- React
- TypeScript
- Vite
- Native Fetch and ReadableStream
- CSS

## Project Structure

```text
WikiPulse/
├── backend/
│   ├── app/
│   │   ├── agent/          # LangGraph workflows and LLM providers
│   │   ├── api/            # FastAPI routes
│   │   ├── clustering/     # Keywords, embeddings, and grouping
│   │   ├── filtering/      # Noise and commercial-safety rules
│   │   ├── models/         # Internal and public contracts
│   │   ├── services/       # Wikimedia and Wikipedia clients
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── App.tsx
│   │   └── index.css
│   ├── tests/
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## Prerequisites

Install:

- Python 3.12
- Node.js 20.19 or newer
- npm
- Git
- An OpenAI API key

The Wikimedia APIs used by the project are public and do not require an API key.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/anjalipandey21/WikiPulse.git
cd WikiPulse
```

### 2. Start the backend

#### macOS or Linux

```bash
cd backend

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export OPENAI_API_KEY="your-openai-api-key"

python -m uvicorn app.main:app \
  --reload \
  --host 127.0.0.1 \
  --port 8001
```

#### Windows PowerShell

```powershell
cd backend

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:OPENAI_API_KEY="your-openai-api-key"

python -m uvicorn app.main:app `
  --reload `
  --host 127.0.0.1 `
  --port 8001
```

Backend API documentation:

```text
http://127.0.0.1:8001/docs
```

### 3. Start the frontend

Open a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite development server proxies `/api` requests to the backend on port `8001`.

## Main API Routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/audience-analysis` | Run a complete analysis |
| `POST` | `/api/audience-analysis/stream` | Run an analysis with live progress |
| `POST` | `/api/audience-reviews` | Start an analyst-review run |
| `GET` | `/api/audience-reviews/{run_id}` | Read review state |
| `POST` | `/api/audience-reviews/{run_id}/commands` | Approve, reject, or edit |
| `POST` | `/api/audience-reviews/{run_id}/questions` | Ask a grounded follow-up question |

## Running Tests

### Backend

From `backend/`:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Frontend

From `frontend/`:

```bash
npm test
npm run lint
npx tsc -b
npm run build
```

## Important Design Decisions

### Clustering happens before the LLM

The LLM does not decide which articles belong together. Topic membership is created through deterministic lexical and embedding-based clustering.

This makes the results easier to inspect and reduces hallucination risk.

### Calculations remain Python-owned

Pageviews, audience-size calculations, evidence ownership, and cluster identity are validated in application code instead of being trusted from model output.

### LLM output is structured and bounded

The workflow uses structured model output and a bounded revision path. Invalid decisions are either revised once or converted into explicit drop outcomes.

### Human review is separate from automatic analysis

Standard Analysis runs automatically.

Analyst Review uses a separate workflow where each candidate can be:

- Approved
- Rejected
- Edited once within an allowlisted set of fields

## Security

- Do not commit `.env` files
- Never expose the OpenAI API key to the frontend
- Keep provider calls server-side
- Do not log private analyst feedback
- Validate all external responses
- Do not expose raw model output or model reasoning

Example local environment file:

```text
OPENAI_API_KEY=your-openai-api-key
```

Keep it in `backend/.env` or your shell environment and ensure it is ignored by Git.

## Current Limitations

This is a challenge prototype intended for local demonstration.

Before production use, it would need:

- Authentication and authorization
- Durable distributed review state
- Persistent database storage
- Distributed locking
- Rate limiting and usage budgets
- Background job processing
- Production monitoring and alerting
- Broader browser-level end-to-end tests

## Deployment Approach

A production version could use:

- React frontend on Vercel, Cloudflare Pages, or S3/CloudFront
- FastAPI backend in Docker
- ECS, Cloud Run, Azure Container Apps, or Kubernetes
- PostgreSQL for durable review state
- Redis for distributed coordination and rate limiting
- A managed secret store for API keys

## Challenge Alignment

The project demonstrates:

- Public API integration
- Full-stack application development
- Custom semantic clustering
- LLM orchestration with LangGraph
- Structured output and deterministic validation
- Human-in-the-loop review
- Clean separation of concerns
- Secure server-side API-key handling

## Acknowledgements

- Wikimedia Pageviews API
- Wikipedia public summaries
- OpenAI
- LangGraph
- Sentence Transformers
- scikit-learn
- FastAPI
- React
- Vite
