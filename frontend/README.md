# WikiPulse dashboard

React and TypeScript dashboard for the WikiPulse audience-analysis API.

## Local development

Start the FastAPI backend on `http://127.0.0.1:8000`, then run:

```bash
npm install
npm run dev
```

Vite proxies browser requests under `/api` to the local FastAPI server.

## Verification

```bash
npm run lint
npm run build
```
