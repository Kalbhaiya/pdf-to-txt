# Smart Question Engine — Railway-ready MVP

Android-browser-friendly FastAPI app for batch extraction of MCQs from scanned/photo-based PDFs using the Gemini API. It includes editable review and TXT, JSON, and basic HTML Mock export.

## Deploy on Railway (recommended path)

This project is configured to build with the included `Dockerfile`. Railway's dashboard deployment path is GitHub-based; this ZIP is not itself a Railway import. From an Android phone:

1. Download and extract this ZIP using the phone's Files app.
2. Create a **private** GitHub repository. Upload the extracted project files and folders to its root (including `Dockerfile`, `railway.json`, `.dockerignore`, `main.py`, `requirements.txt`, and `static/index.html`). Do **not** upload a real `.env` file or API key.
3. Open Railway, create a project, choose **Deploy from GitHub repo**, and select that repository. Railway should detect `railway.json` and the Dockerfile.
4. In the service's **Variables** section, add the variables listed below and deploy/redeploy.
5. In the service Settings → Networking, choose **Generate Domain** to get a public `*.up.railway.app` URL.
6. For persistence across redeploys, create a Railway Volume and mount it at exactly `/app/data`. The app stores job JSON there. A volume is not required just to boot, but without one saved jobs may disappear on a redeploy/restart.

## Required Railway variables

- `GEMINI_API_KEY` — your key from Google AI Studio; never put it in the frontend or GitHub.
- `APP_PASSWORD` — choose your own long, unique password; this is the password entered in the web app.

## Optional variables

- `GEMINI_MODEL=gemini-3.8-flash`
- `MAX_UPLOAD_MB=100`
- `BATCH_PAGES=2`
- `MAX_CONCURRENT_REQUESTS=2` (app clamps this to 1–4)

Do not add quotes around values unless the quotes are actually part of the secret. Never share your API key in chat. If exposed, revoke it and create a new one.

## First run

1. Open the generated Railway domain on your phone.
2. Enter the `APP_PASSWORD` you configured in Railway.
3. Upload a small 5–10 page sample PDF, start with 1 page per batch, and verify the question text, Hindi, math symbols, options, and answers.
4. Once quality looks acceptable, try 2 pages per batch. Review all warnings before using the output as a published mock.
5. Export TXT, JSON, or HTML Mock. PDF-print formatting can be added in a later version; this MVP currently exports the questions as TXT/JSON/HTML.

## Limits and cost notes

- Railway hosting usage and Gemini API usage are separate. Railway Trial credits pay for Railway hosting; they do not pay for Google Gemini API usage.
- Gemini free-tier quotas/availability vary by account and model; check the Google AI Studio/API usage page before running a large PDF.
- This starter sends small PDF page batches inline, not the whole PDF in one request. Keep each batch small and under Gemini's per-request PDF limits.
- A deployment restart during an active extraction can interrupt that job. This MVP is for controlled testing, not yet a durable job queue for unattended large-scale production.
- HTML import is limited to text preview in this version. For image-based HTML, print/save it as PDF first.

## Local Docker test (optional)

```bash
docker build -t smart-question-engine .
docker run --rm -p 8080:8080 \
  -e GEMINI_API_KEY='your-key' \
  -e APP_PASSWORD='your-private-password' \
  smart-question-engine
```

Open `http://localhost:8080`.
