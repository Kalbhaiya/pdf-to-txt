# Smart Question Engine — Railway Edition v0.2

Mobile-friendly FastAPI app for extracting bilingual exam MCQs from scanned/photo-based PDFs using Gemini Vision. This update adds live per-batch results, clear API error details, automatic smaller-batch recovery for truncated JSON, failed-batch retry, stop requests, event logs, question editing/review, search/filter, and TXT/CSV/JSON/HTML Mock exports.

## Deploy/update on Railway from an Android phone

1. Extract the ZIP on your phone.
2. Open the existing **private** GitHub repository connected to Railway. Replace the existing project files with the updated files from this ZIP. Keep `Dockerfile`, `railway.json`, `requirements.txt`, `main.py`, and `static/index.html` at the repository root / expected project root exactly as in the ZIP. Do not upload this ZIP as the only repository file.
3. Commit the changed files. Railway should automatically redeploy from the connected GitHub repository. If not, open Railway → service → Deployments → Redeploy.
4. In Railway Variables, keep/set `GEMINI_API_KEY`, `APP_PASSWORD`, and the optional variables below. Do not paste secrets into GitHub, HTML, README, screenshots, or chat.
5. In Railway Service Settings → Networking, use the existing generated domain.
6. For persistent records/retry, mount a Railway Volume at `/app/data`.

## Required environment variables

- `GEMINI_API_KEY`: secret created through Google AI Studio.
- `APP_PASSWORD`: a long, unique private password used to open the tool.

## Optional variables

```env
GEMINI_MODEL=gemini-3.8-flash
MAX_UPLOAD_MB=100
BATCH_PAGES=2
MAX_CONCURRENT_REQUESTS=1
```

`MAX_CONCURRENT_REQUESTS` is clamped to 1–4. Start at 1, and use 1–2 PDF pages per batch for quality testing. Railway hosting credit does not pay for Gemini API usage. Check the model's current quota/pricing before running large PDFs.

## Features in v0.2

- Live question data becomes visible as each batch finishes; no need to wait for the whole PDF.
- Lightweight status polling + only-new-batch updates to reduce mobile data use.
- Per-batch event log, error details, source-page tracking, token usage display.
- Automatic re-splitting into smaller page batches when Gemini output is truncated or malformed JSON is returned.
- Retry failed page ranges without intentionally rerunning successful batches.
- Stop request skips queued work; already-running Gemini requests may need to finish.
- Editable questions/options/answer/warnings after extraction, live review warnings, search/filter, copy text.
- Export TXT, CSV, JSON, and HTML Mock.
- App password required on job endpoints; API Key stays server-side.

## Important limitations

- HTML upload currently returns a text preview but does not extract linked/embedded question images. For image-based Answer Key HTML, print/export it to PDF and upload the PDF.
- This starter keeps job metadata and the original PDF in `/app/data` so failed batches can be retried. Use a persistent Railway Volume there; storage is finite, so remove old jobs/PDFs when no longer required.
- Stop requests do not abort a Gemini HTTP request already in progress.
- AI OCR is not guaranteed to be error-free. Always review warnings, options, maths symbols, and marked answers before publishing.

## Docker

The included Dockerfile runs Uvicorn on Railway's `$PORT` and creates `/app/data` for jobs. The container listens on `0.0.0.0`.
