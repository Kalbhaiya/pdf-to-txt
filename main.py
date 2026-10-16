import os, re, json, uuid, time, asyncio, base64, threading, csv, io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(APP_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = APP_DIR / "static"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_UPLOAD_MB = max(1, int(os.getenv("MAX_UPLOAD_MB", "100")))
DEFAULT_BATCH_PAGES = max(1, min(5, int(os.getenv("BATCH_PAGES", "2"))))
MAX_CONCURRENT_REQUESTS = max(1, min(4, int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))))
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(title="Smart Question Extraction Engine", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
jobs: dict[str, dict[str, Any]] = {}
lock = threading.RLock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID.")
    return DATA_DIR / f"{job_id}.json"


def safe_job(job_id: str) -> dict[str, Any]:
    path = job_path(job_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if job_id in jobs:
        return jobs[job_id]
    raise HTTPException(status_code=404, detail="Job not found. The job may have expired or the persistent volume is not mounted.")


def persist_job(job_id: str) -> None:
    with lock:
        job = jobs[job_id]
        path = job_path(job_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def patch_job(job_id: str, **updates: Any) -> None:
    with lock:
        jobs[job_id].update(updates)
        jobs[job_id]["updated_at"] = utcnow()
        persist_job(job_id)


def add_event(job_id: str, text: str, level: str = "info", details: str = "") -> None:
    with lock:
        job = jobs[job_id]
        events = job.setdefault("events", [])
        job["event_sequence"] = int(job.get("event_sequence", 0)) + 1
        events.append({"seq": job["event_sequence"], "time": utcnow(), "level": level, "text": text, "details": details[:1200]})
        job["events"] = events[-150:]
        job["updated_at"] = utcnow()
        persist_job(job_id)


def require_app_password(x_app_password: str | None) -> None:
    if not APP_PASSWORD:
        raise HTTPException(status_code=503, detail="Deployment setup incomplete: set APP_PASSWORD in Railway Variables.")
    if not x_app_password or x_app_password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="App password is missing or incorrect.")


def readable_api_error(status_code: int, body: str) -> str:
    try:
        data = json.loads(body)
        detail = data.get("error", {}).get("message") or body
    except Exception:
        detail = body
    detail = str(detail).replace(GEMINI_API_KEY, "[API_KEY_REDACTED]")
    if status_code == 400:
        hint = "Check request format, PDF/page batch size, and model support."
    elif status_code == 401:
        hint = "API key is invalid. Recreate GEMINI_API_KEY in Google AI Studio and update Railway Variables."
    elif status_code == 403:
        hint = "API access may be blocked for this key/project or billing/region settings. Check Google AI Studio."
    elif status_code == 404:
        hint = "The model name or API endpoint was not found. Check GEMINI_MODEL."
    elif status_code == 429:
        hint = "Gemini quota/rate limit reached. Wait, reduce concurrency, or check AI Studio usage."
    elif status_code >= 500:
        hint = "Gemini service temporarily failed; the batch may be retried."
    else:
        hint = "See Gemini API error details below."
    return f"Gemini API HTTP {status_code}. {hint} Details: {detail[:850]}"


def gemini_request(pdf_bytes: bytes, page_start: int, page_end: int, api_key: str) -> dict[str, Any]:
    prompt = f'''You are a careful OCR and question-structure extraction engine for exam answer-key PDFs.
The attached PDF contains original source pages {page_start}-{page_end}. For source_page in the output, count pages inside this attached PDF starting at 1.

Return ONLY valid JSON, with this exact shape:
{{"questions":[{{"question_number":"as printed or empty","question_english":"exact English text or empty","question_hindi":"exact Hindi text or empty","options":[{{"label":"A","english":"exact English option or empty","hindi":"exact Hindi option or empty"}},{{"label":"B","english":"...","hindi":"..."}},{{"label":"C","english":"...","hindi":"..."}},{{"label":"D","english":"...","hindi":"..."}}],"answer":"A/B/C/D or empty only if visibly established","source_page":1,"warnings":[]}}]}}

Rules:
- Extract every visible multiple-choice question in reading order. Do not skip a question because one language is missing.
- Do not invent illegible words, options, answers, dates, or question numbers. Use empty strings for unreadable/missing text and add a short warning.
- Preserve numbers, units, equations, fractions, roots, powers, symbols, tables, and mathematical meaning. Use plain text or simple LaTeX if necessary.
- Keep English and Hindi in their own fields; do not translate or paraphrase.
- Options must be A, B, C, D in order; always include four items even if some text is missing.
- Do not treat headers/footers as questions. If a page contains no question, return no question for that page.
- source_page is the 1-based page number inside this specific attachment.
- Only fill answer when the correct option is explicitly marked/shown in the source. Never solve the question to guess the answer.
- Add warnings when OCR is uncertain, an option is cut off, or content is missing.
- Be concise: preserve question wording but do not include page headers or explanatory commentary.
'''
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "application/pdf", "data": base64.b64encode(pdf_bytes).decode("ascii")}}
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1, "maxOutputTokens": 16384}
    }
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(endpoint, headers=headers, json=body, timeout=240)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = readable_api_error(resp.status_code, resp.text)
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(last_error)
            if not resp.ok:
                raise RuntimeError(readable_api_error(resp.status_code, resp.text))
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                feedback = data.get("promptFeedback", {})
                raise RuntimeError("Gemini returned no candidate. " + json.dumps(feedback, ensure_ascii=False)[:500])
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")
            parts = candidate.get("content", {}).get("parts", [])
            raw = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
            if finish_reason == "MAX_TOKENS":
                raise RuntimeError("OUTPUT_TRUNCATED: Gemini hit the output-token limit. The engine will automatically split this page batch into smaller batches.")
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
            raw = re.sub(r"\s*```$", "", raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as parse_error:
                if not raw:
                    raise RuntimeError("EMPTY_MODEL_OUTPUT: Gemini returned an empty text response.")
                # Repair is only a last resort. Do not silently discard the original parse failure.
                repair_body = {"contents": [{"parts": [{"text": "Convert this OCR result into valid JSON matching {\"questions\":[]} only. Do not invent or remove any question content. Return JSON only:\n" + raw[:45000]}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0, "maxOutputTokens": 16384}}
                repair = requests.post(endpoint, headers=headers, json=repair_body, timeout=120)
                if not repair.ok:
                    raise RuntimeError(f"INVALID_JSON: Gemini output was not valid JSON (at character {parse_error.pos}); repair request also failed: {readable_api_error(repair.status_code, repair.text)}")
                rd = repair.json()
                rparts = rd.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                rtext = "\n".join(p.get("text", "") for p in rparts if p.get("text")).strip()
                rtext = re.sub(r"^```(?:json)?\s*|\s*```$", "", rtext, flags=re.I)
                try:
                    parsed = json.loads(rtext)
                except Exception:
                    raise RuntimeError(f"INVALID_JSON: Gemini returned malformed JSON around character {parse_error.pos}; reduce batch pages. Output sample: {raw[:250]}")
            if not isinstance(parsed, dict) or not isinstance(parsed.get("questions", []), list):
                raise RuntimeError("INVALID_SCHEMA: Gemini output did not contain a questions array.")
            parsed["_usage"] = data.get("usageMetadata", {})
            parsed["_finish_reason"] = finish_reason
            return parsed
        except Exception as exc:
            last_error = str(exc)
            if "OUTPUT_TRUNCATED" in last_error or "INVALID_JSON" in last_error or "INVALID_SCHEMA" in last_error or "EMPTY_MODEL_OUTPUT" in last_error:
                raise RuntimeError(last_error)
            if attempt < 2:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(last_error or "Unknown Gemini API error")


def normalize_question(q: dict[str, Any], global_page_start: int, global_page_end: int) -> dict[str, Any]:
    local_page = q.get("source_page", 1)
    try:
        local_page = int(local_page)
    except Exception:
        local_page = 1
    local_page = max(1, local_page)
    global_page = min(global_page_end, global_page_start + local_page - 1)
    options = q.get("options") if isinstance(q.get("options"), list) else []
    by_label = {}
    for opt in options:
        if isinstance(opt, dict):
            label = str(opt.get("label", "")).upper().replace(")", "").replace(".", "").strip()
            if label in {"A", "B", "C", "D"}:
                by_label[label] = opt
    clean_options = []
    for lab in "ABCD":
        opt = by_label.get(lab, {})
        clean_options.append({"label": lab, "english": str(opt.get("english", "") or "").strip(), "hindi": str(opt.get("hindi", "") or "").strip()})
    warnings = q.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)] if warnings else []
    ans = str(q.get("answer", "") or "").upper().strip()
    if ans not in {"A", "B", "C", "D"}:
        ans = ""
    out = {
        "question_number": str(q.get("question_number", "") or "").strip(),
        "question_english": str(q.get("question_english", "") or "").strip(),
        "question_hindi": str(q.get("question_hindi", "") or "").strip(),
        "options": clean_options,
        "answer": ans,
        "source_page": global_page,
        "warnings": [str(w) for w in warnings if str(w).strip()]
    }
    if not out["question_english"] and not out["question_hindi"]:
        out["warnings"].append("Question text was not detected clearly")
    if any(not o["english"] and not o["hindi"] for o in clean_options):
        out["warnings"].append("One or more options are blank or unreadable")
    out["warnings"] = list(dict.fromkeys(out["warnings"]))
    return out


def rebuild_live_questions(job: dict[str, Any]) -> None:
    questions = []
    batches = sorted(job.get("batch_results", []), key=lambda b: (b.get("start_page", 0), b.get("end_page", 0)))
    for batch in batches:
        for idx, raw_q in enumerate(batch.get("questions", []), 1):
            q = dict(raw_q)
            q["id"] = f"p{q.get('source_page', 0)}-n{q.get('question_number') or 'x'}-{idx}"
            questions.append(q)
    job["questions"] = questions
    job["questions_found"] = len(questions)
    job["needs_review"] = sum(1 for q in questions if q.get("warnings"))


def split_pdf_bytes(chunk: bytes) -> tuple[bytes, bytes] | None:
    try:
        doc = fitz.open(stream=chunk, filetype="pdf")
        if len(doc) <= 1:
            doc.close()
            return None
        mid = len(doc) // 2
        out = []
        for a, b in ((0, mid - 1), (mid, len(doc) - 1)):
            piece = fitz.open()
            piece.insert_pdf(doc, from_page=a, to_page=b)
            out.append(piece.tobytes())
            piece.close()
        doc.close()
        return out[0], out[1]
    except Exception:
        return None


async def request_batch_with_recovery(chunk: bytes, start_page: int, end_page: int) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(None, gemini_request, chunk, start_page, end_page, GEMINI_API_KEY)
        clean_questions = []
        for q in data.get("questions", []):
            if isinstance(q, dict):
                clean_questions.append(normalize_question(q, start_page, end_page))
        usage = data.get("_usage", {}) or {}
        return clean_questions, {"input_tokens": int(usage.get("promptTokenCount", 0) or 0), "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0)}, []
    except Exception as exc:
        error_text = str(exc)
        if any(marker in error_text for marker in ("OUTPUT_TRUNCATED", "INVALID_JSON", "INVALID_SCHEMA")) and end_page > start_page:
            halves = split_pdf_bytes(chunk)
            if halves:
                mid = start_page + (end_page - start_page) // 2
                left_q, left_u, left_notes = await request_batch_with_recovery(halves[0], start_page, mid)
                right_q, right_u, right_notes = await request_batch_with_recovery(halves[1], mid + 1, end_page)
                combined_usage = {"input_tokens": left_u["input_tokens"] + right_u["input_tokens"], "output_tokens": left_u["output_tokens"] + right_u["output_tokens"]}
                note = [f"Original pages {start_page}-{end_page} automatically split after truncated/invalid model output."]
                return left_q + right_q, combined_usage, note + left_notes + right_notes
        raise


def make_batches(file_bytes: bytes, selected_ranges: list[tuple[int, int]] | None = None, batch_pages: int = 2):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    if not total_pages:
        doc.close()
        raise RuntimeError("The PDF has no pages.")
    if total_pages > 1000:
        doc.close()
        raise RuntimeError("This version limits each PDF to 1,000 pages. Split it into smaller files.")
    ranges = selected_ranges or [(s + 1, min(total_pages, s + batch_pages)) for s in range(0, total_pages, batch_pages)]
    batches = []
    for start, end in ranges:
        if start < 1 or end > total_pages or end < start:
            continue
        piece = fitz.open()
        piece.insert_pdf(doc, from_page=start - 1, to_page=end - 1)
        batches.append((start, end, piece.tobytes()))
        piece.close()
    doc.close()
    return total_pages, batches


async def process_pdf_job(job_id: str, file_bytes: bytes, batch_pages: int, selected_ranges: list[tuple[int, int]] | None = None) -> None:
    try:
        total_pages, batches = make_batches(file_bytes, selected_ranges, batch_pages)
        with lock:
            job = jobs[job_id]
            if selected_ranges is None:
                job.update({"status": "processing", "total_pages": total_pages, "total_batches": len(batches), "completed_batches": 0, "batch_results": [], "batch_errors": [], "events": [], "started_at": utcnow(), "cancel_requested": False})
            else:
                # Preserve successful batches and retry only failed page ranges.
                retry_keys = {(s, e) for s, e, _ in batches}
                job["batch_errors"] = [e for e in job.get("batch_errors", []) if tuple(e.get("page_range", [])) not in retry_keys]
                job["batch_results"] = [b for b in job.get("batch_results", []) if (b.get("start_page"), b.get("end_page")) not in retry_keys]
                job["status"] = "processing"
                job["cancel_requested"] = False
                job["message"] = f"Failed batches retry हो रहे हैं: {len(batches)} batches."
                job["completed_batches"] = len(job.get("batch_results", []))
            job["message"] = f"PDF के {total_pages} पेज; इस run में {len(batches)} batches. Live results चालू हैं।"
            rebuild_live_questions(job)
            job["updated_at"] = utcnow()
            persist_job(job_id)
        add_event(job_id, f"Extraction started: {len(batches)} batches queued for {total_pages} pages.")
        if not batches:
            raise RuntimeError("No valid PDF page batches were created.")

        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        done_this_run = 0

        async def one_batch(start_page: int, end_page: int, chunk: bytes):
            nonlocal done_this_run
            async with sem:
                with lock:
                    if jobs[job_id].get("cancel_requested"):
                        done_this_run += 1
                        jobs[job_id]["completed_batches"] = min(jobs[job_id].get("total_batches", 0), jobs[job_id].get("completed_batches", 0) + 1)
                        jobs[job_id]["message"] = "Stop requested. Waiting for in-flight request(s) to finish."
                        persist_job(job_id)
                        return
                started = time.monotonic()
                add_event(job_id, f"Started batch pages {start_page}-{end_page}.")
                try:
                    async with sem_dummy():
                        qlist, usage, notes = await request_batch_with_recovery(chunk, start_page, end_page)
                    elapsed = round(time.monotonic() - started, 1)
                    with lock:
                        job = jobs[job_id]
                        job["batch_result_sequence"] = int(job.get("batch_result_sequence", 0)) + 1
                        batch_seq = job["batch_result_sequence"]
                        for q_index, q in enumerate(qlist, 1):
                            q["id"] = f"p{int(q.get('source_page', 0)):04d}-n{q.get('question_number') or 'x'}-i{q_index:04d}"
                        job.setdefault("batch_results", []).append({"seq": batch_seq, "start_page": start_page, "end_page": end_page, "questions": qlist, "question_count": len(qlist), "elapsed_seconds": elapsed, "finished_at": utcnow(), "usage": usage})
                        job.setdefault("usage", {"input_tokens": 0, "output_tokens": 0})
                        job["usage"]["input_tokens"] += usage["input_tokens"]
                        job["usage"]["output_tokens"] += usage["output_tokens"]
                        job["batch_errors"] = [e for e in job.get("batch_errors", []) if tuple(e.get("page_range", [])) != (start_page, end_page)]
                        job["completed_batches"] = min(job.get("total_batches", len(batches)), job.get("completed_batches", 0) + 1)
                        job["last_batch"] = {"pages": f"{start_page}-{end_page}", "status": "success", "questions": len(qlist), "seconds": elapsed}
                        rebuild_live_questions(job)
                        job["message"] = f"Pages {start_page}-{end_page}: {len(qlist)} questions extracted. Live total: {len(job['questions'])}."
                        job["updated_at"] = utcnow()
                        persist_job(job_id)
                    add_event(job_id, f"Completed pages {start_page}-{end_page}: {len(qlist)} questions in {elapsed}s.", "success", "; ".join(notes))
                except Exception as exc:
                    elapsed = round(time.monotonic() - started, 1)
                    error_text = str(exc)[:1400]
                    with lock:
                        job = jobs[job_id]
                        job.setdefault("batch_errors", []).append({"pages": f"{start_page}-{end_page}", "page_range": [start_page, end_page], "error": error_text, "failed_at": utcnow()})
                        job["completed_batches"] = min(job.get("total_batches", len(batches)), job.get("completed_batches", 0) + 1)
                        job["last_batch"] = {"pages": f"{start_page}-{end_page}", "status": "failed", "questions": 0, "seconds": elapsed}
                        job["message"] = f"Pages {start_page}-{end_page} batch failed. Error details नीचे देखें; बाकी batches जारी हैं।"
                        job["updated_at"] = utcnow()
                        persist_job(job_id)
                    add_event(job_id, f"Failed pages {start_page}-{end_page}.", "error", error_text)
                finally:
                    done_this_run += 1
                    with lock:
                        job = jobs[job_id]
                        job["run_batches_finished"] = done_this_run
                        job["updated_at"] = utcnow()
                        persist_job(job_id)

        # Semaphore is applied by one_batch; this no-op context keeps request recovery readable.
        await asyncio.gather(*(one_batch(s, e, b) for s, e, b in batches))
        with lock:
            job = jobs[job_id]
            rebuild_live_questions(job)
            if job.get("cancel_requested"):
                job["status"] = "cancelled"
                job["message"] = f"Stopped. {len(job.get('questions', []))} questions were extracted before stop."
            elif job.get("batch_errors"):
                job["status"] = "completed_with_errors"
                job["message"] = f"Finished with {len(job['batch_errors'])} failed batch(es); {len(job.get('questions', []))} questions extracted. You can retry failed batches."
            else:
                job["status"] = "completed"
                job["message"] = f"Completed: {len(job.get('questions', []))} questions extracted. Review flagged questions before publishing."
            job["finished_at"] = utcnow()
            job["updated_at"] = utcnow()
            persist_job(job_id)
        add_event(job_id, jobs[job_id]["message"], "success" if jobs[job_id]["status"] == "completed" else "warning")
    except Exception as exc:
        with lock:
            if job_id in jobs:
                jobs[job_id].update({"status": "failed", "error": str(exc)[:1400], "message": "Processing failed. See diagnostic details below.", "finished_at": utcnow(), "updated_at": utcnow()})
                persist_job(job_id)
        add_event(job_id, "Job-level failure.", "error", str(exc))


class sem_dummy:
    """Async no-op context manager; concurrency is governed by one_batch's outer semaphore."""
    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc, tb): return False


def extract_html_text(file_bytes: bytes) -> tuple[str, int]:
    soup = BeautifulSoup(file_bytes, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    text_content = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    image_count = len(soup.find_all("img"))
    return text_content, image_count


@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True, "version": "0.2.0", "model": MODEL, "api_key_configured": bool(GEMINI_API_KEY), "app_password_configured": bool(APP_PASSWORD), "max_upload_mb": MAX_UPLOAD_MB, "max_concurrent_requests": MAX_CONCURRENT_REQUESTS}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    batch_pages: int = Form(DEFAULT_BATCH_PAGES),
    x_app_password: str | None = Header(default=None),
):
    require_app_password(x_app_password)
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured in server Secrets.")
    batch_pages = max(1, min(5, int(batch_pages)))
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds MAX_UPLOAD_MB={MAX_UPLOAD_MB} MB.")
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    job_id = uuid.uuid4().hex[:12]
    if ext == ".pdf" or file.content_type == "application/pdf":
        try:
            test = fitz.open(stream=content, filetype="pdf")
            n_pages = len(test)
            test.close()
        except Exception:
            raise HTTPException(status_code=400, detail="The uploaded file could not be opened as a PDF.")
        (DATA_DIR / f"{job_id}.pdf").write_bytes(content)
        jobs[job_id] = {"id": job_id, "filename": filename, "input_type": "pdf", "status": "queued", "created_at": utcnow(), "updated_at": utcnow(), "total_pages": n_pages, "total_batches": 0, "completed_batches": 0, "questions": [], "questions_found": 0, "needs_review": 0, "batch_results": [], "batch_result_sequence": 0, "batch_errors": [], "events": [], "event_sequence": 0, "usage": {"input_tokens": 0, "output_tokens": 0}, "message": f"PDF queue में है: {n_pages} pages.", "batch_pages": batch_pages, "cancel_requested": False}
        persist_job(job_id)
        asyncio.create_task(process_pdf_job(job_id, content, batch_pages))
        return {"job_id": job_id, "status": "queued", "message": jobs[job_id]["message"]}
    if ext in {".html", ".htm"} or "html" in (file.content_type or ""):
        text_content, image_count = extract_html_text(content)
        return JSONResponse({"error": "html_import_not_ready", "message": "HTML text preview मिल गया है, लेकिन image-based questions के लिए अभी PDF export करके अपलोड करें। ZIP के linked images के साथ HTML import अगले चरण में जोड़ा जा सकता है।", "detected_images": image_count, "extracted_text_preview": text_content[:5000]}, status_code=422)
    raise HTTPException(status_code=400, detail="Please upload a PDF file. For now, export HTML to PDF and upload the PDF.")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, x_app_password: str | None = Header(default=None)):
    require_app_password(x_app_password)
    return safe_job(job_id)


@app.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str, after_event: int = 0, after_batch_seq: int = 0, x_app_password: str | None = Header(default=None)):
    """Lightweight progress endpoint; avoids retransmitting the full question list on every poll."""
    require_app_password(x_app_password)
    job = safe_job(job_id)
    events = [e for e in job.get("events", []) if int(e.get("seq", 0)) > after_event]
    metadata = {k: v for k, v in job.items() if k not in {"questions", "batch_results", "original_pdf_bytes"}}
    metadata["events"] = events
    metadata["event_cursor"] = int(job.get("event_sequence", 0))
    metadata["questions_found"] = int(job.get("questions_found", len(job.get("questions", []))))
    metadata["needs_review"] = int(job.get("needs_review", sum(1 for q in job.get("questions", []) if q.get("warnings"))))
    metadata["batch_meta_cursor"] = int(job.get("batch_result_sequence", 0))
    return metadata


@app.get("/api/jobs/{job_id}/updates")
def get_job_updates(job_id: str, after_seq: int = 0, x_app_password: str | None = Header(default=None)):
    """Returns only newly completed batches, including their questions."""
    require_app_password(x_app_password)
    job = safe_job(job_id)
    batches = [b for b in job.get("batch_results", []) if int(b.get("seq", 0)) > after_seq]
    batches.sort(key=lambda b: int(b.get("seq", 0)))
    return {"batches": batches, "batch_cursor": int(job.get("batch_result_sequence", 0))}


@app.get("/api/jobs/{job_id}/questions")
def get_job_questions(job_id: str, x_app_password: str | None = Header(default=None)):
    require_app_password(x_app_password)
    job = safe_job(job_id)
    return {"questions": job.get("questions", []), "question_count": len(job.get("questions", []))}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, x_app_password: str | None = Header(default=None)):
    require_app_password(x_app_password)
    job = safe_job(job_id)
    if job.get("status") not in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail=f"Cannot stop a job with status {job.get('status')}.")
    with lock:
        jobs[job_id] = job
        jobs[job_id]["cancel_requested"] = True
        jobs[job_id]["message"] = "Stop requested. In-flight Gemini requests will finish; queued batches will be skipped."
        persist_job(job_id)
    add_event(job_id, "User requested stop.", "warning")
    return {"ok": True, "message": "Stop requested."}


@app.post("/api/jobs/{job_id}/retry-failed")
async def retry_failed(job_id: str, x_app_password: str | None = Header(default=None)):
    require_app_password(x_app_password)
    job = safe_job(job_id)
    if job.get("status") in {"processing", "queued"}:
        raise HTTPException(status_code=409, detail="Wait for the current job to finish before retrying.")
    errors = job.get("batch_errors", [])
    pdf_path = DATA_DIR / f"{job_id}.pdf"
    if not errors:
        raise HTTPException(status_code=409, detail="No failed batches to retry.")
    if not pdf_path.exists():
        raise HTTPException(status_code=410, detail="Original PDF is unavailable. Upload the PDF again.")
    ranges = [tuple(e.get("page_range", [])) for e in errors if len(e.get("page_range", [])) == 2]
    if not ranges:
        raise HTTPException(status_code=409, detail="No retryable page ranges were recorded.")
    with lock:
        jobs[job_id] = job
        jobs[job_id]["batch_pages"] = int(job.get("batch_pages", DEFAULT_BATCH_PAGES))
        jobs[job_id]["status"] = "queued"
        jobs[job_id]["message"] = f"Retry queue: {len(ranges)} failed batches."
        jobs[job_id]["updated_at"] = utcnow()
        persist_job(job_id)
    content = pdf_path.read_bytes()
    asyncio.create_task(process_pdf_job(job_id, content, int(job.get("batch_pages", DEFAULT_BATCH_PAGES)), ranges))
    return {"ok": True, "message": f"Retry started for {len(ranges)} failed batch(es)."}


@app.put("/api/jobs/{job_id}/questions")
async def save_questions(job_id: str, payload: dict[str, Any], x_app_password: str | None = Header(default=None)):
    require_app_password(x_app_password)
    job = safe_job(job_id)
    qs = payload.get("questions")
    if not isinstance(qs, list):
        raise HTTPException(status_code=400, detail="questions must be a list")
    clean = []
    for i, q in enumerate(qs, 1):
        if not isinstance(q, dict):
            continue
        page = int(q.get("source_page", 1) or 1)
        item = normalize_question(q, page, page)
        item["id"] = str(q.get("id", i))
        clean.append(item)
    clean_by_id = {str(q.get("id")): q for q in clean}
    with lock:
        jobs[job_id] = job
        jobs[job_id]["questions"] = clean
        # Keep the underlying batch records in sync, so a later failed-batch retry cannot undo edits.
        for batch in jobs[job_id].get("batch_results", []):
            batch["questions"] = [clean_by_id.get(str(old_q.get("id")), old_q) for old_q in batch.get("questions", [])]
            batch["question_count"] = len(batch.get("questions", []))
        jobs[job_id]["questions_found"] = len(clean)
        jobs[job_id]["needs_review"] = sum(1 for q in clean if q.get("warnings"))
        jobs[job_id]["message"] = f"Edited data saved: {len(clean)} questions."
        jobs[job_id]["updated_at"] = utcnow()
        persist_job(job_id)
    return {"ok": True, "question_count": len(clean)}


@app.get("/api/jobs/{job_id}/export/{fmt}")
def export_job(job_id: str, fmt: str, x_app_password: str | None = Header(default=None)):
    require_app_password(x_app_password)
    job = safe_job(job_id)
    questions = job.get("questions", [])
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(job.get("filename", "questions")).stem)[:60] or "questions"
    if fmt == "json":
        data = {"source_file": job.get("filename"), "question_count": len(questions), "questions": questions}
        return Response(json.dumps(data, ensure_ascii=False, indent=2), media_type="application/json; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{base}_questions.json"'})
    if fmt == "csv":
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["question_number", "question_english", "question_hindi", "option_A_english", "option_A_hindi", "option_B_english", "option_B_hindi", "option_C_english", "option_C_hindi", "option_D_english", "option_D_hindi", "answer", "source_page", "warnings"])
        for i, q in enumerate(questions, 1):
            options = {o.get("label"): o for o in q.get("options", [])}
            row = [q.get("question_number") or i, q.get("question_english", ""), q.get("question_hindi", "")]
            for lab in "ABCD":
                row.extend([options.get(lab, {}).get("english", ""), options.get(lab, {}).get("hindi", "")])
            row.extend([q.get("answer", ""), q.get("source_page", ""), "; ".join(q.get("warnings", []))])
            writer.writerow(row)
        return Response("\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{base}_questions.csv"'})
    if fmt == "txt":
        lines = []
        for i, q in enumerate(questions, 1):
            n = q.get("question_number") or str(i)
            lines.append(f"Q{n}. {q.get('question_english','')}")
            if q.get("question_hindi"): lines.append(q["question_hindi"])
            for opt in q.get("options", []):
                label, en, hi = opt.get("label", ""), opt.get("english", ""), opt.get("hindi", "")
                lines.append(f"{label}) {en}".rstrip())
                if hi: lines.append(hi)
            if q.get("answer"): lines.append(f"Answer: {q['answer']}")
            lines.append(f"[Source page: {q.get('source_page','')}]" )
            if q.get("warnings"): lines.append("[REVIEW: " + "; ".join(q["warnings"]) + "]")
            lines.append("")
        return Response("\ufeff" + "\n".join(lines), media_type="text/plain; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{base}_questions.txt"'})
    if fmt == "html":
        payload = json.dumps(questions, ensure_ascii=False).replace("</", "<\\/")
        html = """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Extracted Mock</title><style>body{font-family:Arial,'Noto Sans Devanagari',sans-serif;max-width:900px;margin:auto;padding:18px;line-height:1.5;color:#222}.q{border-bottom:1px solid #ddd;padding:16px 0}.hint{color:#a55;font-size:13px}.opt{margin:5px 0 5px 14px}@media print{.hint{display:none}}</style></head><body><h1>Extracted Mock Questions</h1><div id=\"root\"></div><script>const qs=""" + payload + """;const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));document.getElementById('root').innerHTML=qs.map((q,i)=>`<section class=\"q\"><b>Q${esc(q.question_number||i+1)}.</b> ${esc(q.question_english)}<div>${esc(q.question_hindi)}</div>${q.options.map(o=>`<div class=\"opt\">${esc(o.label)}) ${esc(o.english)}<div>${esc(o.hindi)}</div></div>`).join('')}<div><b>Answer:</b> ${esc(q.answer||'Not provided')}</div>${q.warnings.length?`<div class=\"hint\">Review: ${esc(q.warnings.join('; '))}</div>`:''}</section>`).join('');</script></body></html>"""
        return Response(html, media_type="text/html; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{base}_mock.html"'})
    raise HTTPException(status_code=400, detail="Unsupported format. Use txt, json, csv, or html.")
