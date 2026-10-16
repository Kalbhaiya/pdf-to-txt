import os, io, re, json, uuid, time, asyncio, base64, threading
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
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR = APP_DIR / "static"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
DEFAULT_BATCH_PAGES = int(os.getenv("BATCH_PAGES", "2"))
MAX_CONCURRENT_REQUESTS = max(1, min(4, int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))))
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(title="Smart Question Extraction Engine", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

jobs: dict[str, dict[str, Any]] = {}
lock = threading.Lock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_job(job_id: str) -> dict[str, Any]:
    path = DATA_DIR / f"{job_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if job_id in jobs:
        return jobs[job_id]
    raise HTTPException(status_code=404, detail="Job not found")


def persist_job(job_id: str) -> None:
    job = jobs[job_id]
    tmp = DATA_DIR / f"{job_id}.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DATA_DIR / f"{job_id}.json")


def require_app_password(x_app_password: str | None) -> None:
    if not APP_PASSWORD:
        raise HTTPException(status_code=503, detail="Deployment setup incomplete: set APP_PASSWORD in Secrets before using this app.")
    if not x_app_password or x_app_password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="App password is missing or incorrect.")


def gemini_request(pdf_bytes: bytes, page_start: int, page_end: int, api_key: str) -> dict[str, Any]:
    prompt = f'''You are a careful OCR and question-structure extraction engine for exam answer-key PDFs.
Read every visible multiple-choice question on the attached PDF pages. This batch contains original PDF pages {page_start}-{page_end}. Inside this batch, count pages starting from 1.

Return ONLY valid JSON, no markdown fences, in this exact shape:
{{"questions":[{{"question_number":"as printed or empty","question_english":"exact English text or empty","question_hindi":"exact Hindi text or empty","options":[{{"label":"A","english":"exact English option or empty","hindi":"exact Hindi option or empty"}},{{"label":"B","english":"...","hindi":"..."}},{{"label":"C","english":"...","hindi":"..."}},{{"label":"D","english":"...","hindi":"..."}}],"answer":"A/B/C/D or empty only if visibly established","source_page":1,"warnings":[]}}]}}

Rules:
- Extract all questions visible on these pages, in reading order. Do not skip a question just because one language is missing.
- Do not invent illegible words, options, answers, dates, or question numbers. Use an empty string for unreadable/missing content and add a short warning.
- Preserve numbers, units, equations, fractions, roots, powers, symbols, tables and mathematical meaning. Use plain text or simple LaTeX where necessary.
- Keep English and Hindi in their separate fields; do not translate or paraphrase.
- Options must be A, B, C, D in order. Include four items even when text is missing.
- If the page has no question, return no question for that page. Do not treat headers/footers as questions.
- source_page is the page number within this attached batch, starting at 1.
- Answer must only be filled when the correct option is explicitly marked/shown in the source. Never solve the question to guess the answer.
- Include a warning when OCR is uncertain, an option appears cut off, or text is missing.
'''
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    api_headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "application/pdf", "data": base64.b64encode(pdf_bytes).decode("ascii")}}
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1, "maxOutputTokens": 8192}
    }
    last_error = None
    for attempt in range(4):
        try:
            resp = requests.post(endpoint, headers=api_headers, json=body, timeout=240)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"Gemini API HTTP {resp.status_code}: {resp.text[:500]}"
                time.sleep(min(2 ** attempt, 12))
                continue
            if not resp.ok:
                raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {resp.text[:1000]}")
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError("Gemini returned no candidate. Try a smaller page batch or check API quota.")
            parts = candidates[0].get("content", {}).get("parts", [])
            raw = "\n".join(p.get("text", "") for p in parts if p.get("text"))
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
            raw = re.sub(r"\s*```$", "", raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # One repair attempt with the model's own response as input.
                repair_body = {"contents": [{"parts": [{"text": "Convert the following into valid JSON only, preserving every character and value. Do not add missing information:\n" + raw}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}
                repair = requests.post(endpoint, headers=api_headers, json=repair_body, timeout=120)
                repair.raise_for_status()
                rd = repair.json()
                rparts = rd.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                rtext = "\n".join(p.get("text", "") for p in rparts if p.get("text")).strip()
                rtext = re.sub(r"^```(?:json)?\s*|\s*```$", "", rtext, flags=re.I)
                parsed = json.loads(rtext)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("questions", []), list):
                raise RuntimeError("Gemini output was not in the expected JSON structure.")
            parsed["_usage"] = data.get("usageMetadata", {})
            return parsed
        except Exception as e:
            last_error = str(e)
            if attempt < 3:
                time.sleep(min(2 ** attempt, 12))
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
    if not isinstance(warnings, list): warnings = [str(warnings)] if warnings else []
    ans = str(q.get("answer", "") or "").upper().strip()
    if ans not in {"A", "B", "C", "D"}: ans = ""
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
    # Deduplicate warnings while keeping order.
    out["warnings"] = list(dict.fromkeys(out["warnings"]))
    return out


async def process_pdf_job(job_id: str, file_bytes: bytes, api_key: str, batch_pages: int) -> None:
    loop = asyncio.get_running_loop()
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        if total_pages == 0:
            raise RuntimeError("The PDF has no pages.")
        if total_pages > 1000:
            raise RuntimeError("This starter version limits each PDF to 1,000 pages. Split the PDF and process it as separate jobs.")
        batches = []
        for start in range(0, total_pages, batch_pages):
            end = min(total_pages, start + batch_pages)
            piece = fitz.open()
            piece.insert_pdf(doc, from_page=start, to_page=end - 1)
            batches.append((start + 1, end, piece.tobytes()))
            piece.close()
        doc.close()
        jobs[job_id].update({"status": "processing", "total_pages": total_pages, "total_batches": len(batches), "completed_batches": 0, "questions": [], "message": f"PDF के {total_pages} पेज {len(batches)} बैच में प्रोसेस होंगे।"})
        persist_job(job_id)
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        results: list[tuple[int, int, dict[str, Any] | None, str | None]] = []
        completed = 0
        async def one_batch(start_page: int, end_page: int, chunk: bytes):
            nonlocal completed
            async with sem:
                try:
                    data = await loop.run_in_executor(None, gemini_request, chunk, start_page, end_page, api_key)
                    res = (start_page, end_page, data, None)
                except Exception as exc:
                    res = (start_page, end_page, None, str(exc))
                results.append(res)
                completed += 1
                jobs[job_id]["completed_batches"] = completed
                jobs[job_id]["message"] = f"बैच {completed}/{len(batches)} प्रोसेस हो चुके हैं।"
                persist_job(job_id)
        await asyncio.gather(*(one_batch(s, e, b) for s, e, b in batches))
        all_questions = []
        batch_errors = []
        usage_in = usage_out = 0
        for start, end, data, err in sorted(results, key=lambda x: x[0]):
            if err:
                batch_errors.append({"pages": f"{start}-{end}", "error": err})
                continue
            usage = (data or {}).get("_usage", {})
            usage_in += int(usage.get("promptTokenCount", 0) or 0)
            usage_out += int(usage.get("candidatesTokenCount", 0) or 0)
            for q in (data or {}).get("questions", []):
                if isinstance(q, dict):
                    all_questions.append(normalize_question(q, start, end))
        # Sort primarily by source page, keep detected question order within each batch.
        all_questions = sorted(enumerate(all_questions), key=lambda pair: (pair[1].get("source_page", 0), pair[0]))
        all_questions = [q for _, q in all_questions]
        # Assign stable IDs for client-side editing.
        for i, q in enumerate(all_questions, 1):
            q["id"] = i
        jobs[job_id].update({
            "status": "completed_with_errors" if batch_errors else "completed",
            "questions": all_questions,
            "batch_errors": batch_errors,
            "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
            "message": f"काम पूरा: {len(all_questions)} प्रश्न निकले। पहले Review सूची जाँचें।"
        })
        persist_job(job_id)
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e), "message": "प्रोसेसिंग विफल हुई।"})
        persist_job(job_id)


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
    return {"ok": True, "version": "0.1.0", "model": MODEL, "api_key_configured": bool(GEMINI_API_KEY), "app_password_configured": bool(APP_PASSWORD)}


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
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
        # Validate PDF before sending the job to background processing.
        try:
            test = fitz.open(stream=content, filetype="pdf")
            n_pages = len(test)
            test.close()
        except Exception:
            raise HTTPException(status_code=400, detail="The uploaded file could not be opened as a PDF.")
        jobs[job_id] = {"id": job_id, "filename": filename, "input_type": "pdf", "status": "queued", "created_at": utcnow(), "total_pages": n_pages, "total_batches": 0, "completed_batches": 0, "questions": [], "batch_errors": [], "message": "PDF processing queue में है।"}
        persist_job(job_id)
        background_tasks.add_task(lambda: asyncio.run(process_pdf_job(job_id, content, GEMINI_API_KEY, batch_pages)))
        return {"job_id": job_id, "status": "queued", "message": jobs[job_id]["message"]}
    if ext in {".html", ".htm"} or "html" in (file.content_type or ""):
        text_content, image_count = extract_html_text(content)
        return JSONResponse({
            "error": "html_text_only",
            "message": "इस starter build में HTML से वास्तविक टेक्स्ट पढ़ा जा सकता है, लेकिन अलग image files को सुरक्षित रूप से पढ़ने के लिए HTML और उसकी assets को ZIP में देना होगा या HTML को PDF में बदलकर अपलोड करें।",
            "detected_images": image_count,
            "extracted_text_preview": text_content[:5000]
        }, status_code=422)
    raise HTTPException(status_code=400, detail="Please upload a PDF file. HTML text import is a limited preview in this starter build.")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, x_app_password: str | None = Header(default=None)):
    require_app_password(x_app_password)
    return safe_job(job_id)


@app.put("/api/jobs/{job_id}/questions")
async def save_questions(job_id: str, payload: dict[str, Any], x_app_password: str | None = Header(default=None)):
    require_app_password(x_app_password)
    job = safe_job(job_id)
    if job.get("status") not in {"completed", "completed_with_errors"}:
        raise HTTPException(status_code=409, detail="Questions can only be edited after extraction finishes.")
    qs = payload.get("questions")
    if not isinstance(qs, list):
        raise HTTPException(status_code=400, detail="questions must be a list")
    clean = []
    for i, q in enumerate(qs, 1):
        if not isinstance(q, dict): continue
        item = normalize_question(q, int(q.get("source_page", 1) or 1), int(q.get("source_page", 1) or 1))
        item["id"] = i
        clean.append(item)
    jobs[job_id] = job
    jobs[job_id]["questions"] = clean
    jobs[job_id]["message"] = f"Edited data saved: {len(clean)} questions."
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
    if fmt == "txt":
        lines = []
        for i, q in enumerate(questions, 1):
            n = q.get("question_number") or str(i)
            lines.append(f"Q{n}. {q.get('question_english','')}")
            if q.get("question_hindi"): lines.append(q["question_hindi"])
            for opt in q.get("options", []):
                label = opt.get("label", "")
                en = opt.get("english", "")
                hi = opt.get("hindi", "")
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
    raise HTTPException(status_code=400, detail="Unsupported format. Use json, txt, or html.")
