"""
Evaluate the running Agentic RAG app against evaluation/whitepaper_eval.json.

1. Ask: send each question to the app's /query, one at a time, and save the raw responses.
2. Retrieve: recreate each answer's final retrieval inside the app container with the app's
   own code (simple: top-k for final_query; complex: top-k per sub-query, merged and
   de-duplicated), and map each returned source back to its full chunk text.
3. Judge: score each in-scope answer with one Gemini judge request (temperature 0, its own
   prompt, not the app's grader); out-of-scope questions need no judge call.

Judge requests are budgeted for the free tier's 20 requests per day: at most one request per
in-scope question per run, no retries, and a hard cap (--max-judge-requests). A question
whose judge request fails is left unjudged; re-running judges only the missing ones. A 429
from the judge stops judging for the run.

Usage (from the repo root, with the app running via docker compose):
    python3 evaluation/run_eval.py [--base-url http://localhost:8000] [--container agentic-rag]
                                   [--judge-model gemini-3.5-flash] [--max-judge-requests 20]
                                   [--out DIR]

Results go to evaluation/results/<date>-<commit>/ by default. Only the standard library is
needed. The API key is read from GOOGLE_API_KEY or .env and is never written to any output.
"""

import argparse
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "whitepaper_eval.json"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
JUDGE_MIN_INTERVAL = 8.5  # seconds between judge requests (about 7 per minute)
DEFAULT_MAX_JUDGE_REQUESTS = 20  # the judge model's free-tier daily quota


class JudgeUnavailable(RuntimeError):
    """The judge request failed; the question stays unjudged for a later run."""


class JudgeStopped(RuntimeError):
    """Stop judging for this run (rate limit or request cap)."""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        sys.exit("GOOGLE_API_KEY not set and not found in .env")
    return key


def git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    )
    return proc.stdout.strip() or "unknown"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- 1. Ask the app


def ask(base_url: str, question: str) -> dict:
    req = urllib.request.Request(
        f"{base_url}/query",
        data=json.dumps({"query": question, "include_sources": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            http_status, body = resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        http_status = e.code
        try:
            body = json.load(e)
        except Exception:
            body = {"status": "error", "answer": f"HTTP {e.code}"}
    except Exception as e:
        http_status, body = None, {"status": "error", "answer": f"{type(e).__name__}: {e}"}
    body["http_status"] = http_status
    body["client_latency_ms"] = round((time.time() - start) * 1000, 1)
    return body


def run_queries(base_url: str, items: list[dict], raw_path: Path, pause: float) -> list[dict]:
    done = {r["id"]: r for r in read_jsonl(raw_path)}
    for n, item in enumerate(items, 1):
        if item["id"] in done:
            continue
        print(f"[ask {n}/{len(items)}] {item['id']}: {item['question'][:60]}", flush=True)
        done[item["id"]] = {"id": item["id"], "response": ask(base_url, item["question"])}
        r = done[item["id"]]["response"]
        print(
            f"    status={r.get('status')} path={r.get('query_type')} "
            f"rewrites={r.get('iterations')} revised={r.get('revised')} "
            f"is_grounded={r.get('is_grounded')} {r['client_latency_ms'] / 1000:.1f}s",
            flush=True,
        )
        write_jsonl(raw_path, [done[i["id"]] for i in items if i["id"] in done])
        time.sleep(pause)
    return [done[i["id"]] for i in items]


# ---------------------------------------------------------------- 2. Retrieval (in the container)

RETRIEVE_CODE = r"""
import json, sys
import chromadb
from app.config import get_settings
from app.retrieval.vectorstore import get_vectorstore_manager
s = get_settings()
req = json.load(sys.stdin)
manager = get_vectorstore_manager()
out = {"k": s.retrieval_k, "retrieved": {}}
for qid, q in req["queries"].items():
    if q["sub_queries"]:
        docs = manager.retrieve_for_queries(q["sub_queries"], k=s.retrieval_k)
    else:
        docs = manager.vectorstore.similarity_search(q["final_query"], k=s.retrieval_k)
    out["retrieved"][qid] = [{"page": d.metadata.get("page"), "text": d.page_content} for d in docs]
col = chromadb.PersistentClient(path=s.chroma_persist_directory).get_collection(s.collection_name)
r = col.get(include=["documents", "metadatas"])
out["chunks"] = [{"id": i, "page": m.get("page"), "text": d}
                 for i, m, d in zip(r["ids"], r["metadatas"], r["documents"])]
print("@@JSON@@" + json.dumps(out))
"""


def recreate_retrieval(container: str, queries: dict[str, dict]) -> dict:
    """Recreate each answer's final retrieval with the app's own code, inside its container."""
    proc = subprocess.run(
        ["docker", "exec", "-i", container, "python", "-c", RETRIEVE_CODE],
        input=json.dumps({"queries": queries}),
        capture_output=True,
        text=True,
        timeout=600,
    )
    marker = next((ln for ln in proc.stdout.splitlines() if ln.startswith("@@JSON@@")), None)
    if marker is None:
        sys.exit(f"Retrieval in container failed:\n{proc.stderr[-2000:]}")
    return json.loads(marker[len("@@JSON@@") :])


def full_chunk(source: dict, chunks: list[dict]) -> dict | None:
    """Map a /query source (first 500 characters of a chunk) back to the full chunk."""
    prefix = norm(source.get("content", ""))
    for c in chunks:
        if c["page"] == source.get("page") and norm(c["text"]).startswith(prefix):
            return c
    return None


# ---------------------------------------------------------------- 3. Judge

JUDGE_SYSTEM = """You are an impartial evaluator of a retrieval-augmented question answering system.
You compare the system's answer with a reference answer written from the source document, and
check the answer against the text the system was given. Judge only from the material provided;
do not use outside knowledge to decide whether a claim is supported.

Definitions:
- correctness: "correct" if the answer conveys all key facts of the reference answer without
  contradicting it (extra accurate detail is fine); "partial" if some key facts are missing or
  there is a minor inaccuracy; "incorrect" if it is wrong, contradicts the reference, or does
  not answer (for example, says it could not find the information).
- claims: split the answer into atomic factual claims (ignore greetings, hedges and statements
  about the system itself). For each, "supported" is true only if the GENERATION CONTEXT states
  or directly implies it.
- answer_relevance: "relevant" if the answer addresses the question asked, "partial" if it
  addresses it only in part or drifts, "irrelevant" otherwise.
- retrieved_chunks: for each numbered RETRIEVED CHUNK, "relevant" is true if it contains
  information that helps answer the question."""

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "correctness": {"type": "STRING", "enum": ["correct", "partial", "incorrect"]},
        "correctness_reason": {"type": "STRING"},
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"claim": {"type": "STRING"}, "supported": {"type": "BOOLEAN"}},
                "required": ["claim", "supported"],
            },
        },
        "answer_relevance": {"type": "STRING", "enum": ["relevant", "partial", "irrelevant"]},
        "retrieved_chunks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"index": {"type": "INTEGER"}, "relevant": {"type": "BOOLEAN"}},
                "required": ["index", "relevant"],
            },
        },
    },
    "required": [
        "correctness",
        "correctness_reason",
        "claims",
        "answer_relevance",
        "retrieved_chunks",
    ],
}


def judge_prompt(item: dict, answer: str, gen_context: list[str], retrieved: list[str]) -> str:
    gen = "\n\n".join(f"[G{i}] {t}" for i, t in enumerate(gen_context, 1)) or "(none)"
    ret = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(retrieved, 1)) or "(none)"
    return (
        f"QUESTION:\n{item['question']}\n\nREFERENCE ANSWER:\n{item['ground_truth']}\n\n"
        f"SYSTEM ANSWER:\n{answer}\n\nGENERATION CONTEXT (text the system answered from):\n{gen}\n\n"
        f"RETRIEVED CHUNKS ({len(retrieved)} chunks from the system's final retrieval):\n{ret}"
    )


class Judge:
    """Sends judge requests: exactly one per call, never retried, within a hard cap."""

    def __init__(self, model: str, key: str, max_requests: int):
        self.model, self.key, self.max_requests = model, key, max_requests
        self.requests_sent, self.last = 0, 0.0

    def __call__(self, prompt: str) -> dict:
        if self.requests_sent >= self.max_requests:
            raise JudgeStopped(f"judge request cap reached ({self.max_requests})")
        body = {
            "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": JUDGE_SCHEMA,
            },
        }
        time.sleep(max(0.0, self.last + JUDGE_MIN_INTERVAL - time.time()))
        self.last = time.time()
        self.requests_sent += 1
        req = urllib.request.Request(
            GEMINI_URL.format(model=self.model),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.key},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            return json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            if e.code == 429:
                quota = re.search(r'"quotaId":\s*"([^"]+)"', text)
                raise JudgeStopped(
                    f"judge rate-limited (429, {quota.group(1) if quota else 'quota unknown'})"
                ) from e
            raise JudgeUnavailable(f"judge HTTP {e.code}") from e
        except (
            TimeoutError,
            urllib.error.URLError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as e:
            raise JudgeUnavailable(f"judge {type(e).__name__}") from e


# ---------------------------------------------------------------- Scoring


def base_row(item: dict, response: dict, type_: str) -> dict:
    return {
        "id": item["id"],
        "type": type_,
        "question": item["question"],
        "answer": response.get("answer"),
        "status": response.get("status"),
        "http_status": response.get("http_status"),
        "path": response.get("query_type"),
        "sub_queries": response.get("sub_queries") or [],
        "rewrites": response.get("iterations"),
        "final_query": response.get("final_query"),
        "is_grounded": response.get("is_grounded"),
        "revised": response.get("revised"),
        "caveat": response.get("caveat"),
        "latency_ms": response.get("latency_ms"),
        "client_latency_ms": response.get("client_latency_ms"),
    }


def context_row(item, response, retrieved, chunks) -> dict:
    """Everything about an in-scope answer that doesn't need the judge."""
    gen_chunks = [full_chunk(s, chunks) for s in response.get("sources", [])]
    gen_context = [
        c["text"] if c else s.get("content", "")
        for c, s in zip(gen_chunks, response.get("sources", []), strict=True)
    ]
    quotes = [s["quote"] for s in item["supporting"]]
    in_retrieved = [any(q in norm(c["text"]) for c in retrieved) for q in quotes]
    row = base_row(item, response, item["type"])
    row.update(
        ground_truth=item["ground_truth"],
        sources=[
            {"page": s.get("page"), "chunk_id": c["id"] if c else None}
            for s, c in zip(response.get("sources", []), gen_chunks, strict=True)
        ],
        retrieved=[{"page": c["page"], "chunk_id": c.get("id")} for c in retrieved],
        supporting_quote_in_retrieved=in_retrieved,
        supporting_quote_in_generation_context=[
            any(q in norm(t) for t in gen_context) for q in quotes
        ],
        context_recall=sum(in_retrieved) / len(quotes),
        judge=None,
    )
    return row, gen_context


def apply_verdict(row: dict, verdict: dict, n_retrieved: int) -> None:
    claims = verdict.get("claims", [])
    relevant = {c["index"]: c["relevant"] for c in verdict.get("retrieved_chunks", [])}
    row.update(
        judge=verdict,
        correctness=verdict["correctness"],
        answer_relevance=verdict["answer_relevance"],
        faithfulness=(sum(c["supported"] for c in claims) / len(claims)) if claims else None,
        context_precision=(
            sum(bool(relevant.get(i)) for i in range(1, n_retrieved + 1)) / n_retrieved
            if n_retrieved
            else None
        ),
    )
    row["diagnosis"] = diagnose(row)


def diagnose(row: dict) -> str | None:
    """A first, mechanical cause for answers that were not fully correct."""
    if row.get("correctness") == "correct":
        return None
    if row["status"] == "no_relevant_docs":
        if any(row["supporting_quote_in_retrieved"]):
            return f"grader rejected retrieved supporting chunk(s) ({row['rewrites']} rewrites)"
        return f"retrieval miss for final query, then gave up ({row['rewrites']} rewrites)"
    if not any(row["supporting_quote_in_retrieved"]):
        return "retrieval miss: no supporting passage in the final retrieval"
    if not all(row["supporting_quote_in_generation_context"]):
        return "supporting passage(s) missing from generation context (retrieval or grader)"
    return "generation: supporting passages were in context"


def score_out_of_scope(item, response) -> dict:
    row = base_row(item, response, "out_of_scope")
    refused = row["status"] == "no_relevant_docs"
    row.update(
        refused_correctly=refused,
        diagnosis=None if refused else f"answered instead of refusing (status={row['status']})",
    )
    return row


# ---------------------------------------------------------------- Summary


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered) + 0.5) - 1))]


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def scored(rows: list[dict]) -> list[dict]:
    """Rows with a result: judged answers plus app errors (which count as not correct)."""
    return [r for r in rows if r.get("judge") or r["status"] == "error"]


def block(rows: list[dict], title: str) -> list[str]:
    done = scored(rows)
    judged = [r for r in done if r.get("judge")]
    counts = {
        k: sum(r.get("correctness") == k for r in judged)
        for k in ("correct", "partial", "incorrect")
    }
    rel = {
        k: sum(r.get("answer_relevance") == k for r in judged)
        for k in ("relevant", "partial", "irrelevant")
    }
    n = len(done)
    return [
        f"### {title} (n={len(rows)}, scored {n})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Correct / partial / incorrect | {counts['correct']} / {counts['partial']} / {counts['incorrect']} |",
        f"| Accuracy (correct only) | {fmt(counts['correct'] / n if n else None)} |",
        f"| Score (correct = 1, partial = 0.5) | {fmt((counts['correct'] + 0.5 * counts['partial']) / n if n else None)} |",
        f"| Faithfulness (mean share of supported claims) | {fmt(mean([r.get('faithfulness') for r in judged]))} |",
        f"| Answer relevance (relevant / partial / irrelevant) | {rel['relevant']} / {rel['partial']} / {rel['irrelevant']} |",
        f"| Context recall (mean share of supporting quotes retrieved) | {fmt(mean([r['context_recall'] for r in rows]))} |",
        f"| Context recall (all supporting quotes retrieved) | {sum(r['context_recall'] == 1 for r in rows)}/{len(rows)} |",
        f"| Context precision (mean share of retrieved chunks judged relevant) | {fmt(mean([r.get('context_precision') for r in judged]))} |",
        f"| Errors | {sum(r['status'] == 'error' for r in rows)} |",
        f"| Unjudged | {len(rows) - n} |",
        "",
    ]


def path_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Path | n | Correct / partial / incorrect | Faithfulness | Recall | Precision | p50 latency, s | Avg rewrites | Regenerated |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for path in ("simple", "complex", None):
        group = [r for r in rows if r.get("path") == path]
        if not group:
            continue
        judged = [r for r in group if r.get("judge")]
        cpi = "/".join(
            str(sum(r.get("correctness") == k for r in judged))
            for k in ("correct", "partial", "incorrect")
        )
        lat = [r["latency_ms"] / 1000 for r in group if r.get("latency_ms")]
        lines.append(
            f"| {path or 'unknown (error)'} | {len(group)} | {cpi} | "
            f"{fmt(mean([r.get('faithfulness') for r in judged]))} | "
            f"{fmt(mean([r.get('context_recall') for r in group]))} | "
            f"{fmt(mean([r.get('context_precision') for r in judged]))} | "
            f"{fmt(pct(lat, 0.5), 1)} | {fmt(mean([r['rewrites'] for r in group]))} | "
            f"{sum(bool(r.get('revised')) for r in group)} |"
        )
    return lines + [""]


def summarize(rows: list[dict], meta: dict) -> str:
    core = [r for r in rows if r["type"] in ("factual", "multi_passage")]
    synth = [r for r in rows if r["type"] == "synthesis"]
    in_scope = core + synth
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    ok = [r for r in rows if r["status"] != "error" and r.get("latency_ms")]
    server_lat = [r["latency_ms"] / 1000 for r in ok]
    client_lat = [r["client_latency_ms"] / 1000 for r in rows if r.get("client_latency_ms")]
    revised = [r for r in rows if r.get("revised")]

    pairs = [
        (r["is_grounded"], r["faithfulness"] == 1.0)
        for r in in_scope
        if r.get("is_grounded") is not None and r.get("faithfulness") is not None
    ]
    agree = sum(a == b for a, b in pairs)

    lines = [
        f"# Evaluation: {meta['dataset']}",
        "",
        f"- Run: {meta['started']} to {meta['finished']} against {meta['base_url']} (app commit {meta['commit']})",
        f"- App model: {meta['app_model']}; judge: {meta['judge_model']} (temperature 0)",
        f"- Judge requests sent: {meta['judge_requests']} of a cap of {meta['judge_cap']}"
        + (f"; judging stopped: {meta['judge_stopped']}" if meta["judge_stopped"] else ""),
        f"- Context metrics use each answer's final retrieval, recreated with the app's code (k={meta['k']}; complex: k per sub-query, merged)",
        "",
        "## Answers",
        "",
        *block(core, "In-scope, factual and multi-passage"),
        *block(synth, "Synthesis (q20, reported separately)"),
        "## By path (router label)",
        "",
        "In-scope:",
        "",
        *path_table(in_scope),
        "Out-of-scope router labels: " + ", ".join(f"{r['id']}={r.get('path')}" for r in oos),
        "",
        "## Refusals",
        "",
        f"Out-of-scope refused with no_relevant_docs: {sum(r['refused_correctly'] for r in oos)}/{len(oos)}",
        "",
        "## Self-correction",
        "",
        f"- Answers regenerated: {len(revised)}/{len(rows)}",
        f"- Grounded after regeneration: {sum(r.get('is_grounded') is True for r in revised)}",
        f"- Still ungrounded, returned with a caveat: {sum(bool(r.get('caveat')) for r in rows)}",
        "",
        "## Hallucination check agreement",
        "",
        "App's final `is_grounded` vs judge (grounded = every claim supported):",
        "",
        "| | judge grounded | judge not grounded |",
        "|---|---|---|",
        f"| app is_grounded = true | {sum(a and b for a, b in pairs)} | {sum(a and not b for a, b in pairs)} |",
        f"| app is_grounded = false | {sum((not a) and b for a, b in pairs)} | {sum((not a) and (not b) for a, b in pairs)} |",
        "",
        f"Agreement: {agree}/{len(pairs)}" + (f" ({agree / len(pairs):.0%})" if pairs else ""),
        "",
        "## Latency and rewrites",
        "",
        "| | p50 | p95 |",
        "|---|---|---|",
        f"| Server latency, s (n={len(server_lat)}) | {fmt(pct(server_lat, 0.5), 1)} | {fmt(pct(server_lat, 0.95), 1)} |",
        f"| Client wall time, s (n={len(client_lat)}) | {fmt(pct(client_lat, 0.5), 1)} | {fmt(pct(client_lat, 0.95), 1)} |",
        "",
        f"- Average rewrites, in-scope: {fmt(mean([r['rewrites'] for r in in_scope]))}; out-of-scope: {fmt(mean([r['rewrites'] for r in oos]))}",
        f"- Errors (status error or HTTP failure): {sum(r['status'] == 'error' for r in rows)}",
        "",
        "## Not fully correct, failures, caveats, refusal misses and unjudged",
        "",
    ]
    flagged = [
        r
        for r in rows
        if r.get("diagnosis")
        or r["status"] == "error"
        or r.get("caveat")
        or (r["type"] != "out_of_scope" and not r.get("judge") and r["status"] != "error")
    ]
    if not flagged:
        lines.append("None.")
    for r in flagged:
        outcome = r.get("correctness") or ("unjudged" if r["type"] != "out_of_scope" else "n/a")
        if r["status"] == "error":
            cause = f"app error: {norm(r.get('answer') or '')[:200]}"
        else:
            cause = r.get("diagnosis") or r.get("unjudged_reason") or "returned with a caveat"
        lines += [
            f"### {r['id']} ({r['type']}, path {r.get('path')}): {r['question']}",
            "",
            f"- Outcome: {outcome}, status {r['status']}, {r['rewrites']} rewrites, "
            f"revised={r.get('revised')}, is_grounded={r.get('is_grounded')}",
            f"- Final query: {r.get('final_query')!r}; sub-queries: {r.get('sub_queries')}",
            f"- Mechanical diagnosis: {cause}",
        ]
        if r.get("caveat"):
            lines.append(f"- Caveat shown: {r['caveat']}")
        if r.get("judge"):
            lines.append(f"- Judge: {r['judge']['correctness_reason']}")
        lines += [f"- Answer: {norm(r.get('answer') or '')[:400]}", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- Main


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--container", default="agentic-rag")
    ap.add_argument("--judge-model", default="gemini-3.5-flash")
    ap.add_argument("--max-judge-requests", type=int, default=DEFAULT_MAX_JUDGE_REQUESTS)
    ap.add_argument("--out", help="results directory (default: results/<date>-<commit>)")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between app queries")
    args = ap.parse_args()

    commit = git_commit()
    out = Path(args.out or ROOT / "evaluation" / "results" / f"{dt.date.today()}-{commit}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {out}", flush=True)
    data = json.loads(DATASET.read_text())
    items = data["in_scope"] + data["out_of_scope"]
    started = dt.datetime.now().isoformat(timespec="seconds")

    raw = run_queries(args.base_url, items, out / "raw_responses.jsonl", args.pause)
    responses = {r["id"]: r["response"] for r in raw}

    queries = {
        i["id"]: {
            "final_query": responses[i["id"]].get("final_query") or i["question"],
            "sub_queries": responses[i["id"]].get("sub_queries") or [],
        }
        for i in data["in_scope"]
    }
    print("[retrieve] recreating each answer's final retrieval in the container", flush=True)
    ret = recreate_retrieval(args.container, queries)
    chunk_ids = {norm(c["text"]): c["id"] for c in ret["chunks"]}
    for docs in ret["retrieved"].values():
        for d in docs:
            d["id"] = chunk_ids.get(norm(d["text"]))

    judge = Judge(args.judge_model, api_key(), args.max_judge_requests)
    # Judged rows are saved as they complete; a re-run judges only the questions still missing
    judged_path = out / "judged.jsonl"
    judged = {
        r["id"]: r for r in read_jsonl(judged_path) if r.get("judge_model") == args.judge_model
    }
    stopped = None
    rows = []
    for n, item in enumerate(data["in_scope"], 1):
        if item["id"] in judged:
            rows.append(judged[item["id"]])
            continue
        response = responses[item["id"]]
        retrieved = ret["retrieved"][item["id"]]
        row, gen_context = context_row(item, response, retrieved, ret["chunks"])
        rows.append(row)
        if row["status"] == "error":
            row["diagnosis"] = None
            continue
        if stopped:
            row["unjudged_reason"] = stopped
            continue
        print(f"[judge {n}/{len(data['in_scope'])}] {item['id']}", flush=True)
        try:
            verdict = judge(
                judge_prompt(
                    item, response.get("answer", ""), gen_context, [c["text"] for c in retrieved]
                )
            )
        except JudgeStopped as e:
            stopped = str(e)
            row["unjudged_reason"] = stopped
            print(f"    {stopped}: judging stopped for this run", flush=True)
            continue
        except JudgeUnavailable as e:
            row["unjudged_reason"] = str(e)
            print(f"    {e}: left unjudged (re-run later to judge it)", flush=True)
            continue
        apply_verdict(row, verdict, len(retrieved))
        row["judge_model"] = args.judge_model
        with judged_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"    {row['correctness']} faith={fmt(row.get('faithfulness'))} "
            f"recall={fmt(row['context_recall'])} prec={fmt(row.get('context_precision'))}",
            flush=True,
        )
    rows += [score_out_of_scope(i, responses[i["id"]]) for i in data["out_of_scope"]]
    write_jsonl(out / "results.jsonl", rows)

    app_model = (
        subprocess.run(
            ["docker", "exec", args.container, "printenv", "LLM_MODEL"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "unknown"
    )
    meta = {
        "dataset": DATASET.name,
        "started": started,
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "commit": commit,
        "app_model": app_model,
        "judge_model": args.judge_model,
        "judge_requests": judge.requests_sent,
        "judge_cap": args.max_judge_requests,
        "judge_stopped": stopped,
        "k": ret["k"],
    }
    (out / "summary.md").write_text(summarize(rows, meta))
    unjudged = [
        r["id"]
        for r in rows
        if r["type"] != "out_of_scope" and not r.get("judge") and r["status"] != "error"
    ]
    print(f"\nJudge requests sent: {judge.requests_sent}/{args.max_judge_requests}")
    if unjudged:
        print(f"Unjudged (re-run to judge): {', '.join(unjudged)}")
    print(f"Wrote {out / 'results.jsonl'} and {out / 'summary.md'}")


if __name__ == "__main__":
    main()
