#!/usr/bin/env python3
"""
Terminal support triage agent for the Multi-Domain Support Triage Challenge.

The agent is self-contained: it uses local CSV/text/html/md files only, and never calls
network APIs. It retrieves from the provided support corpus, applies conservative risk
routing, and writes the required output CSV.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ALLOWED_STATUSES = {"replied", "escalated"}
ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
KNOWN_COMPANIES = ("HackerRank", "Claude", "Visa")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "have", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "please", "the", "this", "to", "we", "what", "when", "where", "with",
    "you", "your",
}

COMPANY_TERMS = {
    "HackerRank": {"hackerrank", "test", "assessment", "candidate", "interview", "proctor", "proctoring", "codepair", "challenge", "recruiter", "invitation", "score", "question", "plagiarism"},
    "Claude": {"claude", "anthropic", "chat", "conversation", "workspace", "team", "model", "prompt", "api", "subscription", "plan", "usage", "message"},
    "Visa": {"visa", "card", "credit", "debit", "transaction", "charge", "merchant", "bank", "issuer", "atm", "fraud", "lost", "stolen", "payment", "dispute"},
}

PRODUCT_KEYWORDS = {
    "HackerRank": {
        "Assessments": {"assessment", "test", "candidate", "invite", "invitation", "score", "question", "challenge"},
        "Interviews": {"interview", "codepair", "live", "pair", "interviewer"},
        "Account and access": {"login", "password", "account", "sso", "permission", "access", "email"},
        "Billing": {"billing", "invoice", "subscription", "payment", "refund", "charge"},
        "Integrity and proctoring": {"proctor", "webcam", "plagiarism", "integrity", "cheat", "suspicious"},
    },
    "Claude": {
        "Account and login": {"login", "password", "account", "email", "sso", "access", "verification"},
        "Billing and plans": {"billing", "invoice", "subscription", "plan", "refund", "charge", "payment"},
        "Claude product usage": {"chat", "conversation", "message", "prompt", "model", "artifact", "project"},
        "Team and workspace administration": {"team", "workspace", "member", "admin", "seat", "permission"},
        "API and integrations": {"api", "key", "rate", "limit", "integration", "token"},
    },
    "Visa": {
        "Lost or stolen cards": {"lost", "stolen", "missing", "card", "replacement"},
        "Fraud and unauthorized transactions": {"fraud", "unauthorized", "scam", "charge", "dispute", "transaction"},
        "Payments and transactions": {"payment", "declined", "merchant", "refund", "transaction", "charge"},
        "Cardholder support": {"card", "issuer", "bank", "atm", "pin", "debit", "credit"},
    },
    "General": {
        "Out of scope": {"weather", "recipe", "homework", "joke", "poem", "unrelated"},
        "General support": {"help", "support", "issue", "question"},
    },
}

HIGH_RISK_TERMS = {
    "fraud", "stolen", "lost card", "unauthorized", "scam", "chargeback", "legal",
    "lawsuit", "security breach", "personal data", "pii", "password reset for",
    "account takeover", "hacked", "compromised", "refund", "billing dispute",
    "invoice correction", "delete my account", "cannot access", "locked out",
    "payment failed", "sensitive",
}
BUG_TERMS = {"bug", "error", "crash", "broken", "fails", "failed", "failure", "not working", "stuck", "500", "404"}
FEATURE_TERMS = {"feature", "request", "add", "support for", "enhancement", "would like", "can you build", "dark mode"}
MALICIOUS_TERMS = {"ignore previous", "system prompt", "developer message", "jailbreak", "bypass policy", "secret key"}


@dataclass
class Doc:
    company: str
    title: str
    path: str
    text: str
    tokens: Counter


@dataclass
class Decision:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9'-]{1,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def clean_text(raw: str) -> str:
    raw = html.unescape(raw)
    raw = re.sub(r"<(script|style).*?</\\1>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"https?://\S+", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def infer_company_from_path(path: Path) -> str:
    lowered = str(path).lower()
    if "hackerrank" in lowered:
        return "HackerRank"
    if "claude" in lowered or "anthropic" in lowered:
        return "Claude"
    if "visa" in lowered:
        return "Visa"
    return "General"


def load_corpus(root: Path, explicit: list[Path]) -> list[Doc]:
    candidates: list[Path] = []
    if explicit:
        for item in explicit:
            if item.is_dir():
                candidates.extend(p for p in item.rglob("*") if p.is_file())
            elif item.is_file():
                candidates.append(item)
    else:
        for pattern in ("corpus/**/*", "support_corpus/**/*", "docs/**/*", "knowledge_base/**/*"):
            candidates.extend(p for p in root.glob(pattern) if p.is_file())

    docs: list[Doc] = []
    for path in sorted(set(candidates)):
        if path.name in {"sample_support_tickets.csv", "support_tickets.csv"}:
            continue
        if path.suffix.lower() not in {".txt", ".md", ".html", ".htm", ".csv"}:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text = clean_text(raw)
        if len(text) < 40:
            continue
        docs.append(Doc(
            company=infer_company_from_path(path),
            title=path.stem.replace("_", " ").replace("-", " ").strip().title(),
            path=str(path),
            text=text,
            tokens=Counter(tokenize(text)),
        ))
    return docs


def infer_company(company_field: str, text: str) -> str:
    normalized = (company_field or "").strip().lower()
    for company in KNOWN_COMPANIES:
        if normalized == company.lower():
            return company
    token_set = set(tokenize(text))
    scores = {company: len(token_set & terms) for company, terms in COMPANY_TERMS.items()}
    best, score = max(scores.items(), key=lambda item: item[1])
    return best if score > 0 else "General"


def classify_request_type(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in MALICIOUS_TERMS):
        return "invalid"
    if len(tokenize(text)) < 2:
        return "invalid"
    if any(term in lowered for term in FEATURE_TERMS):
        return "feature_request"
    if any(term in lowered for term in BUG_TERMS):
        return "bug"
    return "product_issue"


def classify_product_area(company: str, text: str) -> str:
    token_set = set(tokenize(text))
    areas = PRODUCT_KEYWORDS.get(company, PRODUCT_KEYWORDS["General"])
    best, score = max(((area, len(token_set & terms)) for area, terms in areas.items()), key=lambda item: item[1])
    if score:
        return best
    return "General support" if company == "General" else f"{company} support"


def retrieve(text: str, company: str, docs: list[Doc], limit: int = 3) -> list[tuple[float, Doc]]:
    query = Counter(tokenize(text))
    if not query:
        return []
    query_norm = math.sqrt(sum(v * v for v in query.values()))
    scored: list[tuple[float, Doc]] = []
    for doc in docs:
        if doc.company not in {company, "General"} and company != "General":
            continue
        dot = sum(query[t] * doc.tokens.get(t, 0) for t in query)
        if dot <= 0:
            continue
        doc_norm = math.sqrt(sum(v * v for v in doc.tokens.values())) or 1.0
        company_bonus = 1.15 if doc.company == company else 1.0
        scored.append(((dot / (query_norm * doc_norm)) * company_bonus, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


def is_high_risk(text: str, request_type: str, company: str, product_area: str) -> bool:
    lowered = text.lower()
    if request_type == "invalid":
        return False
    if any(term in lowered for term in HIGH_RISK_TERMS):
        return True
    if company == "Visa" and product_area in {"Lost or stolen cards", "Fraud and unauthorized transactions"}:
        return True
    if "someone else" in lowered and ("account" in lowered or "card" in lowered):
        return True
    return False


def safe_excerpt(doc: Doc, query: str, max_chars: int = 360) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", doc.text)
    query_terms = set(tokenize(query))
    ranked = sorted(sentences, key=lambda s: len(query_terms & set(tokenize(s))), reverse=True)
    excerpt = next((s.strip() for s in ranked if len(s.strip()) > 30), doc.text[:max_chars])
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return excerpt


def build_response(text: str, company: str, product_area: str, request_type: str, retrieved: list[tuple[float, Doc]], escalate: bool) -> str:
    if request_type == "invalid":
        return "I can only help with support questions for HackerRank, Claude, or Visa using the provided support corpus. Please send a relevant support issue."
    if company == "General" and not retrieved:
        return "I cannot answer this safely from the provided HackerRank, Claude, or Visa support corpus. Please contact the relevant support team with more details."
    if escalate:
        source = f" Relevant corpus match: {retrieved[0][1].title}." if retrieved else ""
        return f"This looks like a {product_area.lower()} case for {company} and should be handled by a human support specialist. Please avoid sharing sensitive account, payment, assessment, or card details in chat and use the official {company} support channel for secure handling.{source}"
    if not retrieved:
        return f"I do not have enough matching {company} support documentation in the provided corpus to answer this safely. Please escalate this to support."
    top_doc = retrieved[0][1]
    excerpt = safe_excerpt(top_doc, text)
    return f"Based on the provided {company} support documentation, the most relevant guidance is from '{top_doc.title}': {excerpt} If this does not resolve the issue, contact support with the exact error, account context, and steps already tried."


def justify(company: str, product_area: str, request_type: str, retrieved: list[tuple[float, Doc]], escalate: bool) -> str:
    reason = "high-risk or sensitive content" if escalate else "answerable from retrieved corpus"
    if request_type == "invalid":
        reason = "irrelevant, unsafe, or insufficient support request"
    elif not retrieved and not escalate:
        reason = "no sufficiently relevant corpus match"
    docs = ", ".join(doc.title for _, doc in retrieved[:2]) or "none"
    return f"Classified as {request_type}; company={company}; area={product_area}; decision based on {reason}; retrieved={docs}."


def decide(row: dict[str, str], docs: list[Doc]) -> Decision:
    text = f"{row.get('subject', '')}\n{row.get('issue', '')}".strip()
    company = infer_company(row.get("company", ""), text)
    request_type = classify_request_type(text)
    product_area = classify_product_area(company, text)
    retrieved = retrieve(text, company, docs)
    high_risk = is_high_risk(text, request_type, company, product_area)
    unsupported = request_type != "invalid" and not retrieved and company != "General"
    status = "escalated" if high_risk or unsupported else "replied"
    response = build_response(text, company, product_area, request_type, retrieved, status == "escalated")
    return Decision(status, product_area, response, justify(company, product_area, request_type, retrieved, status == "escalated"), request_type)


def run(input_csv: Path, output_csv: Path, docs: list[Doc]) -> None:
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "product_area", "response", "justification", "request_type"])
        writer.writeheader()
        for row in rows:
            decision = decide(row, docs)
            assert decision.status in ALLOWED_STATUSES
            assert decision.request_type in ALLOWED_REQUEST_TYPES
            writer.writerow(decision.__dict__)


def evaluate(sample_csv: Path, docs: list[Doc]) -> None:
    if not sample_csv.exists():
        print(f"No sample file found at {sample_csv}")
        return
    with sample_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    totals = Counter()
    correct = Counter()
    for row in rows:
        pred = decide(row, docs)
        for field in ("status", "request_type"):
            expected = (row.get(field) or "").strip()
            if expected:
                totals[field] += 1
                correct[field] += int(getattr(pred, field) == expected)
    for field in ("status", "request_type"):
        total = totals[field]
        score = correct[field] / total if total else 0
        print(f"{field}: {correct[field]}/{total} ({score:.1%})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local support ticket triage.")
    parser.add_argument("--input", default="support_tickets.csv", help="Input ticket CSV.")
    parser.add_argument("--output", default="triage_output.csv", help="Required output CSV.")
    parser.add_argument("--sample", default="sample_support_tickets.csv", help="Sample labeled CSV.")
    parser.add_argument("--corpus", action="append", default=[], help="Corpus file or directory. Repeatable.")
    parser.add_argument("--eval", action="store_true", help="Evaluate status/request_type on the sample CSV.")
    args = parser.parse_args()

    root = Path.cwd()
    docs = load_corpus(root, [Path(p) for p in args.corpus])
    if not docs:
        print("Warning: no corpus documents found. Add corpus files under corpus/, support_corpus/, docs/, or pass --corpus.")

    if args.eval:
        evaluate(Path(args.sample), docs)

    input_csv = Path(args.input)
    if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}")
        return 2
    run(input_csv, Path(args.output), docs)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
