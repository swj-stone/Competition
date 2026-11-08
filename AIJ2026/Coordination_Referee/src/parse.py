"""Semantic parsing of Coordination Referee runs.

Every message is mapped to a coarse semantic category (assign, deliver, ack,
work, iterate, wait, disagree, drift, ...) together with the subtask and the
counterpart agent it talks about.  The categories are recovered from the text
prefix (robust to truncation) and fall back to protocol type / refs / intent
when the text is too short.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

TOOLS = {"web_search", "ticket_api", "email_api", "calendar_api", "doc_reader",
         "crm_api", "db_query", "file_store", "policy_lookup", "calculator"}

DOMAIN_SUBTASKS = {
    "support": ["apply_workaround", "identify_root_cause", "notify_customer", "pull_logs",
                "reproduce_issue", "triage_ticket", "verify_fix", "write_postmortem"],
    "onboarding": ["assign_buddy", "assign_equipment", "collect_documents", "confirm_day_one",
                   "create_accounts", "grant_access", "run_security_briefing",
                   "schedule_training"],
    "sales": ["build_pricing", "check_discount_policy", "collect_usage_data", "draft_proposal",
              "legal_precheck", "qualify_lead", "schedule_demo", "send_proposal"],
    "procurement": ["check_budget", "collect_requirements", "draft_purchase_order",
                    "negotiate_terms", "obtain_approval", "request_quotes",
                    "shortlist_vendors", "verify_compliance"],
    "legal": ["check_data_processing", "check_liability", "compare_playbook", "draft_redlines",
              "extract_clauses", "flag_risks", "parse_contract", "sign_off"],
    "compliance": ["collect_evidence", "draft_report", "log_findings", "management_response",
                   "rate_severity", "sample_records", "scope_audit", "test_controls"],
}
SUB2DOM = {s: d for d, ss in DOMAIN_SUBTASKS.items() for s in ss}
DOMAINS = list(DOMAIN_SUBTASKS)

ART_DOMAIN = {
    "purchase_order": "procurement", "vendor_shortlist": "procurement",
    "compliance_note": "procurement", "budget_check": "procurement",
    "quote_table": "procurement",
    "postmortem": "support", "customer_email": "support", "rca_report": "support",
    "triage_note": "support", "log_bundle": "support",
    "meeting_invite": "sales", "pricing_sheet": "sales", "proposal_deck": "sales",
    "discount_approval": "sales", "usage_report": "sales",
    "signoff_memo": "legal", "playbook_diff": "legal", "redline_doc": "legal",
    "risk_matrix": "legal", "clause_extract": "legal",
    "readiness_checklist": "onboarding", "training_plan": "onboarding",
    "equipment_manifest": "onboarding", "access_matrix": "onboarding",
    "account_list": "onboarding",
    "audit_report": "compliance", "control_test_log": "compliance",
    "evidence_pack": "compliance", "scope_memo": "compliance",
    "findings_register": "compliance",
}

DRIFT_TOPICS = ["office relocation", "next year travel policy", "marketing campaign",
                "datacenter migration", "unrelated customer escalation",
                "internal wiki cleanup", "quarterly hiring plan", "competitor launch"]


def goal_domain(goal: str) -> str:
    g = goal.lower()
    # compliance first: "Run the Q1 compliance sweep over incident logs ..." / "Audit the incident logs
    # control set ..." mention "incident" but are compliance goals (their subtasks are compliance ones)
    if "compliance" in g or "audit" in g:
        return "compliance"
    if "incident" in g or "ticket" in g:
        return "support"
    if "onboard" in g:
        return "onboarding"
    if "renewal" in g or "proposal" in g:
        return "sales"
    if "purchase order" in g or "provision" in g:
        return "procurement"
    if "compliance" in g or "audit" in g:
        return "compliance"
    if "legal" in g or "agreement" in g or "contract" in g or "clause" in g:
        return "legal"
    return "unknown"


# --- categories --------------------------------------------------------------
ASSIGN, DELIVER, RECOVER, ACK, WORK, ITER, CHAT, DRIFT_SOFT, DRIFT_HARD, \
    SET, DISAGREE, WAIT, FINAL, STOP, OTHER = range(15)
CAT_NAMES = ["assign", "deliver", "recover", "ack", "work", "iter", "chat", "drift_soft",
             "drift_hard", "set", "disagree", "wait", "final", "stop", "other"]

_SUBRE = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
_AGRE = re.compile(r"\ba\d+\b")

_HANDOFF_ASSIGN = ("Assigning", "Task alloc", "Handing")
_HANDOFF_RECOVER = ("Recovery", "Escalating", "Detected", "Mitigation", "Repairing", "Fallback")
_HANDOFF_DELIVER_PREFIX = ("Output for", "Finished", "Delivering")
_ACK = ("Acknowledged", "Got it", "Taking", "Roger", "Confirmed")
_WORK = ("Working through", "Processing", "Called", "Ran ", "Intermediate")
_ITER = ("Another pass", "Iterating", "Re-checking", "One more attempt", "Repeating",
         "Looping back")
_CHAT = ("Sounds good", "Noted", "Understood", "Ok, cont", "Fine by me", "Ack.", "Alright",
         "Copy that")
_DRIFT_SOFT = ("While here", "Expanding scope")
_DRIFT_HARD = ("Side note", "Deprioriti", "New direction", "Pivoting")
_SET = ("Recorded", "Shared state", "Setting", "State patch", "Writing")
_DIS = ("Overriding", "That contradicts", "I disagree", "Conflict on", "Rejecting")


def _starts(text: str, prefixes) -> bool:
    return any(text.startswith(p) or (len(text) >= 4 and p.startswith(text)) for p in prefixes)


def categorize(m: Dict[str, Any]) -> int:
    typ = m.get("type", "")
    text = m.get("text", "") or ""
    if typ == "final":
        return FINAL
    if typ == "system":
        return STOP
    if typ == "status":
        return WAIT
    if typ == "handoff":
        if _AGRE.match(text) and (", please" in text[:6] or text[2:4] in (": ", ", ", " o")
                                  or re.match(r"a\d+[:,]? ", text) or re.match(r"a\d+ own", text)):
            return ASSIGN
        if _starts(text, _HANDOFF_ASSIGN):
            return ASSIGN
        if _starts(text, _HANDOFF_RECOVER):
            return RECOVER
        if _starts(text, _HANDOFF_DELIVER_PREFIX):
            return DELIVER
        if m.get("refs"):
            return DELIVER
        # "<sub> is done..." / "<sub> complete" / "<sub> closed out"
        if re.match(r"[a-z_]+ (is|com|clo|c$|i$)", text) or re.match(r"[a-z]+_[a-z_]*\s*$", text):
            if re.match(r"[a-z_]+\s*$", text):
                return DELIVER if m.get("from") != "a1" else OTHER
            return DELIVER
        if re.match(r"a\d", text):
            return ASSIGN
        return OTHER
    if typ == "state_update":
        if _starts(text, _DIS) or "is incorrect" in text:
            return DISAGREE
        if _starts(text, _SET) or "now equals" in text:
            return SET
        if re.match(r"[a-z_]+=\d+ is", text):
            return DISAGREE
        return OTHER
    if typ == "inform":
        if _starts(text, _ITER):
            return ITER
        if _starts(text, _DRIFT_HARD):
            return DRIFT_HARD
        if _starts(text, _DRIFT_SOFT):
            return DRIFT_SOFT
        if _starts(text, _CHAT):
            return CHAT
        if text.startswith("Confirmed") or (len(text) >= 4 and "Confirmed".startswith(text)):
            return ACK          # "Confirmed. X is in progress." is an acknowledgement
        if _starts(text, _WORK) or " in progress" in text:
            return WORK
        if _starts(text, _ACK) or " accepted" in text or " picked up" in text:
            return ACK
        if re.match(r"[a-z_]+ acc", text):
            return ACK
        if re.match(r"[a-z_]+ in", text):
            return WORK
        return OTHER
    return OTHER


def text_subtask(text: str) -> str:
    for s in _SUBRE.findall(text):
        if s in TOOLS or s.startswith("art_"):
            continue
        if s in SUB2DOM:
            return s
    return ""


def text_agent(text: str) -> str:
    x = _AGRE.findall(text)
    return x[0] if x else ""


def drift_topic(text: str) -> str:
    t = text.lower()
    for d in DRIFT_TOPICS:
        if d in t:
            return d
    return ""


def parse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    run = {"run_id": row.get("run_id"), "goal": row.get("goal", "") or ""}
    for f in ("agents", "shared_state", "messages", "artifacts", "topology"):
        v = row.get(f)
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                v = None
        if v is None:
            v = {} if f == "topology" else []
        run[f] = v
    if not isinstance(run["topology"], dict):
        run["topology"] = {}
    msgs = []
    for i, m in enumerate(run["messages"]):
        if not isinstance(m, dict):
            continue
        m = dict(m)
        m.setdefault("t", i)
        text = m.get("text", "") or ""
        m["text"] = text
        m["cat"] = categorize(m)
        intent = m.get("intent", "") or ""
        sub = intent if intent in SUB2DOM else ""
        if not sub:
            sub = text_subtask(text)
        m["sub"] = sub
        m["tagent"] = text_agent(text)
        m["refs"] = m.get("refs") or []
        msgs.append(m)
    run["messages"] = msgs
    run["domain"] = goal_domain(run["goal"])
    return run
