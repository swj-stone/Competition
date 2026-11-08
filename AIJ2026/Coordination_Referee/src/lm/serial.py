"""Run -> text serialization for the LM member (shared by offline training and in-package inference).

One line per message turn "[t] from>to type intent | text (tool) (refs)", followed by that turn's shared-state ops and
artifacts; artifact hashes become per-run ids h1, h2, ... (equality kept, hex dropped).  Returns the text and the char
offset of every message line, keyed by message t, for the pointer head."""
import json

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift", "runaway_loop"]


def _j(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x


def serialize(row):
    agents = _j(row.get("agents")) or []
    msgs = _j(row.get("messages")) or []
    state = _j(row.get("shared_state")) or []
    arts = _j(row.get("artifacts")) or []
    topo = _j(row.get("topology")) or {}
    head = ["goal: %s" % row.get("goal", "")]
    if isinstance(topo, dict):
        edges = " ".join("%s>%s" % tuple(e[:2]) for e in topo.get("edges", []) if isinstance(e, (list, tuple)) and len(e) >= 2)
        head.append("topology: %s %s" % (topo.get("type", ""), edges))
    head.append("agents: " + "; ".join("%s %s [%s]" % (a.get("id", ""), a.get("role", ""), ",".join(a.get("tools", []) or []))
                                       for a in agents if isinstance(a, dict)))
    extra = {}
    for s in state:
        if isinstance(s, dict):
            extra.setdefault(s.get("t"), []).append("  state %s %s %s=%s" % (s.get("agent", ""), s.get("op", ""),
                                                                            s.get("key", ""), str(s.get("value", ""))[:40]))
    hid = {}
    for a in arts:
        if isinstance(a, dict):
            h = hid.setdefault(a.get("hash", ""), "h%d" % (len(hid) + 1))
            extra.setdefault(a.get("t"), []).append("  artifact %s %s %s %s %s %s" % (a.get("id", ""), a.get("by", ""), a.get("type", ""),
                                                                                    a.get("subtask", ""), a.get("status", ""), h))
    lines, offs, pos = [], {}, 0
    text = "\n".join(head) + "\n"
    pos = len(text)
    seen = set()
    for m in msgs:
        if not isinstance(m, dict):
            continue
        t = m.get("t")
        line = "[%s] %s>%s %s %s | %s" % (t, m.get("from", ""), m.get("to", ""), m.get("type", ""), m.get("intent", ""), m.get("text", ""))
        if m.get("tool"):
            line += " (tool %s)" % m["tool"]
        if m.get("refs"):
            line += " (refs %s)" % ",".join(str(r) for r in m["refs"])
        if t not in offs:
            offs[t] = pos
        block = [line] + (extra.get(t, []) if t not in seen else [])
        seen.add(t)
        s = "\n".join(block) + "\n"
        lines.append(s)
        pos += len(s)
    tail = [x for t, v in extra.items() if t not in seen for x in v]
    body = "".join(lines) + ("\n".join(tail) + "\n" if tail else "")
    return text + body, offs
