"""Evals: automatic tests for the agent's *behavior* against the real model (needs GROQ_API_KEY).

Each case is a scripted conversation. After it runs we check what the agent did (which tools it
called) and what it said (e.g. that numbers match the tool output, and assumptions are stated).
Every final answer also goes through the grounding check: numbers that no tool produced are reported.
Models are not deterministic, so run more than once.
Run:  python evals/run_evals.py               (all cases)
      python evals/run_evals.py anc           (only cases whose name contains "anc")
      python evals/run_evals.py --repeat 2    (everything twice)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import Agent  # noqa: E402
from grounding import normalize_text  # noqa: E402

CASES = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))


def number_forms(value):
    """Ways a tool value may appear in a reply: 25.0 -> '25.0'/'25'; 6219 -> '6219'/'6,219'/'6,200'."""
    forms = {str(value)}
    if isinstance(value, float) and value.is_integer():
        forms.add(str(int(value)))
    if isinstance(value, int):
        forms |= {f"{value:,}", f"{round(value, -2):,}"}
    return forms


def run_case(case):
    agent = Agent()
    replies, last_turn_start = [], 0
    for turn in case["turns"]:
        last_turn_start = len(agent.trace)
        replies.append(agent.reply(turn))
    called = [t["tool"] for t in agent.trace]
    last_called = called[last_turn_start:]
    # Models often write curly apostrophes and narrow spaces as thousands separators.
    last = normalize_text(replies[-1].replace("\u2019", "'"))
    failures = []

    for tool in case.get("must_call", []):
        if tool not in called:
            failures.append(f"expected a call to {tool}")
    if case.get("must_not_call_any") and called:
        failures.append(f"should not call any tool, called {called}")
    wanted = case.get("must_call_in_last_turn_any")
    if wanted and not set(wanted) & set(last_called):
        failures.append(f"last turn should call one of {wanted}, called {last_called}")
    if "reply_must_contain_any" in case and not any(s.lower() in last.lower() for s in case["reply_must_contain_any"]):
        failures.append(f"last reply should mention one of {case['reply_must_contain_any']}")
    check = case.get("reply_must_contain_tool_value")
    if check:
        outputs = [json.loads(t["output"]) for t in agent.trace if t["tool"] == check["tool"] and not t["error"]]
        if not outputs:
            failures.append(f"no successful {check['tool']} output to compare with")
        elif not any(form in last for form in number_forms(outputs[-1][check["field"]])):
            failures.append(f"reply does not contain {check['field']}={outputs[-1][check['field']]} from the tool")
    codes_check = case.get("tool_output_codes")
    if codes_check:
        seen = [{a["code"] for a in json.loads(t["output"]).get("airports", [])}
                for t in agent.trace if t["tool"] == codes_check["tool"] and not t["error"]]
        if set(codes_check["codes"]) not in seen:
            failures.append(f"{codes_check['tool']} should compare {codes_check['codes']}, compared {seen}")
    if case.get("expect_assumptions") and not any(w in last.lower() for w in ("assumption", "limitation")):
        failures.append("reply has no assumptions/limitations section")
    return failures, called, replies, agent.last_unverified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filter", nargs="?", default="")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    selected = [c for c in CASES if args.filter in c["name"]]
    passes = {c["name"]: 0 for c in selected}
    for run in range(1, args.repeat + 1):
        print(f"--- run {run}/{args.repeat}")
        passed = grounded = 0
        for case in selected:
            try:
                failures, called, replies, unverified = run_case(case)
            except Exception as e:  # provider error (rate limit, request too large...) counts as a failure
                failures, called, replies, unverified = [f"{type(e).__name__}: {str(e)[:200]}"], [], [""], []
            grounded += not unverified
            ok = not failures
            passed += ok
            passes[case["name"]] += ok
            print(f"{'PASS' if ok else 'FAIL'}  {case['name']}   tools={called}"
                  + (f"   unverified numbers={unverified}" if unverified else ""))
            for f in failures:
                print(f"      - {f}")
            if not ok:
                print(f"      last reply: {replies[-1][:400]}")
        print(f"{passed}/{len(selected)} passed; {grounded}/{len(selected)} final answers fully grounded "
              "(every number found in a tool output)")
    if args.repeat > 1:
        print("\nPer case: " + ", ".join(f"{n} {p}/{args.repeat}" for n, p in passes.items()))


if __name__ == "__main__":
    main()
