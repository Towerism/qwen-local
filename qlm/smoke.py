"""`qlm smoke`: eight API checks against the running server (docs/architecture.md, "What qlm smoke checks").

Each request and reply goes to a log under ~/.local/state/qlm/smoke/. Headers are never logged,
so the API key stays out of it.
"""

import json
from dataclasses import dataclass

from qlm import files, service
from qlm.system import SERVER, ServerUnreachable

MODEL = "qwen3.8-27b"
GENERATION_TIMEOUT_S = 180
OTHER_TIMEOUT_S = 10
MAX_TOKENS = 4096
LOG_BODY_LIMIT = 64 * 1024

PROMPT = "What is the shipping status of order A-7731? Use the lookup_order tool."
ORDER_ID = "A-7731"
TRACKING_CODE = "ZQ-48213"
TOOL_RESULT = {"order_id": ORDER_ID, "status": "shipped", "tracking_code": TRACKING_CODE}
TOOL_NAME = "lookup_order"
TOOL_DESCRIPTION = "Look up the shipping status of an order by its ID."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {"order_id": {"type": "string", "pattern": "^[A-Z]-[0-9]{4}$"}},
    "required": ["order_id"],
    "additionalProperties": False,
}
THINKING_PROMPT = "What is 17 multiplied by 23? Reply with only the number."
THINKING_ANSWER = "391"
MTP_PROMPT = "Write the numbers from 1 to 80, separated by commas, and nothing else."
MTP_MAX_TOKENS = 400

NAMES = {
    1: "health", 2: "models", 3: "tool call", 4: "tool loop",
    5: "anthropic tool call", 6: "thinking", 7: "mtp", 8: "auth required",
}


@dataclass
class Check:
    number: int
    status: str
    detail: str

    def line(self):
        return f"{self.status} {self.number} {NAMES[self.number]}: {self.detail}"


class CheckFailed(Exception):
    pass


class _Log:
    """The smoke log, created owner-only and never overwritten."""

    def __init__(self, ctx):
        d = ctx.paths.state_dir / "smoke"
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = ctx.clock.now().strftime("%Y%m%dT%H%M%SZ")
        self.path = files.create_new(d, lambda n: f"smoke-{files.stamped(stamp, n)}.log")

    def write(self, method, path, status, seconds, body, response):
        with open(self.path, "a") as f:
            f.write(f"=== {method} {path} -> {status} in {seconds:.1f} s\n")
            if body is not None:
                f.write("request: " + json.dumps(body) + "\n")
            f.write("response: " + response[:LOG_BODY_LIMIT].decode(errors="replace") + "\n")


class _Client:
    def __init__(self, ctx, key, log):
        self.ctx, self.key, self.log = ctx, key, log
        self.unauthorized = 0

    def call(self, method, path, body=None, auth="bearer", timeout_s=OTHER_TIMEOUT_S, expect=200):
        headers = {}
        if auth == "bearer":
            headers["Authorization"] = f"Bearer {self.key}"
        elif auth == "x-api-key":
            headers["x-api-key"] = self.key
        t0 = self.ctx.clock.now()
        try:
            r = self.ctx.http.request(method, f"{SERVER}{path}", headers=headers, json_body=body, timeout_s=timeout_s)
        except ServerUnreachable as e:
            self.log.write(method, path, "no answer", 0, body, str(e).encode())
            raise CheckFailed(str(e)) from None
        self.log.write(method, path, r.status, (self.ctx.clock.now() - t0).total_seconds(), body, r.body)
        if r.status != expect:
            if r.status == 401 and auth:
                self.unauthorized += 1
            raise CheckFailed(f"HTTP {r.status}")
        if expect != 200:
            return None
        try:
            return json.loads(r.body)
        except ValueError:
            raise CheckFailed("the reply is not JSON") from None


def _message(reply):
    try:
        choice = reply["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise CheckFailed("the reply has no choices[0].message") from None
    if choice.get("finish_reason") == "length":
        raise CheckFailed("the reply hit the token limit")
    return message


def _chat(prompt_messages, **extra):
    return {"model": MODEL, "messages": prompt_messages, "max_tokens": MAX_TOKENS, **extra}


def _openai_tool():
    return {"type": "function", "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS}}


def _user(text):
    return [{"role": "user", "content": text}]


def check_models(client):
    reply = client.call("GET", "/v1/models")
    ids = [m.get("id") for m in reply.get("data", []) if isinstance(m, dict)]
    if MODEL not in ids:
        raise CheckFailed(f"{MODEL} not listed (got {', '.join(map(str, ids)) or 'none'})")
    return f"{MODEL} listed"


def check_tool_call(client):
    body = _chat(_user(PROMPT), tools=[_openai_tool()], tool_choice="auto")
    message = _message(client.call("POST", "/v1/chat/completions", body, timeout_s=GENERATION_TIMEOUT_S))
    calls = [c for c in message.get("tool_calls") or [] if c.get("function", {}).get("name") == TOOL_NAME]
    if not calls:
        raise CheckFailed("no tool call in the reply")
    call = calls[0]
    try:
        arguments = json.loads(call["function"].get("arguments", ""))
    except (TypeError, ValueError):
        raise CheckFailed("the arguments are not JSON") from None
    if arguments != {"order_id": ORDER_ID}:
        raise CheckFailed(f"expected exactly order_id {ORDER_ID}, got {json.dumps(arguments)}")
    return call, message


def check_tool_loop(client, call, message):
    call_id = call.get("id") or "call_0"
    assistant = {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": [{**call, "id": call_id}],
    }
    tool = {"role": "tool", "tool_call_id": call_id, "content": json.dumps(TOOL_RESULT)}
    body = _chat(_user(PROMPT) + [assistant, tool], tools=[_openai_tool()], tool_choice="auto")
    final = _message(client.call("POST", "/v1/chat/completions", body, timeout_s=GENERATION_TIMEOUT_S))
    if final.get("tool_calls"):
        raise CheckFailed("the model called a tool again instead of answering")
    if TRACKING_CODE not in (final.get("content") or ""):
        raise CheckFailed(f"the answer doesn't contain {TRACKING_CODE} from the tool result")
    return f"the answer uses {TRACKING_CODE} from the tool result"


def check_anthropic(client):
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "tools": [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": TOOL_PARAMETERS}],
        "messages": _user(PROMPT),
    }
    reply = client.call("POST", "/v1/messages", body, auth="x-api-key", timeout_s=GENERATION_TIMEOUT_S)
    if reply.get("stop_reason") == "max_tokens":
        raise CheckFailed("the reply hit the token limit")
    for block in reply.get("content") or []:
        if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
            if block.get("input") != {"order_id": ORDER_ID}:
                raise CheckFailed(f"expected input order_id {ORDER_ID}, got {json.dumps(block.get('input'))}")
            return f"tool_use block for {TOOL_NAME}"
    raise CheckFailed("no tool_use block in the reply")


def check_thinking(client):
    body = _chat(_user(THINKING_PROMPT), chat_template_kwargs={"enable_thinking": True})
    message = _message(client.call("POST", "/v1/chat/completions", body, timeout_s=GENERATION_TIMEOUT_S))
    reasoning, content = message.get("reasoning_content"), message.get("content") or ""
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise CheckFailed("no reasoning_content in the reply")
    if "<think>" in content:
        raise CheckFailed("thinking leaked into content")
    if THINKING_ANSWER not in content:
        raise CheckFailed(f"the answer doesn't contain {THINKING_ANSWER}")
    return "reasoning_content is separate from the answer"


def check_mtp(client):
    body = _chat(
        _user(MTP_PROMPT), chat_template_kwargs={"enable_thinking": False}, temperature=0, max_tokens=MTP_MAX_TOKENS
    )
    reply = client.call("POST", "/v1/chat/completions", body, timeout_s=GENERATION_TIMEOUT_S)
    _message(reply)
    timings = reply.get("timings") or {}
    accepted, drafted = timings.get("draft_n_accepted"), timings.get("draft_n")
    if not accepted:
        raise CheckFailed("no accepted draft tokens in timings, so MTP isn't working")
    return f"{accepted} of {drafted} draft tokens accepted"


def check_auth(client):
    body = {"messages": _user("hi"), "max_tokens": 1}
    try:
        client.call("POST", "/v1/chat/completions", body, auth=None, expect=401)
    except CheckFailed as e:
        if str(e).startswith("HTTP 2"):
            raise CheckFailed("the server accepted a request without the API key") from None
        raise
    return "a request without the key gets 401"


def run(ctx, failed_names=None):
    """Run the eight checks. If `failed_names` is a list, the names of failed checks are added to it."""
    try:
        return _run(ctx, failed_names)
    except service.Stop as stop:
        return ctx.fail(stop.what, stop.todo)
    except BrokenPipeError:
        raise
    except OSError as e:
        return ctx.fail(f"file error: {e}", "Fix the cause and try again")


def _run(ctx, failed_names):
    p = ctx.paths
    if not p.api_key.exists() or not p.api_key.read_text().strip():
        raise service.Stop(f"the API key file {p.api_key} is missing or empty", "Run qlm install")
    key = p.api_key.read_text().strip()
    mtp_on = "LLAMA_ARG_SPEC_TYPE" in service.active_profile(ctx).settings
    log = _Log(ctx)
    client = _Client(ctx, key, log)
    ctx.say("Note: these requests replace Con's cached prompt in the server's single slot, "
            "so Con's next turn re-reads its context.")

    results = {}

    def report(check):
        results[check.number] = check
        ctx.say(check.line())

    def attempt(number, fn, *args, detail=None):
        try:
            value = fn(*args)
        except CheckFailed as e:
            report(Check(number, "FAIL", str(e)))
            return None
        except (AttributeError, KeyError, TypeError, IndexError):
            # A JSON reply that isn't shaped like llama-server's (a list where an object belongs...)
            report(Check(number, "FAIL", "unexpected reply shape; see the log"))
            return None
        report(Check(number, "PASS", detail or (value if isinstance(value, str) else "ok")))
        return value

    def skip(number, why):
        report(Check(number, "SKIP", why))

    def health():
        status = service.health_status(ctx)
        if status != 200:
            raise CheckFailed("no answer" if status is None else f"HTTP {status}")
        log.write("GET", "/health", 200, 0, None, b'{"status":"ok"}')
        return "server is ready"

    if attempt(1, health) is None:
        for n in range(2, 9):
            skip(n, "the server isn't ready")
    else:
        attempt(2, check_models, client)
        called = attempt(3, check_tool_call, client, detail=f"{TOOL_NAME} called with order_id {ORDER_ID}")
        if called is None:
            skip(4, "check 3 failed")
        else:
            attempt(4, check_tool_loop, client, *called)
        attempt(5, check_anthropic, client)
        attempt(6, check_thinking, client)
        if mtp_on:
            attempt(7, check_mtp, client)
        else:
            skip(7, "MTP is off in this profile")
        attempt(8, check_auth, client)

    ctx.say(f"Log: {log.path}")
    failed = [results[n] for n in range(1, 9) if results[n].status == "FAIL"]
    if not failed:
        return 0
    names = ", ".join(f"{c.number} {NAMES[c.number]}" for c in failed)
    if failed_names is not None:
        failed_names.append(names)
    if results[1].status == "FAIL":
        todo = "Start the server with qlm start, then run qlm smoke again"
    elif client.unauthorized:
        todo = (f"The server refused the key in {p.api_key}; it may have loaded a different key. "
                "Restart it with qlm stop and qlm start")
    else:
        todo = f"See the log at {log.path}"
    return ctx.fail(f"smoke checks failed: {names}", todo)
