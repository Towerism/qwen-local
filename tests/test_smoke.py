"""qlm smoke against a fake llama-server that answers like the pinned build."""

import copy
import json
import stat

import pytest

from qlm import cli
from qlm.system import ServerUnreachable
from tests.fakes import FakeClock, FakeRunner, FakeServer, make_ctx, reply
from tests.conftest import REPO
from tests.test_service import install

KEY = "k" * 43
FIXTURES = REPO / "tests/fixtures/server"


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def classify(body):
    """Which smoke check a /v1/chat/completions body belongs to."""
    kwargs = body.get("chat_template_kwargs", {})
    if any(m["role"] == "tool" for m in body["messages"]):
        return "tool_loop"
    if body.get("tools"):
        return "tool_call"
    if kwargs.get("enable_thinking") is True:
        return "thinking"
    if kwargs.get("enable_thinking") is False:
        return "mtp"
    return "auth"


def make_server(**overrides):
    """A FakeServer that passes all eight checks; each keyword replaces one check's reply.

    Override keys: health, models, tool_call, tool_loop, messages, thinking, mtp, auth. A value is a
    LocalResponse, an exception, or a dict sent as a 200 JSON body.
    """
    answers = {
        "health": reply(200, {"status": "ok"}),
        "models": reply(200, fixture("models")),
        "tool_call": reply(200, fixture("tool_call")),
        "tool_loop": reply(200, fixture("tool_loop")),
        "messages": reply(200, fixture("messages_tool_use")),
        "thinking": reply(200, fixture("thinking")),
        "mtp": reply(200, fixture("mtp")),
        "auth": reply(401, {"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}),
    }
    for name, value in overrides.items():
        answers[name] = reply(200, value) if isinstance(value, dict) else value

    def keyed(name):
        def answer(record):
            if name != "health" and name != "auth" and not record["headers"]:
                return reply(401, {"error": {"type": "authentication_error"}})
            return answers[name]

        return answer

    def chat(record):
        name = classify(record["json"])
        if name == "auth":
            return answers["auth"] if not record["headers"] else reply(200, fixture("thinking"))
        return keyed(name)(record)

    return FakeServer({
        ("GET", "/health"): keyed("health"),
        ("GET", "/v1/models"): keyed("models"),
        ("POST", "/v1/chat/completions"): chat,
        ("POST", "/v1/messages"): keyed("messages"),
    })


def run_smoke(home, repo, server, profile="agent"):
    install(home, repo, profile=profile, key=KEY)
    ctx = make_ctx(home, repo, runner=FakeRunner(), http=server, clock=FakeClock(advance=True))
    code = cli.main(["smoke"], ctx)
    return code, ctx


def lines(ctx):
    return [l for l in ctx.out.getvalue().splitlines() if l.split(" ", 1)[0] in ("PASS", "FAIL", "SKIP")]


def status_of(ctx, number):
    return next(l for l in lines(ctx) if l.split()[1] == str(number)).split()[0]


def test_a_healthy_server_passes_all_eight_checks(home, repo):
    code, ctx = run_smoke(home, repo, make_server())
    assert code == 0
    assert [l.split()[:2] for l in lines(ctx)] == [["PASS", str(n)] for n in range(1, 9)]
    assert "Con's cached prompt" in ctx.out.getvalue().splitlines()[0]


def test_requests_carry_the_key_the_way_each_api_expects(home, repo):
    server = make_server()
    run_smoke(home, repo, server)
    assert server.to("GET", "/health")[0]["headers"] == {}
    assert server.to("GET", "/v1/models")[0]["headers"] == {"Authorization": f"Bearer {KEY}"}
    assert server.to("POST", "/v1/messages")[0]["headers"] == {"x-api-key": KEY}
    chats = server.to("POST", "/v1/chat/completions")
    assert [bool(c["headers"]) for c in chats] == [True, True, True, True, False]


def test_the_tool_and_prompts_are_the_specified_ones(home, repo):
    server = make_server()
    run_smoke(home, repo, server)
    first = server.to("POST", "/v1/chat/completions")[0]["json"]
    tool = first["tools"][0]["function"]
    assert tool["name"] == "lookup_order"
    assert tool["parameters"] == {
        "type": "object",
        "properties": {"order_id": {"type": "string", "pattern": "^[A-Z]-[0-9]{4}$"}},
        "required": ["order_id"],
        "additionalProperties": False,
    }
    assert first["tool_choice"] == "auto" and first["max_tokens"] == 4096
    assert "temperature" not in first
    assert first["messages"] == [{"role": "user", "content": "What is the shipping status of order A-7731? Use the lookup_order tool."}]
    anthropic = server.to("POST", "/v1/messages")[0]["json"]
    assert anthropic["tools"][0]["input_schema"] == tool["parameters"]


def test_the_tool_loop_sends_back_the_call_and_its_result(home, repo):
    server = make_server()
    run_smoke(home, repo, server)
    loop = server.to("POST", "/v1/chat/completions")[1]["json"]["messages"]
    assert loop[1]["role"] == "assistant" and loop[1]["tool_calls"][0]["id"] == "call_8f2a"
    assert loop[2]["role"] == "tool" and loop[2]["tool_call_id"] == "call_8f2a"
    assert json.loads(loop[2]["content"]) == {"order_id": "A-7731", "status": "shipped", "tracking_code": "ZQ-48213"}


def test_an_unreachable_server_fails_health_and_skips_the_rest(home, repo):
    code, ctx = run_smoke(home, repo, make_server(health=ServerUnreachable("Connection refused")))
    assert code == 1
    assert status_of(ctx, 1) == "FAIL"
    assert [status_of(ctx, n) for n in range(2, 9)] == ["SKIP"] * 7
    assert "qlm start" in ctx.err.getvalue()


def bad_tool_call(arguments=None, calls=True):
    body = copy.deepcopy(fixture("tool_call"))
    msg = body["choices"][0]["message"]
    if not calls:
        del msg["tool_calls"]
        msg["content"] = '<tool_call>{"name": "lookup_order"}</tool_call>'
        body["choices"][0]["finish_reason"] = "stop"
    else:
        msg["tool_calls"][0]["function"]["arguments"] = arguments
    return body


@pytest.mark.parametrize(
    "body, reason",
    [
        (bad_tool_call('{"order_id": "A-7731", "note": "x"}'), "order_id"),
        (bad_tool_call("not json"), "JSON"),
        (bad_tool_call(calls=False), "no tool call"),
    ],
)
def test_a_wrong_tool_call_fails_and_skips_the_loop(home, repo, body, reason):
    code, ctx = run_smoke(home, repo, make_server(tool_call=body))
    assert code == 1
    assert status_of(ctx, 3) == "FAIL" and reason in next(l for l in lines(ctx) if l.split()[1] == "3")
    assert status_of(ctx, 4) == "SKIP"


def test_a_call_without_an_id_gets_call_0(home, repo):
    body = copy.deepcopy(fixture("tool_call"))
    del body["choices"][0]["message"]["tool_calls"][0]["id"]
    server = make_server(tool_call=body)
    code, ctx = run_smoke(home, repo, server)
    loop = server.to("POST", "/v1/chat/completions")[1]["json"]["messages"]
    assert loop[1]["tool_calls"][0]["id"] == "call_0" and loop[2]["tool_call_id"] == "call_0"
    assert code == 0


def with_message(name, **changes):
    body = copy.deepcopy(fixture(name))
    body["choices"][0]["message"].update(changes)
    return body


def test_a_final_answer_without_the_tracking_code_fails(home, repo):
    code, ctx = run_smoke(home, repo, make_server(tool_loop=with_message("tool_loop", content="It shipped.")))
    assert code == 1 and status_of(ctx, 4) == "FAIL"


def test_an_anthropic_reply_with_only_text_fails(home, repo):
    body = copy.deepcopy(fixture("messages_tool_use"))
    body["content"] = [{"type": "text", "text": "I can't look that up."}]
    body["stop_reason"] = "end_turn"
    code, ctx = run_smoke(home, repo, make_server(messages=body))
    assert code == 1 and status_of(ctx, 5) == "FAIL"


@pytest.mark.parametrize(
    "changes", [{"reasoning_content": ""}, {"content": "<think>hmm</think>391"}, {"content": "392"}]
)
def test_thinking_must_be_separate_and_correct(home, repo, changes):
    code, ctx = run_smoke(home, repo, make_server(thinking=with_message("thinking", **changes)))
    assert code == 1 and status_of(ctx, 6) == "FAIL"


def test_mtp_is_skipped_without_a_draft_model(home, repo):
    server = make_server()
    code, ctx = run_smoke(home, repo, server, profile="long")
    assert code == 0
    assert status_of(ctx, 7) == "SKIP" and "MTP is off in this profile" in ctx.out.getvalue()
    assert not any(classify(r["json"]) == "mtp" for r in server.to("POST", "/v1/chat/completions"))


@pytest.mark.parametrize("timings", [{}, {"draft_n": 100, "draft_n_accepted": 0}])
def test_mtp_needs_accepted_draft_tokens(home, repo, timings):
    body = copy.deepcopy(fixture("mtp"))
    body["timings"] = {"predicted_n": 230, **timings}
    code, ctx = run_smoke(home, repo, make_server(mtp=body))
    assert code == 1 and status_of(ctx, 7) == "FAIL"


def test_mtp_detail_shows_accepted_out_of_drafted(home, repo):
    code, ctx = run_smoke(home, repo, make_server())
    assert "152 of 180" in next(l for l in lines(ctx) if l.split()[1] == "7")


def test_a_server_that_accepts_requests_without_the_key_fails(home, repo):
    code, ctx = run_smoke(home, repo, make_server(auth=reply(200, fixture("thinking"))))
    assert code == 1 and status_of(ctx, 8) == "FAIL"
    assert "without the API key" in ctx.out.getvalue()


def test_errors_and_timeouts_fail_only_their_own_check(home, repo):
    server = make_server(models=reply(500, "boom"), tool_call=ServerUnreachable("timed out after 180 s"))
    code, ctx = run_smoke(home, repo, server)
    assert code == 1
    assert "HTTP 500" in next(l for l in lines(ctx) if l.split()[1] == "2")
    assert "timed out after 180 s" in next(l for l in lines(ctx) if l.split()[1] == "3")
    assert status_of(ctx, 4) == "SKIP"
    assert [status_of(ctx, n) for n in (5, 6, 7, 8)] == ["PASS"] * 4
    assert "2 models" in ctx.err.getvalue() and "3 tool call" in ctx.err.getvalue()


def test_a_reply_cut_off_by_the_token_limit_says_so(home, repo):
    body = copy.deepcopy(fixture("thinking"))
    body["choices"][0]["finish_reason"] = "length"
    code, ctx = run_smoke(home, repo, make_server(thinking=body))
    assert "token limit" in next(l for l in lines(ctx) if l.split()[1] == "6")


def test_a_wrong_key_is_explained(home, repo):
    unauthorized = reply(401, {"error": {"type": "authentication_error"}})
    server = make_server(models=unauthorized, tool_call=unauthorized, messages=unauthorized, thinking=unauthorized, mtp=unauthorized)
    code, ctx = run_smoke(home, repo, server)
    assert code == 1
    assert "api-key" in ctx.err.getvalue() and "qlm stop" in ctx.err.getvalue()


def test_timeouts_are_180_s_for_generation_and_10_s_otherwise(home, repo):
    server = make_server()
    run_smoke(home, repo, server)
    assert server.to("GET", "/v1/models")[0]["timeout_s"] == 10
    assert {r["timeout_s"] for r in server.to("POST", "/v1/messages")} == {180}
    chats = server.to("POST", "/v1/chat/completions")
    assert [r["timeout_s"] for r in chats] == [180, 180, 180, 180, 10]


def test_the_log_records_each_request_and_never_the_key(home, repo):
    code, ctx = run_smoke(home, repo, make_server())
    logs = list((home / ".local/state/qlm/smoke").iterdir())
    assert len(logs) == 1 and logs[0].name == "smoke-20261017T120000Z.log"
    assert stat.S_IMODE(logs[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(logs[0].parent.stat().st_mode) == 0o700
    text = logs[0].read_text()
    assert text.count("POST /v1/chat/completions") == 5 and "GET /v1/models" in text and "ZQ-48213" in text
    assert KEY not in text and KEY not in ctx.out.getvalue() and KEY not in ctx.err.getvalue()
    assert str(logs[0]) in ctx.out.getvalue().splitlines()[-1]


def test_two_runs_in_one_second_get_separate_logs(home, repo):
    run_smoke(home, repo, make_server())
    ctx = make_ctx(home, repo, runner=FakeRunner(), http=make_server(), clock=FakeClock(advance=True))
    cli.main(["smoke"], ctx)
    assert sorted(p.name for p in (home / ".local/state/qlm/smoke").iterdir()) == [
        "smoke-20261017T120000Z-1.log", "smoke-20261017T120000Z.log"
    ]


def test_a_missing_key_file_fails_before_any_request(home, repo):
    install(home, repo)
    (home / ".config/qlm/api-key").unlink()
    server = make_server()
    ctx = make_ctx(home, repo, runner=FakeRunner(), http=server, clock=FakeClock(advance=True))
    assert cli.main(["smoke"], ctx) == 1
    assert server.requests == [] and "qlm install" in ctx.err.getvalue()
