"""Play twenty questions against the loaded model, and watch J space while it answers."""
import copy
from collections import Counter
import html
import json
import logging
import math
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import gradio as gr

from chatlab.extension_api import write_private_text
from chatlab.extensions.hangman.game import reopened as hangman_reopened, rewound as hangman_rewound
from chatlab.extensions.hangman.page import Games, shown
from . import jspace
from .game import (
    GIVE_UP, MAX_FILE_BYTES, OPENING, PROBE_PREFILL, QUESTIONS, RECHECK_PREFILL, SYSTEM, answer_of, asked, check,
    finish_turn, load, messages_for, new_game, probe_thing, read_verdict, reasoning_of, recheck_messages,
    recheck_summary, revealed, saved,
)

logger = logging.getLogger(__name__)

STALE_TOKEN = "Select a token in the current response again."
MALFORMED = "That file is not a valid saved twenty questions game."
PROBE_TOKENS, RECHECK_TOKENS = 24, 8
# Probability cut points for the five sequential fills, coolest first.
STEPS = (0.01, 0.05, 0.15, 0.4)
WAITING = '<div class="tq-note">J space appears here after each reply.</div>'

CSS = """
#twenty-questions-page {overflow-y:auto; min-height:0; padding:12px;}
#twenty-questions-page .tq-check {max-width:1000px;}
#twenty-questions-page .tq-jspace {display:flex; flex-wrap:wrap; gap:12px;}
#twenty-questions-page .tq-jspace figure {margin:0; flex:1 1 300px; min-width:0;}
#twenty-questions-page .tq-jspace figcaption {font-weight:600; margin-bottom:4px;}
#twenty-questions-page .tq-jspace table {border-collapse:collapse; font-size:12px; width:100%;}
#twenty-questions-page .tq-jspace th, #twenty-questions-page .tq-jspace td {padding:1px 4px; text-align:left;
  vertical-align:top; white-space:nowrap;}
#twenty-questions-page .tq-jspace td.tq-cell {color:#111; border:1px solid var(--background-fill-primary);}
#twenty-questions-page .tq-jspace .tq-scroll {max-height:420px; overflow:auto;}
#twenty-questions-page .tq-note {font-size:12px; opacity:.8; margin:4px 0;}
"""


def chat_messages(game):
    messages = []
    for number, turn in enumerate(game["turns"], 1):
        messages.append({"role": "user", "content": turn["question"]})
        reasoning = reasoning_of(turn["text"], turn.get("reasoning_prefilled", False))
        if reasoning:
            messages.append({"role": "assistant", "content": reasoning,
                             "metadata": {"title": f"Reasoning · response {number}"}})
        answer = turn.get("answer") or ("…" if turn.get("finish_reason") is None else "(no answer)")
        messages.append({"role": "assistant", "content": answer})
    return messages


def check_text(game):
    if not game["turns"]:
        return ("Start a game. The page reads the **Answer:** and **Questions left:** lines of each reply and "
                "checks them against the ones before.")
    questions = asked(game)
    tally = Counter(verdict for _, _, verdict in questions)
    parts = [f"**Questions answered:** {len(questions)} · yes {tally['yes']} · no {tally['no']}"]
    left = next((t["left"] for t in reversed(game["turns"]) if t.get("left") is not None), None)
    if left is not None:
        parts[0] += f" · **Questions left:** {left} of {QUESTIONS}"
    thing = revealed(game)
    if thing:
        parts.append(f"**Thing named:** {shown(thing)}")
    problems = check(game)
    parts.append("**Contradictions:**\n" + "\n".join(f"- Response {n}: {m}" for n, m in problems)
                 if problems else "**Contradictions:** none found.")
    unread = [str(n) for n, t in enumerate(game["turns"], 1)
              if n > 1 and t.get("finish_reason") and t.get("verdict") is None and not t.get("revealed")]
    if unread:
        parts.append(f"Responses without a readable answer: {', '.join(unread)}.")
    if game.get("recheck"):
        recheck = game["recheck"]
        agree, read, disagree = recheck_summary(recheck)
        lines = [f"**Recheck against {shown(recheck['thing'])}:** {agree} of {read} answers agree"
                 f" (asked fresh, greedy, by {shown(recheck.get('model_id') or 'an unrecorded model')})."]
        lines += [f"- Response {row['turn'] + 1}, {shown(row['question'])}: the game said "
                  f"{row['game_verdict'].upper()}, asked about {shown(recheck['thing'])} the model says "
                  f"{row['verdict'].upper()}." for row in disagree]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def turn_choices(game):
    return [(f"Response {n} · {t['question'][:40]}", n - 1) for n, t in enumerate(game["turns"], 1)]


def turn_note(turn):
    sampling = turn.get("sampling", {})
    parts = [f"Model {shown(turn.get('model_id') or 'unrecorded')}",
             f"temperature {shown(sampling.get('temperature'))}", f"seed {shown(sampling.get('seed'))}",
             f"{len(turn.get('metrics', [])) - turn.get('forced_prefix_tokens', 0)} sampled tokens",
             f"finish: {shown(turn.get('finish_reason') or 'running')}"]
    if turn.get("branch"):
        branch = turn["branch"]
        parts.append(f"branched at token {branch['token_index'] + 1} with {shown(branch['forced_tokens'])} "
                     "replayed tokens")
    if turn.get("probes"):
        counts = Counter(probe["thing"] or "(unreadable)" for probe in turn["probes"])
        parts.append(f"asked what it is {len(turn['probes'])} times: "
                     + ", ".join(f"{shown(thing)} ×{count}" for thing, count in counts.most_common()))
    return " · ".join(parts)


def fill(log_prob, palette):
    probability = math.exp(log_prob)
    return palette["sequential"][sum(probability >= step for step in STEPS)]


def jspace_html(turn, number, palette, display):
    """What the lens read at each position of response ``number``, block by block."""
    reading = turn.get("jspace")
    if reading is None:
        note = turn.get("jspace_error") or "Not read. Turn on **Read J space after every reply**, or read the game again."
        return f'<div class="tq-note">Response {number}: {html.escape(note)}</div>'
    figures = []
    for position in reading["positions"]:
        rows = []
        for block in reversed(position["blocks"]):
            cells = "".join(
                f'<td class="tq-cell" style="background:{fill(top["log_prob"], palette)}" '
                f'title="{html.escape(top["text"])} · log p {top["log_prob"]:.2f}">'
                f'{html.escape(display(top["text"]))}</td>' for top in block["top"])
            rows.append(f'<tr><th>{block["block"] + 1}</th>{cells}</tr>')
        figures.append(
            f'<figure><figcaption>{html.escape(position["label"])} · after '
            f'<code>{html.escape(display(position.get("token", "")))}</code></figcaption>'
            f'<div class="tq-scroll"><table><thead><tr><th>Block</th><th colspan="{jspace.TOP}">Top tokens through '
            f'the lens</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></figure>')
    lens = html.escape(reading.get("lens") or "the imported lens")
    return (f'<div class="tq-note">Response {number}, read through {lens}. Deepest block first; '
            'warmer cells hold more probability. A lens readout is not the model saying the word.</div>'
            f'<div class="tq-jspace">{"".join(figures)}</div>')


def watch_html(game, palette):
    """Each watched thing's mean lens log probability as each reply chooses its answer (or starts)."""
    rows, things = [], []
    for turn in game["turns"]:
        for thing in (turn.get("jspace") or {}).get("watch", {}):
            if thing not in things:
                things.append(thing)
    if not things:
        return ('<div class="tq-note">Add things to the watch list to follow their probability in J space '
                'across the game.</div>')
    for number, turn in enumerate(game["turns"], 1):
        watch = (turn.get("jspace") or {}).get("watch", {})
        cells = []
        for thing in things:
            values = watch.get(thing)
            value = values[-1] if values else None
            cells.append('<td class="tq-cell">–</td>' if value is None else
                         f'<td class="tq-cell" style="background:{fill(value, palette)}">{value:.1f}</td>')
        rows.append(f'<tr><th>{number}</th><td>{html.escape(turn["question"][:40])}</td>{"".join(cells)}</tr>')
    head = "".join(f"<th>{html.escape(thing)}</th>" for thing in things)
    return ('<div class="tq-note">Mean log probability over the lens\'s blocks, read where each reply chooses '
            'its answer, or where it starts when it gave none.</div>'
            f'<div class="tq-jspace"><figure><div class="tq-scroll"><table><thead><tr><th>Response</th>'
            f'<th>Question</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></figure></div>')


def watch_list(text):
    return list(dict.fromkeys(t.strip() for t in (text or "").replace("\n", ",").split(",") if t.strip()))


def rewound(game, turn_count):
    child = hangman_rewound(game, turn_count)
    # A recheck answered the parent's questions, not the child's.
    child.pop("recheck", None)
    return child


def build_page(context):
    games = Games()
    selections = context.tokens.selections()
    menu = context.tokens.menu("twenty-questions-tokens")
    palette, display = context.tokens.palette, context.tokens.display_text

    def forget(owner):
        games.cancel(owner)
        selections.forget(owner)

    with gr.Column(elem_id="twenty-questions-page"):
        owner = gr.State(value=selections.new_session, delete_callback=forget)
        game_state = gr.State(new_game(SYSTEM))
        token_state = gr.State(("", []))
        picked = gr.State(None)
        gr.Markdown("# Twenty questions\nThe model thinks of a thing and answers your questions. Nothing on this "
                    "page knows the thing: each reply is checked against the replies before it, and J space shows "
                    "what the imported Jacobian lens reads while the model answers.")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                models = gr.Button("Open Models", size="sm")
                context.navigation.open_models(models)
                with gr.Accordion("Prompt", open=False):
                    system = gr.Textbox(value=SYSTEM, label="System prompt", lines=10,
                                        info="Fixed when a game starts. Leave empty to send no system message.")
                    opening = gr.Textbox(value=OPENING, label="Opening message", lines=2)
                temperature = gr.Slider(0, 2, value=1.0, step=0.05, label="Temperature")
                seed = gr.Number(value=42, precision=0, minimum=0, label="Seed",
                                 info="Response n is sampled with this seed plus n - 1.")
                max_tokens = gr.Number(value=2048, precision=0, minimum=1, maximum=32768,
                                       label="Tokens per response")
                start = gr.Button("New game", variant="primary")
                with gr.Accordion("J space", open=True):
                    auto_read = gr.Checkbox(value=True, label="Read J space after every reply",
                                            info="Needs a Jacobian lens imported in Chat for the loaded model.")
                    watched = gr.Textbox(label="Watch list", placeholder="cat, piano, Eiffel Tower",
                                         info="Things to follow through the lens, separated by commas.")
                    reread = gr.Button("Read J space for the whole game again", size="sm")
                with gr.Accordion("Ask what it is", open=False):
                    gr.Markdown("Asks the model, as a side question that never joins the game, what the thing is, "
                                f"writing `{PROBE_PREFILL}` to start its answer. A model holding one thing names it "
                                "every time; one that has chosen nothing scatters.")
                    samples = gr.Slider(1, 20, value=5, step=1, label="Samples")
                    ask_thing = gr.Button("Ask after the latest response", size="sm")
                with gr.Accordion("Recheck the answers", open=False):
                    gr.Markdown("Asks every yes-or-no question again, in a fresh conversation about one thing, "
                                "and compares the answers with the game's.")
                    candidate = gr.Textbox(label="Thing", placeholder="Leave empty for the thing the game named")
                    recheck = gr.Button("Recheck", size="sm")
                with gr.Accordion("Saved games", open=False):
                    gr.Markdown("Every finished response saves the game. Uploading one opens a copy: new "
                                "questions continue it under whichever model is loaded, and the original file is left alone.")
                    upload = gr.File(label="Open a saved game", file_types=[".json"], type="filepath")
                    download = gr.File(label="This game", interactive=False)
            with gr.Column(scale=2, min_width=360):
                chat = gr.Chatbot(type="messages", label="Game", height=420, render_markdown=False)
                with gr.Row():
                    question = gr.Textbox(placeholder="A yes-or-no question, or your guess", show_label=False,
                                          lines=1, max_lines=1, scale=4, submit_btn=False)
                    send = gr.Button("Ask", variant="primary", scale=1)
                    stop = gr.Button("Stop", scale=1)
                jspace_panel = gr.HTML(WAITING, label="J space")
                with gr.Accordion("Watch list across the game", open=True):
                    watch_panel = gr.HTML(watch_html(new_game(SYSTEM), palette))
                check_panel = gr.Markdown(check_text(new_game(SYSTEM)), elem_classes=["tq-check"])
            with gr.Column(scale=2, min_width=360):
                picker = gr.Dropdown(choices=[], value=None, label="Response", interactive=True)
                note = gr.Markdown("")
                strip = gr.HighlightedText(label="Click a token to inspect it; right-click to branch the response there",
                                           combine_adjacent=False, show_legend=True, elem_id="twenty-questions-tokens",
                                           color_map=context.tokens.color_map, elem_classes=menu.strip_classes)
                menu_request, menu_response, menu_action = menu.bridges()
                detail = gr.Markdown("Select a token.")
                alternatives = gr.Dataframe(headers=["Token ID", "Text", "Raw probability"], interactive=False)
                token_jspace = gr.Button("Read J space after the selected token", size="sm")
                rewind = gr.Button("Rewind to this response", size="sm")
                gr.Markdown("Rewinding and branching start a new game from the selected response; "
                            "the game you leave stays saved.")
                with gr.Accordion("Full response", open=False):
                    raw = gr.Textbox(show_label=False, lines=8, max_lines=20, interactive=False)
                with gr.Accordion("Context sent to the model", open=False) as context_pane:
                    context_note = gr.Markdown("Open this to read the prompt behind the selected response.")
                    context_refresh = gr.Button("Show the selected response's context", size="sm")
                    context_body = gr.Textbox(show_label=False, lines=12, max_lines=30, interactive=False)

    def view(game, session_id, index):
        """Everything that shows one response of one game, but the token detail."""
        if index is None or not 0 <= index < len(game["turns"]):
            payload, _ = selections.view(session_id, (game["id"], None), [])
            return gr.update(choices=turn_choices(game), value=None), "", payload, [], "", WAITING
        turn = game["turns"][index]
        note, token_strip = turn_note(turn), context.tokens.strip(turn["metrics"])
        space = jspace_html(turn, index + 1, palette, display) if turn.get("finish_reason") else \
            '<div class="tq-note">J space is read when the reply finishes.</div>'
        payload, _ = selections.view(session_id, (game["id"], index), turn["metrics"])
        return (gr.update(choices=turn_choices(game), value=index), note, payload, token_strip, turn["text"], space)

    def cleared():
        return "Select a token.", [], None

    def frame(game, session_id, index, *, path=gr.skip(), check=True):
        return (game, chat_messages(game), check_text(game) if check else gr.skip(),
                watch_html(game, palette) if check else gr.skip(), *view(game, session_id, index), path)

    outputs = [game_state, chat, check_panel, watch_panel, picker, note, token_state, strip, raw, jspace_panel,
               download]
    inspector = [detail, alternatives, picked]

    staging = {"directory": None, "owner": None}

    def save(game):
        """Save the game, and stage the one copy the interface may serve, as hangman does."""
        text = saved(game)
        if len(text.encode("utf-8")) > MAX_FILE_BYTES:
            raise OSError("the game is larger than a saved game can be opened at")
        path = context.data_dir / f"{game['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_private_text(path, text)
        if staging["directory"] is None or not staging["directory"].is_dir():
            staging["owner"] = tempfile.TemporaryDirectory(prefix="chatlab-twenty-questions-")
            staging["directory"] = Path(staging["owner"].name)
        copy_path = staging["directory"] / path.name
        partial = staging["directory"] / f".{path.name}.{uuid4().hex}.tmp"
        try:
            write_private_text(partial, text)
            partial.replace(copy_path)
        finally:
            partial.unlink(missing_ok=True)
        for earlier in staging["directory"].glob("*.json"):
            if earlier != copy_path:
                earlier.unlink(missing_ok=True)
        return str(copy_path)

    def saved_path(game):
        try:
            return save(game)
        except OSError as exc:
            logger.warning("Could not save twenty questions game %s: %s", game["id"], exc)
            gr.Warning(f"This game was not saved: {exc}.")
            return gr.skip()

    def read_jspace(session, turn, prompt_ids, things):
        """Read the finished ``turn``'s J space onto it; a reading that cannot be made says why."""
        turn.pop("jspace", None)
        turn.pop("jspace_error", None)
        try:
            ids = list(prompt_ids) + [m["token_id"] for m in turn["metrics"]]
            turn["jspace"] = jspace.read(session, ids, jspace.positions_of(turn, len(prompt_ids)), things)
        except Exception as exc:  # noqa: BLE001 - the game goes on without the reading
            turn["jspace_error"] = str(exc) or type(exc).__name__
            logger.info("Twenty questions: J space not read: %s", turn["jspace_error"])

    def respond(game, session_id, text, temp, random_seed, token_limit, read_space, things_text, edit=None):
        """Generate one reply to ``text`` at the end of ``game``, then read its J space."""
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Type a question first.")
            integer_seed, limit = int(random_seed), int(token_limit)
            if integer_seed < 0 or not 1 <= limit <= 32768:
                raise ValueError("Seed must be nonnegative and tokens per response between 1 and 32768.")
            cancel = games.start(session_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise gr.Error(str(exc)) from exc
        game = copy.deepcopy(game)
        game.pop("recheck", None)
        index = len(game["turns"])
        turn = dict(question=text.strip(), text="", metrics=[], forced_prefix_tokens=0, reasoning_prefilled=False,
                    finish_reason=None, started_at=time.time(),
                    sampling=dict(temperature=float(temp), top_p=1.0, top_k=0, max_new_tokens=limit,
                                  seed=integer_seed + index))
        game["turns"].append(turn)
        failure, showing = None, False
        try:
            with context.models.open_session() as session:
                games.attach(session_id, session)
                turn.update(model_id=session.model_id, load_id=session.load_id)
                forced = []
                if edit is not None:
                    forced = branch_ids(session, edit)
                    turn["branch"] = dict(parent=edit["parent"], turn=edit["turn"], token_index=edit["token_index"],
                                          forced_tokens=len(forced), replacement=edit.get("label"))
                stop_ids = session.stop_token_ids
                showing = True
                yield (*frame(game, session_id, index, check=False), "")
                stream = session.generate(messages_for(game, index), forced_ids=forced, **turn["sampling"])
                prompt_ids = None
                try:
                    for update in stream:
                        prompt_ids = update.prompt_ids
                        turn.update(text=update.text, metrics=update.metrics,
                                    forced_prefix_tokens=update.forced_prefix_tokens,
                                    reasoning_prefilled=update.reasoning_prefilled)
                        finish_turn(turn)
                        yield (*frame(game, session_id, index, check=False), gr.skip())
                finally:
                    stream.close()
                last = turn["metrics"][-1]["token_id"] if turn["metrics"] else None
                turn["finish_reason"] = "stopped" if cancel.is_set() else "stop" if last in stop_ids else "length"
                if read_space and turn["metrics"] and prompt_ids and not cancel.is_set():
                    read_jspace(session, turn, prompt_ids, watch_list(things_text))
        except Exception as exc:
            failure = str(exc) or type(exc).__name__
            logger.warning("Twenty questions game %s: response %s failed: %s", game["id"], index + 1, failure)
        finally:
            games.finish(session_id)
            turn["seconds"] = time.time() - turn["started_at"]
        if not turn["metrics"] and (failure is not None or cancel.is_set()):
            # Nothing was generated, so there is no response to keep.
            game["turns"].pop()
            if showing:
                yield (*frame(game, session_id, index - 1 if index else None), gr.skip())
            if failure is not None:
                raise gr.Error(failure)
            return
        if failure is not None:
            turn.update(finish_reason="error", error=failure)
        finish_turn(turn)
        yield (*frame(game, session_id, index, path=saved_path(game)), gr.skip())
        if failure is not None:
            raise gr.Error(failure)

    def branch_ids(session, edit):
        """The kept tokens of the edited response, then the replacement."""
        if edit["load_id"] != session.load_id:
            raise ValueError("That response came from a different model load, or from a saved game, so its "
                             "tokens cannot be replayed. Rewind to the response before it and ask again instead.")
        kept = edit["kept_ids"]
        if edit.get("candidate_id") is not None:
            candidate = edit["candidate_id"]
            if candidate in session.hidden_token_ids and candidate not in session.stop_token_ids:
                raise ValueError("The loaded model never shows that token in a response. "
                                 "Type the replacement text instead.")
            replacement = [candidate]
        else:
            replacement = session.encode_replacement(kept, edit["text"])
        if not replacement:
            raise ValueError("Enter replacement text or choose an alternative.")
        return kept + replacement

    def start_game(system_text, opening_text, session_id, temp, random_seed, token_limit, read_space, things):
        game = new_game(system_text)
        yield from respond(game, session_id, opening_text or OPENING, temp, random_seed, token_limit,
                           read_space, things)

    def play(game, text, session_id, temp, random_seed, token_limit, read_space, things):
        if not game["turns"]:
            raise gr.Error("Start a new game first.")
        yield from respond(game, session_id, text, temp, random_seed, token_limit, read_space, things)

    settings = [temperature, seed, max_tokens, auto_read, watched]
    serial = dict(concurrency_id="twenty-questions-view", show_progress="hidden")
    for event in (start.click, send.click, question.submit, menu_action.input):
        event(cleared, None, inspector, queue=False)
    start.click(start_game, [system, opening, owner, *settings], [*outputs, question], **serial)
    send.click(play, [game_state, question, owner, *settings], [*outputs, question], **serial)
    question.submit(play, [game_state, question, owner, *settings], [*outputs, question], **serial)
    stop.click(games.cancel, owner, [], queue=False)

    def held(session_id):
        """The model, held for a side job in this view, and the event that stops it."""
        try:
            cancel = games.start(session_id)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return cancel

    def reread_game(game, session_id, things):
        """Read J space again for every response this model load wrote, with today's watch list."""
        if not game["turns"]:
            raise gr.Error("Start a game first.")
        cancel = held(session_id)
        game = copy.deepcopy(game)
        try:
            with context.models.open_session() as session:
                games.attach(session_id, session)
                mine = [i for i, t in enumerate(game["turns"]) if t.get("load_id") == session.load_id and t["metrics"]]
                if not mine:
                    raise ValueError("No response in this game was written by the model load in memory, so its "
                                     "tokens cannot be read again.")
                for index in mine:
                    if cancel.is_set():
                        break
                    read_jspace(session, game["turns"][index], session.prompt_ids(messages_for(game, index)),
                                watch_list(things))
        except Exception as exc:
            raise gr.Error(str(exc) or type(exc).__name__) from exc
        finally:
            games.finish(session_id)
        index = len(game["turns"]) - 1
        return frame(game, session_id, index, path=saved_path(game))

    reread.click(reread_game, [game_state, owner, watched], outputs, **serial)

    def ask_what(game, session_id, count, temp):
        """Ask what the thing is ``count`` times after the latest response, recording each answer on it."""
        if not game["turns"] or not game["turns"][-1].get("finish_reason"):
            raise gr.Error("Start a game first.")
        cancel = held(session_id)
        game = copy.deepcopy(game)
        turn = game["turns"][-1]
        asking = dict(game, turns=game["turns"] + [dict(question=GIVE_UP)])
        messages = messages_for(asking, len(game["turns"]))
        probes = []
        try:
            with context.models.open_session() as session:
                games.attach(session_id, session)
                for sample in range(int(count)):
                    if cancel.is_set():
                        break
                    probe_seed = 1000 * turn.get("sampling", {}).get("seed", 0) + sample
                    stream = session.generate(messages, temperature=float(temp), top_p=1.0, top_k=0,
                                              max_new_tokens=PROBE_TOKENS, seed=probe_seed,
                                              answer_prefill=PROBE_PREFILL)
                    update = None
                    try:
                        for update in stream:
                            pass
                    finally:
                        stream.close()
                    if update is None or cancel.is_set():
                        break
                    last = update.metrics[-1]["token_id"] if update.metrics else None
                    found = dict(seed=probe_seed, text=update.text, reasoning_prefilled=update.reasoning_prefilled,
                                 finish_reason="stop" if last in session.stop_token_ids else "length")
                    found["thing"] = probe_thing(found)
                    probes.append(found)
        except Exception as exc:
            raise gr.Error(str(exc) or type(exc).__name__) from exc
        finally:
            games.finish(session_id)
        turn["probes"] = turn.get("probes", []) + probes
        return frame(game, session_id, len(game["turns"]) - 1, path=saved_path(game))

    ask_thing.click(ask_what, [game_state, owner, samples, temperature], outputs, **serial)

    def recheck_game(game, session_id, thing):
        """Ask every answered question again about one thing, greedily, in a fresh conversation."""
        thing = (thing or "").strip() or revealed(game)
        if not thing:
            raise gr.Error("Name a thing to recheck against; the game has not named one yet.")
        questions = asked(game)
        if not questions:
            raise gr.Error("The game has no answered yes-or-no questions to recheck.")
        cancel = held(session_id)
        game = copy.deepcopy(game)
        rows = []
        try:
            with context.models.open_session() as session:
                games.attach(session_id, session)
                for index, text, verdict in questions:
                    if cancel.is_set():
                        break
                    stream = session.generate(recheck_messages(thing, text), temperature=0.0, top_p=1.0, top_k=0,
                                              max_new_tokens=RECHECK_TOKENS, seed=0, answer_prefill=RECHECK_PREFILL)
                    update = None
                    try:
                        for update in stream:
                            pass
                    finally:
                        stream.close()
                    if update is None:
                        break
                    answer = answer_of(update.text, update.reasoning_prefilled)
                    rows.append(dict(turn=index, question=text, game_verdict=verdict, text=update.text,
                                     verdict=read_verdict(answer)))
                game["recheck"] = dict(thing=thing, model_id=session.model_id, load_id=session.load_id, rows=rows)
        except Exception as exc:
            raise gr.Error(str(exc) or type(exc).__name__) from exc
        finally:
            games.finish(session_id)
        return frame(game, session_id, len(game["turns"]) - 1, path=saved_path(game))

    recheck.click(recheck_game, [game_state, owner, candidate], outputs, **serial)

    def select_response(game, session_id, index):
        return (*view(game, session_id, index), *cleared())

    view_outputs = [picker, note, token_state, strip, raw, jspace_panel, *inspector]
    picker.input(select_response, [game_state, owner, picker], view_outputs, **serial)

    def inspect_token(session_id, payload, event: gr.SelectData):
        index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
        return (*selections.inspect(session_id, payload, event), index)

    strip.select(inspect_token, [owner, token_state], [detail, alternatives, picked], queue=False)

    def token_space(game, session_id, payload, index):
        """J space after one chosen token of the shown response; shown, not saved."""
        try:
            view_id, index, _ = selections.resolve(session_id, payload, index)
            if view_id[0] != game["id"] or view_id[1] is None:
                raise ValueError(STALE_TOKEN)
        except (TypeError, ValueError) as exc:
            raise gr.Error("Click a token in the shown response first.") from exc
        turn_index = view_id[1]
        turn = game["turns"][turn_index]
        held(session_id)
        try:
            with context.models.open_session() as session:
                games.attach(session_id, session)
                if turn.get("load_id") != session.load_id:
                    raise ValueError("That response came from another model load, so its tokens cannot be read again.")
                prompt = session.prompt_ids(messages_for(game, turn_index))
                ids = list(prompt) + [m["token_id"] for m in turn["metrics"]]
                reading = jspace.read(session, ids, [("Chosen token", len(prompt) + index)])
        except Exception as exc:
            raise gr.Error(str(exc) or type(exc).__name__) from exc
        finally:
            games.finish(session_id)
        return jspace_html(dict(jspace=reading), turn_index + 1, palette, display)

    token_jspace.click(token_space, [game_state, owner, token_state, picked], jspace_panel, **serial)

    def offer_menu(game, session_id, payload, request_id, event: gr.SelectData):
        index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
        try:
            view_id, index, metric = selections.resolve(session_id, payload, index)
        except ValueError:
            return menu.refuse(request_id, STALE_TOKEN)
        if view_id[0] != game["id"] or view_id[1] is None:
            return menu.refuse(request_id, STALE_TOKEN)
        return menu.offer(request_id, dict(view_id=list(view_id), index=index),
                          text=metric.get("text", ""), candidates=metric.get("top_candidates", []),
                          verb="Branch this response at", label="Your own replacement text",
                          submit="Replace token and regenerate")

    strip.select(offer_menu, [game_state, owner, token_state, menu_request], menu_response, queue=False)

    def branch(game, session_id, payload, action, temp, random_seed, token_limit, read_space, things):
        """Regenerate the selected response from one of its tokens, in a new game."""
        try:
            chosen = json.loads(action)
            selection = chosen["selection"]
            view_id, index, metric = selections.resolve(session_id, payload, selection["index"])
            if list(view_id) != selection["view_id"] or view_id[0] != game["id"] or view_id[1] is None:
                raise ValueError(STALE_TOKEN)
        except (KeyError, TypeError, ValueError) as exc:
            raise gr.Error(STALE_TOKEN) from exc
        turn_index = view_id[1]
        turn = game["turns"][turn_index]
        edit = dict(parent=game["id"], turn=turn_index, token_index=index, load_id=turn.get("load_id"),
                    kept_ids=[m["token_id"] for m in turn["metrics"][:index]])
        if chosen.get("kind") == "candidate":
            candidates = metric.get("top_candidates", [])
            position = chosen.get("index")
            if not isinstance(position, int) or not 0 <= position < len(candidates):
                raise gr.Error("Choose an alternative for the selected token.")
            edit.update(candidate_id=candidates[position]["token_id"], label=candidates[position]["text"])
        elif chosen.get("kind") == "text" and isinstance(chosen.get("text"), str) and chosen["text"]:
            edit.update(text=chosen["text"], label=chosen["text"])
        else:
            raise gr.Error("Choose an alternative or type replacement text.")
        child = rewound(game, turn_index)
        yield from respond(child, session_id, turn["question"], temp, random_seed, token_limit, read_space, things,
                           edit=edit)

    menu_action.input(branch, [game_state, owner, token_state, menu_action, *settings], [*outputs, question],
                      **serial)

    def rewind_to(game, session_id, index):
        if index is None or not 0 <= index < len(game["turns"]):
            raise gr.Error("Select a response to rewind to.")
        return (*frame(rewound(game, index + 1), session_id, index, path=None), *cleared())

    rewind.click(rewind_to, [game_state, owner, picker], [*outputs, *inspector], **serial)

    def open_saved(path, session_id):
        if not path:
            return (gr.skip(),) * (len(outputs) + len(inspector))
        try:
            game = load(path)
        except (OSError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise gr.Error(MALFORMED) from exc
        try:
            for turn in game["turns"]:
                context.tokens.strip(turn["metrics"])
                for metric in turn["metrics"]:
                    context.tokens.describe(metric)
            # A new id, so continuing it never writes over the file that was opened.
            child = hangman_reopened(game)
            shown_game = frame(child, session_id, len(child["turns"]) - 1 if child["turns"] else None, path=None)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise gr.Error(MALFORMED) from exc
        return (*shown_game, *cleared())

    upload.upload(open_saved, [upload, owner], [*outputs, *inspector], **serial)

    def show_context(game, index):
        if index is None or not 0 <= index < len(game["turns"]):
            return "Select a response first.", ""
        turn = game["turns"][index]
        text, load_id = context.models.prompt_text(messages_for(game, index))
        if text is None:
            return "Load a model to render the prompt through its chat template.", ""
        if load_id == turn.get("load_id"):
            return f"The prompt response {index + 1} was generated from.", text
        return (f"Rendered by the model loaded now. Response {index + 1} came from "
                f"{shown(turn.get('model_id') or 'an unrecorded model')} under another load, "
                "so its template may have differed."), text

    context_refresh.click(show_context, [game_state, picker], [context_note, context_body], show_progress="hidden")
    context_pane.expand(show_context, [game_state, picker], [context_note, context_body], show_progress="hidden")
