"""Play hangman against the loaded model and read every reply token by token."""
import copy
import json
import logging
import re
import threading
import time

import gradio as gr

from extension_api import write_private_text
from .game import (
    MAX_FILE_BYTES, OPENING, SYSTEM, check, dictionary, finish_turn, fitting_words, latest_board, load,
    messages_for, new_game, reasoning_of, reopened, rewound, saved, WORD_LIST,
)

logger = logging.getLogger(__name__)

STALE_TOKEN = "Select a token in the current response again."
MALFORMED = "That file is not a valid saved hangman game."
EXAMPLES = 12

CSS = """
#hangman-page {overflow-y:auto; min-height:0; padding:12px;}
#hangman-page .hangman-check {max-width:1000px;}
"""


class Games:
    """One running response per view, and a way to stop it from another event."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = {}

    def start(self, owner):
        with self._lock:
            if owner in self._active:
                raise ValueError("A response is already being generated in this view.")
            self._active[owner] = [threading.Event(), None]
            return self._active[owner][0]

    def attach(self, owner, session):
        with self._lock:
            active = self._active[owner]
            active[1] = session
            if active[0].is_set():
                session.cancel()

    def finish(self, owner):
        with self._lock:
            self._active.pop(owner, None)

    def cancel(self, owner):
        with self._lock:
            active = self._active.get(owner)
            if active:
                active[0].set()
                if active[1] is not None:
                    active[1].cancel()


def chat_messages(game):
    messages = []
    for number, turn in enumerate(game["turns"], 1):
        messages.append({"role": "user", "content": turn["guess"]})
        reasoning = reasoning_of(turn["text"], turn.get("reasoning_prefilled", False))
        if reasoning:
            messages.append({"role": "assistant", "content": reasoning,
                             "metadata": {"title": f"Reasoning · response {number}"}})
        answer = turn.get("answer") or ("…" if turn.get("finish_reason") is None else "(no answer)")
        messages.append({"role": "assistant", "content": answer})
    return messages


def check_text(game):
    if not game["turns"]:
        return "Start a game. The page reads the **Board:** line of each reply and checks it against the ones before."
    parts = []
    board, guessed = latest_board(game)
    guessed = sorted(guessed)
    if board is None:
        parts.append("**Board:** no readable `Board:` line yet.")
    else:
        parts.append(f"**Board:** `{' '.join(c.upper() if c != '_' else '_' for c in board)}` "
                     f"({len(board)} letters) · **Letters guessed:** {', '.join(guessed).upper() or 'none'}")
        words = fitting_words(game)
        if not dictionary():
            parts.append(f"**Words that fit:** no word list at `{WORD_LIST}`.")
        else:
            shown = ", ".join(words[:EXAMPLES]) + (", …" if len(words) > EXAMPLES else "")
            parts.append(f"**Words that fit the latest board:** {len(words)} in `{WORD_LIST}`"
                         + (f" ({shown})" if words else ""))
    revealed = next((t["revealed_word"] for t in reversed(game["turns"]) if t.get("revealed_word")), None)
    if revealed:
        parts.append(f"**Revealed word:** {revealed.upper()}")
    problems = check(game)
    if problems:
        parts.append("**Contradictions:**\n" + "\n".join(f"- Response {n}: {m}" for n, m in problems))
    else:
        parts.append("**Contradictions:** none found.")
    unread = [str(n) for n, t in enumerate(game["turns"], 1) if t.get("finish_reason") and t.get("board") is None]
    if unread:
        parts.append(f"Responses without a readable board: {', '.join(unread)}.")
    return "\n\n".join(parts)


def turn_choices(game):
    return [(f"Response {n} · {t['guess'][:40]}", n - 1) for n, t in enumerate(game["turns"], 1)]


def shown(value):
    """A recorded value as Markdown that shows its characters and nothing else.

    An opened file can carry any string here, and in the note a string such as
    ``![x](https://host/pixel)`` would load that address. Anything beyond plain
    words and numbers goes in a code span, which Markdown never renders.
    """
    text = " ".join(str(value).split())
    return text if re.fullmatch(r"[A-Za-z0-9.+ -]*", text) else "`" + text.replace("`", "'") + "`"


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
    return " · ".join(parts)


def build_page(context):
    games = Games()
    selections = context.tokens.selections()
    menu = context.tokens.menu("hangman-tokens")

    def forget(owner):
        games.cancel(owner)
        selections.forget(owner)

    with gr.Column(elem_id="hangman-page"):
        owner = gr.State(value=selections.new_session, delete_callback=forget)
        game_state = gr.State(new_game(SYSTEM))
        token_state = gr.State(("", []))
        gr.Markdown("# Hangman\nThe model thinks of a word and hosts the game. Nothing on this page knows the word: "
                    "each reply is checked only against the replies before it.")
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
                with gr.Accordion("Saved games", open=False):
                    gr.Markdown("Every finished response saves the game. Uploading one opens a copy: new "
                                "guesses continue it under whichever model is loaded, and the original file is left alone.")
                    upload = gr.File(label="Open a saved game", file_types=[".json"], type="filepath")
                    download = gr.File(label="This game", interactive=False)
            with gr.Column(scale=2, min_width=360):
                chat = gr.Chatbot(type="messages", label="Game", height=460, render_markdown=False)
                with gr.Row():
                    guess = gr.Textbox(placeholder="A letter, a word, or anything to say", show_label=False,
                                       lines=1, max_lines=1, scale=4, submit_btn=False)
                    send = gr.Button("Guess", variant="primary", scale=1)
                    stop = gr.Button("Stop", scale=1)
                check_panel = gr.Markdown(check_text(new_game(SYSTEM)), elem_classes=["hangman-check"])
            with gr.Column(scale=2, min_width=360):
                picker = gr.Dropdown(choices=[], value=None, label="Response", interactive=True)
                note = gr.Markdown("")
                strip = gr.HighlightedText(label="Click a token to inspect it; right-click to branch the response there",
                                           combine_adjacent=False, show_legend=True, elem_id="hangman-tokens",
                                           color_map=context.tokens.color_map, elem_classes=menu.strip_classes)
                menu_request, menu_response, menu_action = menu.bridges()
                detail = gr.Markdown("Select a token.")
                alternatives = gr.Dataframe(headers=["Token ID", "Text", "Raw probability"], interactive=False)
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
        """Everything that shows one response of one game, but the token detail.

        Streamed frames never carry the probability table: Gradio's Dataframe
        fails in the browser on a streamed update, and with it every component
        after it in that frame. Whatever starts a new response clears the
        detail in a separate event instead, through ``cleared``.
        """
        if index is None or not 0 <= index < len(game["turns"]):
            payload, _ = selections.view(session_id, (game["id"], None), [])
            return (gr.update(choices=turn_choices(game), value=None), "", payload, [], "")
        turn = game["turns"][index]
        payload, _ = selections.view(session_id, (game["id"], index), turn["metrics"])
        return (gr.update(choices=turn_choices(game), value=index), turn_note(turn), payload,
                context.tokens.strip(turn["metrics"]), turn["text"])

    def cleared():
        return "Select a token.", []

    def frame(game, session_id, index, *, path=gr.skip(), check=True):
        # The check reads every board and filters the word list, so a
        # streaming response leaves it alone until the response is finished.
        return (game, chat_messages(game), check_text(game) if check else gr.skip(),
                *view(game, session_id, index), path)

    outputs = [game_state, chat, check_panel, picker, note, token_state, strip, raw, download]
    inspector = [detail, alternatives]

    def save(game):
        text = saved(game)
        if len(text.encode("utf-8")) > MAX_FILE_BYTES:
            raise OSError("the game is larger than a saved game can be opened at")
        path = context.data_dir / f"{game['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_private_text(path, text)
        return str(path)

    def respond(game, session_id, text, temp, random_seed, token_limit, edit=None):
        """Generate one reply to ``text`` at the end of ``game``, streaming frames.

        ``edit`` replays a kept prefix of an earlier response ending in a
        replacement: its token IDs are only meaningful under the load that
        produced them, which is checked once the model is held.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Type a guess first.")
            integer_seed, limit = int(random_seed), int(token_limit)
            if integer_seed < 0 or not 1 <= limit <= 32768:
                raise ValueError("Seed must be nonnegative and tokens per response between 1 and 32768.")
            cancel = games.start(session_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise gr.Error(str(exc)) from exc
        game = copy.deepcopy(game)
        index = len(game["turns"])
        turn = dict(guess=text.strip(), text="", metrics=[], forced_prefix_tokens=0, reasoning_prefilled=False,
                    finish_reason=None, started_at=time.time(),
                    sampling=dict(temperature=float(temp), top_p=1.0, top_k=0, max_new_tokens=limit,
                                  seed=integer_seed + index))
        game["turns"].append(turn)
        failure, shown = None, False
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
                shown = True
                yield (*frame(game, session_id, index, check=False), "")
                stream = session.generate(messages_for(game, index), forced_ids=forced, **turn["sampling"])
                try:
                    for update in stream:
                        turn.update(text=update.text, metrics=update.metrics,
                                    forced_prefix_tokens=update.forced_prefix_tokens,
                                    reasoning_prefilled=update.reasoning_prefilled)
                        finish_turn(turn)
                        yield (*frame(game, session_id, index, check=False), gr.skip())
                finally:
                    stream.close()
                last = turn["metrics"][-1]["token_id"] if turn["metrics"] else None
                turn["finish_reason"] = "stopped" if cancel.is_set() else "stop" if last in stop_ids else "length"
        except Exception as exc:
            failure = str(exc) or type(exc).__name__
            logger.warning("Hangman game %s: response %s failed: %s", game["id"], index + 1, failure)
        finally:
            games.finish(session_id)
            turn["seconds"] = time.time() - turn["started_at"]
        if not turn["metrics"] and (failure is not None or cancel.is_set()):
            # Nothing was generated, whether it failed or was stopped during
            # the prompt, so there is no response to keep: the next prompt
            # would otherwise carry an empty assistant turn.
            game["turns"].pop()
            if shown:
                yield (*frame(game, session_id, index - 1 if index else None), gr.skip())
            if failure is not None:
                raise gr.Error(failure)
            return
        if failure is not None:
            turn.update(finish_reason="error", error=failure)
        finish_turn(turn)
        path = gr.skip()
        try:
            path = save(game)
        except OSError as exc:
            logger.warning("Could not save hangman game %s: %s", game["id"], exc)
            gr.Warning(f"This response was not saved: {exc}.")
        yield (*frame(game, session_id, index, path=path), gr.skip())
        if failure is not None:
            raise gr.Error(failure)

    def branch_ids(session, edit):
        """The kept tokens of the edited response, then the replacement."""
        if edit["load_id"] != session.load_id:
            raise ValueError("That response came from a different model load, so its tokens cannot be replayed. "
                             "Rewind instead, or load the model that wrote it.")
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

    def start_game(system_text, opening_text, session_id, temp, random_seed, token_limit):
        game = new_game(system_text)
        yield from respond(game, session_id, opening_text or OPENING, temp, random_seed, token_limit)

    def play(game, text, session_id, temp, random_seed, token_limit):
        if not game["turns"]:
            raise gr.Error("Start a new game first.")
        yield from respond(game, session_id, text, temp, random_seed, token_limit)

    settings = [temperature, seed, max_tokens]
    serial = dict(concurrency_id="hangman-view", show_progress="hidden")
    for event in (start.click, send.click, guess.submit, menu_action.input):
        event(cleared, None, inspector, queue=False)
    start.click(start_game, [system, opening, owner, *settings], [*outputs, guess], **serial)
    send.click(play, [game_state, guess, owner, *settings], [*outputs, guess], **serial)
    guess.submit(play, [game_state, guess, owner, *settings], [*outputs, guess], **serial)
    stop.click(games.cancel, owner, [], queue=False)

    def select_response(game, session_id, index):
        return (*view(game, session_id, index), *cleared())

    view_outputs = [picker, note, token_state, strip, raw, *inspector]
    picker.input(select_response, [game_state, owner, picker], view_outputs, **serial)

    def inspect_token(session_id, payload, event: gr.SelectData):
        return selections.inspect(session_id, payload, event)

    strip.select(inspect_token, [owner, token_state], [detail, alternatives], queue=False)

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

    def branch(game, session_id, payload, action, temp, random_seed, token_limit):
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
        yield from respond(child, session_id, turn["guess"], temp, random_seed, token_limit, edit=edit)

    menu_action.input(branch, [game_state, owner, token_state, menu_action, *settings], [*outputs, guess], **serial)

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
        # load checks the shape of the file, not every field of every token it
        # records, so the game is rendered here in full before any of it is
        # shown: a file wrong deeper down is refused whole, not half drawn.
        try:
            # A new id, so continuing it never writes over the file that was opened.
            child = reopened(game)
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
                f"`{turn.get('model_id') or 'an unrecorded model'}` under another load, "
                "so its template may have differed."), text

    context_refresh.click(show_context, [game_state, picker], [context_note, context_body], show_progress="hidden")
    context_pane.expand(show_context, [game_state, picker], [context_note, context_body], show_progress="hidden")
