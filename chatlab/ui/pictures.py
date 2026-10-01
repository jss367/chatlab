"""Pictures in the message box: pasting one, attaching one, and the strip that shows them.

A picture reaches the composer three ways - pasted while typing, dropped on
the box, or chosen with **Attach** - and all three go through the one upload
the Attach button already makes. The paste and the drop are handed to that
button's own file input (:data:`PICTURES_JS`), so a screenshot travels to
the server as a file upload rather than as a data URL inside an event, and
there is one handler to get right.

What is attached waits in a state of names (see :mod:`attachments`), drawn
as a row of thumbnails above the text, until the message is sent; sending
empties it in the same frame that empties the text.
"""

from __future__ import annotations

import html

import gradio as gr

from chatlab import attachments

ATTACH_ID = "attach-picture"
STRIP_ID = "composer-pictures"
# The hidden box a thumbnail's remove button writes its picture's name into.
REMOVE_BRIDGE_ID = "picture-remove-bridge"
DRAG_CLASS = "dragging-picture"


def picture_url(name: str) -> str:
    """Where the browser fetches a stored picture from.

    The directory is one of Gradio's static paths (see ``ui.layout``), so the
    file is served where it lies rather than copied into Gradio's cache on
    every frame that draws it.
    """

    return f"/gradio_api/file={attachments.image_path(name)}"


def strip_html(names: list[str] | None) -> str:
    """The thumbnails of what is attached, each with a button that takes it off."""

    chips = []
    for name in names or []:
        escaped = html.escape(name, quote=True)
        chips.append(
            f'<div class="picture-chip" title="Attached picture">'
            f'<img src="{html.escape(picture_url(name), quote=True)}" alt="Attached picture">'
            f'<button type="button" class="picture-remove" data-name="{escaped}" '
            f'aria-label="Remove this picture">×</button></div>'
        )
    return "".join(chips)


def attach_pictures(files, names: list[str] | None):
    """Store what was pasted, dropped or chosen, and add it to the message."""

    names = list(names or [])
    paths = files if isinstance(files, list) else [files] if files else []
    problem = None
    for path in paths:
        if len(names) >= attachments.MAX_IMAGES_PER_MESSAGE:
            problem = f"A message holds at most {attachments.MAX_IMAGES_PER_MESSAGE} pictures."
            break
        try:
            name = attachments.store_file(path)
        except attachments.AttachmentError as error:
            problem = str(error)
            continue
        if name not in names:
            names.append(name)
    return names, strip_html(names), problem if problem else gr.skip()


def remove_picture(payload: str | None, names: list[str] | None):
    """Take one picture off the message; ``payload`` is its name and a nonce."""

    name = (payload or "").split("|", 1)[0]
    names = [kept for kept in names or [] if kept != name]
    return names, strip_html(names)


PICTURES_JS = f"""
() => {{
  if (window.__chatlabPictures) {{ return; }}
  window.__chatlabPictures = true;
  // The Attach button's own file input sits beside the button rather than
  // inside it. Handing it the files and announcing the change is the upload
  // a click and a choice would have made.
  const upload = (files) => {{
    const button = document.getElementById('{ATTACH_ID}');
    const input = button?.parentElement?.querySelector('input[type=file]');
    if (!input || !files.length) {{ return false; }}
    const transfer = new DataTransfer();
    files.forEach((file) => transfer.items.add(file));
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', {{bubbles: true}}));
    return true;
  }};
  const pictures = (list) => Array.from(list || []).filter((file) => file.type.startsWith('image/'));
  const composer = (node) => node?.closest?.('#composer');
  document.addEventListener('paste', (event) => {{
    // Typing a message, or nothing focused at all with the chat on screen:
    // a screenshot pasted anywhere else (a system prompt, a search box) is
    // left to that box.
    const target = event.target;
    const idle = target === document.body && document.getElementById('composer')?.offsetParent;
    if (!composer(target) && !idle) {{ return; }}
    const files = pictures(event.clipboardData?.files);
    if (!files.length) {{ return; }}
    event.preventDefault();
    upload(files);
  }});
  const carriesFiles = (event) => Array.from(event.dataTransfer?.types || []).includes('Files');
  document.addEventListener('dragover', (event) => {{
    const box = composer(event.target);
    if (!box || !carriesFiles(event)) {{ return; }}
    event.preventDefault();
    box.classList.add('{DRAG_CLASS}');
  }});
  document.addEventListener('dragleave', (event) => {{
    const box = composer(event.target);
    if (box && !box.contains(event.relatedTarget)) {{ box.classList.remove('{DRAG_CLASS}'); }}
  }});
  document.addEventListener('drop', (event) => {{
    const box = composer(event.target);
    if (!box) {{ return; }}
    box.classList.remove('{DRAG_CLASS}');
    const files = pictures(event.dataTransfer?.files);
    if (!files.length) {{ return; }}
    event.preventDefault();
    upload(files);
  }});
  document.addEventListener('click', (event) => {{
    const remove = event.target.closest?.('#{STRIP_ID} .picture-remove');
    if (!remove) {{ return; }}
    const input = document.querySelector('#{REMOVE_BRIDGE_ID} textarea, #{REMOVE_BRIDGE_ID} input');
    if (!input) {{ return; }}
    const prototype = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    // The nonce makes a second removal of the same name a change too.
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, `${{remove.dataset.name}}|${{Date.now()}}`);
    input.dispatchEvent(new Event('input', {{bubbles: true}}));
  }});
}}
"""
