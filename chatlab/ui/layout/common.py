"""What the wiring in every part of the page shares."""


# One queue for everything that rewrites the forks or the conversation in one
# step: the branch buttons, the list, Clear all, Undo, Stop, the loaders, and
# the two listeners on the states. Gradio
# runs events that share a concurrency id one at a time, in the order they
# were queued, and reads a State input when the event runs rather than when
# it was queued. So a redraw queued by a streaming frame can no longer run
# after a click on New with the forks as they were before the click, and
# hand that older pane back over the new one. The generation handlers stay
# out of it: they hold their own slot for the whole reply, and the redraw
# has to run between their frames.
CONVERSATION_PANE_QUEUE = "conversation-pane"


# What a handler that repaints on a timer passes instead of nothing.
#
# Gradio fades every output component of a running event down to 20% opacity
# for as long as it is in flight, and fades it back when the event lands.
# show_progress="hidden" does not turn that off: it only hides the spinner and
# the runtime counter. The fade is a class on the component, and which
# components get it is show_progress_on, which defaults to every output. So a
# handler on a timer hands its outputs a fade in and out on every tick, which
# at the quarter-second the conversation pane polls at reads as a blink
# several times a second, on text nobody asked to have redrawn. An empty
# show_progress_on says to mark nothing as pending, which is what a poll the
# reader did not ask for should be doing.
QUIET_TICK = {"show_progress": "hidden", "show_progress_on": []}
