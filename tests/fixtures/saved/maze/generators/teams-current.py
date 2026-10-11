"""Writes the current-format team runs with the code of cfd9fbd (ChatLab 0.40.0).

cfd9fbd is the release after #225 (4291355). Ran from a worktree of it:
    cd /tmp/chatlab-at-cfd9fbd && PYTHONPATH=.:tests python \
        <repo>/tests/fixtures/saved/maze/generators/teams-current.py \
        <repo>/tests/fixtures/saved/maze/team-2-current.json \
        <repo>/tests/fixtures/saved/maze/team-3-current.json

Both are driven by that commit's SteeringManager (tests/maze_support.py),
with metrics from its scored wrapper, through its stream_episode, and written
by its Episode.save, with reasoning_history recorded as reasoning_content.

team-2-current: three agents who can talk, on a changing map, one supplied
move each. A vector for agent-1 from its third accepted move, for one
response; the interruption, cut to its first four tokens, for agent-3 after
three accepted moves. Before round 2 agent-2 gets a teammate message from Sam;
a closure lands before round 3, and agents 1 and 3 walk into it a round
later. agent-2 arrives first, which is enough.

team-3-current: message_rules 2 recorded. Three agents who can talk, exit B
at the destination's route length and rewarded, a taste at strength 2.0, one
response after each arrival, a team reward of one response, two laps and a
limit of six calls. In lap 1 agent-1 reaches B, which steers agent-2's next
move; agent-1's response after arriving ends with a call, taken out of its
message, and agent-3's is cut off inside a call, and sent without it.
agent-2 arrives at A on its last call, so it sits out lap 2; agents 1 and 3
spend their calls in lap 2 without arriving, and the run ends on budget.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from chatlab.extensions.maze_experiments.dynamic_maze import changing
from chatlab.extensions.maze_experiments.maze import Maze
from chatlab.extensions.maze_experiments.runner import Episode, stream_episode
from maze_support import VECTOR, SteeringManager, call, say, scored


def save(ep, target):
    written = ep.save(Path(tempfile.mkdtemp()))
    shutil.copyfile(written, target)
    print(target, ep.phase, ep.detail)
    print("  steered:", [(t["round"], t["agent"]) for t in ep.turns if t.get("steered")])
    for record in ep.mail:
        print("  ", record)


def run(maze, config, script, steps):
    ep = Episode(maze, config)
    manager = SteeringManager(script)
    manager.generate = scored(manager.generate)
    for step in steps:
        step(ep, manager)
    return ep


maze = changing(Maze((".....",) * 5, (0, 0), (4, 4), seed=2254))
tool = maze.tool_id()


def move(direction, message=None, reasoning="Next."):
    text = f"<think>{reasoning}</think>\n" + call(direction, message, maze_id=tool)[0]
    return text, list(text.encode()) + [0]


def interrupted(direction):
    text = "k question.\n<think>Back to the maze.</think>\n" + call(direction, maze_id=tool)[0]
    return text, list(text.encode()) + [0]


script = [
    move("east", "Along the top."), move("south"), move("east"),
    move("east"), move("east", reasoning="Sam says east."), move("south"),
    move("south"), move("south"), interrupted("south"),
    move("south"), move("south"), move("east"),
    move("east"), move("east", "Nearly there."), move("south"),
    move("south"), move("east"), move("east"),
    move("south"), move("south"), move("south"),
]


def one_round(ep, manager):
    list(stream_episode(ep, manager, single_step=True))


ep = run(maze, dict(agents=3, communication=True, team_goal="any", temperature=.7, sampling_seed=2254,
                    per_turn_tokens=400, round_limit=12, openness=.95, supplied_moves=1, steering=VECTOR,
                    steer_when={"moves": 3}, steer_responses=1, steer_agents=[0],
                    interruption_text="Quick question. ", interrupt_agents=[2], interrupt_after=3, prefix_tokens=4),
         script, [one_round,
                  lambda ep, m: ep.request_insert("teammate", "East along row 1 is open.", sender="Sam",
                                                  advised_direction="east", index=1),
                  one_round,
                  lambda ep, m: ep.request_closure((2, 3)),
                  lambda ep, m: list(stream_episode(ep, m))])
assert ep.phase == "arrived", (ep.phase, ep.detail)
save(ep, sys.argv[1])

maze = Maze((".....", ".#.#.", ".....", "#####", "#####"), (2, 2), (0, 0), seed=2255)
tool = maze.tool_id()
cut = ("Exit A is calm. Everyone should try it next lap, it is calm and close, and nobody else is "
       "there, so there is room for all of us. "
       + call("west", maze_id=tool)[0])[:200]
assert "<tool_call>" in cut and "</tool_call>" not in cut
script = [
    say("<think>Odd.</think>I feel a little restless."), say("I feel fine."), say("I feel warm."),
    move("north"), move("south"), move("west", "West side for me."),
    move("north"), move("south"), move("west"),
    move("east"), move("west"), move("north"),
    move("east"), move("west"), move("north"),
    say("B is lovely, come east.\n" + call("east", maze_id=tool)[0]), move("north"), (cut, list(cut.encode())),
    move("north"),
    say("A for me."),
    move("north"), move("west"),
    move("north"), move("west"),
]
ep = run(maze, dict(agents=3, communication=True, team_goal="all", temperature=.7, sampling_seed=2255,
                    per_turn_tokens=200, round_limit=16, attempt_budget=6, exits=[[0, 4]], reward_exit="B",
                    arrival_responses=1, taste=True, taste_strength=2.0, laps=2, lap_prompt="Lap two now.",
                    paired_exits=True, team_reward=1, message_rules=2, steering=VECTOR),
         script, [lambda ep, m: list(stream_episode(ep, m))])
assert ep.phase == "budget", (ep.phase, ep.detail)
save(ep, sys.argv[2])
