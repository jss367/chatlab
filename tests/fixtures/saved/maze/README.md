# Saved maze runs from earlier ChatLab versions

Each file was written by the ChatLab of its era: a script in `generators/` ran in a worktree of the commit named below, drove an episode with that commit's own scripted test model, and saved it with that commit's own writer. The files are committed as they came out and are never edited; `tests/test_saved_maze.py` reads them the way Load a saved run does and asserts what they say. To make another, add a generator, run it at its commit (the command is in its header), and add a row here and a test there. Trial files are inputs people write, so their generators write them in the shape that commit read and check that it reads them.

| File | Format | Era: predates | Commit | Exercises |
|---|---|---|---|---|
| run-1-before-goal-mode | run-1 | goal_mode, prompts, recovery window, history field | b206b5e (#49) | interruption and recovery, wall, wrong maze id, invalid direction, call inside reasoning, recovered openness |
| run-1-hint-before-prompts | run-1 | system_prompt/instruction | 37d4d92 (#55) | hint mode and its tool id, coordinate id refused, interruption asked for by hand, abandoned |
| run-1-fork-before-recovery-window | run-1 | recovery window | 7e88a08 (#80) | hidden goal, own wording, token-edit fork, length cut-off |
| run-1-trial-before-recovery-window, trials-1-original | run-1, trials-1 | recovery window | 9cb00e8 (#84) | trial stamp and checksum, openness from the trial, stock and own wording |
| run-1-recovery-window-not-recovered | run-1 | history field | 5d86958 (#101) | recorded window, partial prefix, not recovered, text after a call |
| run-2-closures | run-2 | history field | 03b826a (#130) | closure, dropped closure, queued closure, paused |
| run-2-waypoint-steering, run-1-steering-stopped | run-2, run-1 | history field | 22361e8 (#148) | waypoint, steering at a cell and after moves, closure by the waypoint, stop at a steered opening frame |
| trials-1-waypoint-steering | trials-1 | | 22361e8 (#148) | named and inline vectors, waypoint, recovery window and wording in trials |
| run-3-inserts, run-3-inserts-changing-map, run-3-stopped | run-3 | history field | ffcbd7b (#156) | note, teammate and user messages and their outcomes, inserts on a changing map, stop after a message |
| team-1-shared-budget | team-1 | team-2, per-agent limit | 06d69fe (#149) | messages (one to no reader), message too long, cut off, abandoned |
| team-1-token-limit, team-1-starved | team-1 | team-2, per-agent limit | 06d69fe (#149) | shared limit spent to the token; too little left for a round |
| team-1-steering-checkpoint | team-1 | team-2 | 33bbe42 (#154) | required checkpoint, steering for some agents |
| team-1-agent-budget | team-1 | team-2 | 8e3db58 (#168) | five agents, per-agent limit, out of tokens |
| team-2-interventions | team-2 | history field | a736089 (#171) | supplied moves, waypoint, interruptions (one by hand), not recovered, out of calls |
| team-2-map-change-messages | team-2 | history field | 0f97a99 (#172) | team closure, dropped and queued closures, messages to agents |
| team-2-fork, team-2-fork-round-limit | team-2 | history field | 1bb1b78 (#173) | fork at an interrupted and an uninterrupted response, steering cell for one agent, round limit |
| run-1-prefilled-reasoning-before-history-field, team-2-stopped-before-history-field | run-1, team-2 | history field (02b0d32, #200) | 930f350 (#197) | template-opened reasoning, incomplete stream, team stopped mid-round |
| team-3-rule-1, team-3-rule-1-dropouts | team-3 | message rule 2 (2306785, #203) | 61d1f4b (#201) | exits, reward exit, taste, laps; arrival message carrying a call and one cut off; dropouts, owed responses cancelled by the limit, round limit |
| team-3-rule-2-unrecorded | team-3 | message_rules (4291355, #225) | 2306785 (#203) | cut-off and call-carrying arrival messages under rule 2 |
| team-3-team-reward, team-3-laps-limits | team-3 | message_rules | 1ebe9d0 (#204) | paired exits, team reward; arriving on the last token, sitting out a lap, round limit between laps |
| run-1-current, run-3-current-changing-map, team-2-current, team-3-current | run-1, run-3, team-2, team-3 | | cfd9fbd (0.40.0) | reasoning_content history; recorded message_rules 2 |
| team-3-assigned-exits | team-3 | | 431c09d (claude/maze-first-lap-exits) | assigned exits for two of three laps, an agent walking onto an exit it is not assigned, the first free lap |

No era was synthesized from current code. There is no team-3 file without the history field: #201 added team-3 after #200 added the field.
