"""Maps that close cells during a run, under an identifier that does not change."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .maze import Maze

FORMAT = "chatlab-maze-run-2"


@dataclass(frozen=True)
class ChangingMaze(Maze):
    """A maze that keeps one tool identifier across every version of itself.

    ``Maze.tool_id`` hashes the grid, so a wall closing mid-run would rename
    the maze under the model: its next call would carry the identifier it was
    given and come back ``wrong_maze``, telling it that it had addressed some
    other maze rather than that this one had changed. The identifier here is
    fixed when the run starts and answers for the environment across its
    versions, which is what the model is navigating.

    It names no destination in any goal mode, so a run that conceals the goal
    discloses no more through this identifier than through the one
    ``Maze.tool_id`` gives a hidden-goal run.
    """

    environment_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.environment_id:
            raise ValueError("A changing map needs an environment identifier.")

    def tool_id(self, goal_mode="coordinates"):
        return self.environment_id


def environment_id(maze):
    """The handle a map keeps while its walls change.

    Derived from the version the run starts as, rather than chosen, so the
    same scenario sends the model the same prompt bytes every time it is run;
    and derived without the goal, so it says nothing extra under a concealed
    destination.
    """
    return Maze.tool_id(maze, "hidden")


def changing(maze):
    """The same maze, ready to change, identified by the version it starts as."""
    return ChangingMaze(maze.grid, maze.start, maze.goal, maze.seed, environment_id(maze))


def load_maze(data):
    """A changing map read back from a saved run, under its recorded identifier.

    The identifier is derived again rather than taken as written, because it is
    what the model was addressing: a file naming an identifier the starting grid
    does not produce describes a run this replay cannot reconstruct.
    """
    maze = changing(Maze.from_dict(data))
    if data.get("environment_id", maze.environment_id) != maze.environment_id:
        raise ValueError("The run's environment identifier does not belong to the map it starts from.")
    return maze


def close_cell(maze, cell):
    """The same map with one open cell walled off.

    Every version of a changing map is itself a maze the run could have begun
    with, so ``Maze.__post_init__`` refuses a closure that leaves the
    destination unreachable from the start.
    """
    if not isinstance(cell, (list, tuple)) or len(cell) != 2 or any(type(x) is not int for x in cell):
        raise ValueError("A closure names one cell as a row and a column.")
    cell = tuple(cell)
    if not maze.open(cell):
        raise ValueError("Only an open cell inside the maze can be closed.")
    if cell in (maze.start, maze.goal):
        raise ValueError("The start and the destination are never closed.")
    row, column = cell
    grid = list(maze.grid)
    grid[row] = grid[row][:column] + "#" + grid[row][column + 1:]
    return replace(maze, grid=tuple(grid))


def check_closure(maze, position, cell):
    """The map after a closure, or why that closure cannot happen here.

    A closure narrows the problem rather than ending it, so the character has
    to be able to reach the destination from where it is standing afterwards.
    Walling in the cell it occupies would put it inside a wall, which is not a
    position the simulator could report.
    """
    position = tuple(position)
    if tuple(cell) == position:
        raise ValueError("The cell the character is standing in cannot be closed.")
    changed = close_cell(maze, cell)
    if position not in changed.distances(changed.goal):
        raise ValueError("That closure would cut the character off from the destination.")
    return changed


def maze_at_turn(maze, updates, index):
    """The map as the response at ``index`` found it, or its latest version for None.

    A closure recorded at boundary ``n`` landed before response ``n``
    generated, so that response acted on the map it produced.
    """
    for update in updates:
        if index is not None and update["before_turn"] > index:
            break
        maze = close_cell(maze, update["closed_cell"])
    return maze


def validate_pending(maze, cell, updates):
    """Check a closure a saved run was still waiting to apply.

    The cell is read against the map as it stands, because a queued closure
    blocks another until it lands and so meets the same map it was queued
    against. Where the character is standing is not asked: an autosave written
    between a response and the next one carries a cell the character has since
    moved onto, which is a closure about to be dropped rather than a file that
    could not have been written.
    """
    if not cell:
        return
    if not isinstance(cell, (list, tuple)) or len(cell) != 2 or any(type(x) is not int for x in cell):
        raise ValueError("A pending closure names one cell as a row and a column.")
    close_cell(maze_at_turn(maze, updates, None), cell)


def validate_drops(maze, drops, updates, turns):
    """Check a run's record of the closures that never happened.

    The reason is not checked. A dropped closure is one the map never took, so
    there is no version of the map holding its consequences the way an applied
    one holds its grid, and the reason may be about the run rather than the map
    at all, as it is for a closure an episode ended under.

    Everything else is. One closure is queued at a time, so no two drops land
    on one response boundary, none shares a boundary with a closure the map
    accepted, and none lies past the responses the file records. And each cell
    has to be one the run could have queued: a queued closure blocks another
    until it lands, so the map at its boundary is the map it was queued
    against, and an open cell that is neither the start nor the destination is
    what queueing one required.
    """
    if not isinstance(drops, list):
        raise ValueError("A run's dropped closures must be a list.")
    taken = {update["before_turn"] for update in updates}
    previous = -1
    for drop in drops:
        if not isinstance(drop, dict):
            raise ValueError("Each dropped closure must be an object.")
        boundary = drop.get("before_turn")
        if type(boundary) is not int or not previous < boundary <= len(turns) or boundary in taken:
            raise ValueError("Each dropped closure names one free response boundary, in order.")
        previous = boundary
        cell = drop.get("cell")
        if not isinstance(cell, list) or len(cell) != 2 or any(type(x) is not int for x in cell):
            raise ValueError("A dropped closure names one cell as a row and a column.")
        close_cell(maze_at_turn(maze, updates, boundary - 1), cell)
        if not isinstance(drop.get("reason"), str) or not drop["reason"].strip():
            raise ValueError("A dropped closure records why it was dropped.")


def validate_updates(maze, updates, turns, events):
    """Check a recorded run of closures against the path the run recorded.

    Each closure is replayed against the position the run's own transitions
    put the character in, and the map it claims to have produced is compared
    with the one its closure actually produces, so a file cannot describe a
    map the model was never navigating.
    """
    previous = -1
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError("Each map change must be an object.")
        boundary = update.get("before_turn")
        if type(boundary) is not int or not previous < boundary <= len(turns):
            raise ValueError("Each map change names one response boundary, in order.")
        previous = boundary
        accepted = [e for e in events if e["accepted"] and e.get("turn", -1) < boundary]
        position = tuple(accepted[-1]["after"]) if accepted else maze.start
        if list(position) != update.get("position"):
            raise ValueError("A map change records a position the run's path never reached.")
        maze = check_closure(maze, position, update.get("closed_cell"))
        if list(maze.grid) != update.get("grid"):
            raise ValueError("A map change records a map its own closure does not produce.")
    return maze
