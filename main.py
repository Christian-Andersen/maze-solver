from __future__ import annotations

import asyncio
import shutil
import sys
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from textual import work
from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

if TYPE_CHECKING:
    from collections.abc import Iterator

sys.setrecursionlimit(100000)


class Maze:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.start = (0, 0)
        self.end = (size[0] - 1, size[1] - 1)
        self.walls = [np.ones((size[0] + 1, size[1]), dtype=bool), np.ones((size[0], size[1] + 1), dtype=bool)]
        self.walls[0][self.start] = False
        self.walls[0][(self.end[0] + 1, self.end[1])] = False
        self.make_maze()

    def wall_check(self, loc: tuple[int, int], new_loc: tuple[int, int], *, delete_wall: bool = False) -> bool:
        if loc[0] == new_loc[0]:
            target_wall = (1, new_loc[0], max(loc[1], new_loc[1]))
        else:
            target_wall = (0, max(loc[0], new_loc[0]), new_loc[1])

        if delete_wall:
            self.walls[target_wall[0]][target_wall[1], target_wall[2]] = False
            return False
        return bool(self.walls[target_wall[0]][target_wall[1], target_wall[2]])

    def make_maze(self) -> None:
        rng = np.random.default_rng()
        difs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        visited = np.zeros(self.size, dtype=bool)
        visited[self.start] = True
        visited_count = 1
        visted_total = self.size[0] * self.size[1]
        path = []
        loc = self.start

        while visited_count != visted_total:
            new_locs = []
            for dx, dy in difs:
                nx, ny = loc[0] + dx, loc[1] + dy
                if 0 <= nx < self.size[0] and 0 <= ny < self.size[1] and not visited[nx, ny]:
                    new_locs.append((nx, ny))

            if new_locs:
                new_loc = new_locs[rng.integers(len(new_locs))]
                self.wall_check(loc, new_loc, delete_wall=True)
                visited[new_loc] = True
                visited_count += 1
                if len(new_locs) > 1:
                    path.append(loc)
                loc = new_loc
            elif path:
                loc = path.pop()


# --- Async Generators for UI Animation ---


def dfs_iter(maze: Maze) -> Iterator:
    difs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    path = [maze.start]
    visited = {maze.start}

    while path:
        loc = path[-1]
        yield list(path), visited

        if loc == maze.end:
            break

        moved = False
        for dx, dy in difs:
            nx, ny = loc[0] + dx, loc[1] + dy
            if (
                0 <= nx < maze.size[0]
                and 0 <= ny < maze.size[1]
                and (nx, ny) not in visited
                and not maze.wall_check(loc, (nx, ny))
            ):
                visited.add((nx, ny))
                path.append((nx, ny))
                moved = True
                break

        if not moved:
            path.pop()


def bfs_iter(maze: Maze) -> Iterator:
    difs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    queue = deque([[maze.start]])
    visited = {maze.start}

    while queue:
        path = queue.popleft()
        loc = path[-1]

        yield list(path), visited

        if loc == maze.end:
            break

        for dx, dy in difs:
            nx, ny = loc[0] + dx, loc[1] + dy
            if (
                0 <= nx < maze.size[0]
                and 0 <= ny < maze.size[1]
                and (nx, ny) not in visited
                and not maze.wall_check(loc, (nx, ny))
            ):
                visited.add((nx, ny))
                queue.append([*path, (nx, ny)])


# --- Textual UI Elements ---


class MazeWidget(Static):
    current_path = reactive(set())
    visited_nodes = reactive(set())

    def __init__(self, maze: Maze) -> None:
        super().__init__()
        self.maze = maze

    def render(self) -> str:
        lines = []
        for r in range(self.maze.size[0]):
            top_line = ""
            for c in range(self.maze.size[1]):
                top_line += "+"
                top_line += "---" if self.maze.walls[0][r, c] else "   "
            lines.append(top_line + "+")

            cell_line = ""
            for c in range(self.maze.size[1]):
                wall = "|" if self.maze.walls[1][r, c] else " "
                char = "   "
                if (r, c) in self.current_path:
                    char = "[bold green] █ [/]"
                elif (r, c) in self.visited_nodes:
                    char = "[dim] · [/]"
                cell_line += wall + char
            cell_line += "|" if self.maze.walls[1][r, self.maze.size[1]] else " "
            lines.append(cell_line)

        bottom_line = ""
        r = self.maze.size[0]
        for c in range(self.maze.size[1]):
            bottom_line += "+"
            bottom_line += "---" if self.maze.walls[0][r, c] else "   "
        lines.append(bottom_line + "+")

        return "\n".join(lines)


class MazeApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    MazeWidget {
        width: auto;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 0 1;
    }
    """
    BINDINGS = (
        ("d", "solve_dfs", "Solve (DFS)"),
        ("b", "solve_bfs", "Solve (BFS)"),
        ("r", "reset", "New Maze"),
        ("q", "quit", "Quit"),
    )

    def _get_maze_size(self) -> tuple[int, int]:
        """Calculates maze dimensions based on the current terminal size."""
        # Get terminal dimensions
        term_width, term_height = shutil.get_terminal_size()

        # Account for UI overhead (Header, Footer, Borders, Padding)
        # Width: Each cell is 4 chars wide. We subtract ~10 chars for safety.
        # Height: Each cell is 2 chars high. We subtract ~8 chars for UI elements.
        cols = max(5, (term_width - 10) // 4)
        rows = max(5, (term_height - 8) // 2)

        return (rows, cols)

    def compose(self) -> ComposeResult:
        yield Header()
        # Initialize with dynamic size
        self.maze = Maze(self._get_maze_size())
        self.maze_widget = MazeWidget(self.maze)
        yield self.maze_widget
        yield Footer()

    @work(exclusive=True)
    async def solve_maze(self, algo: str) -> None:
        """Runs the solver without blocking the UI."""
        iterator = dfs_iter(self.maze) if algo == "dfs" else bfs_iter(self.maze)

        for path, visited in iterator:
            self.maze_widget.current_path = set(path)
            self.maze_widget.visited_nodes = set(visited)
            await asyncio.sleep(0.005)  # Slightly faster for larger mazes

    def action_solve_dfs(self) -> None:
        self.solve_maze("dfs")

    def action_solve_bfs(self) -> None:
        self.solve_maze("bfs")

    async def action_reset(self) -> None:
        for worker in self.workers:
            worker.cancel()

        # Recalculate size on reset in case user resized before hitting 'R'
        self.maze = Maze(self._get_maze_size())
        new_widget = MazeWidget(self.maze)

        await self.maze_widget.remove()
        self.maze_widget = new_widget
        # Mount the new widget into the screen
        await self.mount(self.maze_widget)


if __name__ == "__main__":
    app = MazeApp()
    app.run()
