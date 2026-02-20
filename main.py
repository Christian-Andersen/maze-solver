#!/usr/bin/env python3
import sys
import time

import cv2
import numpy as np

sys.setrecursionlimit(100000)


class Node:
    def __init__(
        self, name: tuple[int, int], parent: "Node | None" = None, children: list["Node"] | None = None
    ) -> None:
        self.name = name
        self.parent = parent
        if children:
            self.children = children


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
        visted = np.zeros(self.size, dtype=bool)
        visted[self.start] = True
        visted_count = 1
        visted_total = self.size[0] * self.size[1]
        path = []
        loc = self.start
        while visted_count != visted_total:
            new_locs = []
            for dx, dy in difs:
                new_loc = (loc[0] + dx, loc[1] + dy)
                try:
                    if visted[new_loc]:
                        continue
                except IndexError:
                    continue
                if (new_loc[0] < 0) or (new_loc[1] < 0):
                    continue
                new_locs.append(new_loc)
            if new_locs:
                new_loc = new_locs[rng.integers(len(new_locs))]
                self.wall_check(loc, new_loc, delete_wall=True)
                visted[new_loc] = True
                visted_count += 1
                if len(new_locs) > 1:
                    path.append(loc)
                loc = new_loc
            else:
                loc = path.pop()

    def get_image(self, squares: list[tuple[int, int]] | None = None) -> np.ndarray:
        if squares is None:
            squares = []
        image = np.zeros((10 + 100 * self.size[1], 10 + 100 * self.size[1]), dtype=np.uint8)
        self._draw_squares(image, squares)
        self._draw_corners(image)
        self._draw_walls(image)
        return image

    def _draw_squares(self, image: np.ndarray, squares: list[tuple[int, int]]) -> None:
        for square in squares:
            image[
                (10 + square[0] * 100) : (100 + square[0] * 100), (10 + square[1] * 100) : (100 + square[1] * 100)
            ] = 128

    def _draw_corners(self, image: np.ndarray) -> None:
        for row_idx in range(self.size[0] + 1):
            for col_idx in range(self.size[1] + 1):
                image[(row_idx * 100) : (10 + row_idx * 100), (col_idx * 100) : (10 + col_idx * 100)] = 255

    def _draw_walls(self, image: np.ndarray) -> None:
        for row_idx, row in enumerate(self.walls[0]):
            for col_idx, col in enumerate(row):
                if col:
                    image[(100 * row_idx) : (10 + 100 * row_idx), (10 + 100 * col_idx) : (100 + 100 * col_idx)] = 255
        for row_idx, row in enumerate(self.walls[1]):
            for col_idx, col in enumerate(row):
                if col:
                    image[(10 + 100 * row_idx) : (100 + 100 * row_idx), (100 * col_idx) : (10 + 100 * col_idx)] = 255

    def show_image(self, squares: list[tuple[int, int]]) -> None:
        cv2.imshow("Maze", self.get_image(squares))
        cv2.waitKey()


class DepthFirstSolve:
    def __init__(self, maze: Maze) -> None:
        self.maze = maze

    def solve(self) -> list[tuple[int, int]]:
        difs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        ways: dict[tuple[int, int], list[tuple[int, int]]] = {}
        solution = [self.maze.start]
        while True:
            loc = solution[-1]
            if loc not in ways:
                ways[loc] = []
                for dx, dy in difs:
                    new_loc = (loc[0] + dx, loc[1] + dy)
                    try:
                        if new_loc == solution[-2]:
                            continue
                    except IndexError:
                        pass
                    if (
                        (new_loc[0] < 0)
                        or (new_loc[1] < 0)
                        or (new_loc[0] >= self.maze.size[0])
                        or (new_loc[1] >= self.maze.size[1])
                    ):
                        continue
                    if not self.maze.wall_check(loc, new_loc):
                        if new_loc == self.maze.end:
                            return [*solution, new_loc]
                        ways[loc].append(new_loc)
            if ways[loc]:
                solution.append(ways[loc].pop())
            else:
                solution.pop()


class WidthFirstSolve:
    def __init__(self, maze: Maze) -> None:
        self.maze = maze
        self.current_node = Node(maze.start)

    def solve(self) -> list[tuple[int, int]]:
        nodes = [self.current_node]
        visted = {self.current_node.name}
        while True:
            new_nodes = []
            for node in nodes:
                loc = node.name
                for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    new_loc = (loc[0] + dx, loc[1] + dy)
                    if new_loc in visted:
                        continue
                    if (
                        (new_loc[0] < 0)
                        or (new_loc[1] < 0)
                        or (new_loc[0] >= self.maze.size[0])
                        or (new_loc[1] >= self.maze.size[1])
                    ):
                        continue
                    if self.maze.wall_check(loc, new_loc):
                        continue
                    if new_loc == self.maze.end:
                        visted.add(new_loc)
                        solution = [new_loc]
                        curr: Node | None = node
                        while curr is not None:
                            solution.append(curr.name)
                            curr = curr.parent
                        return list(reversed(solution))
                    new_nodes.append(Node(new_loc, node))
                    visted.add(new_loc)
            nodes = new_nodes


def main() -> None:
    n = 500
    times: dict[str, list[float]] = {"Maze Making": [], "Depth First": [], "Width First": []}
    print(len(times["Maze Making"]) + 1)
    start_time = time.time()
    maze = Maze((n, n))
    times["Maze Making"].append(time.time() - start_time)
    sols = []
    start_time = time.time()
    sols.append(tuple(DepthFirstSolve(maze).solve()))
    times["Depth First"].append(time.time() - start_time)
    start_time = time.time()
    sols.append(tuple(WidthFirstSolve(maze).solve()))
    times["Width First"].append(time.time() - start_time)
    if len(set(sols)) != 1:
        for sol in sols:
            print(sol)
        msg = "ALL SOLUTIONS DO NOT MATCH"
        print(msg)
        raise RuntimeError(msg)
    for key, value in times.items():
        print(f"{key} - {(sum(value) / len(value)):.2f}s")


if __name__ == "__main__":
    main()
