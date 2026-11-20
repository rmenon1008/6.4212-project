"""The Push-T task: push a T-shaped block into a goal pose with a round pusher.

Modified from the Push-T environment in the Colab notebook provided in
https://github.com/real-stanford/diffusion_policy/blob/main/README.md
(itself adapted from Implicit Behavior Cloning, https://implicitbc.github.io/).
The physics is unchanged, so the authors' demonstrations and pretrained policy
apply as is.  What is gone is everything that only drew it -- pygame, OpenCV,
the gym wrapper -- in favor of matplotlib.

MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pymunk
import shapely.geometry as sg
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Polygon

######################################################################
## Code for nobody but Python to read
######################################################################

# The workspace is a 512 x 512 box (units are pixels of the original renderer).
# y increases downward, as in an image; plots invert their y-axis to match.
WORKSPACE = 512
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])  # x, y, theta
AGENT_RADIUS = 15
SIM_HZ = 100
CONTROL_HZ = 10
K_P, K_V = 100, 20  # the pusher's PD controller
SUCCESS_COVERAGE = 0.95


def _tee_vertices(scale=30, length=4):
    """The T's two boxes, in its body frame: the crossbar, then the stem."""
    bar = [
        (-length * scale / 2, scale),
        (length * scale / 2, scale),
        (length * scale / 2, 0),
        (-length * scale / 2, 0),
    ]
    stem = [
        (-scale / 2, scale),
        (-scale / 2, length * scale),
        (scale / 2, length * scale),
        (scale / 2, scale),
    ]
    return bar, stem


def apply_transform(X_AB, p_BC):
    """p_AC = X_AB p_BC: the position of point C in frame A, given in frame B.

    X_AB is the pose of frame B in frame A, as a planar (x, y, theta); for
    the T in the world, X_WT is an observation's last three entries.  The T
    frame's origin is the middle of the crossbar's outer edge; its x-axis runs
    along the crossbar and its y-axis down the stem.  (The world's y-axis
    points down the screen, as in an image.)  p_BC may be one point, (2,), or
    many, (N, 2).
    """
    x, y, theta = X_AB
    c, s = np.cos(theta), np.sin(theta)
    R_AB = np.array([[c, -s], [s, c]])
    return np.array([x, y]) + np.asarray(p_BC, dtype=float) @ R_AB.T


def tee_polygons(X_WT):
    """The T's two boxes, as world-frame vertex arrays, at pose X_WT."""
    return [apply_transform(X_WT, box) for box in _tee_vertices()]


def coverage(block_pose, goal_pose=GOAL_POSE):
    """Fraction of the goal T's area that the block covers."""
    block = sg.MultiPolygon([sg.Polygon(p) for p in tee_polygons(block_pose)])
    goal = sg.MultiPolygon([sg.Polygon(p) for p in tee_polygons(goal_pose)])
    return goal.intersection(block).area / goal.area


class PushTEnv:
    """Push-T, stepped at CONTROL_HZ.

    The state and observation are both
        [agent_x, agent_y, block_x, block_y, block_angle],
    and an action is the position the pusher's PD controller drives toward.
    """

    def __init__(self, state=None, seed=None):
        self.reset(state, seed)

    def reset(self, state=None, seed=None):
        """Start from `state`, or a random one (the seed picks which)."""
        self._setup()
        if state is None:
            # The legacy RandomState keeps seeds compatible with the dataset.
            rs = np.random.RandomState(seed=seed)
            state = np.array(
                [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ]
            )
        self.set_state(state)
        return self.observation()

    def set_state(self, state):
        state = [float(s) for s in state]
        self.agent.position = state[:2]
        # Setting the angle rotates about the center of mass, so it goes first.
        self.block.angle = state[4]
        self.block.position = state[2:4]
        self.space.step(1.0 / SIM_HZ)

    def observation(self):
        return np.array(
            [*self.agent.position, *self.block.position, self.block.angle % (2 * np.pi)]
        )

    def step(self, action):
        """Track `action` for one control period; return (obs, coverage, done)."""
        dt = 1.0 / SIM_HZ
        action = pymunk.Vec2d(*action)
        for _ in range(SIM_HZ // CONTROL_HZ):
            acceleration = K_P * (action - self.agent.position) - K_V * self.agent.velocity
            self.agent.velocity += acceleration * dt
            self.space.step(dt)
        c = coverage([*self.block.position, self.block.angle])
        return self.observation(), c, c > SUCCESS_COVERAGE

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        walls = [
            ((5, 506), (5, 5)),
            ((5, 5), (506, 5)),
            ((506, 5), (506, 506)),
            ((5, 506), (506, 506)),
        ]
        self.space.add(*[pymunk.Segment(self.space.static_body, a, b, 2) for a, b in walls])

        self.agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.agent.position = (256, 400)
        self.space.add(self.agent, pymunk.Circle(self.agent, AGENT_RADIUS))

        bar, stem = _tee_vertices()
        mass = 1
        # (Yes, the bar's inertia twice.  It is what the demonstrations used.)
        inertia = 2 * pymunk.moment_for_poly(mass, vertices=bar)
        self.block = pymunk.Body(mass, inertia)
        shapes = [pymunk.Poly(self.block, bar), pymunk.Poly(self.block, stem)]
        self.block.center_of_gravity = (
            shapes[0].center_of_gravity + shapes[1].center_of_gravity
        ) / 2
        self.block.position = (256, 300)
        self.space.add(self.block, *shapes)


######################################################################
## Drawing
######################################################################

GOAL_COLOR = "#90ee90"  # LightGreen
BLOCK_COLOR = "#778899"  # LightSlateGray
AGENT_COLOR = "#4169e1"  # RoyalBlue


def setup_axes(ax):
    ax.set_xlim(0, WORKSPACE)
    ax.set_ylim(WORKSPACE, 0)  # y down, as in an image
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_state(ax, state, goal=True, alpha=1.0):
    """Draw the goal, the block and the pusher for state."""
    if goal:
        for p in tee_polygons(GOAL_POSE):
            ax.add_patch(Polygon(p, color=GOAL_COLOR, zorder=1))
    for p in tee_polygons(state[2:5]):
        ax.add_patch(
            Polygon(p, facecolor=BLOCK_COLOR, edgecolor="#5a6773", alpha=alpha, zorder=2)
        )
    ax.add_patch(Circle(state[:2], AGENT_RADIUS, color=AGENT_COLOR, alpha=alpha, zorder=4))


def draw_paths(ax, paths, cmap="plasma", linewidth=1.5, alpha=0.8, duration=None):
    """Draw each (T, 2) pusher path, colored by time from dark to bright.
    The colors span `duration` steps, or by default the path's own length."""
    for path in paths:
        path = np.asarray(path)
        segments = np.stack([path[:-1], path[1:]], axis=1)
        lines = LineCollection(
            segments,
            cmap=cmap,
            array=np.arange(len(segments)) / max(duration or len(segments), 1),
            norm=Normalize(0, 1),
            linewidth=linewidth,
            alpha=alpha,
            zorder=3,
        )
        ax.add_collection(lines)


PLAN_COLOR = "#d62728"


def make_animation(
    states, trail=True, interval=1000 // CONTROL_HZ, ncols=4, plans=None
):
    """A matplotlib animation, in real time, of a (T, 5) sequence of states,
    or of N of them, (N, T, 5), side by side in a grid.

    plans: optionally, (N, T, S, H, 2) pusher paths to draw at each frame,
    such as the S action sequences a policy was choosing among."""
    import matplotlib.pyplot as plt

    states = np.asarray(states)
    if states.ndim == 2:
        states = states[np.newaxis]
        plans = None if plans is None else np.asarray(plans)[np.newaxis]
    n, steps = states.shape[:2]
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows + 0.4), squeeze=False
    )
    for ax in axes.flat[n:]:
        ax.axis("off")

    def frame(t):
        for i, (ax, s) in enumerate(zip(axes.flat, states)):
            ax.clear()
            setup_axes(ax)
            draw_state(ax, s[t])
            if plans is not None:
                ax.add_collection(
                    LineCollection(
                        plans[i, t], color=PLAN_COLOR, linewidth=0.8, alpha=0.4,
                        zorder=5,
                    )
                )
            if trail and t > 0:
                draw_paths(ax, [s[: t + 1, :2]], duration=steps - 1)
        fig.suptitle(f"t = {t / CONTROL_HZ:.1f} s")

    fig.tight_layout()
    return FuncAnimation(fig, frame, frames=steps, interval=interval)
