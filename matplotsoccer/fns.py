import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc
from matplotlib.pyplot import cm

spadl_config = {
    "length": 105,
    "width": 68,
    "penalty_box_length": 16.5,
    "penalty_box_width": 40.32,
    "six_yard_box_length": 5.5,
    "six_yard_box_width": 18.3,
    "penalty_spot_distance": 11,
    "goal_width": 7.3,
    "goal_length": 2,
    "origin_x": 0,
    "origin_y": 0,
    "circle_radius": 9.15,
}

zline = 8000

zfield = -5000

zheatmap = 7000

zaction = 9000

ztext = 9500

zvisible = -4000


def _plot_rectangle(x1, y1, x2, y2, ax, color):
    ax.plot([x1, x1], [y1, y2], color=color, zorder=zline)
    ax.plot([x2, x2], [y1, y2], color=color, zorder=zline)
    ax.plot([x1, x2], [y1, y1], color=color, zorder=zline)
    ax.plot([x1, x2], [y2, y2], color=color, zorder=zline)


def field(color="white", figsize=None, ax=None, show=True):
    if color == "white":
        return _field(
            ax=ax,
            linecolor="black",
            fieldcolor="white",
            alpha=1,
            figsize=figsize,
            field_config=spadl_config,
            show=show,
        )
    elif color == "green":
        return _field(
            ax=ax,
            linecolor="white",
            fieldcolor="green",
            alpha=0.4,
            figsize=figsize,
            field_config=spadl_config,
            show=show,
        )
    else:
        raise Exception("Invalid field color")


def _field(
    ax=None,
    linecolor="black",
    fieldcolor="white",
    alpha=1,
    figsize=None,
    field_config=spadl_config,
    show=True,
):
    cfg = field_config

    # Create figure
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    # Pitch Outline & Centre Line
    x1, y1, x2, y2 = (
        cfg["origin_x"],
        cfg["origin_y"],
        cfg["origin_x"] + cfg["length"],
        cfg["origin_y"] + cfg["width"],
    )

    d = cfg["goal_length"]
    rectangle = plt.Rectangle(
        (x1 - 2 * d, y1 - 2 * d),
        cfg["length"] + 4 * d,
        cfg["width"] + 4 * d,
        fc=fieldcolor,
        alpha=alpha,
        zorder=zfield,
    )
    ax.add_patch(rectangle)
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)
    ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1, y2], color=linecolor, zorder=zline)

    # Left Penalty Area
    x1 = cfg["origin_x"]
    x2 = cfg["origin_x"] + cfg["penalty_box_length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["penalty_box_width"] / 2
    y2 = m + cfg["penalty_box_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Right Penalty Area
    x1 = cfg["origin_x"] + cfg["length"] - cfg["penalty_box_length"]
    x2 = cfg["origin_x"] + cfg["length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["penalty_box_width"] / 2
    y2 = m + cfg["penalty_box_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Left 6-yard Box
    x1 = cfg["origin_x"]
    x2 = cfg["origin_x"] + cfg["six_yard_box_length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["six_yard_box_width"] / 2
    y2 = m + cfg["six_yard_box_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Right 6-yard Box
    x1 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"]
    x2 = cfg["origin_x"] + cfg["length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["six_yard_box_width"] / 2
    y2 = m + cfg["six_yard_box_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Left Goal
    x1 = cfg["origin_x"] - cfg["goal_length"]
    x2 = cfg["origin_x"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["goal_width"] / 2
    y2 = m + cfg["goal_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Right Goal
    x1 = cfg["origin_x"] + cfg["length"]
    x2 = cfg["origin_x"] + cfg["length"] + cfg["goal_length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["goal_width"] / 2
    y2 = m + cfg["goal_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Prepare Circles
    mx, my = (cfg["origin_x"] + cfg["length"]) / 2, (cfg["origin_y"] + cfg["width"]) / 2
    centreCircle = plt.Circle(
        (mx, my), cfg["circle_radius"], color=linecolor, fill=False, zorder=zline
    )
    centreSpot = plt.Circle((mx, my), 0.4, color=linecolor, zorder=zline)

    lx = cfg["origin_x"] + cfg["penalty_spot_distance"]
    leftPenSpot = plt.Circle((lx, my), 0.4, color=linecolor, zorder=zline)
    rx = cfg["origin_x"] + cfg["length"] - cfg["penalty_spot_distance"]
    rightPenSpot = plt.Circle((rx, my), 0.4, color=linecolor, zorder=zline)

    # Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    # Prepare Arcs
    r = cfg["circle_radius"] * 2
    leftArc = Arc(
        (lx, my),
        height=r,
        width=r,
        angle=0,
        theta1=307,
        theta2=53,
        color=linecolor,
        zorder=zline,
    )
    rightArc = Arc(
        (rx, my),
        height=r,
        width=r,
        angle=0,
        theta1=127,
        theta2=233,
        color=linecolor,
        zorder=zline,
    )

    # Draw Arcs
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)

    # Tidy Axes
    plt.axis("off")

    # Display Pitch
    if figsize:
        h, w = fig.get_size_inches()
        newh, neww = figsize, w / h * figsize
        fig.set_size_inches(newh, neww, forward=True)

    if show:
        plt.show()

    return ax


def heatmap(
    matrix,
    ax=None,
    figsize=None,
    alpha=1,
    cmap="Blues",
    linecolor="black",
    cbar=False,
    show=True,
):
    if ax is None:
        ax = _field(
            figsize=figsize, linecolor=linecolor, fieldcolor="white", show=False
        )

    cfg = spadl_config
    x1, y1, x2, y2 = (
        cfg["origin_x"],
        cfg["origin_y"],
        cfg["origin_x"] + cfg["length"],
        cfg["origin_y"] + cfg["width"],
    )
    extent = (x1, x2, y1, y2)

    limits = ax.axis()
    imobj = ax.imshow(
        matrix, extent=extent, aspect="auto", alpha=alpha, cmap=cmap, zorder=zheatmap
    )
    ax.axis(limits)

    if cbar:
        # dirty hack
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        colorbar = plt.gcf().colorbar(
            imobj, ax=ax, fraction=0.035, aspect=15, pad=-0.05
        )
        colorbar.minorticks_on()

    plt.axis("scaled")
    if show:
        plt.show()
    return ax


def heatmap_green(matrix, ax=None, figsize=None, show=True):
    if ax is None:
        ax = _field(linecolor="white", fieldcolor="white", show=False)
    return heatmap(matrix, ax=ax, show=show, cmap="RdYlGn_r")


def get_lines(labels):
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    assert labels.ndim == 2

    labels = list([list([str(lb) for lb in label]) for label in labels])
    maxlen = {i: 0 for i in range(len(labels[0]))}
    for label in labels:
        for i, lb in enumerate(label):
            maxlen[i] = max(maxlen[i], len(lb))

    labels = [[lb.ljust(maxlen[i]) for i, lb in enumerate(label)] for label in labels]

    return [" | ".join(label) for label in labels]


def actions_individual(
    data: pd.DataFrame,
    teamView: str,
    label: list,
    labeltitle: list,
    visible_area: pd.DataFrame,
    save_dir: str,
    filename: str,
    color="white",
    legloc="right",
    ax=None,
    figsize=None,
    show=True,
    show_legend=True,
):
    ax = field(ax=ax, color=color, figsize=figsize, show=False)
    fig = plt.gcf()
    figsize, _ = fig.get_size_inches()
    arrowsize = math.sqrt(figsize)

    # SANITIZING INPUT
    location = np.asarray(data.loc[:, "start_x":"end_y"])

    action_type = data["type_name"]
    if action_type is None:
        m, n = location.shape
        action_type = ["pass" for i in range(m)]
        if label is None:
            show_legend = False
    action_type = np.asarray(action_type)

    teams = data["team_name"]
    if teams is None:
        teams = ["Team X" for t in action_type]
    teams = np.asarray(teams)
    assert teams.ndim == 1

    result = data["result_name"] == "success"
    if result is None:
        result = [1 for t in action_type]
    result = np.asarray(result)
    assert result.ndim == 1

    if label is None:
        label = [[t] for t in action_type]
    label = np.asarray(label)
    lines = get_lines(label)
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    assert label.ndim == 2
    indexa = np.asarray([list(range(1, len(label) + 1))]).reshape(-1, 1)
    label = np.concatenate([indexa, label], axis=1)
    if labeltitle is not None:
        labeltitle = list(labeltitle)
        labeltitle.insert(0, "")
        labeltitle = [labeltitle]
        label = np.concatenate([labeltitle, label])
        lines = get_lines(label)
        titleline = lines[0]
        plt.plot(np.NaN, np.NaN, "-", color="none", label=titleline)
        plt.plot(np.NaN, np.NaN, "-", color="none", label="-" * len(titleline))
        lines = lines[1:]
    else:
        lines = get_lines(label)

    m, n = location.shape
    if n != 2 and n != 4:
        raise ValueError("Location must have 2 or 4 columns")
    assert location.shape[1] == 4

    eventmarkers = itertools.cycle(["s", "p", "h"])
    event_types = set(action_type)
    eventmarkerdict = {"pass": "o"}
    for eventtype in event_types:
        if eventtype != "pass":
            eventmarkerdict[eventtype] = next(eventmarkers)

    markersize = figsize * 2

    def get_color(type_name, team):
        if type_name == "dribble":
            return "black"
        elif team == teamView:
            return "blue"
        else:
            return "orange"

    colors = np.array([get_color(ty, te) for ty, te in zip(action_type, teams)])
    blue_n = np.sum(colors == "blue")
    orange_n = np.sum(colors == "orange")
    blue_markers = iter(list(cm.Blues(np.linspace(0.1, 0.8, blue_n))))
    orange_markers = iter(list(cm.Oranges(np.linspace(0.1, 0.8, orange_n))))

    for ty, r, loc, color, line in zip(action_type, result, location, colors, lines):
        [sx, sy, ex, ey] = loc
        if color == "blue":
            c = next(blue_markers)
        elif color == "orange":
            c = next(orange_markers)
        else:
            c = "black"

        if ty == "dribble":
            ax.plot(
                [sx, ex],
                [sy, ey],
                color=c,
                linestyle="--",
                linewidth=2,
                label=line,
                zorder=zaction,
            )
        else:
            ec = "black" if r else "red"
            m = eventmarkerdict[ty]
            ax.plot(
                sx,
                sy,
                linestyle="None",
                marker=m,
                markersize=markersize,
                label=line,
                color=c,
                mec=ec,
                zorder=zaction,
            )

            if abs(sx - ex) > 1 or abs(sy - ey) > 1:
                ax.arrow(
                    sx,
                    sy,
                    ex - sx,
                    ey - sy,
                    head_width=arrowsize,
                    head_length=arrowsize,
                    linewidth=1,
                    fc=ec,
                    ec=ec,
                    length_includes_head=True,
                    zorder=zaction,
                )

    coordinates = data["freeze_frame_360"].iat[2]
    if coordinates is not None:
        players_atx = np.asarray(coordinates[0:22:2])
        players_aty = np.asarray(coordinates[1:23:2])
        players_dfx = np.asarray(coordinates[22:44:2])
        players_dfy = np.asarray(coordinates[23:45:2])
        if (teams[2] == teamView) & (
            action_type[2]
            in [
                "interception",
                "tackle",
                "clearance",
                "foul",
                "keeper_claim",
                "keeper_punch",
                "keeper_save",
                "keeper_pick_up",
            ]
        ):
            dc = "blue"
            ac = "orange"
        elif teams[2] == teamView:
            ac = "blue"
            dc = "orange"
        else:
            ac = "orange" "blue"
            dc = "blue"

    if visible_area is not None:
        visible_area = data["visible_area_360"].iat[2]
        if visible_area == 0:
            visible_area_xs = np.empty(1)
            visible_area_ys = np.empty(1)
        else:
            MAX_LENGTH_VISIBLE = 120
            visible_area_xs = np.asarray(visible_area[::2])
            visible_area_ys = np.asarray(visible_area[1::2])
            if action_type[2] == "take_on":
                visible_area_xs = MAX_LENGTH_VISIBLE - visible_area_xs
            visible_area_xs = visible_area_xs - (
                MAX_LENGTH_VISIBLE - spadl_config["length"]
            )
            if action_type[2] != "take_on":
                visible_area_ys = spadl_config["width"] - visible_area_ys
    else:
        visible_area_xs = np.empty(1)
        visible_area_ys = np.empty(1)

    ax.fill(
        visible_area_xs,
        visible_area_ys,
        color="lightgrey",
        zorder=zvisible,
    )  # coloring the camera visible area
    ax.scatter(
        players_atx,
        players_aty,
        zorder=zaction,
        color=ac,
        s=50,
    )
    ax.scatter(
        players_dfx,
        players_dfy,
        zorder=zaction,
        color=dc,
        s=50,
    )

    # leg = plt.legend(loc=9, prop={"family": "monospace", "size": 12})
    if show_legend:
        if legloc == "top":
            plt.legend(
                bbox_to_anchor=(0.5, 1.05),
                loc="lower center",
                prop={"family": "monospace"},
            )
        elif legloc == "right":
            plt.legend(
                bbox_to_anchor=(1.05, 0.5),
                loc="center left",
                prop={"family": "monospace"},
            )

    if show:
        plt.show()

    if save_dir is not None:
        plt.savefig(save_dir + filename + ".png", bbox_inches="tight")
        plt.close()
        print(save_dir + filename + ".png was plotted")
