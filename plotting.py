"""Classes to handle plotting during the training."""
from __future__ import print_function, division
import math
import cPickle as pickle
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import time

class History(object):
    def __init__(self):
        self.line_groups = OrderedDict()

    @staticmethod
    def from_string(s):
        return pickle.loads(s)

    def to_string(self):
        return pickle.dumps(self, protocol=-1)

    @staticmethod
    def load_from_filepath(fp):
        #return json.loads(open(, "r").read())
        with open(fp, "r") as f:
            history = pickle.load(f)
        return history

    def save_to_filepath(self, fp):
        with open(fp, "w") as f:
            pickle.dump(self, f, protocol=-1)

    def add_group(self, group_name, line_names, increasing=True):
        self.line_groups[group_name] = LineGroup(group_name, line_names, increasing=increasing)

    def add_value(self, group_name, line_name, x, y):
        self.line_groups[group_name].lines[line_name].append(x, y)

    def get_group_names(self):
        return list(self.line_groups.iterkeys())

    def get_groups_increasing(self):
        return [group.increasing for group in self.line_groups.itervalues()]

    def get_max_x(self):
        return max([group.get_max_x() for group in self.line_groups.itervalues()])

    def get_recent_average(self, group_name, line_name, nb_points):
        ys = self.line_groups[group_name].lines[line_name].ys[-nb_points:]
        return np.average(ys)

class LineGroup(object):
    def __init__(self, group_name, line_names, increasing=True):
        self.group_name = group_name
        self.lines = OrderedDict([(name, Line()) for name in line_names])
        self.increasing = increasing

    def get_line_names(self):
        return list(self.lines.iterkeys())

    def get_line_xs(self):
        return [line.xs for line in self.lines.itervalues()]

    def get_line_ys(self):
        return [line.ys for line in self.lines.itervalues()]

    def get_max_x(self):
        return max([max(line.xs) if len(line.xs) > 0 else 0 for line in self.lines.itervalues()])

class Line(object):
    def __init__(self, xs=None, ys=None, datetimes=None):
        self.xs = xs if xs is not None else []
        self.ys = ys if ys is not None else []
        self.datetimes = datetimes if datetimes is not None else []

    def append(self, x, y):
        self.xs.append(x)
        self.ys.append(float(y)) # float to get rid of numpy
        self.datetimes.append(time.time())

class LossPlotter(object):
    def __init__(self, titles, increasing, save_to_fp):
        assert len(titles) == len(increasing)
        n_plots = len(titles)
        self.titles = titles
        self.increasing = dict([(title, incr) for title, incr in zip(titles, increasing)])
        self.colors = ["red", "blue", "cyan", "magenta", "orange", "black"]

        self.nb_points_max = 500
        self.save_to_fp = save_to_fp
        self.start_batch_idx = 0
        self.autolimit_y = False
        self.autolimit_y_multiplier = 5

        #self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        nrows = max(1, int(math.sqrt(n_plots)))
        ncols = int(math.ceil(n_plots / nrows))
        width = ncols * 10
        height = nrows * 10

        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))

        if nrows == 1 and ncols == 1:
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flat

        title_to_ax = dict()
        for idx, (title, ax) in enumerate(zip(self.titles, self.axes)):
            title_to_ax[title] = ax
        self.title_to_ax = title_to_ax

        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.05)

    def plot(self, history):
        for plot_idx, title in enumerate(self.titles):
            ax = self.title_to_ax[title]
            group_name = title
            group_increasing = self.increasing[title]
            if isinstance(history, History):
                group = history.line_groups[title]
                line_names = group.get_line_names()
                line_xs = group.get_line_xs()
                line_ys = group.get_line_ys()
            else:
                line_names = list(history[title].iterkeys())
                line_xs = [[x for (x, y) in history[title][line_name]] for line_name in line_names]
                line_ys = [[y for (x, y) in history[title][line_name]] for line_name in line_names]
            self._plot_group(ax, group_name, group_increasing, line_names, line_xs, line_ys)
        self.fig.savefig(self.save_to_fp)

    def _line_to_xy(self, line_x, line_y, limit_y_min=None, limit_y_max=None):
        point_every = max(1, int(len(line_x) / self.nb_points_max))
        points_x = []
        points_y = []
        curr_sum = 0
        counter = 0
        last_idx = len(line_x) - 1
        for i in range(len(line_x)):
            batch_idx = line_x[i]
            if batch_idx > self.start_batch_idx:
                curr_sum += line_y[i]
                counter += 1
                if counter >= point_every or i == last_idx:
                    points_x.append(batch_idx)
                    y = curr_sum / counter
                    if limit_y_min is not None and limit_y_max is not None:
                        y = np.clip(y, limit_y_min, limit_y_max)
                    elif limit_y_min is not None:
                        y = max(y, limit_y_min)
                    elif limit_y_max is not None:
                        y = min(y, limit_y_max)
                    points_y.append(y)
                    counter = 0
                    curr_sum = 0

        return points_x, points_y

    def _plot_group(self, ax, group_name, group_increasing, line_names, line_xs, line_ys):
        ax.cla()
        ax.grid()

        if self.autolimit_y and any([len(line_xs) > 0 for line_xs in line_xs]):
            min_x = min([np.min(line_x) for line_x in line_xs])
            max_x = max([np.max(line_x) for line_x in line_xs])
            min_y = min([np.min(line_y) for line_y in line_ys])
            max_y = max([np.max(line_y) for line_y in line_ys])

            if group_increasing:
                if max_y > 0:
                    limit_y_max = None
                    limit_y_min = max_y / self.autolimit_y_multiplier
                    if min_y > limit_y_min:
                        limit_y_min = None
            else:
                if min_y > 0:
                    limit_y_max = min_y * self.autolimit_y_multiplier
                    limit_y_min = None
                    if max_y < limit_y_max:
                        limit_y_max = None

            if limit_y_min is not None:
                ax.plot((min_x, max_x), (limit_y_min, limit_y_min), c="purple")

            if limit_y_max is not None:
                ax.plot((min_x, max_x), (limit_y_max, limit_y_max), c="purple")

            # y achse range begrenzen
            yaxmin = min_y if limit_y_min is None else limit_y_min
            yaxmax = max_y if limit_y_max is None else limit_y_max
            yrange = yaxmax - yaxmin
            yaxmin = yaxmin - (0.05 * yrange)
            yaxmax = yaxmax + (0.05 * yrange)
            ax.set_ylim([yaxmin, yaxmax])
        else:
            limit_y_min = None
            limit_y_max = None

        for line_name, line_x, line_y, line_col in zip(line_names, line_xs, line_ys, self.colors):
            x, y = self._line_to_xy(line_x, line_y, limit_y_min=limit_y_min, limit_y_max=limit_y_max)
            ax.plot(x, y, color=line_col, linewidth=1.0)

        ax.set_title(group_name)
