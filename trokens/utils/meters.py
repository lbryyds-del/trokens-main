#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import trokens.utils.logging as logging
import trokens.utils.metrics as metrics
import trokens.utils.misc as misc


logger = logging.get_logger(__name__)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(
        self,
        epoch_iters,
        cfg,
        metric1_name="top1_err",
        metric2_name="top5_err",
        extra_metrics_prefix="multi_loss",
    ):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.multi_loss = MultiLossMeter(cfg.LOG_PERIOD, prefix=extra_metrics_prefix)
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        self.metric1_name = metric1_name
        self.metric2_name = metric2_name
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.multi_loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size, extra_metrics=None):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        dloss, loss = prepare_loss_dict(loss)
        self.multi_loss.add_value(dloss)
        if extra_metrics is not None:
            self.multi_loss.add_value(extra_metrics)
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL and self.metric1_name and top1_err is not None:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
        if not self._cfg.DATA.MULTI_LABEL and self.metric2_name and top5_err is not None:
            self.mb_top5_err.add_value(top5_err)
            self.num_top5_mis += top5_err * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }

        stats.update(self.multi_loss.get_win_median())

        if not self._cfg.DATA.MULTI_LABEL and self.metric1_name:
            stats[self.metric1_name] = self.mb_top1_err.get_win_median()
        if not self._cfg.DATA.MULTI_LABEL and self.metric2_name:
            stats[self.metric2_name] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }

        stats.update(self.multi_loss.get_global_avg())

        if not self._cfg.DATA.MULTI_LABEL:
            avg_loss = self.loss_total / self.num_samples
            if self.metric1_name:
                stats[self.metric1_name] = self.num_top1_mis / self.num_samples
            if self.metric2_name:
                stats[self.metric2_name] = self.num_top5_mis / self.num_samples
            stats["loss"] = avg_loss
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(
        self,
        max_iter,
        cfg,
        metric1_name="top1_err",
        metric2_name="top5_err",
        metric1_mode="min",
        metric2_mode="min",
        extra_metrics_prefix="extra_meter",
    ):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        self.metric1_name = metric1_name
        self.metric2_name = metric2_name
        self.metric1_mode = metric1_mode
        self.metric2_mode = metric2_mode
        self.extra_metrices = MultiLossMeter(cfg.LOG_PERIOD, prefix=extra_metrics_prefix)
        # Best values over the full val set.
        self.best_metric1 = float("-inf") if metric1_mode == "max" else float("inf")
        self.best_metric2 = float("-inf") if metric2_mode == "max" else float("inf")
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.extra_metrices.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def _best_key(self, name, mode):
        if name is None:
            return None
        return f"best_{name}" if mode == "max" else f"min_{name}"

    def _update_best(self, current_best, value, mode):
        return max(current_best, value) if mode == "max" else min(current_best, value)

    def update_stats(self, top1_err, top5_err, mb_size, extra_metrices={}):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        if self.metric1_name and top1_err is not None:
            self.mb_top1_err.add_value(top1_err)
            self.num_top1_mis += top1_err * mb_size
        if self.metric2_name and top5_err is not None:
            self.mb_top5_err.add_value(top5_err)
            self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size
        self.extra_metrices.add_value(extra_metrices)

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL and self.metric1_name:
            stats[self.metric1_name] = self.mb_top1_err.get_win_median()
        if not self._cfg.DATA.MULTI_LABEL and self.metric2_name:
            stats[self.metric2_name] = self.mb_top5_err.get_win_median()
        stats.update(self.extra_metrices.get_win_median())
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            # Single-path episodes have no global classifier predictions.
            # Their Base/Novel/HM AP is accumulated by the episodic evaluator.
            if not self._cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ENABLE:
                stats["map"] = get_map(
                    torch.cat(self.all_preds).cpu().numpy(),
                    torch.cat(self.all_labels).cpu().numpy(),
                )
        else:
            if self.metric1_name:
                metric1_value = self.num_top1_mis / self.num_samples
                self.best_metric1 = self._update_best(
                    self.best_metric1, metric1_value, self.metric1_mode
                )
                stats[self.metric1_name] = metric1_value
                stats[self._best_key(self.metric1_name, self.metric1_mode)] = self.best_metric1
            if self.metric2_name:
                metric2_value = self.num_top5_mis / self.num_samples
                self.best_metric2 = self._update_best(
                    self.best_metric2, metric2_value, self.metric2_mode
                )
                stats[self.metric2_name] = metric2_value
                stats[self._best_key(self.metric2_name, self.metric2_mode)] = self.best_metric2
        stats.update(self.extra_metrices.get_global_avg())

        logging.log_json_stats(stats)


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)


class MultiLossMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size, prefix = None):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.window_size = window_size
        self.prefix = prefix
        self.losses = dict()

    def add_prefix(self, d):
        if self.prefix is None: return d
        return {f"{self.prefix}_{k}":v for k,v in d.items()}

    def reset(self):
        """
        Reset the deque.
        """
        for l in self.losses.values(): l.reset()

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        value = loss_dict_to_float(value)
        for k, v in value.items():
            if k not in self.losses:
                self.losses[k] = ScalarMeter(self.window_size)
            self.losses[k].add_value(v)

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return self.add_prefix({k:v.get_win_median() for k,v in self.losses.items()})

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return self.add_prefix({k:v.get_win_avg() for k,v in self.losses.items()})

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.add_prefix({k:v.get_global_avg() for k,v in self.losses.items()})

def to_float(v):
    if isinstance(v, torch.Tensor): v = v.item()
    return v

def loss_dict_to_float(d):
    ret = {}
    for k,v in d.items():
        ret[k] = to_float(v)
    return ret

def prepare_loss_dict(loss):
    if isinstance(loss ,dict):
        dloss = loss
        if 'loss' in dloss: loss = dloss['loss']
        else: loss = sum(dloss.values())
    else:
        dloss = {}
    dloss = loss_dict_to_float(dloss)
    return dloss, to_float(loss)


##
