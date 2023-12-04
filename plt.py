import os
import math
import json
import matplotlib.pyplot as plt
from typing import List, Optional
from transformers.trainer import TRAINER_STATE_NAME
from param_dict import output_dir
# from llmtuner.extras.logging import get_logger
#
#
# logger = get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5) # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: Optional[List[str]] = ["loss"]) -> None:

    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        # if len(metrics) == 0:
        #     logger.warning(f"No metric {key} to plot.")
        #     continue

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(save_dictionary, "training_{}.png".format(key)), format="png", dpi=100)
        print("Figure saved:", os.path.join(save_dictionary, "training_{}.png".format(key)))

def plot_clf_loss(save_dictionary: os.PathLike, keys: Optional[List[str]] = ["loss"]):

    with open(os.path.join(save_dictionary, "clf_state.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data)):
            if key in data[i]:
                steps.append(data[i]["step"])
                metrics.append(data[i][key])

        # if len(metrics) == 0:
        #     logger.warning(f"No metric {key} to plot.")
        #     continue

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(save_dictionary, "clf_training_{}.png".format(key)), format="png", dpi=100)
        print("Figure saved:", os.path.join(save_dictionary, "clf_training_{}.png".format(key)))

def save_state(state):
    # example
    # data = {
    #     "epoch": 0.0,
    #     "learning_rate": 1.998630699712447e-06,
    #     "loss": 2.4518,
    #     "step": 10
    # }

    # log_history = []
    # clf_log_history.append(clf_state)
    with open(os.path.join(output_dir, "clf_state.json"), "a") as f:
        f.write(json.dumps(state, indent=4) + "\n")
        f.close()


# if __name__ == '__main__':
    #print(os.listdir())
    # plot_loss(r"test_results/checkpoint-1000/")
    # plot_clf_loss(r"./")

    # clf_log_history = []
    # clf_state = {
    #     "epoch": 0,
    #     "step": 0,
    #     "loss": 0.001
    # }
    # clf_log_history.append(clf_state)
    # with open(os.path.join("./sft", "clf_state.json"), "a") as f:
    #     f.write(json.dumps(clf_log_history, indent=4) + "\n")
