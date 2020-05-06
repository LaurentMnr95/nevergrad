# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import sqrt, tan, pi
from pathlib import Path

import numpy as np
import nevergrad as ng
import PIL.Image
import os
from nevergrad.common.typetools import ArrayLike
from .. import base
import torch.nn as nn
import torch


class Image(base.ExperimentFunction):
    def __init__(self, problem_name: str = "recovering", index: int = 0, params: dict = {}) -> None:
        """
        problem_name: the type of problem we are working on.
           recovering: we directly try to recover the target image.
        index: the index of the problem, inside the problem type.
           For example, if problem_name is "recovering" and index == 0,
           we try to recover the face of O. Teytaud.
        """

        # Storing high level information.
        self.domain_shape = (256, 256, 3)
        self.problem_name = problem_name
        self.index = index

        # Storing data necessary for the problem at hand.
        assert problem_name == "recovering"  # For the moment we have only this one.
        assert index == 0  # For the moment only 1 target.
        # path = os.path.dirname(__file__) + "/headrgb_olivier.png"
        path = Path(__file__).with_name("headrgb_olivier.png")
        image = PIL.Image.open(path).resize((self.domain_shape[0], self.domain_shape[1]), PIL.Image.ANTIALIAS)
        self.data = np.asarray(image)[:, :, :3]  # 4th Channel is pointless here, only 255.

        array = ng.p.Array(init=128 * np.ones(self.domain_shape), mutable_sigma=True, )
        array.set_mutation(sigma=35)
        array.set_bounds(lower=0, upper=255.99, method="clipping", full_range_sampling=True)
        max_size = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
        array.set_recombination(ng.p.mutation.Crossover(axis=(0, 1), max_size=max_size)).set_name("")  # type: ignore

        if problem_name == "adversarial":
            self.targeted = params["targeted"] if ("targeted" in params) else False
            self.epsilon = params["epsilon"] if ("epsilon" in params) else 0.05
            self.image_size = params["image_size"]
            self.domain_shape = (self.image_size, self.image_size, 3)
            self.image = params["image"]
            self.label = params["label"]
            self.classifier = params["classifier"]
            # TODO: changes params
            array = ng.p.Array(init=128 * np.ones(self.domain_shape), mutable_sigma=True, )
            array.set_mutation(sigma=35)
            array.set_bounds(lower=-self.epsilon, upper=self.epsilon, method="clipping", full_range_sampling=True)
            max_size = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
            array.set_recombination(ng.p.mutation.Crossover(axis=(0, 1),
                                                            max_size=max_size)).set_name("")  # type: ignore

        super().__init__(self._loss, array)
        self.register_initialization(problem_name=problem_name, index=index)
        self._descriptors.update(problem_name=problem_name, index=index)

    def _loss(self, x: np.ndarray) -> float:
        if self.problem_name == "recovering":
            x = np.array(x, copy=False).ravel()
            x = x.reshape(self.domain_shape)
            assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
            # Define the loss, in case of recovering: the goal is to find the target image.
            assert self.index == 0
            value = np.sum(np.fabs(np.subtract(x, self.data)))
            
        if self.problem_name == "adversarial":
            x = torch.Tensor(x)
            image_adv = self.image + x
            image_adv = image_adv.view(1, image_adv.shape).cuda()
            output_adv = self.classifier(image_adv)
            if self.targeted:
                value = nn.CrossEntropyLoss(output_adv, self.label)
            else:
                value = -nn.CrossEntropyLoss(output_adv, self.label)
            value = value.item()
        return value
