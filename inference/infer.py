import abc
from dataclasses import dataclass
from enum import Enum
import json
import os
import cv2
from typing import List
import argparse

@dataclass
class Box:
    x: int
    y: int
    dx: int
    dy: int


class Operator(Enum):
    ADD: '{0}+{1}'
    SUB: '{0}-{1}'
    MUL: '{0} \\cdot {1}'
    FRAC: '\\frac{{ {0} }}{{ {1} }}'
    SQRT: '\\sqrt{{ {0} }}'
    INT: '\\int {0}'
    PAR: '\\left({0}\\right)'
    FPAR: 'f\\left({0}\\right)'
    FFPAR: 'F\\left({0}\\right)'
    GPAR: 'G\\left({0}\\right)'


@dataclass
class Prediction:
    op: Operator
    bboxes: List[Box]


class Model(abc.ABC):
    def __init__(self, raw_model):
        self._raw_model = raw_model

    @abc.abstractclassmethod
    def predict(input) -> Prediction:
        pass


class ModelExample(Model):
    def predict(self, input) -> Prediction:
        return Operator.ADD, [Box(25, 25, 17, 17), Box(34, 34, 17, 17)]


class InferenceManager:
    def __init__(self, raw_model, operators_data):
        self._model = ModelExample(raw_model)

    def infer(self, input) -> str:
        op, bboxes = self._model.predict(input)
        bboxes = list(map(lambda x: input[x[0]:x[0]+x[2], x[1]:x[1]+x[3]], bboxes))
        return op.format(*list(map(lambda x: self._model.predict(x), bboxes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--operators', help='operators JSON config',
                        required=True, type=str)
    parser.add_argument('--in_dir', help='input inference folder path',
                        required=True, type=str)
    cmd_args = parser.parse_args()

    with open(cmd_args.operators, 'r') as ops_cfg_file:
        ops_config = json.load(ops_cfg_file)

    raw_model = None
    mgr = InferenceManager(raw_model, ops_config)

    for f in os.listdir(cmd_args.in_dir):
        img = cv2.imread(f)
        print(mgr.infer(img))
