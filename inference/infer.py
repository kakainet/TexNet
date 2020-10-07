import abc
from dataclasses import dataclass
from enum import Enum
import json

@dataclass
class Box:
    x: int
    y: int
    dx: int
    dy: int


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
    def predict(input) -> Prediction:
        return Operator.ADD, [Box(25, 25, 17, 17), Box(34, 34, 17, 17)]


class InferenceManager:
    def __init__(self, raw_model, operators_data):
        self._model = ModelExample(raw_model)

    def infer(input) -> str:
        pass

    

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

    for f in os.listdir(args.in_dir):
        img = cv2.imread(f)
        print(mgr.infer(img))
