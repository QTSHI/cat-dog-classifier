from collections import OrderedDict
from pathlib import Path

import numpy as np

from layers import(
    Affine,
    Convolution,
    Flatten,
    MaxPooling,
    ReLU,
    SoftmaxAndLoss
)

class CatVsDog:
    def __init__(self,seed = 42):
        rng = np.random.default_rng(seed)

        input_channels = 3
        input_height = 128
        input_width = 128

        filter1_number = 8
        filter2_number =16
        filter_size = 3

        final_height = input_height//4
        final_width = input_width//4

        affine_input_size = filter2_number*final_height*final_width

        self.params={}
        #第一个卷积层权重 
        self.params["W1"] = (
            rng.standard_normal(
                (
                    filter1_number,
                    input_channels,
                    filter_size,
                    filter_size,
                )
            )
            *np.sqrt(
                2.0
                /(
                    input_channels
                    *filter_size
                    *filter_size
                )
            )
        ).astype(np.float32)

        self.params["b1"] = np.zeros(
            filter1_number,
            dtype=np.float32,
        )
        #第二个卷积层权重
        self.params["W2"] = (
            rng.standard_normal(
                (
                    filter2_number,
                    filter1_number,
                    filter_size,
                    filter_size,
                )
            )
            * np.sqrt(
                2.0
                / (
                    filter1_number
                    * filter_size
                    * filter_size
                )
            )
        ).astype(np.float32)

        self.params["b2"] = np.zeros(
            filter2_number,
            dtype=np.float32,
        )
        #全链接层权重
        self.params["W3"] = (
            rng.standard_normal(
                (affine_input_size, 2)
            )
            * np.sqrt(1.0 / affine_input_size)
        ).astype(np.float32)

        self.params["b3"] = np.zeros(
            2,
            dtype=np.float32,
        )

        self.layers = OrderedDict()

        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            stride=1,
            pad=1,
        )

        self.layers["ReLU1"] = ReLU()

        self.layers["Pool1"] = MaxPooling(
            pool_height=2,
            pool_width=2,
            stride=2,
        )

        self.layers["Conv2"] = Convolution(
            self.params["W2"],
            self.params["b2"],
            stride=1,
            pad=1,
        )

        self.layers["ReLU2"] = ReLU()

        self.layers["Pool2"] = MaxPooling(
            pool_height=2,
            pool_width=2,
            stride=2,
        )

        self.layers["Flatten"] = Flatten()

        self.layers["Affine"] = Affine(
            self.params["W3"],
            self.params["b3"],
        )

        self.loss_layer = SoftmaxAndLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self,x,lables):
        scores = self.predict(x)

        return self.loss_layer.forward(scores,lables)

    def accuracy(self,x,lables):
        scores = self.predict(x)
        predictions = np.argmax(scores,axis=1)

        return np.mean(predictions == lables)

    def gradient(self,x,lables):
        self.loss(x,lables)
        dout = self.loss_layer.backward()

        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        gradients = {
            "W1": self.layers["Conv1"].dweight,
            "b1": self.layers["Conv1"].dbias,
            "W2": self.layers["Conv2"].dweight,
            "b2": self.layers["Conv2"].dbias,
            "W3": self.layers["Affine"].dweight,
            "b3": self.layers["Affine"].dbias,
        }

        return gradients
    
    def load_parameters(self, model_path):
    

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"找不到模型文件：{model_path}"
            )

        with np.load(model_path) as saved_params:
            for name in self.params:
                if name not in saved_params:
                    raise KeyError(
                        f"模型文件缺少参数：{name}"
                    )

                loaded_parameter = saved_params[name]

                if (
                    loaded_parameter.shape
                    != self.params[name].shape
                ):
                    raise ValueError(
                        f"{name} 形状不一致："
                        f"模型需要 {self.params[name].shape}，"
                        f"文件中为 {loaded_parameter.shape}"
                    )

            # [...] 表示原地复制到现有数组
                self.params[name][...] = (
                    loaded_parameter
                )
    