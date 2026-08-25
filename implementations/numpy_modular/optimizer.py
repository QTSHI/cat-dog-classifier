class SGD:
    def __init__(self,learning_rate = 0.005):
        self.learning_rate = learning_rate

    def update(self,params,gradients):
        """
        新参数 = 旧参数 - 学习率 * 梯度
        """
        for name in params:
            if name not in gradients:
                raise KeyError(
                    f"找不到参数 {name} 的梯度"
                )

            if (
                params[name].shape!=gradients[name].shape
            ):
                raise ValueError(f"{name} 的参数与梯度形状不一致")

            params[name] -=(
                self.learning_rate
                *gradients[name]
            )

        