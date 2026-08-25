import numpy as np

def im2col(input_data,filter_height,filter_width,stride = 1,pad = 0,):
    N,C,H,W = input_data.shape
    output_height =(H + 2*pad - filter_height)//stride + 1 
    output_width =(W + 2*pad - filter_width)//stride + 1 

    padded_image = np.pad(
        input_data,
        (
            (0,0),
            (0,0),
            (pad,pad),
            (pad,pad),
        ),
        mode="constant",
    )

    col = np.zeros(
        (
            N,C,filter_height,filter_width,output_height,output_width
        ),
        dtype=input_data.dtype
    )
    for y in range(filter_height):
        y_max = y + stride*output_height

        for x in range(filter_width):
            x_max = x + stride*output_width

            col[:,:,y,x,:,:] = padded_image[:,:,y:y_max:stride,x:x_max:stride,]

    col = col.transpose(0,4,5,1,2,3)
    col = col.reshape(N * output_height * output_width,-1)

    return col

def col2im(
    col,
    input_shape,
    filter_height,
    filter_width,
    stride=1,
    pad=0,
):
    """
    把 im2col 格式的数据恢复成图片形状。

    注意：卷积窗口重叠的区域会进行梯度累加。
    """

    N, C, H, W = input_shape

    output_height = (
        H + 2 * pad - filter_height
    ) // stride + 1

    output_width = (
        W + 2 * pad - filter_width
    ) // stride + 1

    # 恢复 im2col 展开前的多维排列
    col = col.reshape(
        N,
        output_height,
        output_width,
        C,
        filter_height,
        filter_width,
    )

    col = col.transpose(0, 3, 4, 5, 1, 2)

    # stride - 1 是为了容纳最后一个滑动窗口
    image = np.zeros(
        (
            N,
            C,
            H + 2 * pad + stride - 1,
            W + 2 * pad + stride - 1,
        ),
        dtype=col.dtype,
    )

    for y in range(filter_height):
        y_max = y + stride * output_height

        for x in range(filter_width):
            x_max = x + stride * output_width

            # 使用 +=，因为不同窗口可能覆盖同一个像素
            image[
                :,
                :,
                y:y_max:stride,
                x:x_max:stride,
            ] += col[:, :, y, x, :, :]

    # 去掉之前添加的 padding
    return image[
        :,
        :,
        pad:H + pad,
        pad:W + pad,
    ]


class Convolution:
    def __init__(self,weight,bias,stride = 1,pad = 0):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.pad = pad

        self.x = None 
        self.col = None
        self.col_weight = None 

        self.dweight = None
        self.dbias = None

    def forward(self,x):
        filter_number,C,FH,FW = self.weight.shape
        N,C,H,W, = x.shape

        output_height = (H + 2*self.pad -FH)//self.stride + 1
        output_width = (W + 2*self.pad -FW)//self.stride + 1

        col = im2col(x,FH,FW,self.stride,self.pad)

        col_weight = self.weight.reshape(filter_number,-1).T

        out = col @ col_weight + self.bias

        out =out.reshape(N,output_height,output_width,filter_number)
        out = out.transpose(0,3,1,2)

        self.x = x
        self.col = col
        self.col_weight = col_weight

        return out

    def backward(self,dout):
        filter_number, C, FH, FW = self.weight.shape

        # 将输出梯度转换成矩阵乘法需要的形状
        dout = dout.transpose(0, 2, 3, 1)
        dout = dout.reshape(-1, filter_number)

        # 偏置梯度：对所有样本和空间位置求和
        self.dbias = np.sum(dout, axis=0)

        # 权重梯度
        dweight = self.col.T @ dout

        # 恢复成卷积核原来的形状
        self.dweight = dweight.T.reshape(
            filter_number,
            C,
            FH,
            FW,
        )

        # 传给输入的梯度
        dcol = dout @ self.col_weight.T

        dx = col2im(
            dcol,
            self.x.shape,
            FH,
            FW,
            self.stride,
            self.pad,
        )

        return dx

class MaxPooling:
    def __init__(self,pool_height =2,pool_width=2,stride = 2,pad=0):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.pad = pad

        self.x =None
        self.argmax = None

    def forward(self,x):
        N,C,H,W = x.shape

        output_height = (H + 2*self.pad - self.pool_height)//self.stride + 1
        output_width = (W + 2*self.pad - self.pool_width)//self.stride + 1

        col = im2col(x,self.pool_height,self.pool_width,self.stride,self.pad)

        pool_size =(self.pool_height*self.pool_width)

        col = col.reshape(-1,pool_size)

        argmax = np.argmax(col,axis=1)

        out = np.max(col,axis=1)

        out = out.reshape(N,output_height,output_width,C)

        out = out.transpose(0,3,1,2)

        self.x = x
        self.argmax = argmax

        return out
    
    def backward(self,dout):
        dout = dout.transpose(0,2,3,1)

        pool_size =(self.pool_height*self.pool_width)

        dmax = np.zeros(
            (
                self.argmax.size,
                pool_size,
            ),
            dtype=dout.dtype,
        )
        dmax[
            np.arange(self.argmax.size),
            self.argmax,
        ] = dout.reshape(-1)

        N,C,H,W =self.x.shape

        output_height = (
            H + 2 * self.pad - self.pool_height
        ) // self.stride + 1

        output_width = (
            W + 2 * self.pad - self.pool_width
        ) // self.stride + 1

        dcol = dmax.reshape(N*output_height*output_width,C*pool_size)

        dx = col2im(dcol,self.x.shape,self.pool_height,self.pool_width,self.stride,self.pad)

        return dx

class ReLU:

    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = x <=0

        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self,dout):

        dx = dout.copy()
        dx[self.mask] = 0

        return dx

class Flatten:

    def __init__(self):
        self.original_shape = None 

    def forward(self,x):
        self.original_shape = x.shape
        batch_size = x.shape[0]

        return x.reshape(batch_size,-1)

    def backward(self,dout):
        return dout.reshape(self.original_shape)

class Affine:

    def __init__(self,weight,bias):
        self.weight = weight
        self.bias = bias

        self.x=None

        self.dweight = None 
        self.dbias = None

    def forward(self,x):
        self.x = x
        out = x @ self.weight + self.bias

        return out
    
    def backward(self,dout):
        dx = dout @ self.weight.T
        self.dweight = self.x.T @ dout
        self.dbias = np.sum(dout,axis=0)

        return dx

def softmax(x):
    if x.ndim ==1:
        shifted_x = x-np.max(x)
        exp_x = np.exp(shifted_x)

        return exp_x / np.sum(exp_x)

    shifted_x = x -np.max(x,axis=1,keepdims=True)
    exp_x=np.exp(shifted_x)
    probabilities = exp_x /np.sum(exp_x,axis=1,keepdims=True)

    return probabilities

def cross_entropy_loss(probabilities,labels):
    if probabilities.ndim ==1:
        probabilities = probabilities.reshape(1,-1)

    labels = np.asarray(labels,dtype=np.int64).reshape(-1)

    batch_size = probabilities.shape[0]

    if len(labels) !=batch_size:
        raise ValueError("标签数量必须与样本数量相同")

    correct_probabilities = probabilities[np.arange(batch_size),labels]

    loss = -np.mean(np.log(correct_probabilities +1e-7))

    return loss

class SoftmaxAndLoss:
    def __init__(self):
        self.loss = None 
        self.probabilities =None
        self.lables = None

    def forward(self,scores,lables):
        self.lables = np.asarray(lables,dtype=np.int64).reshape(-1)

        self.probabilities = softmax(scores)
        self.loss = cross_entropy_loss(self.probabilities,self.lables)

        return self.loss

    def backward(self,dout =1.0):
        batch_size = self.lables.shape[0]

        dx =self.probabilities.copy()

        dx[np.arange(batch_size),self.lables] -=1

        dx *= dout /batch_size

        return dx

    
