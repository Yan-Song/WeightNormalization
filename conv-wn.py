import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer.functions.connection import linear
from chainer.functions.connection import convolution_2d
from chainer.training import extensions


class WNLinear(chainer.Chain):
    def __init__(self, m, n, initial_L=1.0, debug=False):
        super().__init__()
        self.add_param('b', n, dtype='f', initializer=I.Normal())
        self.add_param('L', (), dtype='f', initializer=I.Constant(initial_L))
        self.add_param('V', (n, m), dtype='f', initializer=I.Normal())
        self.debug = debug
        self.times = 0

    def __call__(self, x):
        if self.debug:
            self.times += 1
            if self.times % 100 == 0:
                print("L={}".format(self.L.data))
        norm = xp.linalg.norm(self.V.data)
        W = F.broadcast_to(self.L, self.V.data.shape) * self.V / norm
        return linear.linear(x, W, self.b)


class WNConvolution2D(chainer.Chain):

    def __init__(self, m, n, ksize, stride=1, pad=0,
                 use_cudnn=True, initial_L=1.0, deterministic=False, debug=False):
        super().__init__()
        self.ksize = ksize
        self.stride = (stride, stride)
        self.pad = (pad, pad)
        self.use_cudnn = use_cudnn
        self.deterministic = deterministic

        self.add_param('b', n, dtype='f', initializer=I.Normal())
        self.add_param('L', (), dtype='f', initializer=I.Constant(initial_L))
        self.add_param('V', (n, m, ksize, ksize), dtype='f', initializer=I.Normal())
        self.debug = debug
        self.times = 0

    def __call__(self, x):
        if self.debug:
            self.times += 1
            if self.times % 100 == 0:
                print("L={}".format(self.L.data))
        norm = xp.linalg.norm(self.V.data)
        W = F.broadcast_to(self.L, self.V.data.shape) * self.V / norm
        return convolution_2d.convolution_2d(
            x, W, self.b, self.stride, self.pad, self.use_cudnn,
            deterministic=self.deterministic)


class Model(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv1=WNConvolution2D(3, 8, 5, initial_L=3.0),
            conv2=WNConvolution2D(8, 32, 3, initial_L=3.0),
            conv3=WNConvolution2D(32, 64, 3, initial_L=3.0),
            bn1=L.BatchNormalization(8),
            bn2=L.BatchNormalization(32),
            bn3=L.BatchNormalization(64),
            out=WNLinear(64, 10, initial_L=3.2)
        )

    def __call__(self, x, t):
        h = x  # (3, 32, 32)
        h = F.elu(self.bn1(F.max_pooling_2d(self.conv1(h), 2)))  # (8, 14, 14)
        h = F.elu(self.bn2(F.max_pooling_2d(self.conv2(h), 3)))  # (32, 4, 4)
        h = F.elu(self.bn3(F.max_pooling_2d(self.conv3(h), 2)))  # (64, 1, 1)
        h = self.out(h)

        loss = F.softmax_cross_entropy(h, t)
        acc = F.accuracy(h, t)

        chainer.report({'loss': loss, 'acc': acc}, self)

        return loss


train, test = chainer.datasets.get_cifar10()

gpu_device = 2
xp = chainer.cuda.cupy

model = Model()
chainer.cuda.get_device(gpu_device).use()
model.to_gpu()

opt = chainer.optimizers.SGD(lr=0.5)
opt.setup(model)

bs = 20
train_iter = chainer.iterators.SerialIterator(train, bs)
test_iter = chainer.iterators.SerialIterator(test, bs, repeat=False)
updater = chainer.training.StandardUpdater(train_iter, opt, device=gpu_device)
trainer = chainer.training.Trainer(updater, (30, 'epoch'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_device), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport(log_name='wn.log'))
trainer.extend(extensions.PrintReport([
    'epoch', 'main/loss', 'main/acc',
    'validation/main/loss', 'validation/main/acc']))

trainer.run()
