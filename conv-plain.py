import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class Model(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv1=L.Convolution2D(3, 8, 5),
            conv2=L.Convolution2D(8, 32, 3),
            conv3=L.Convolution2D(32, 64, 3),
            bn1=L.BatchNormalization(8),
            bn2=L.BatchNormalization(32),
            bn3=L.BatchNormalization(64),
            out=L.Linear(None, 10)
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

gpu_device = 3

model = Model()
chainer.cuda.get_device(gpu_device).use()
model.to_gpu()

# opt = chainer.optimizers.SGD()  # lr=0.01
opt = chainer.optimizers.SGD(lr=0.5)
opt.setup(model)

bs = 20
train_iter = chainer.iterators.SerialIterator(train, bs)
test_iter = chainer.iterators.SerialIterator(test, bs, repeat=False)
updater = chainer.training.StandardUpdater(train_iter, opt, device=gpu_device)
trainer = chainer.training.Trainer(updater, (30, 'epoch'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_device), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport(log_name='plain.log'))
trainer.extend(extensions.PrintReport([
    'epoch', 'main/loss', 'main/acc',
    'validation/main/loss', 'validation/main/acc']))

trainer.run()
