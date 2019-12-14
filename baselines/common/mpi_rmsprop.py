import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
import time
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class MpiRMSProp(object):
    def __init__(self, var_list, *, decay=0.9, epsilon=1e-10, scale_grad_by_procs=True, comm=None):
        # shape (3, )
        self.var_list = var_list
        self.decay = decay
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        # shape(2, 5)
        # U.numel(v) for v in var_list 对于每一个variable返回他们乘积 int值
        size = sum(U.numel(v) for v in var_list)
        self.cache = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None and MPI is not None else comm

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        if self.comm is not None:
            globalg = np.zeros_like(localg)
            self.comm.Allreduce(localg, globalg, op=MPI.SUM)
            if self.scale_grad_by_procs:
                globalg /= self.comm.Get_size()
        else:
            globalg = np.copy(localg)

        # mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
        # mom = momentum * mom{t-1} + learning_rate * g_t / sqrt(mean_square + epsilon)
        # delta = - mom
       
        self.t += 1
        self.cache  = self.decay * self.cache + (1 - self.decay)  * (globalg * globalg)
        step = (- stepsize) * globalg / (np.sqrt(self.cache) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        if self.comm is None:
            return
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm is None:
            return
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

@U.in_session
def test_RMSprop():
    np.random.seed(0)
    tf.set_random_seed(0)
    # shape 为(3,) 的矩阵
    a = tf.Variable(np.random.randn(3).astype('float32'))
    # shape 为(2, 5) 的矩阵
    b = tf.Variable(np.random.randn(2,5).astype('float32'))
    # 求对a的平方和sin(b)的平均值
    loss = tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    # 步长1e-2
    stepsize = 1e-2
    # 用tensorflow自带的优化
    update_op = tf.train.RMSPropOptimizer(stepsize).minimize(loss)
    # 就是把value传进placeholder
    do_update = U.function([], loss, updates=[update_op])

    # 初始化参数
    tf.get_default_session().run(tf.global_variables_initializer())
    losslist_ref = []
    time_start = time.time()
    for i in range(1000):
        # 做优化，并记录打印
        l = do_update()
        print(i, l)
        losslist_ref.append(l)
    time_end = time.time()
    print('time cost',time_end-time_start,'s')
    print("------------------------------------------------------")

    # 获得随机种子
    tf.set_random_seed(0)
    # 初始化参数
    tf.get_default_session().run(tf.global_variables_initializer())

    var_list = [a,b]
    # 就是把value传进placeholder
    lossandgrad = U.function([], [loss, U.flatgrad(loss, var_list)])
    rmsprop = MpiRMSProp(var_list)

    losslist_test = []
    time_start1 = time.time()
    for i in range(1000):
        l,g = lossandgrad()
        rmsprop.update(g, stepsize)
        print(i,l)
        losslist_test.append(l)
    time_end1 = time.time()
    print('time cost',time_end1-time_start1,'s')
    #np.testing.assert_allclose(np.array(losslist_ref), np.array(losslist_test), atol=1e-4)


if __name__ == '__main__':
    test_RMSprop()
