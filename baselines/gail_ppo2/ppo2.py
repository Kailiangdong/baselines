import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import Dataset, dataset, explained_variance, fmt_row, zipsame, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.mpi_adam import MpiAdam
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.gail_ppo2.runner import Runner
from contextlib import contextmanager
import numpy as np
import baselines.common.tf_util as U
from baselines.common import colorize

def constfn(val):
    def f(_):
        return val
    return f

def learn(env, network, total_timesteps, reward_giver, expert_dataset ,g_step , d_step, mpi_rank_weight = 1, 
            eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval = 1, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, comm=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    reward_giver                      reward given by discriminator

    d_step                            discriminator training step

    g_step                            generator training step

    dataset                           expert dataset

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''
    # 设置全局种子
    set_global_seeds(seed)
    # 设置 lr得function, 为了后面fraction

    # if isinstance(lr, float): lr = constfn(lr)
    # else: assert callable(lr)
    # if isinstance(cliprange, float): cliprange = constfn(cliprange)
    # else: assert callable(cliprange)

    # set mpi
    nworkers = MPI.COMM_WORLD.Get_size()
    #rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # 把总步数int化
    total_timesteps = int(total_timesteps)
    # 根据网络建立policy pi， 这里应该不用变
    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env， 获得环境数量,这就是并行multi processing
    # 一般情况下就是1
    nenvs = env.num_envs
    # Get state_space and action_space， 获得动作和状态的空间
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size， 获得batch size等会儿每个个的batch size传进去的
    # nstep = timesteps per actor per update
    # nbatch 总的batch size
    # nenvs  = 8 , nsteps = 2048, nbatch  = 16384
    nbatch = nenvs * nsteps
    # 总共要train的batch数量
    # nminibatches 每次小训练的次数
    nbatch_train = nbatch // nminibatches
    # 看是不是mpi根程序，假如是mpi但是rank = 2，3，4也不是根程序
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    # 初始化model
    if model_fn is None:
        from baselines.gail_ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)
    
    # 这句话可以无视，每次我们都创建model
    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    # runner里面就是采样
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, reward_giver = reward_giver)
    # 建立一个双向链表
    epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    # 开始计时，一种浮点数计时方式
    tfirststart = time.perf_counter()

    # 总的步数除以每次nbatch，总的一次episode更新次数
    nupdates = total_timesteps//nbatch
    # 更新次数是nupdates, 那我gail应该是以nbatch为一次更新单位，但是不限制nupdates的次数
    # 迭代循环就这样吧

        # set adam optimizer for discriminator
    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out
    d_adam.sync()
    # update是每次更新的index
    for update in range(1, nupdates+1):
        # 假如他们不整除要报警
        assert nbatch % nminibatches == 0

        logger.log("********** Iteration %i ************"%update)
        # ------------------ Update G ------------------
        for _ in range(g_step):
            logger.log("...Optimizing Policy...")
            # Start timer
            # 每次小更新里面的计时
            tstart = time.perf_counter()
            # 每次更新的比例
            # frac = 1.0 - (update - 1.0) / nupdates
            # Calculate the learning rate
            # # 获得比例对应的学习率和clip值
            # lrnow = lr(frac)
            # # Calculate the cliprange
            # cliprangenow = cliprange(frac)
            # log_intercal是10也就是说每过10次，然后根程序要log一次日志
            # 也就是说多进程了每次update
            #if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

            # Get minibatch
            # 获得轨迹， 并获得奖励加成优势函数带来的return
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

            # log_intercal是10也就是说每过10次，然后根程序要log一次日志, 与环境的交互结束
            #if update % log_interval == 0 and is_mpi_root: logger.info('Done.')
            # 将 环境交互的信息导入到队列里
            epinfobuf.extend(epinfos)

            # Here what we're going to do is for each minibatch calculate the loss and append it.
            mblossvals = []
            
            if states is None: # nonrecurrent version
                # Index of each element of batch_size
                # Create the indices array
                # 对于一个nbatch返回它的index
                inds = np.arange(nbatch)
                # noptepochs对于每次generator update 的优化次数
                for _ in range(noptepochs):
                    # Randomize the indexes
                    np.random.shuffle(inds)
                    # 0 to batch_size with batch_train_size step
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        # 获得采样值
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        # 返回的是model的 loss值
                        mblossvals.append(model.train(lr, cliprange, *slices))
            
            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)
            # End timer
            tnow = time.perf_counter()
            # Calculate the fps (frame per second)
            fps = int(nbatch / (tnow - tstart))

            # 每过10次log或者 第一次log
            # if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                # 一个评价指标
            logger.log("...Optimizing done...")
            
        ev = explained_variance(values, returns)
        logger.logkv("misc/serial_timesteps", update*nsteps)
        logger.logkv("misc/nupdates", update)
        logger.logkv("misc/total_timesteps", update*nbatch)
        logger.logkv("fps", fps)
        logger.logkv("misc/explained_variance", float(ev))
        logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
        logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
        logger.logkv('epgivenrewmean',safemean([epinfo['fr'] for epinfo in epinfobuf]))
        logger.logkv('misc/time_elapsed', tnow - tfirststart)
        # 打印一些loss, loss都是ppo的loss
        # loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
        for (lossval, lossname) in zip(lossvals, model.loss_names):
            logger.logkv('loss/' + lossname, lossval)
        
        logger.dumpkvs()

        #return model

         # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        # 打印reward_giver的loss名称
        logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(obs))
        batch_size = len(obs) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        # 和上面同样， 进行batch_size迭代循环
        for ob_batch, ac_batch in dataset.iterbatches((obs, actions),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            # 导入同样大小的expert data
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for reward_giver,感觉就是求平均
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            # 更新了discriminator
            d_adam.update(allmean(g), lr)
            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    # 取出平均值
    return np.nan if len(xs) == 0 else np.mean(xs)
