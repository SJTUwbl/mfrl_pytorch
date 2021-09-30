from . import mfac
from . import q_learning
from . import ac

MFQ = q_learning.MFQ
DQN = q_learning.DQN
MFAC = mfac.MFAC
AC = ac.AC

def spawn_ai(algo_name, env, handle, human_name, max_steps):
    if algo_name == 'mfq':
        model = MFQ(handle, env, max_steps, memory_size=80000)
    elif algo_name == 'dqn':
        model = DQN(handle, env, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC(handle, env)
    elif algo_name =="ac":
        model = AC(handle, env)
    return model
