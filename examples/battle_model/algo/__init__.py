from . import ac
from . import q_learning


MFAC = ac.MFAC
MFQ = q_learning.MFQ


def spawn_ai(algo_name, env, handle, human_name, max_steps):
    if algo_name == 'mfq':
        model = MFQ(handle, env, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC(handle, env)
    return model
