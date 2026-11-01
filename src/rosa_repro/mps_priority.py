"""Explicit per-client MPS priority hints, scoped to synchronous engine launch."""
from contextlib import contextmanager
import os


def attest(settings, environ=None):
    env=os.environ if environ is None else environ
    priority=settings['priority']
    if type(priority) is not int or priority not in (0,1):raise ValueError('Invalid MPS client priority')
    if not all(env.get(k,'').startswith('/') for k in ('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY')):
        raise ValueError('Explicit MPS pipe and log paths required')
    if env.get('CUDA_MPS_CLIENT_PRIORITY')!=str(priority):raise ValueError('Worker MPS priority differs from launch configuration')
    return dict(protocol='mps_client_priority_v1',role=settings['role'],priority=priority,pid=os.getpid(),
        pipe_directory=env['CUDA_MPS_PIPE_DIRECTORY'],scheduling_guaranteed=False)


@contextmanager
def engine_launch(settings):
    """All GroupCore model constructors run serially before request admission."""
    name='CUDA_MPS_CLIENT_PRIORITY';previous=os.environ.get(name)
    try:
        os.environ[name]=str(settings['priority'])
        evidence=attest(settings)
        yield evidence
    finally:
        if previous is None:os.environ.pop(name,None)
        else:os.environ[name]=previous
