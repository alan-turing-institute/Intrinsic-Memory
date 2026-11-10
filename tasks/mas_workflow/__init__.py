from mas.mas import MetaMAS
from .autogen import AutoGen
from .macnet import MacNet
from .dylan import DyLAN, DyLANPrivateMemory, DyLANRoleAgnosticMemory, DyLANSharedMemory

MAS = {
    'autogen': AutoGen,
    'macnet': MacNet,
    'dylan': DyLAN,
    'dylan-shared': DyLANSharedMemory,
    'dylan-private': DyLANPrivateMemory,
    'dylan-private-agnostic': DyLANRoleAgnosticMemory,
}

def get_mas(mas_type: str) -> MetaMAS:

    if MAS.get(mas_type) is None:
        raise ValueError('Unsupported mas type.')
    return MAS.get(mas_type)() 