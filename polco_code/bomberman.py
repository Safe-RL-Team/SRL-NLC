from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.base import *
import numpy as np

# bomb collection constraint. 
class BMSafetyConstraint(SafetyConstraint, Serializable):

    def __init__(self, max_value=1., **kwargs):
        self.max_value = max_value
        Serializable.quick_init(self, locals())
        super(BMSafetyConstraint,self).__init__(max_value, **kwargs)

    def evaluate(self, path):
        return path['env_infos']['cost']
