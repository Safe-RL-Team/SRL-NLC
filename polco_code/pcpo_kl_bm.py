from sandbox.cpo.algos.safe.policy_gradient_safe_discrete import PolicyGradientSafeDiscrete
from sandbox.cpo.optimizers.conjugate_constraint_optimizer_pcpo_discrete import ConjugateConstraintOptimizerPCPO_Discrete
from rllab.core.serializable import Serializable


class PCPO_KL_BM(PolicyGradientSafeDiscrete, Serializable):
    """
    Constrained Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            safety_constraint=None,
            safety_tradeoff=False,
            learn_safety_tradeoff_coeff=False,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateConstraintOptimizerPCPO_Discrete(**optimizer_args)


        if 'safety_constrained_optimizer' in kwargs.keys():
            kwargs.pop('safety_constrained_optimizer')

        super(PCPO_KL_BM, self).__init__(optimizer=optimizer, 
                                   safety_constrained_optimizer=True,
                                   safety_constraint=safety_constraint,
                                   safety_tradeoff=safety_tradeoff,
                                   learn_safety_tradeoff_coeff=learn_safety_tradeoff_coeff,
                                   **kwargs)