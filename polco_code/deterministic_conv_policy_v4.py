from rllab.core.lasagne_powered import LasagnePowered
import lasagne.layers as L
from rllab.core.network import ConvNetwork
from rllab.core.network import MLP_Direct
from rllab.core.network import MLP_Direct_v2
from rllab.core.network import MLP
from rllab.core.network import TextLSTM
from rllab.distributions.categorical import Categorical
from rllab.policies.base import StochasticPolicy
from rllab.misc import tensor_utils
from rllab.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
import numpy as np
import lasagne.nonlinearities as NL
import theano.tensor as TT


TEXT_LENGTH = 34

class CategoricalConvPolicy(StochasticPolicy, LasagnePowered):
    def __init__(
            self,
            env_spec,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_sizes_prob,
            hidden_sizes=[],
            num_embedding_object = 4,
            size_embedding=10,
            hidden_nonlinearity=NL.rectify,
            output_nonlinearity=NL.softmax,
            prob_network=None,
            conv_network=None,
            enable_hcMLP = True,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        self._env_spec = env_spec

        # process images
        # output is (None, 10)
        conv_network = ConvNetwork(
            input_shape=env_spec.observation_space.shape,
            output_dim=size_embedding,#size_embedding,#env_spec.action_space.n,
            conv_filters=conv_filters,
            conv_filter_sizes=conv_filter_sizes,
            conv_strides=conv_strides,
            conv_pads=conv_pads,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            #output_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
            name="conv_network",
        )

        # process hc
        # output is (None, 10)
        hc_network = TextLSTM(
            input_shape=(TEXT_LENGTH,),
            output_dim=size_embedding,
            name="hc_network",
        )

        if enable_hcMLP:
            # combine hc and images
            prob_network = MLP(
                input_shape=(size_embedding+size_embedding,),
                input_layer=L.ConcatLayer([conv_network.output_layer,hc_network.output_layer]),
                #input_layer=L.ElemwiseMergeLayer([conv_network.output_layer,hc_network.output_layer], TT.mul),
                output_dim=env_spec.action_space.n,
                hidden_sizes=hidden_sizes_prob,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.softmax,
                name="prob_network",
            )
        else:
            # only images
            prob_network = MLP(
                input_shape=(size_embedding,),
                input_layer=L.ConcatLayer([conv_network.output_layer]),
                output_dim=env_spec.action_space.n,
                hidden_sizes=hidden_sizes_prob,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.softmax,
                name="prob_network",
            )

        self._l_prob = prob_network.output_layer
        #self._l_prob = conv_network.output_layer
        
        self._l_obs = conv_network.input_layer

        self._l_obs_hc = hc_network.input_layer

        self._f_prob = ext.compile_function(
            [conv_network.input_layer.input_var, hc_network.input_layer.input_var],
            L.get_output(prob_network.output_layer)
            #L.get_output(conv_network.output_layer)
        )

        self._dist = Categorical(env_spec.action_space.n)

        super(CategoricalConvPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [prob_network.output_layer])
        #LasagnePowered.__init__(self, [conv_network.output_layer])


    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, obs_hc_var, state_info_vars=None):
        return dict(
            prob=L.get_output(
                self._l_prob,
                {self._l_obs: obs_var,
                 self._l_obs_hc : obs_hc_var.astype('int32')
                }
            )
        )

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        obs = observation['image'] # the size here is 7 by 7 by 4

        # get the value of hc
        hc = observation['mission'].split(',')#(instruction;hc;object)
        text = hc[0]
        text = text.split(' ')
        text_list = []
        for i in text:
            text_list.append(int(i))
        


        zero_text = list(np.zeros(TEXT_LENGTH - len(text_list)).astype('int'))
        text_list = text_list+zero_text
        text_list = np.array(text_list)

        # the observation was flattern
        # the effect is equal to obs.reshape((-1,196) where 196 = 7 * 7 * 4
        flat_obs = self.observation_space.flatten(obs)#(196,) 
        prob = self._f_prob([flat_obs],[text_list])[0]

        action = self.action_space.weighted_sample(prob)



        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist
