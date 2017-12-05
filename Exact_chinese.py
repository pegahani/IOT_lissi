import util
import time
from numpy import linalg as LA

__author__ = 'pegah'

import numpy as np
ftype = np.float32

class exact:
    def __init__(self, _mdp, _states_list, _lambda, _discount= 1.0, _height= 63):
        self.mdp = _mdp
        self.d = self.mdp.d
        self.discount = _discount
        self.height = _height

        self.Lambda = np.zeros(len(_lambda), dtype=ftype)
        self.Lambda[:] = _lambda

        self.states = _states_list
        self.values = util.Counter(1) # A Counter is a dict with default [0, 0, .., 0]

        self.query_counter_ = 0

    # the exact value iteration goes here

    def computeQValueFromValues_(self, state, action, reward):
        value = [0]
        transitionFunction = self.mdp.getTransitionStatesAndProbs(state,action)

        for nextState, probability in transitionFunction:
            rewards = (np.array(reward)).dot(self.Lambda)
            value += probability * (np.array(rewards) + (self.discount * np.array(self.values[nextState])))

        return value

    def exact_value_iteraion(self, matrix):

        query_iteration = []
        qos_iteration = []

        result = open("result_exact" + ".txt", "w")

        states_list = self.states
        optimal_policy = {i:None for i in states_list}
        _time = self.height

        while _time > -1:

            print >> result, 'tour', _time
            result.flush()

            valuesCopy = self.values.copy()
            for state in (states_list):
                _V_best = [0]

                possible_actions = self.mdp.getPossibleActions(state)

                for action in possible_actions:

                    tempo = self.mdp.getQoS(state, action, _time,possible_actions, matrix)
                    rewards_list = tempo[0:2]
                    matrix = tempo[2]

                    for reward_value in rewards_list[1]:
                        Q_d = self.computeQValueFromValues_(state, action, reward_value)

                        if Q_d[0] > _V_best[0]:
                            _V_best = Q_d
                            optimal_policy[state] = action

                        valuesCopy[state] = _V_best

            query_iteration.append(self.query_counter_)

            self.values = valuesCopy
            _time -= 1

        print >> result, '************'
        print >> result, 'optimal policy', optimal_policy
        print >> result, "final vector value", self.values
        print >> result, 'final values with initial distribution', \
            sum([(1.0/len(self.states))*np.array(value) for value in self.values.itervalues()])

        print >> result, "************"
        print >> result, "quiries in time" , query_iteration
        result.flush()

        result.close()
        return self.values