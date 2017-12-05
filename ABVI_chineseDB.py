import cplex
import operator
import time
from numpy import linalg as LA

__author__ = 'pegah'

import numpy as np
import util
import timeit

ftype = np.float32

class abvi_chinese:
    def __init__(self, _mdp, _states_list, _lambda, _discount= 1.0, _height= 63):

        self.mdp = _mdp
        self.d = self.mdp.d
        self.discount = _discount
        self.height = _height

        self.Lambda = np.zeros(len(_lambda), dtype=ftype)
        self.Lambda[:] = _lambda

        self.states = _states_list
        self.values = util.Counter(self.d) # A Counter is a dict with default [0, 0, .., 0]

        # it is a global variable for definig if an action selected for a state in the optimal policy
        self.change_checker = False

        #print [i for i in self.values.iterkeys()]

        self.query_counter_ = 0
        self.Lambda_inequalities = []

        """initialize linear programming as a minimization problem"""
        self.prob = cplex.Cplex()
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

        constr, rhs = [], []
        _d = self.d

        self.prob.variables.add(lb=[0.0] * _d, ub=[1.0] * _d)

        self.prob.set_results_stream(None)
        self.prob.set_log_stream(None)
        self.prob.set_warning_stream(None)

        """add sum(lambda)_i = 1 on problem constraints"""
        c = [[j, 1.0] for j in range(0,_d)]
        constr.append(zip(*c))
        rhs.append(1)

        """inside this function E means the added constraint is an equality equation
        there are three options for sense:
        G: constraint is greater than rhs,
        L: constraint is lesser than rhs,
        E: constraint is equal to rhs"""

        self.prob.linear_constraints.add(lin_expr=constr, senses="E" * len(constr), rhs=rhs)
        # self.prob.write("show-Ldominance.lp")
        self.pareto = 0
        self.kd = 0
        self.queries = 0

        self.wen = open("output_weng" + ".txt", "w")
        self.prob.write("show-Ldominance.lp")


    ##########################
    # beginning
    # the vector comparison part
    def pareto_comparison(self, a, b):
        a = np.array(a, dtype=ftype)
        b = np.array(b, dtype=ftype)

        assert len(a) == len(b), \
            "two vectors don't have the same size"

        return all(a >= b)

    def cplex_K_dominance_check(self, _V_best, Q):

        _d = len(_V_best)

        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

        ob = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
        self.prob.objective.set_linear(ob)
        self.prob.write("show-Ldominance-ob.lp")
        self.prob.solve()

        result = self.prob.solution.get_objective_value()
        if result < 0.0:
            return False
        # print >> self.wen, _V_best - Q, ">> 0"
        return True

    def is_already_exist(self, inequality_list, new_constraint):
        """

        :param inequality_list: list of inequalities. list of lists of dimension d+1
        :param new_constraint: new added constraint to list of inequalities of dimension d+1
        :return: True if new added constraint is already exist in list of constraints for Lambda polytope
        """
        #global u, v
        if new_constraint in inequality_list:
            return True
        else:
            for i in range(len(inequality_list)):
                first = True
                for x, y in zip(inequality_list[i], new_constraint)[1:]:
                    if x == 0 and y == 0:
                        continue
                    if first:
                        u, v = x , y
                        first = False
                    elif (u * y != x * v):
                        break
                else :
                    return True
        #print "new_constraint", new_constraint
        return False

    def Query(self, _V_best, Q):

        bound = [0.0]
        _d = len(_V_best)

        constr = []
        rhs = []

        Vscal = self.Lambda.dot(_V_best)
        Qscal = self.Lambda.dot(Q)
        # choice of the best policy
        keep = (Vscal > Qscal)

        if keep:
            new_constraints = bound + map(operator.sub, _V_best, Q)
        else:
            new_constraints = bound + map(operator.sub, Q, _V_best)
            self.change_checker = True

        # memorizing it
        if not self.is_already_exist(self.Lambda_inequalities, new_constraints):
            if keep:
                c = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
                #print >> self.wen,  "Constrainte", self.query_counter_, _V_best - Q, "|> 0"
            else:
                c = [(j, float(Q[j] - _V_best[j])) for j in range(0, _d)]
                #print >> self.wen, "Constrainte", self.query_counter_, Q - _V_best, "|> 0"
            self.query_counter_ += 1
            constr.append(zip(*c))
            rhs.append(0.0)
            self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
            self.Lambda_inequalities.append(new_constraints)

        # return the result
        if keep:
            return _V_best
        else:
            return Q

    def get_best(self, _V_best, Q):

        if np.array_equal( _V_best, Q):
             return Q

        if self.pareto_comparison(_V_best, Q):
            self.pareto += 1
            return _V_best

        if self.pareto_comparison(Q, _V_best):
            self.pareto += 1
            self.change_checker = True
            return Q

        if self.cplex_K_dominance_check(Q, _V_best):
            self.kd += 1
            self.change_checker = True
            return Q

        if self.cplex_K_dominance_check(_V_best, Q):
            self.kd += 1
            return _V_best

        query = self.Query(_V_best, Q)
        self.queries += 1
        # self.query_counter_ += 1 ## seulement si existe --> dans Query

        return query

    ##########################
    #end
    # the vector comparison part

    def computeQValueFromValues(self, state, action, reward):

        value = [0.0]*self.mdp.d
        transitionFunction = self.mdp.getTransitionStatesAndProbs(state,action)

        for nextState, probability in transitionFunction:
            rewards = reward
            value += probability * (np.array(rewards) + (self.discount * np.array(self.values[nextState])) )

        return value

    "value iteration for discrete time MDP with a finite horizon"
    def interactive_value_iteration(self, matrix):

        errors_iteration = []
        qos_iteration = []
        query_iteration = []

        #variables for checking calculation time
        global data_time, step_time, states_time, actions_time, qos_time

        result = open("result_ivi" + ".txt", "w")

        states_list = self.states
        d = self.d

        #initial optimal policy
        optimal_policy = {i:None for i in states_list}

        "the iteration start here"
        _time = self.height #stop at the bottom of the horizontal length

        while _time > -1:
            print >> result,'tour = ', _time
            result.flush()

            valuesCopy = self.values.copy()

            state_counter = 0

            for state in (states_list):
                #if state != 'ASterminal':

                state_counter += 1

                _V_best_d = self.values[state] #np.zeros(d, dtype=ftype)
                possible_actions = self.mdp.getPossibleActions(state)

                print "state_counter = ", state_counter
                print state, " = ", len(possible_actions), possible_actions

                for action in possible_actions:

                    start = timeit.default_timer()
                    #tempo = self.mdp.getQoS(state, action, _time, possible_actions, matrix)
                    #rewards_list = tempo[0:2]
                    #matrix = tempo[2]

                    tempo = self.mdp.getQoS(state, action, _time, possible_actions, matrix)
                    rewards_list = tempo[1]
                    stop = timeit.default_timer()

                    print "time for QoS = ", stop - start

                    start_2 = timeit.default_timer()
                    for reward_value in rewards_list:

                        #print '******** ', action,' ', _time,' ', reward_value

                        Q_d = self.computeQValueFromValues(state, action, reward_value)
                        _V_best_d = self.get_best(_V_best_d, Q_d)

                        "if the action for state has changed"
                        if self.change_checker == True:
                            optimal_policy[state] = action
                            self.change_checker = False

                        valuesCopy[state] = _V_best_d

                    stop_2 = timeit.default_timer()

                    print "time for finding the best action = ", stop_2 - start_2


            self.values = valuesCopy
            #print 'self.values', self.values
            _time -= 1

            query_iteration.append(self.queries)
            #print 'errors_iteration', query_iteration
            temp_vect = sum([(1.0/len(self.states))*np.array(value) for value in self.values.itervalues()])
            #qos_iteration.append(LA.norm( temp_vect, np.inf))
            #print 'qos_iteration', qos_iteration
            #errors_iteration.append(temp_vect)
            errors_iteration.append(self.values)
            #print 'errors_iteration', errors_iteration


        print >> result, '************'
        print >> result, 'optimal policy', optimal_policy
        print >> result, 'final vector value', self.values

        print >> result, '******************'
        print >> result, "queries in time", query_iteration
        print >> result, 'for computing error', errors_iteration

        result.flush()
        result.close()
        return self.values
