# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.Q = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if state in self.Q:
          if action in self.Q[state]:
            return self.Q[state][action]
        return 0.0


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if state in self.Q:
          if len(self.getLegalActions(state)) == 0:
            return 0.0
          m = float("-inf")
          for a in self.getLegalActions(state):
            m = max(m,self.getQValue(state,a))
          return m
        else:
          return 0.0


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        action = None
        if state in self.Q:
          m = float("-inf")
          for a in self.getLegalActions(state):
            q = self.getQValue(state,a)
            if q > m:
              m = q
              action = a
        return action


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        legalActions = self.getLegalActions(state)
        action = None

        # If No Legal Actions, return None
        if len(legalActions) == 0:
          return None

        # With probability epsilon, choose random legal move
        elif util.flipCoin(self.epsilon):
          return random.choice(legalActions)

        # Otherwise, choose action with highest Q value
        else:
          best_Q = float("-inf")
          for a in legalActions:
            q = self.getQValue(state,a)
            if q > best_Q:
              action = a
              best_Q = q

          return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        if state not in self.Q:
          self.Q[state] = util.Counter()
          for a in self.getLegalActions(state):
            self.Q[state][a] = 0.0

        if nextState not in self.Q:
          self.Q[nextState] = util.Counter()
          for a in self.getLegalActions(nextState):
            self.Q[nextState][a] = 0.0

        m = self.computeValueFromQValues(nextState)
        Qsa = self.getQValue(state,action)
        self.Q[state][action] = Qsa + self.alpha*(reward + self.discount*m - Qsa)


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        w = self.getWeights()
        f = self.featExtractor.getFeatures(state, action)
        keys = f.keys()
        return sum([w[i]*f[i] for i in keys])

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        #print(self.getWeights())
        #print(self.featExtractor.getFeatures(state, action).keys())
        m = float("-inf")
        if len(self.getLegalActions(nextState)) == 0:
          m = 0.0
        else:
          for a in self.getLegalActions(nextState):
            m = max(m,self.getQValue(nextState,a))

        d = (reward + self.discount*m) - self.getQValue(state,action)
        f = self.featExtractor.getFeatures(state, action)

        keys = f.keys()
        for i in keys:
          if i in self.weights:
            self.weights[i] = self.weights[i] + self.alpha*d*f[i]
          else:
            self.weights[i] = self.alpha*d*f[i]



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
