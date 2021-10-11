# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Repeat for given numbe of iterations
        for _ in range(self.iterations):


          values2 = self.values.copy()
          # Iterate over all States
          for s in self.mdp.getStates():

            # If Terminal, update utility of this state to be the reward of this terminal
            if self.mdp.isTerminal(s):
              values2[s] = self.mdp.getReward(s,None,None)

            else:

              # Determine Update for each State by iterating over possible Actions
              m = float("-inf")
              for a in self.mdp.getPossibleActions(s):

                # Calculate Expected Utility of taking this Action
                x = 0
                for t in self.mdp.getTransitionStatesAndProbs(s,a):
                  x += t[1]*self.getValue(t[0])

                # Keep track of the maximum expected utility of each of the Actions
                m = max(m,x)

              # Calculate/Save Updated Utilities
              update = self.mdp.getReward(s,None,None) + self.discount*m
              values2[s] = update

          # Now Update Utilities
          for k in values2.keys():
            self.values[k] = values2[k]



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        # Find expected utility of making this move
        x = 0
        for t in self.mdp.getTransitionStatesAndProbs(state,action):
          x += t[1] * self.getValue(t[0])


        # Return Reward + discounted expected utility
        return self.mdp.getReward(state,None,None) + self.discount*x


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        best_move = None
        best_Q = float("-inf")

        for a in self.mdp.getPossibleActions(state):
          q = self.computeQValueFromValues(state,a)
          if q > best_Q:
            best_Q = q
            best_move = a

        return best_move


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
