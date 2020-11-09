# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print('new pos:', newPos)
        # print('new food:', newFood)
        # print('new ghost state:', newGhostStates)
        # for ghost in newGhostStates:
        #     print(ghost.getPosition())
        # print('new scare time:', newScaredTimes)
        # print('new pos:', successorGameState)


        # print(currentGameState.getPacmanPosition(), '->', successorGameState.getPacmanPosition())
        # print(currentGameState.getPacmanState(), '->', successorGameState.getPacmanState())

        food_score = 0
        ghost_score = 0
        better_pos = 0

        if newPos in currentGameState.getFood().asList():
            food_score += 5 * (1 + 4 * ((0.9) ** min(30, len(currentGameState.getFood().asList()))))

        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) < 2:
                ghost_score += -25 * ((0.7) ** manhattanDistance(newPos, ghost.getPosition()))

        if newPos == currentGameState.getPacmanPosition():
            better_pos += -10
        elif food_score < 5:
            min_food_distance = math.inf
            for food in newFood.asList():
                if manhattanDistance(newPos, food) < min_food_distance:
                    min_food_distance = manhattanDistance(newPos, food)
            better_pos += ((0.99) ** min_food_distance) * 5

        return food_score + ghost_score + better_pos*(random.randint(7,10)/10)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    BEST_SCORE = "BEST_SCORE"
    BEST_MOVE = "BEST_MOVE"

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # print(gameState.getNumAgents())
        self.agents_count = gameState.getNumAgents() - 1 
        # print( gameState.getLegalActions(agentIndex))
        # return Directions.STOP
        return self.maxChoice(1, gameState, 0)[MinimaxAgent.BEST_MOVE]

    def minChoice(self, depth, gamestate, agent_id):
        # Bascause
        if self.is_game_finished(gamestate, depth):
            return {
                MinimaxAgent.BEST_SCORE: self.evaluationFunction(gamestate),
                MinimaxAgent.BEST_MOVE: Directions.STOP
            }
        
        scores = []

        if agent_id < self.agents_count:
            actions = gamestate.getLegalActions(agent_id)
            scores = [self.minChoice(depth, gamestate.generateSuccessor(agent_id, action), agent_id + 1) for action in actions]
            scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best

            return {
                MinimaxAgent.BEST_SCORE: bestScore,
                MinimaxAgent.BEST_MOVE: actions[chosenIndex]
            }

        else:
            actions = gamestate.getLegalActions(agent_id)
            depth += 1
            scores = [self.maxChoice(depth, gamestate.generateSuccessor(agent_id, action), 0) for action in actions]
            scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best

            return {
                MinimaxAgent.BEST_SCORE: bestScore,
                MinimaxAgent.BEST_MOVE: actions[chosenIndex]
            }
            

    def maxChoice(self, depth, gamestate, agent_id):
        # Basecase
        if self.is_game_finished(gamestate, depth):
            return {
                MinimaxAgent.BEST_SCORE: self.evaluationFunction(gamestate),
                MinimaxAgent.BEST_MOVE: Directions.STOP
            }
        
        scores = []

        # print(gamestate.getLegalActions(0))
        # print(gamestate.getLegalActions(1))
        # gamestate = gamestate.generateSuccessor(agent_id, gamestate.getLegalActions(agent_id)[1])
        # print(gamestate.getLegalActions(0))
        # print(gamestate.getLegalActions(1))

        actions = gamestate.getLegalActions(agent_id)
        scores = [self.minChoice(depth, gamestate.generateSuccessor(agent_id, action), 1) for action in actions]
        scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return {
            MinimaxAgent.BEST_SCORE: bestScore,
            MinimaxAgent.BEST_MOVE: actions[chosenIndex]
        }

    def is_game_finished(self, gamestate, d):
        return d > self.depth or gamestate.isLose() or gamestate.isWin()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # print(gameState.getNumAgents())
        self.agents_count = gameState.getNumAgents() - 1 
        # print( gameState.getLegalActions(agentIndex))
        # return Directions.STOP

        return self.maxChoice(1, gameState, 0, -1 * math.inf, math.inf)[MinimaxAgent.BEST_MOVE]

    def minChoice(self, depth, gamestate, agent_id, a, b):
        # Bascause
        if self.is_game_finished(gamestate, depth):
            return {
                MinimaxAgent.BEST_SCORE: self.evaluationFunction(gamestate),
                MinimaxAgent.BEST_MOVE: Directions.STOP
            }
        
        scores = []

        if agent_id < self.agents_count:
            actions = gamestate.getLegalActions(agent_id)
            for act in actions:
                new_state =gamestate.generateSuccessor(agent_id, act)
                new_option = self.minChoice(depth, new_state, agent_id + 1, a, b)
                scores += [new_option]
                if new_option[MinimaxAgent.BEST_SCORE] < a:
                    return {
                        MinimaxAgent.BEST_SCORE: new_option[MinimaxAgent.BEST_SCORE],
                        MinimaxAgent.BEST_MOVE:  new_option[MinimaxAgent.BEST_MOVE]
                    }
                if new_option[MinimaxAgent.BEST_SCORE] <= b:
                    b = new_option[MinimaxAgent.BEST_SCORE]
            scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best

            return {
                MinimaxAgent.BEST_SCORE: bestScore,
                MinimaxAgent.BEST_MOVE: actions[chosenIndex]
            }

        else:
            actions = gamestate.getLegalActions(agent_id)
            depth += 1
            for act in actions:
                new_state =gamestate.generateSuccessor(agent_id, act)
                new_option = self.maxChoice(depth, new_state, 0, a, b)
                scores += [new_option]
                if new_option[MinimaxAgent.BEST_SCORE] < a:
                    return {
                        MinimaxAgent.BEST_SCORE: new_option[MinimaxAgent.BEST_SCORE],
                        MinimaxAgent.BEST_MOVE:  new_option[MinimaxAgent.BEST_MOVE]
                    }
                if new_option[MinimaxAgent.BEST_SCORE] <= b:
                    b = new_option[MinimaxAgent.BEST_SCORE]
            scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best

            return {
                MinimaxAgent.BEST_SCORE: bestScore,
                MinimaxAgent.BEST_MOVE: actions[chosenIndex]
            }
            

    def maxChoice(self, depth, gamestate, agent_id, a, b):
        # Basecase
        if self.is_game_finished(gamestate, depth):
            return {
                MinimaxAgent.BEST_SCORE: self.evaluationFunction(gamestate),
                MinimaxAgent.BEST_MOVE: Directions.STOP
            }
        
        scores = []


        actions = gamestate.getLegalActions(agent_id)
        for act in actions:
            new_state =gamestate.generateSuccessor(agent_id, act)
            new_option = self.minChoice(depth, new_state, 1, a, b)
            scores += [new_option]
            if new_option[MinimaxAgent.BEST_SCORE] > b:
                return {
                    MinimaxAgent.BEST_SCORE: new_option[MinimaxAgent.BEST_SCORE],
                    MinimaxAgent.BEST_MOVE:  new_option[MinimaxAgent.BEST_MOVE]
                }
            if new_option[MinimaxAgent.BEST_SCORE] >= a:
                a = new_option[MinimaxAgent.BEST_SCORE]
        scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return {
            MinimaxAgent.BEST_SCORE: bestScore,
            MinimaxAgent.BEST_MOVE: actions[chosenIndex]
        }

    def is_game_finished(self, gamestate, d):
        return d > self.depth or gamestate.isLose() or gamestate.isWin()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # print(gameState.getNumAgents())
        self.agents_count = gameState.getNumAgents() - 1 
        # print( gameState.getLegalActions(agentIndex))
        # return Directions.STOP
        return self.maxChoice(1, gameState, 0)[MinimaxAgent.BEST_MOVE]

    def avgChoice(self, depth, gamestate, agent_id):
        # Bascause
        if self.is_game_finished(gamestate, depth):
            return {
                MinimaxAgent.BEST_SCORE: self.evaluationFunction(gamestate),
                MinimaxAgent.BEST_MOVE: Directions.STOP
            }
        
        scores = []

        if agent_id < self.agents_count:
            actions = gamestate.getLegalActions(agent_id)
            scores = [self.avgChoice(depth, gamestate.generateSuccessor(agent_id, action), agent_id + 1) for action in actions]
            scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
            bestScore = sum(scores)/len(scores)

            return {
                MinimaxAgent.BEST_SCORE: bestScore,
                MinimaxAgent.BEST_MOVE: None
            }

        else:
            actions = gamestate.getLegalActions(agent_id)
            depth += 1
            scores = [self.maxChoice(depth, gamestate.generateSuccessor(agent_id, action), 0) for action in actions]
            scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
            bestScore = sum(scores)/len(scores)

            return {
                MinimaxAgent.BEST_SCORE: bestScore,
                MinimaxAgent.BEST_MOVE: None
            }
            

    def maxChoice(self, depth, gamestate, agent_id):
        # Basecase
        if self.is_game_finished(gamestate, depth):
            return {
                MinimaxAgent.BEST_SCORE: self.evaluationFunction(gamestate),
                MinimaxAgent.BEST_MOVE: Directions.STOP
            }
        
        scores = []

        actions = gamestate.getLegalActions(agent_id)
        scores = [self.avgChoice(depth, gamestate.generateSuccessor(agent_id, action), 1) for action in actions]
        scores = list(map(lambda x: x[MinimaxAgent.BEST_SCORE], scores))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return {
            MinimaxAgent.BEST_SCORE: bestScore,
            MinimaxAgent.BEST_MOVE: actions[chosenIndex]
        }

    def is_game_finished(self, gamestate, d):
        return d > self.depth or gamestate.isLose() or gamestate.isWin()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # print('new pos:', newPos)
    # print('new food:', newFood)
    # print('new ghost state:', newGhostStates)
    # for ghost in newGhostStates:
    #     print(ghost.getPosition())
    # print('new scare time:', newScaredTimes)
    # print('new pos:', currentGameState)



    food_score = 0
    ghost_score = 0
    better_pos = 0

    if newPos in currentGameState.getFood().asList():
        food_score += 5 * (3 * ((0.9) ** min(50, len(currentGameState.getFood().asList()))))
        neighbor = [(newPos[0] + 1, newPos[1]), (newPos[0] - 1, newPos[1]), (newPos[0], newPos[1] + 1), (newPos[0], newPos[1] - 1)]
        neighbor = neighbor + [(newPos[0] + 2, newPos[1]), (newPos[0] - 2, newPos[1]), (newPos[0], newPos[1] + 2), (newPos[0], newPos[1] - 2)]
        neighbor = neighbor + [(newPos[0] + 1, newPos[1] + 1), (newPos[0] - 1, newPos[1] - 1), (newPos[0] -1, newPos[1] + 1), (newPos[0] + 1, newPos[1] - 1)]
        
        for node in neighbor:
            if node in currentGameState.getFood().asList():
                food_score += 5 * (3 * ((0.9) ** min(50, len(currentGameState.getFood().asList()))))
            else:
                food_score += -4 * (3 * ((0.9) ** min(50, len(currentGameState.getFood().asList()))))

    for ghost in newGhostStates:
        if manhattanDistance(newPos, ghost.getPosition()) < 3:
            ghost_score += -25 * ((0.9) ** manhattanDistance(newPos, ghost.getPosition()))

    if newPos == currentGameState.getPacmanPosition():
        better_pos += -5

    # print("--------------------------------")
    # print(breadthFirstSearch(newPos, 20, currentGameState))
    # print("--------------------------------")

    return breadthFirstSearch(newPos, 4, currentGameState) + ghost_score + better_pos*(random.randint(7,10)/10)
    
ALPHA = 0.5

def breadthFirstSearch(pos, depth, gameState):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    score = 0
    # BFS Queue
    queue = []
    # Visited states
    visited = []
    # Main
    queue.append((pos, 0, 1 if gameState.getFood()[pos[0]][pos[1]] else 0))
    while len(queue) != 0:
        state = queue[0]
        del(queue[0])
        d = state[1] + 1
        score += state[2]*(2*((ALPHA)**depth))
        if state[1] == 0:
            visited.append(state[0])
        # Check if it is a go
        if d > depth:
            return score
        for child in getBFSSuccessors(state[0], d, gameState):
            if child[0] not in visited:
                queue.append(child)
                visited.append(child[0])                
    return score

def getBFSSuccessors(pos, depth, gameState):
    successors = []
    food, walls = gameState.getFood(), gameState.getWalls()
    for action in [(0, 1), (-1, 0), (1, 0), (0, -1)]:
        x,y = pos
        dx, dy = action
        nextx, nexty = int(x + dx), int(y + dy)
        if not walls[nextx][nexty]:
            nextState = (nextx, nexty)
            successors.append( ( nextState, depth, 1 if food[nextx][nexty] else 0) )
    return successors

# Abbreviation
better = betterEvaluationFunction
