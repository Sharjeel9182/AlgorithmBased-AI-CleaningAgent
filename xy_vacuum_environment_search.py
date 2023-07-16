import os.path
from tkinter import *

import numpy as np

from agents import *
from search import *
import sys
import copy
from utils import PriorityQueue

"""
1- BFS: Breadth first search. Using tree or graph version, whichever makes more sense for the problem
2- DFS: Depth-First search. Again using tree or graph version.
3- UCS: Uniform-Cost-Search. Using the following cost function to optimise the path, from initial to current state.
4- A*:  Using A star search.
"""
searchTypes = ['None', 'BFS', 'DFS', 'UCS', 'A*']


class VacuumPlanning(Problem):
    """ The problem of find the next room to clean in a grid of m x n rooms.
    A state is represented by state of the grid. Each room is specified by index set
    (i, j), i in range(m) and j in range (n). Final goal is to find all dirty rooms. But
     we go by sub-goal, meaning finding next dirty room to clean, at a time."""

    def __init__(self, env, searchtype):
        """ Define goal state and initialize a problem
            initial is a pair (i, j) of where the agent is
            goal is next pair(k, l) where map[k][l] is dirty
        """
        self.solution = None
        self.env = env
        self.state = env.agent.location
        super().__init__(self.state)
        self.map = env.things
        self.searchType = searchtype
        self.agent = env.agent
        self.height = hig
        self.width = wid

    def generateSolution(self):
        """ generate search engien based on type of the search chosen by user"""
        self.env.read_env()
        self.state = env.agent.location
        super().__init__(self.state)
        if self.searchType == 'BFS':
            try:
                path, explored = breadth_first_graph_search(self)
                sol = path.solution()
                self.env.set_solution(sol)
                self.env.display_explored(explored)
                self.env.color(sol)
            except:
                raise Exception("Unreachable Room")
        elif self.searchType == 'DFS':
            try:
                path1, explored = depth_first_graph_search(self)
                sol = path1.solution()
                self.env.set_solution(sol)
                self.env.display_explored(explored)
                self.env.color(sol)
            except:
                raise Exception("Unreachable Room ")

        elif self.searchType == 'UCS':
            try:
                path2, explored = best_first_graph_search(self, lambda node: node.path_cost)
                sol = path2.solution()
                self.env.set_solution(sol)
                self.env.display_explored(explored)
                self.env.color(sol)
            except:
                raise Exception("Unreachable Room")

        elif self.searchType == 'A*':
            try:
                path1, explored = astar_search(self)
                sol = path1.solution()
                self.env.set_solution(sol)
                self.env.display_explored(explored)
                self.env.color(sol)
            except:
                raise Exception("Unreachable Room")

        else:
            raise 'NameError'

    def generateNextSolution(self):
        self.generateSolution()

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_neighbors = self.env.things_near(state)
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'Suck']


        current_location = state

        neighbors = [(current_location[0] - 1, current_location[1]),  # up 0
                     (current_location[0] + 1, current_location[1]),  # down 1
                     (current_location[0], current_location[1] - 1),  # Left 2
                     (current_location[0], current_location[1] + 1)]  # Right 3

        if env.some_things_at(neighbors[1], Wall):
            possible_actions.remove('RIGHT')

            # print("right")
        if env.some_things_at(neighbors[0], Wall):
            possible_actions.remove('LEFT')

            # print("left")
        if env.some_things_at(neighbors[3], Wall):
            possible_actions.remove('UP')

            # print("UP")
        if env.some_things_at(neighbors[2], Wall):
            possible_actions.remove('DOWN')


        #print(possible_actions)
        # print(env.dirtCount, "is total dirt")
        # print(self.agent.location)
        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        new_state = list(state)
        x, y = new_state

        if action == 'UP':
            y = y + 1

        elif action == 'DOWN':
            y = y - 1

        elif action == 'LEFT':
            x = x - 1

        elif action == 'RIGHT':
            x = x + 1

        new_state[0] = x
        new_state[1] = y

        return new_state

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """
        return self.env.some_things_at(state, Dirt)

    def path_cost(self, c, state1, action, state2):
        """To be used for UCS and A* search. Returns the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. For our problem
        state is (x, y) coordinate pair.
        To make our problem more interesting we are going to associate
        a height to each state as z = sqrt(x*x + y*y). This effectively means our grid is a bowl shape and
        the center of the grid is the center of the bowl. So now the distance between 2 states become the
        square of Euclidean distance as distance = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2"""
        # print("path_cost: students must modify this cost using the above comment.")

        '''
        agt=self.agent.location
        pt=0
        if (action=='UP'):
            if self.agent.direction.direction=='right':
                pt=pt+1            
            elif self.agent.direction=='left':
                pt=pt+1
            elif self.agent.direction=='down':
                pt=pt+2
            else:
                pt=pt+0
        if (action=='DOWN'):
            if self.agent.direction.direction == 'right':
                pt = pt + 1
            elif self.agent.direction == 'left':
                pt = pt + 1
            elif self.agent.direction == 'up':
                pt = pt + 2
            else:
                pt = pt + 0

        if (action=='LEFT'):
            if self.agent.direction.direction == 'right':
                pt = pt + 2
            elif self.agent.direction == 'down':
                pt = pt + 1
            elif self.agent.direction == 'up':
                pt = pt + 1
            else:
                pt = pt + 0

        if (action=='right'):
            if self.agent.direction.direction == 'left':
                pt = pt + 2
            elif self.agent.direction == 'down':
                pt = pt + 1
            elif self.agent.direction == 'up':
                pt = pt + 1
            else:
                pt = pt + 0
        
        delta_x = state1[0] - state2[0]
        delta_y = state1[1] - state2[1]
        z1 = np.sqrt(state1[0] * state1[0] + state1[1] * state1[1])  # state1
        z2 = np.sqrt(state2[0] * state2[0] + state2[1] * state2[1])  # state2
        distance = (delta_x ** 2 + delta_y ** 2 + (z1 - z2) * (z1 - z2))
        cost = c + distance +pt
        return cost
        '''

        '''
        if action == 'UP':
            if agent.direction.direction == 'left':
                agent.direction += Direction.R 
                agent.performance -= 1
            elif agent.direction.direction == 'right':
                agent.direction += Direction.L 
                agent.performance -= 1
            elif agent.dirbction.direction == "down':
                agent.direction += Direction.L 
                agent.direction += Direction.L 
                agent.performance -= 2
            agent.bump = self.move_to(agent, agent.direction.move_forward (agent. location))
            if not agent.bump:
                self.buttons[y1][x1].config(text="")
                xf, yf = agent. location
                self.buttons [yf][xf1.conf1g(text=agent_label (agent))
                agent.performance -= 1

        '''

        delta_x = state1[0] - state2[0]
        delta_y = state1[1] - state2[1]
        z1 = np.sqrt(state1[0] * state1[0] + state1[1] * state1[1])  # state1
        z2 = np.sqrt(state2[0] * state2[0] + state2[1] * state2[1])  # state2
        distance = (delta_x ** 2 + delta_y ** 2 + (z1 - z2) * (z1 - z2))
        cost = c + distance
        return cost

    def h(self, node):
        """ to be used for A* search. Return the heuristic value for a given state. For this problem use minimum Manhattan
        distance to all the dirty rooms + absolute value of height distance as described above in path_cost() function. .
    """
        ##Gettting all the
        '''
        result = []
        x_start, y_start = (0, 0)
        x_end, y_end = self.width, self.height
        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                if (env.some_things_at((x, y), Dirt)):
                    loc = (y, x)
                    result.append(loc)
                    #cprint("The dirt is present at", result)

        state1 = node.state
        minimum=[]
        for x in result:
            #print(x[0])
            delta_x = state1[0] - x[0]
            delta_y = state1[1] - x[1]
            z1 = np.sqrt(state1[0] * state1[0] + state1[1] * state1[1])  # state1
            z2 = np.sqrt(x[0] * x[0] + x[1] * x[1])  # state2
            distance = (delta_x ** 2 + delta_y ** 2 + (z1 - z2) * (z1 - z2))
            minimum.append(distance)
        find_m=min(minimum)
        return find_m
        '''
        path = node.path()
        state1 = (path[0].state)
        state2 = (path[len(path) - 1].state)

        if env.some_things_at(state2, Dirt):
            A, B = state1[0], state1[1]
            K, L = state2[0], state2[1]
            z1 = np.sqrt(A * A + B * B)
            z2 = np.sqrt(K * K + L * L)
            distance = ((A - K) * (A - K) + (B - L) * (B - L) + (z1 - z2) * (z1 - z2))
            return distance
        return 0


# ______________________________________________________________________________

def agent_label(agt):
    """creates a label based on direction"""
    dir = agt.direction
    lbl = '^'
    if dir.direction == Direction.D:
        lbl = 'v'
    elif dir.direction == Direction.L:
        lbl = '<'
    elif dir.direction == Direction.R:
        lbl = '>'

    return lbl


def is_agent_label(lbl):
    """determines if the label is one of the labels tht agents have: ^ v < or >"""
    return lbl == '^' or lbl == 'v' or lbl == '<' or lbl == '>'


class Gui(VacuumEnvironment):
    """This is a two-dimensional GUI environment. Each location may be
    dirty, clean or can have a wall. The user can change these at each step.
    """
    xi, yi = (0, 0)

    perceptible_distance = 1

    def __init__(self, root, width, height):
        self.searchAgent = None
        print("creating xv with width ={} and height={}".format(width, height))
        super().__init__(width, height)

        self.agent = None
        self.root = root
        self.create_frames(height)
        self.create_buttons(width)
        self.create_walls()
        self.setupTestEnvironment()

    def setupTestEnvironment(self):
        """ first reset the agent"""

        if self.agent is not None:
            xi, yi = self.agent.location
            self.buttons[yi][xi].config(bg='white', text='', state='normal')
            x = self.width // 2
            y = self.height // 2
            self.agent.location = (x, y)
            self.buttons[y][x].config(bg='white', text=agent_label(self.agent), state='normal')
            self.searchType = searchTypes[0]
            self.agent.performance = 0

        """next create a random number of block walls inside the grid as well"""
        roomCount = (self.width - 1) * (self.height - 1)
        blockCount = random.choice(range(roomCount // 10, roomCount // 5))
        for _ in range(blockCount):
            rownum = random.choice(range(1, self.height - 1))
            colnum = random.choice(range(1, self.width - 1))
            self.buttons[rownum][colnum].config(bg='red', text='W', disabledforeground='black')

        self.create_dirts()
        self.stepCount = 0
        self.searchType = None
        self.solution = []
        self.explored = set()
        self.read_env()

    def create_frames(self, h):
        """Adds h row frames to the GUI environment."""
        self.frames = []
        for _ in range(h):
            frame = Frame(self.root, bg='blue')
            frame.pack(side='bottom')
            self.frames.append(frame)

    def create_buttons(self, w):
        """Adds w buttons to the respective row frames in the GUI."""
        self.buttons = []
        for frame in self.frames:
            button_row = []
            for _ in range(w):
                button = Button(frame, bg='white', state='normal', height=1, width=1, padx=1, pady=1)
                button.config(command=lambda btn=button: self.toggle_element(btn))
                button.pack(side='left')
                button_row.append(button)
            self.buttons.append(button_row)

    def create_walls(self):
        """Creates the outer boundary walls which do not move. Also create a random number of
        internal blocks of walls."""
        for row, button_row in enumerate(self.buttons):
            if row == 0 or row == len(self.buttons) - 1:
                for button in button_row:
                    button.config(bg='red', text='W', state='disabled', disabledforeground='black')
            else:
                button_row[0].config(bg='red', text='W', state='disabled', disabledforeground='black')
                button_row[len(button_row) - 1].config(bg='red', text='W', state='disabled', disabledforeground='black')

    def create_dirts(self):
        """ set a small random number of rooms to be dirty at random location on the grid
        This function should be called after create_walls()"""
        self.read_env()  # this is needed to make sure wall objects are created
        roomCount = (self.width - 1) * (self.height - 1)
        self.dirtCount = random.choice(range(5, 15))
        dirtCreated = 0
        while dirtCreated != self.dirtCount:
            rownum = random.choice(range(1, self.height - 1))
            colnum = random.choice(range(1, self.width - 1))
            if self.some_things_at((colnum, rownum)):
                continue
            self.buttons[rownum][colnum].config(bg='grey', text='D')
            dirtCreated += 1

    def setSearchEngine(self, choice):
        """sets the chosen search engine for solving this problem"""
        self.searchType = choice
        self.searchAgent = VacuumPlanning(self, self.searchType)
        self.searchAgent.generateSolution()
        self.done = False

    def set_solution(self, sol):
        self.solution = list(reversed(sol))

    def display_explored(self, explored):
        """display explored slots in a light pink color"""
        if len(self.explored) > 0:  # means we have explored list from previous search. So need to clear their visual fist
            for (x, y) in self.explored:
                self.buttons[y][x].config(bg='white')

        self.explored = explored
        for (x, y) in explored:
            self.buttons[y][x].config(bg='pink')

    def color(self, sol):
        loc = list(self.agent.location)
        sol_list = []
        sol_list.append(loc)

        for i in range(len(sol)):
            if (sol[i] == 'DOWN'):
                y = loc[1] - 1
                x = loc[0]
                loc[0] = x
                loc[1] = y
                #print(loc, "loc")
                sol_list.append((x, y))
            elif (sol[i] == 'UP'):
                y = loc[1] + 1
                x = loc[0]
                loc[0] = x
                loc[1] = y
                #print(loc, "loc")
                sol_list.append((x, y))
            elif (sol[i] == 'LEFT'):
                x = loc[0] - 1
                y = loc[1]
                loc[0] = x
                loc[1] = y
                #print(loc, "loc")
                sol_list.append((x, y))
            elif (sol[i] == 'RIGHT'):
                x = loc[0] + 1
                y = loc[1]
                loc[0] = x
                loc[1] = y
                #print(loc, "loc")
                sol_list.append((x, y))

        for (x, y) in sol_list:
            if not env.some_things_at((x, y), Dirt) or env.some_things_at((x, y), Wall):
                self.buttons[y][x].config(bg='blue')

    def add_agent(self, agt, loc):
        """add an agent to the GUI"""
        self.add_thing(agt, loc)
        # Place the agent at the provided location.
        lbl = agent_label(agt)
        self.buttons[loc[1]][loc[0]].config(bg='white', text=lbl, state='normal')
        self.agent = agt

    def toggle_element(self, button):
        """toggle the element type on the GUI."""
        bgcolor = button['bg']
        txt = button['text']
        if is_agent_label(txt):
            if bgcolor == 'grey':
                button.config(bg='white', state='normal')
            else:
                button.config(bg='grey')
        else:
            if bgcolor == 'red':
                button.config(bg='grey', text='D')
            elif bgcolor == 'grey':
                button.config(bg='white', text='', state='normal')
            elif bgcolor == 'white':
                button.config(bg='red', text='W')

    def execute_action(self, agent, action):
        """Determines the action the agent performs."""

        xi, yi = (self.xi, self.yi)
        self.read_env()
        xi, yi = self.agent.location
        #print(action)
        #print(env.dirtCount, "is total dirt")
        print("agent at location", agent.location, "and action", action)
        if action == 'Suck':
            dirt_list = self.list_things_at(agent.location, Dirt)
            if dirt_list:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_thing(dirt)
                self.buttons[yi][xi].config(bg='white')

        else:
            agent.bump = False
            '''
            if action == 'UP':
                if agent.direction.direction == 'left':
                    agent.direction += Direction.R
                    agent.performance -= 1
                elif agent.direction.direction == 'right':
                    agent.direction += Direction.L
                    agent.performance -= 1
                elif agent.dirbction.direction == 'down':
                    agent.direction += Direction.L
                    agent.direction += Direction.L
                    agent.performance -= 2
                agent.bump = self.move_to(agent, agent.direction.move_forward(agent.location))
                if not agent.bump:
                    self.buttons[yi][xi].config(text="")
                    xf, yf = agent.location
                    self.buttons[yf][xf].config(text=agent_label(agent))
                    agent.performance -= 1
                    
            elif action=='DOWN':
                if agent.direction.direction=='up':
                    agent.direction += Direction.R
                    agent.direction += Direction.R
                    agent.performance -= 2
                elif agent.direction.direction=='left':
                    agent.direction += Direction.L
                    agent.performance -= 1
                elif agent.direction.direction == 'right':
                    agent.direction += Direction.R
                    agent.performance -= 1
                agent.bump = self.move_to(agent, agent.direction.move_forward(agent.location))
                    if not agent.bump:
                        self.buttons[yi][xi].config(text="")
                        xf, yf = agent.location
                        self.buttons[yf][xf].config(text=agent_label(agent))
                        agent.performance -= 1
                '''

            if action == 'RIGHT':
                self.buttons[yi][xi].config(text='')
                xi = xi + 1
                self.buttons[yi][xi].config(text='>')
                self.agent.location = xi, yi

            elif action == 'LEFT':
                self.buttons[yi][xi].config(text='')
                xi = xi - 1
                self.buttons[yi][xi].config(text='<')
                self.agent.location = xi, yi

            elif action == 'DOWN':
                self.buttons[yi][xi].config(text='')
                yi = yi - 1
                self.buttons[yi][xi].config(text='v')
                self.agent.location = xi, yi

            elif action == 'UP':
                self.buttons[yi][xi].config(text='')
                yi = yi + 1
                self.buttons[yi][xi].config(text='^')
                self.agent.location = xi, yi


            if action != 'NoOp':
                agent.performance -= 1

        NumSteps_label.config(text=str(self.stepCount))
        TotalCost_label.config(text=str(self.agent.performance))

    def read_env(self):
        """read_env: This sets proper wall or Dirt status based on bg color"""
        """Reads the current state of the GUI environment."""
        self.dirtCount = 0
        for j, btn_row in enumerate(self.buttons):
            for i, btn in enumerate(btn_row):
                if (j != 0 and j != len(self.buttons) - 1) and (i != 0 and i != len(btn_row) - 1):
                    if self.some_things_at((i, j)):  # and (i, j) != agt_loc:
                        for thing in self.list_things_at((i, j)):
                            if not isinstance(thing, Agent):
                                self.delete_thing(thing)
                    if btn['bg'] == 'grey':  # adding dirt
                        self.add_thing(Dirt(), (i, j))
                        self.dirtCount += 1
                    elif btn['bg'] == 'red':  # adding wall
                        self.add_thing(Wall(), (i, j))

    def update_env(self):
        """Updates the GUI environment according to the current state."""
        self.read_env()
        self.step()
        self.stepCount += 1

    def step(self):
        """updates the environment one step. Currently it is associated with one click of 'Step' button.
        """
        if env.dirtCount == 0:
            print("Everything is clean. DONE!")
            self.done = True
            return

        if len(self.solution) == 0:
            self.execute_action(self.agent, 'Suck')
            self.read_env()
            if env.dirtCount > 0 and self.searchAgent is not None:
                self.searchAgent.generateNextSolution()
                self.running = False
        else:
            move = self.solution.pop()
            self.execute_action(self.agent, move)

    def run(self, steps=500, delay=0.1):
        """Run the Environment for given number of time steps,"""
        self.running = True

        for step in range(steps):
            if self.is_done() or self.running is False:
                if (env.dirtCount == 0):
                    print("everything is clean")
                    break

            self.update_env()
            sleep(delay)
            Tk.update(self.root)

    def reset_env(self):
        """Resets the GUI and agents environment to the initial clear state."""
        self.running = False
        NumSteps_label.config(text=str(0))
        TotalCost_label.config(text=str(0))

        for j, btn_row in enumerate(self.buttons):
            for i, btn in enumerate(btn_row):
                if (j != 0 and j != len(self.buttons) - 1) and (i != 0 and i != len(btn_row) - 1):
                    if self.some_things_at((i, j)):
                        for thing in self.list_things_at((i, j)):
                            if not isinstance(thing, Agent):
                                self.delete_thing(thing)
                    btn.config(bg='white', text='', state='normal')

        self.setupTestEnvironment()


"""
Our search Agents ignore ignore environment percepts for planning. The planning is done based ons static
 data from environment at the beginning. The environment if fully observable
 """


def XYSearchAgentProgram(percept):
    pass


class XYSearchAgent(Agent):
    """The modified SimpleRuleAgent for the GUI environment."""

    def __init__(self, program, loc):
        super().__init__(program)
        self.location = loc
        self.direction = Direction("up")
        self.searchType = searchTypes[0]
        self.stepCount = 0


if __name__ == "__main__":
    win = Tk()
    win.title("Searching Cleaning Robot")
    win.geometry("800x750+50+50")
    win.resizable(True, True)
    frame = Frame(win, bg='black')
    frame.pack(side='bottom')
    topframe = Frame(win, bg='black')
    topframe.pack(side='top')

    wid = 10
    if len(sys.argv) > 1:
        wid = int(sys.argv[1])

    hig = 10
    if len(sys.argv) > 2:
        hig = int(sys.argv[2])

    env = Gui(win, wid, hig)

    theAgent = XYSearchAgent(program=XYSearchAgentProgram, loc=(hig // 2, wid // 2))
    x, y = theAgent.location
    env.add_agent(theAgent, (y, x))

    NumSteps_label = Label(topframe, text='NumSteps: 0', bg='green', fg='white', bd=2, padx=2, pady=2)
    NumSteps_label.pack(side='left')
    TotalCost_label = Label(topframe, text='TotalCost: 0', bg='blue', fg='white', padx=2, pady=2)
    TotalCost_label.pack(side='right')
    reset_button = Button(frame, text='Reset', height=2, width=5, padx=2, pady=2)
    reset_button.pack(side='left')
    next_button = Button(frame, text='Next', height=2, width=5, padx=2, pady=2)
    next_button.pack(side='left')
    run_button = Button(frame, text='Run', height=2, width=5, padx=2, pady=2)
    run_button.pack(side='left')

    next_button.config(command=env.update_env)
    reset_button.config(command=env.reset_env)
    run_button.config(command=env.run)

    searchTypeStr = StringVar(win)
    searchTypeStr.set(searchTypes[0])
    searchTypeStr_dropdown = OptionMenu(frame, searchTypeStr, *searchTypes, command=env.setSearchEngine)
    searchTypeStr_dropdown.pack(side='left')

    win.mainloop()

