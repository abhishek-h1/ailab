"""
#drive link for refercing the code:- https://drive.google.com/drive/folders/1sSUbeoZsvnZxgYbzJ7LFcvXxkMJBr7-m?usp=sharing

EXP1 Implementation of toy problems (Banana and camel problem) (see more available toy problems below)

A person has 3000 bananas and a camel. The person wants to transport the maximum number of bananas to a destination which is 1000 KMs away, using only the camel as a mode of transportation. The camel cannot carry more than 1000 bananas at a time and eats a banana every km it travels. What is the maximum number of bananas that can be transferred to the destination using only camel (no other mode of transportation is allowed). 

Solution: 
Let’s see what we can infer from the question:

We have a total of 3000 bananas.
The destination is 1000KMs
Only 1 mode of transport.
Camel can carry a maximum of 1000 banana at a time.
Camel eats a banana every km it travels.
With all these points, we can say that person won’t be able to transfer any banana to the destination as the camel is going to eat all the banana on its way to the destination.

But the trick here is to have intermediate drop points, then, the camel can make several short trips in between. 

Also, we try to maintain the number of bananas at each point to be multiple of 1000.

Let’s have 2 drop points in between the source and destination.

With 3000 bananas at the source. 2000 at a first intermediate point and 1000 at 2nd intermediate point.

Source————–IP1—————–IP2———————-Destination

3000       x km        2000     y km           1000          z km

——————–>  |    —————> | ———————–>

<——————-    |    <————–  |

——————->    |    —————> |

<——————      |                             |

——————->     |                             |

To go from source to IP1 point camel has to take a total of 5 trips 3 forward and 2 backward. Since we have 3000 bananas to transport.
The same way from IP1 to IP2 camel has to take a total of 3 trips, 2 forward and 1 backward. Since we have 2000 bananas to transport.
At last from IP2 to a destination only 1 forward move.
Let’s see the total number of bananas consumed at every point.

From the source to IP1 its 5x bananas, as the distance between the source and IP1 is x km and the camel had 5 trips.
From IP1 to IP2 its 3y bananas, as the distance between IP1 and IP2 is y km and the camel had 3 trips.
From IP2 to destination its z bananas.
We now try to calculate the distance between the points:

3000 – 5x = 2000 so we get x = 200
2000-3y = 1000 so we get y = 333.33 but here the distance is also the number of bananas and it cannot be fraction so we take y =333 and at IP2 we have the number of bananas equal 1001, so its 2000-3y = 1001
So the remaining distance to the market is 1000 -x-y =z i.e  1000-200-333 => z =467.
Now, there are 1001 bananas at IP2. However the camel can carry only 1000 bananas at a time, so we need to leave one banana behind.

So from IP2 to the destination point camel eats 467 bananas. The remaining bananas are 1000-467=533.
 

So the maximum number of bananas that can be transferred is 533.


"""


a=3000                     #total bananas
l=1000
s=0
while(a>l):
    n=(a/l)*2-1            #no. of trips
    x=l//n                 #distance of check post trips
    s=s+x                  #storing no of bananas consumed
    a=a-l                  #next load of banana
kmleft=a-s
bananasneeded=kmleft
bananasleft=a-bananasneeded
print(bananasleft)


#---------------------------------------------------------------------------------------------------------------------
"""
EXP1 Implementation of toy problems (Maximum Ticket Price Problem) (see more available toy problems below)

Maximize the profit after selling the tickets
Difficulty Level : Medium
Last Updated : 02 Feb, 2022
Given array seats[] where seat[i] is the number of vacant seats in the ith row in a stadium for a cricket match. There are N people in a queue waiting to buy the tickets. Each seat costs equal to the number of vacant seats in the row it belongs to. The task is to maximize the profit by selling the tickets to N people.

Examples: 

Input: seats[] = {2, 1, 1}, N = 3 
Output: 4 
Person 1: Sell the seat in the row with 
2 vacant seats, seats = {1, 1, 1} 
Person 2: All the rows have 1 vacant 
seat each, seats[] = {0, 1, 1} 
Person 3: seats[] = {0, 0, 1}

Input: seats[] = {2, 3, 4, 5, 1}, N = 6 
Output: 22 
 

Recommended: Please try your approach on {IDE} first, before moving on to the solution.
Approach: In order to maximize the profit, the ticket must be for the seat in a row which has the maximum number of vacant seats and the number of vacant seats in that row will be decrement by 1 as one of the seats has just been sold. All the persons can be sold a seat ticket until there are vacant seats. This can be computed efficiently with the help of a priority_queue.

Below is the implementation of the above approach: 

"""

#---------------------------------------------------------------------------------------------------------------------

def maxAmount(M, N, seats): 

	q = [] 

	for i in range(M): 
		q.append(seats[i]) 

	ticketSold = 0

	ans = 0

	q.sort(reverse = True) 
	while (ticketSold < N and q[0] > 0): 
		ans = ans + q[0] 
		temp = q[0] 
		q = q[1:] 
		q.append(temp - 1) 
		q.sort(reverse = True) 
		ticketSold += 1

	return ans

if __name__ == '__main__': 

	seats = []

	rows = int(input("Enter number of rows available : ")) 

	for i in range(0, rows):
		empty = int(input())
		seats.append(empty)

	print(seats)
	M = len(seats) 
	N = int(input("Enter the number of People standing in the queue : "))

	print("Maximum Profit generated = ", maxAmount(N, M, seats))


#---------------------------------------------------------------------------------------------------------------------

"""
EXP1 Implementation of toy problems (N-queens) 

The N Queen is the problem of placing N chess queens on an N×N chessboard so that no two queens attack each other. For example, following is a solution for 4 Queen problem.

Backtracking Algorithm 
The idea is to place queens one by one in different columns, starting from the leftmost column. When we place a queen in a column, we check for clashes with already placed queens. In the current column, if we find a row for which there is no clash, we mark this row and column as part of the solution. If we do not find such a row due to clashes then we backtrack and return false.
 

1) Start in the leftmost column
2) If all queens are placed
    return true
3) Try all rows in the current column. 
   Do following for every tried row.
    a) If the queen can be placed safely in this row 
       then mark this [row, column] as part of the 
       solution and recursively check if placing
       queen here leads to a solution.
    b) If placing the queen in [row, column] leads to
       a solution then return true.
    c) If placing queen doesn't lead to a solution then
       unmark this [row, column] (Backtrack) and go to 
       step (a) to try other rows.
4) If all rows have been tried and nothing worked,
   return false to trigger backtracking.
   

"""

# Python3 program to solve N Queen
# Problem using backtracking
global N
N = 4

def printSolution(board):
	for i in range(N):
		for j in range(N):
			print (board[i][j], end = " ")
		print()

# A utility function to check if a queen can
# be placed on board[row][col]. Note that this
# function is called when "col" queens are
# already placed in columns from 0 to col -1.
# So we need to check only left side for
# attacking queens
def isSafe(board, row, col):

	# Check this row on left side
	for i in range(col):
		if board[row][i] == 1:
			return False

	# Check upper diagonal on left side
	for i, j in zip(range(row, -1, -1),
					range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	# Check lower diagonal on left side
	for i, j in zip(range(row, N, 1),
					range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	return True

def solveNQUtil(board, col):
	
	# base case: If all queens are placed
	# then return true
	if col >= N:
		return True

	# Consider this column and try placing
	# this queen in all rows one by one
	for i in range(N):

		if isSafe(board, i, col):
			
			# Place this queen in board[i][col]
			board[i][col] = 1

			# recur to place rest of the queens
			if solveNQUtil(board, col + 1) == True:
				return True

			# If placing queen in board[i][col
			# doesn't lead to a solution, then
			# queen from board[i][col]
			board[i][col] = 0

	# if the queen can not be placed in any row in
	# this column col then return false
	return False

# This function solves the N Queen problem using
# Backtracking. It mainly uses solveNQUtil() to
# solve the problem. It returns false if queens
# cannot be placed, otherwise return true and
# placement of queens in the form of 1s.
# note that there may be more than one
# solutions, this function prints one of the
# feasible solutions.
def solveNQ():
	board = [ [0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0] ]

	if solveNQUtil(board, 0) == False:
		print ("Solution does not exist")
		return False

	printSolution(board)
	return True

# Driver Code
solveNQ()

# This code is contributed by Divyanshu Mehta

#---------------------------------------------------------------------------------------------------------------------


"""
EXP2 Implementation of constraint satisfaction problems
Aim: To create a program that solves graph coloring which is a problem where the graph vertex are colored so that no adjacent vertex are the same color
 
1. Color first vertex with first color.
2. For  V-1 vertices, consider the currently picked vertex and color it with the
lowest numbered color that has not been used on any previously colored vertices adjacent to it.
3.If all previously used colors
appear on vertices adjacent to v, assign a new color to it.

"""

def addEdge(adj, v, w):
     
    adj[v].append(w)
     
    # Note: the graph is undirected
    adj[w].append(v)
    return adj
 
# Assigns colors (starting from 0) to all
# vertices and prints the assignment of colors
def greedyColoring(adj, V):
     
    result = [-1] * V
 
    # Assign the first color to first vertex
    result[0] = 0;
 
 
    # A temporary array to store the available colors.
    # True value of available[cr] would mean that the
    # color cr is assigned to one of its adjacent vertices
    available = [False] * V
 
    # Assign colors to remaining V-1 vertices
    for u in range(1, V):
         
        # Process all adjacent vertices and
        # flag their colors as unavailable
        for i in adj[u]:
            if (result[i] != -1):
                available[result[i]] = True
 
        # Find the first available color
        cr = 0
        while cr < V:
            if (available[cr] == False):
                break
             
            cr += 1
             
        # Assign the found color
        result[u] = cr
 
        # Reset the values back to false
        # for the next iteration
        for i in adj[u]:
            if (result[i] != -1):
                available[result[i]] = False
 
    # Print the result
    for u in range(V):
        print("Vertex", u, " --->  Color", result[u])
 
# Driver Code
if __name__ == '__main__':
     
    g1 = [[] for i in range(5)]
    g1 = addEdge(g1, 0, 1)
    g1 = addEdge(g1, 0, 2)
    g1 = addEdge(g1, 1, 2)
    g1 = addEdge(g1, 1, 3)
    g1 = addEdge(g1, 2, 3)
    g1 = addEdge(g1, 3, 4)
    print("Coloring of graph 1 ")
    greedyColoring(g1, 5)
 
    g2 = [[] for i in range(5)]
    g2 = addEdge(g2, 0, 1)
    g2 = addEdge(g2, 0, 2)
    g2 = addEdge(g2, 1, 2)
    g2 = addEdge(g2, 1, 4)
    g2 = addEdge(g2, 2, 4)
    g2 = addEdge(g2, 4, 3)
    print("\nColoring of graph 2")
    greedyColoring(g2, 5)


#---------------------------------------------------------------------------------------------------------------------


"""
EXP3 Implementation of Constraint Satisfaction Problem (CSP)
Aim:- To implement the Constraint Satisfaction Problem based on the given constraints.

Manual Procedure: SEND + MORE = MONEY

            5 4  3  2 1
              S  E  N D
              M  O  R E
        +     c3 c2 c1
        -------------------
            M O N E Y
        -------------------
        
1)From Column 5, M=1, since it is only carry-over possible from sum of 2 single digit number in column 4.
2)To produce a carry from column 4 to column 5 'S + M' is atleast 9 so 'S=8or9' so 'S+M=9or10' & so 'O = 0 or 1'. But 'M=1', so 'O = 0'.
3)If there is carry from Column 3 to 4 then 'E=9' & so 'N=0'. But 'O = 0' so there is no carry & 'S=9' & 'c3=0'.
4)If there is no carry from column 2 to 3 then 'E=N' which is impossible, therefore there is carry & 'N=E+1' & 'c2=1'.
5)If there is carry from column 1 to 2 then 'N+R=E mod 10' & 'N=E+1' so 'E+1+R=E mod 10', so 'R=9' but 'S=9', so there must be carry from column 1 to 2. Therefore 'c1=1' & 'R=8'.
6)To produce carry 'c1=1' from column 1 to 2, we must have 'D+E=10+Y' as Y cannot be 0/1 so D+E is atleast 12. As D is atmost 7 & E is atleast 5 (D cannot be 8 or 9 as it is already assigned). N is atmost 7 & 'N=E+1' so 'E=5or6'.
7)If E were 6 & D+E atleast 12 then D would be 7, but 'N=E+1' & N would also be 7 which is impossible. Therefore 'E=5' &
'N=6'.
8)D+E is atleast 12 for that we get 'D=7' & 'Y=2'.

Results
The Constraint Satisfaction Problem was implemented successfully where the possible solutions were displayed based on user input.
"""

import itertools

def get_value(word, substitution):
    s = 0
    factor = 1
    for letter in reversed(word):
        s += factor * substitution[letter]
        factor *= 10
    return s

def solve2(equation):
    left, right = equation.lower().replace(' ', '').split('=')
    left = left.split('+')
    letters = set(right)
    for word in left:
        for letter in word:
            letters.add(letter)
    letters = list(letters)

    digits = range(10)
    for perm in itertools.permutations(digits, len(letters)):
        sol = dict(zip(letters, perm))

        if sum(get_value(word, sol) for word in left) == get_value(right, sol):
            print(' + '.join(str(get_value(word, sol)) for word in left) + " = {} (mapping: {})".format(get_value(right, sol), sol))
                                                                                                        
print('EAT + THAT = APPLE ')
solve2('POINT + ZERO = ENERGY ')    

""" OR The other code is below (Dont Implement both codes) """

def solutions():
    # letters = ('s', 'e', 'n', 'd', 'm', 'o', 'r', 'y')
    all_solutions = list()
    for s in range(9, -1, -1):
        for e in range(9, -1, -1):
            for n in range(9, -1, -1):
                for d in range(9, -1, -1):
                    for m in range(9, 0, -1):
                        for o in range(9, -1, -1):
                            for r in range(9, -1, -1):
                                for y in range(9, -1, -1):
                                    if len(set([s, e, n, d, m, o, r, y])) == 8:
                                        send = 1000 * s + 100 * e + 10 * n + d
                                        more = 1000 * m + 100 * o + 10 * r + e
                                        money = 10000 * m + 1000 * o + 100 * n + 10 * e + y

                                        if send + more == money:
                                            all_solutions.append((send, more, money))
    return all_solutions

print(solutions())                                            


#---------------------------------------------------------------------------------------------------------------------

"""
EXP4 Implementation and Analysis of DFS and BFS for same application

AIM:- To Implement both DFS and BFS of a same application, the application choosen is 
Robot In a Maze

------------------------------------------------------------

Explanation for DFS:-

Backtracking Algorithm: Backtracking is an algorithmic-technique for solving problems recursively by trying to build a solution incrementally. Solving one piece at a time, and removing those solutions that fail to satisfy the constraints of the problem at any point of time (by time, here, is referred to the time elapsed till reaching any level of the search tree) is the process of backtracking.

Approach: Form a recursive function, which will follow a path and check if the path reaches the destination or not. If the path does not reach the destination then backtrack and try other paths. 

Algorithm:  

----> Write following as points:-

Create a solution matrix, initially filled with 0’s.
Create a recursive function, which takes initial matrix, output matrix and position of rat (i, j).
if the position is out of the matrix or the position is not valid then return.
Mark the position output[i][j] as 1 and check if the current position is destination or not. If destination is reached print the output matrix and return.
Recursively call for position (i+1, j) and (i, j+1).
Unmark position (i, j), i.e output[i][j] = 0.


------------------------------------------------------------

Explanation for BFS:-

The idea is inspired from Lee algorithm and uses BFS.  

----> Write following as points:-
We start from the source cell and calls BFS procedure.
We maintain a queue to store the coordinates of the matrix and initialize it with the source cell.
We also maintain a Boolean array visited of same size as our input matrix and initialize all its elements to false.
We LOOP till queue is not empty
Dequeue front cell from the queue
Return if the destination coordinates have reached.
For each of its four adjacent cells, if the value is 1 and they are not visited yet, we enqueue it in the queue and also mark them as visited.
Note that BFS works here because it doesn’t consider a single path at once. It considers all the paths starting from the source and moves ahead one unit in all those paths at the same time which makes sure that the first time when the destination is visited, it is the shortest path.
Below is the implementation of the idea –  

------------------------------------------------------------


Time Complexity of BFS = O(V+E) where V is vertices and E is edges. 
Time Complexity of DFS is also O(V+E) where V is vertices and E is edges
"""

# Python3 program to solve Rat in a Maze
# problem using backtracking

# Maze size
N = 4

# A utility function to print solution matrix sol
def printSolution( sol ):
	
	for i in sol:
		for j in i:
			print(str(j) + " ", end ="")
		print("")

# A utility function to check if x, y is valid
# index for N * N Maze
def isSafe( maze, x, y ):
	
	if x >= 0 and x < N and y >= 0 and y < N and maze[x][y] == 1:
		return True
	
	return False

""" This function solves the Maze problem using Backtracking.
	It mainly uses solveMazeUtil() to solve the problem. It
	returns false if no path is possible, otherwise return
	true and prints the path in the form of 1s. Please note
	that there may be more than one solutions, this function
	prints one of the feasible solutions. """
def solveMaze( maze ):
	
	# Creating a 4 * 4 2-D list
	sol = [ [ 0 for j in range(4) ] for i in range(4) ]
	
	if solveMazeUtil(maze, 0, 0, sol) == False:
		print("Solution doesn't exist");
		return False
	
	printSolution(sol)
	return True
	
# A recursive utility function to solve Maze problem
def solveMazeUtil(maze, x, y, sol):
	
	# if (x, y is goal) return True
	if x == N - 1 and y == N - 1 and maze[x][y]== 1:
		sol[x][y] = 1
		return True
		
	# Check if maze[x][y] is valid
	if isSafe(maze, x, y) == True:
		# Check if the current block is already part of solution path.
		if sol[x][y] == 1:
			return False
		
		# mark x, y as part of solution path
		sol[x][y] = 1
		
		# Move forward in x direction
		if solveMazeUtil(maze, x + 1, y, sol) == True:
			return True
			
		# If moving in x direction doesn't give solution
		# then Move down in y direction
		if solveMazeUtil(maze, x, y + 1, sol) == True:
			return True
		
		# If moving in y direction doesn't give solution then
		# Move back in x direction
		if solveMazeUtil(maze, x - 1, y, sol) == True:
			return True
			
		# If moving in backwards in x direction doesn't give solution
		# then Move upwards in y direction
		if solveMazeUtil(maze, x, y - 1, sol) == True:
			return True
		
		# If none of the above movements work then
		# BACKTRACK: unmark x, y as part of solution path
		sol[x][y] = 0
		return False

# Driver program to test above function
if __name__ == "__main__":
	# Initialising the maze
	maze = [ [1, 0, 0, 0],
			[1, 1, 0, 1],
			[0, 1, 0, 0],
			[1, 1, 1, 1] ]
			
	solveMaze(maze)

""" ----------> The above code is for DFS the below code is for BFS """

import sys

#Maze in binary representation
matrix =[ [ 1, 1, 0, 0, 0, 1, 1],
          [ 0, 1, 1, 1, 1, 1, 1],
          [ 1, 0, 0, 1, 0, 1, 1],
          [ 0, 1, 1, 1, 0, 0, 1],
          [ 0, 1, 0, 1, 1, 1, 1],
          [ 0, 1, 0, 0, 1, 0, 0],
          [ 1, 0, 1, 1, 1, 1, 1] ]

#2D Array mapping to mark visited cell
visited = [[0 for x in range(len(matrix[0]))] for y in range(len(matrix))]

#Source and destination cell
start =(0,0)
end = (6,6)

#Initially storing the max possible integer as the shortest path length
shortLength = sys.maxsize
length=0
hasPath =False

#Function to initiate the search
def findPath():
    visit(start[0], start[1])

#Function to visit a cell and recursively make next move
def visit(x, y):
  global length, shortLength, visited, hasPath

  #Base Condition - Reached the destination cell
  if x==end[0]and y==end[1]:
    #Update hasPath to True (Maze has a solution)
    hasPath = True
    #Store the minimum of the path length
    shortLength = min(length, shortLength)
    #Backtrack to explore more possible paths
    return
  
  #Mark current cell as visited
  visited[x][y] = 1
  #Increment the current path length by 1
  length +=1

  #Check for next move:
  #1.Right
  if canVisit(x+1, y):
    visit(x+1, y)
  
  #2.Down
  if canVisit(x, y+1):
    visit(x, y+1)
  
  #3.Left
  if canVisit(x-1, y):
    visit(x-1, y)

  #4.Up
  if canVisit(x, y-1):
    visit(x, y-1)

  #Backtrack by unvisiting the current cell and
  #decrementing the value of current path length
  visited[x][y] = 0
  length -= 1

#Function checks if (x,y) cell is valid cell or not
def canVisit(x, y):
  #check maze boundaries
  if x<0 or y<0 or x>=len(matrix[0]) or y>=len(matrix):
    return False
  #check 0 or already visited
  if matrix[x][y]==0 or visited[x][y]==1:
    return False
  return True

#Driver code
if __name__ == '__main__':
  findPath()

  #output only if any path to the destination was found 
  if hasPath:
    print(f"Shortest Path Length: {shortLength}")
  else:
    print("No Path Possible")


#---------------------------------------------------------------------------------------------------------------------
"""
EXP5 Implementation of A* algorithm for a real world problem

The goal of this graph problem is to find the shortest path between a starting location
and destination location. A map has been used to create a graph with actual distances between locations. 
The A* algorithm uses a Graph class, a Node class and heuristics to find the shortest path in a fast manner. Heuristics is calculated as straight-line distances 
(air-travel distances) between locations, air-travel distances will never be larger than actual distances.

Time Complexity:-
The time complexity of A* Search Algorithm depends on the heuristic.
In the worst case of an unbounded search space, the number of nodes expanded is exponential in the depth 
of the solution (the shortest path) d: O(b^d), where b is the branching factor (the average number of successors per state). 
This assumes that a goal state exists at all, and is reachable from the start state; if it is not, and the state space is infinite, the algorithm will not terminate.

Space Complexity:-
The space complexity of A* Search Algorithm is roughly the same as that of all other graph search algorithms i.e. O(b^d), as it keeps all generated nodes in memory.


"""

# graph class
class Graph:
    
    # init class
    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    # create undirected graph by adding symmetric edges
    def make_undirected(self):
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.graph_dict.setdefault(b, {})[a] = dist

    # add link from A and B of given distance, and also add the inverse link if the graph is undirected
    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        if not self.directed:
            self.graph_dict.setdefault(B, {})[A] = distance

    # get neighbors or a neighbor
    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    # return list of nodes in the graph
    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)
    
# node class
class Node:

    # init class
    def __init__(self, name:str, parent:str):
        self.name = name
        self.parent = parent
        self.g = 0 # distance to start node
        self.h = 0 # distance to goal node
        self.f = 0 # total cost

    # compare nodes
    def __eq__(self, other):
        return self.name == other.name

    # sort nodes
    def __lt__(self, other):
         return self.f < other.f

    # print node
    def __repr__(self):
        return ('({0},{1})'.format(self.name, self.f))

# A* search
def astar_search(graph, heuristics, start, end):
    
    # lists for open nodes and closed nodes
    open = []
    closed = []

    # a start node and an goal node
    start_node = Node(start, None)
    goal_node = Node(end, None)

    # add start node
    open.append(start_node)
    
    # loop until the open list is empty
    while len(open) > 0:

       
        open.sort()                                 # sort open list to get the node with the lowest cost first
        current_node = open.pop(0)                  # get node with the lowest cost
        closed.append(current_node)                 # add current node to the closed list
        
        # check if we have reached the goal, return the path
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.name + ': ' + str(current_node.g))
                current_node = current_node.parent
            path.append(start_node.name + ': ' + str(start_node.g))
            return path[::-1]
        neighbors = graph.get(current_node.name)    # get neighbours
        
        # loop neighbors
        for key, value in neighbors.items():
            neighbor = Node(key, current_node)      # create neighbor node
            if(neighbor in closed):                 # check if the neighbor is in the closed list
                continue

            # calculate full path cost
            neighbor.g = current_node.g + graph.get(current_node.name, neighbor.name)
            neighbor.h = heuristics.get(neighbor.name)
            neighbor.f = neighbor.g + neighbor.h

            # check if neighbor is in open list and if it has a lower f value
            if(add_to_open(open, neighbor) == True):

                # everything is green, add neighbor to open list
                open.append(neighbor)

    # return None, no path is found
    return None

# check if a neighbor should be added to open list
def add_to_open(open, neighbor):
    for node in open:
        if (neighbor == node and neighbor.f > node.f):
            return False
    return True

# create a graph
graph = Graph() # user-based input for edges will be updated in the upcoming days
# create graph connections (Actual distance)
graph.connect('Frankfurt', 'Wurzburg', 111)
graph.connect('Frankfurt', 'Mannheim', 85)
graph.connect('Wurzburg', 'Nurnberg', 104)
graph.connect('Wurzburg', 'Stuttgart', 140)
graph.connect('Wurzburg', 'Ulm', 183)
graph.connect('Mannheim', 'Nurnberg', 230)
graph.connect('Mannheim', 'Karlsruhe', 67)
graph.connect('Karlsruhe', 'Basel', 191)
graph.connect('Karlsruhe', 'Stuttgart', 64)
graph.connect('Nurnberg', 'Ulm', 171)
graph.connect('Nurnberg', 'Munchen', 170)
graph.connect('Nurnberg', 'Passau', 220)
graph.connect('Stuttgart', 'Ulm', 107)
graph.connect('Basel', 'Bern', 91)
graph.connect('Basel', 'Zurich', 85)
graph.connect('Bern', 'Zurich', 120)
graph.connect('Zurich', 'Memmingen', 184)
graph.connect('Memmingen', 'Ulm', 55)
graph.connect('Memmingen', 'Munchen', 115)
graph.connect('Munchen', 'Ulm', 123)
graph.connect('Munchen', 'Passau', 189)
graph.connect('Munchen', 'Rosenheim', 59)
graph.connect('Rosenheim', 'Salzburg', 81)
graph.connect('Passau', 'Linz', 102)
graph.connect('Salzburg', 'Linz', 126)
# make graph undirected, create symmetric connections
graph.make_undirected()
# create heuristics (straight-line distance, air-travel distance)
heuristics = {}
heuristics['Basel'] = 204
heuristics['Bern'] = 247
heuristics['Frankfurt'] = 215
heuristics['Karlsruhe'] = 137
heuristics['Linz'] = 318
heuristics['Mannheim'] = 164
heuristics['Munchen'] = 120
heuristics['Memmingen'] = 47
heuristics['Nurnberg'] = 132
heuristics['Passau'] = 257
heuristics['Rosenheim'] = 168
heuristics['Stuttgart'] = 75
heuristics['Salzburg'] = 236
heuristics['Wurzburg'] = 153
heuristics['Zurich'] = 157
heuristics['Ulm'] = 0
# run the search algorithm
path = astar_search(graph, heuristics, 'Frankfurt', 'Ulm')
print("Path:", path)


#---------------------------------------------------------------------------------------------------------------------
'''
EXP6 Uncertain methods

Aim : To solve Sudoku using Uncertain Method
Description: Sudoku is a well-known puzzle game and popular for explaining search problems. 
Given an initial 9x9 grid of cells containing numbers between 1 and 9 or blanks, all blanks must be filled with numbers. 
You win Sudoku if you find all values such that every row, column, and 3x3 sub square contains the numbers 1–9, each with a single occurrence.

'''
size = 9
#empty cells have value zero
matrix = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9]]

#print sudoku
def print_sudoku():
    for i in matrix:
        print (i)
#assign cells and check
def number_unassigned(row, col):
    num_unassign = 0
    for i in range(0,size):
        for j in range (0,size):
            #cell is unassigned
            if matrix[i][j] == 0:
                row = i
                col = j
                num_unassign = 1
                a = [row, col, num_unassign]
                return a
    a = [-1, -1, num_unassign]
    return a
#check validity of number 
def is_safe(n, r, c):
    #checking in row
    for i in range(0,size):
        #there is a cell with same value
        if matrix[r][i] == n:
            return False
    #checking in column
    for i in range(0,size):
        #there is a cell with same value
        if matrix[i][c] == n:
            return False
    row_start = (r//3)*3
    col_start = (c//3)*3;
    #checking submatrix
    for i in range(row_start,row_start+3):
        for j in range(col_start,col_start+3):
            if matrix[i][j]==n:
                return False
    return True

#check validity of number
def solve_sudoku():
    row = 0
    col = 0
    #if all cells are assigned then the sudoku is already solved
    #pass by reference because number_unassigned will change the values of row and col
    a = number_unassigned(row, col)
    if a[2] == 0:
        return True
    row = a[0]
    col = a[1]
    #number between 1 to 9
    for i in range(1,10):
        #if we can assign i to the cell or not
        #the cell is matrix[row][col]
        if is_safe(i, row, col):
            matrix[row][col] = i
            #backtracking
            if solve_sudoku():
                return True
            #f we can't proceed with this solution
            #reassign the cell
            matrix[row][col]=0
    return False

if solve_sudoku():
    print_sudoku()
else:
    print("No solution")
    
#-----------------------------------------------------------------------------------------

'''
EXP7 PART A UNIFICATION

Aim:- To perform unification in real world applications

Problem Formulation:-

* To find a mapping between the two expressions that may both contain variables.

* Bind the variables variables to their values in the given expression until no bound variables remain.

Initial State:-

expression 1 = f(X,h(x),Y,g(Y))

expression 2 = f(g(z),w,z,X)

Final State:-

X : g(z)
w : h(x)
Y : Z

expr 1 = f(g(z),h(g(z),z,g(z)))
expr 2 = f(g(z), h(g(z)), z, g(x))

Problem Solving:-

1. Unify f(X,h(x),Y,g(Y)) and f(g(z),w,z,x)
2. It would loop through each argument 
3. Unify (X,g(z)) is invoked
{
    X is a variable therefore substitute X = g(z)
}
4. Unify (h(x),w) is invoked.
{
    W is a variable 
    Therefore, Substitute W = h(x)
}
5. The substitution are mapped to a python dictionary and it expands as
  
  { X = g(z) , w = h(X)}
  
6. Unify (Y,Z) is invoked

  {
    Both Y and Z are variable , hence are added directly to the dictionary.
    
    {X = g(z) , W = h(x) , Y = Z}
   }
7. Unify (g(Y), X) is invoked.

{
    X is a variable but is already present in the dictionary.
}
Therefore, the unify would be on the substituted value if it is not a variable , i.e. , if the substituted value is not a variable 

Unify (g(Y),g(Z)) [Both the terms have g] 

Unify Y & Z [It is already present in the map] 

8. All the variables are bounded , unification is completed successfully.

Final Result:-

{
    X = g(z), W = h(x) , Y = Z
}         

'''
def get_index_comma(string):
    index_list = list()
    par_count = 0

    for i in range(len(string)):
        if string[i] == ',' and par_count == 0:
            index_list.append(i)
        elif string[i] == '(':
            par_count += 1
        elif string[i] == ')':
            par_count -= 1

    return index_list


def is_variable(expr):
    for i in expr:
        if i == '(' or i == ')':
            return False

    return True


def process_expression(expr):
    expr = expr.replace(' ', '')
    index = None
    for i in range(len(expr)):
        if expr[i] == '(':
            index = i
            break
    predicate_symbol = expr[:index]
    expr = expr.replace(predicate_symbol, '')
    expr = expr[1:len(expr) - 1]
    arg_list = list()
    indices = get_index_comma(expr)

    if len(indices) == 0:
        arg_list.append(expr)
    else:
        arg_list.append(expr[:indices[0]])
        for i, j in zip(indices, indices[1:]):
            arg_list.append(expr[i + 1:j])
        arg_list.append(expr[indices[len(indices) - 1] + 1:])

    return predicate_symbol, arg_list


def get_arg_list(expr):
    _, arg_list = process_expression(expr)

    flag = True
    while flag:
        flag = False

        for i in arg_list:
            if not is_variable(i):
                flag = True
                _, tmp = process_expression(i)
                for j in tmp:
                    if j not in arg_list:
                        arg_list.append(j)
                arg_list.remove(i)

    return arg_list


def check_occurs(var, expr):
    arg_list = get_arg_list(expr)
    if var in arg_list:
        return True

    return False


def unify(expr1, expr2):

    if is_variable(expr1) and is_variable(expr2):
        if expr1 == expr2:
            return 'Null'
        else:
            return False
    elif is_variable(expr1) and not is_variable(expr2):
        if check_occurs(expr1, expr2):
            return False
        else:
            tmp = str(expr2) + '/' + str(expr1)
            return tmp
    elif not is_variable(expr1) and is_variable(expr2):
        if check_occurs(expr2, expr1):
            return False
        else:
            tmp = str(expr1) + '/' + str(expr2)
            return tmp
    else:
        predicate_symbol_1, arg_list_1 = process_expression(expr1)
        predicate_symbol_2, arg_list_2 = process_expression(expr2)

        # Step 2
        if predicate_symbol_1 != predicate_symbol_2:
            return False
        # Step 3
        elif len(arg_list_1) != len(arg_list_2):
            return False
        else:
            # Step 4: Create substitution list
            sub_list = list()

            # Step 5:
            for i in range(len(arg_list_1)):
                tmp = unify(arg_list_1[i], arg_list_2[i])

                if not tmp:
                    return False
                elif tmp == 'Null':
                    pass
                else:
                    if type(tmp) == list:
                        for j in tmp:
                            sub_list.append(j)
                    else:
                        sub_list.append(tmp)

            # Step 6
            return sub_list


if __name__ == '__main__':
    
    f1 = 'Q(a, g(x, a), f(y))'
    f2 = 'Q(a, g(f(b), a), x)'
    # f1 = input('f1 : ')
    # f2 = input('f2 : ')

    result = unify(f1, f2)
    if not result:
        print('The process of Unification failed!')
    else:
        print('The process of Unification successful!')
        print(result)


'''
EXP7 PART-B Resolution
Input :- (Give the File name as input.txt)

2
Friends(Alice,Bob,Charlie,Diana)
Friends(Diana,Charlie,Bob,Alice)
2
Friends(a,b,c,d)
NotFriends(a,b,c,d)  
                     
'''

'''

Aim:- Implementation of Resolution using Predicative logic in real world applications


Problem Formulation:-

By Building reputation proofs, i.e. proofs by contradictions prove a conclusion of those given statements based on conjunctive normal form.
or clausal form.


Initial State:-

John likes all kind of food.
Apple and vegetable are food
Anything anyone eats and not killed is food.
Anil eats peanuts and still alive
Harry eats everything that Anil eats.
Prove by resolution that:
John likes peanuts.


Step-1: Conversion of Facts into FOL

In the first step we will convert all the given statements into its first order logic.

Resolution in FOL
Step-2: Conversion of FOL into CNF

In First order logic resolution, it is required to convert the FOL into CNF as CNF form makes easier for resolution proofs.



Eliminate all implication (→) and rewrite
∀x ¬ food(x) V likes(John, x)
food(Apple) Λ food(vegetables)
∀x ∀y ¬ [eats(x, y) Λ ¬ killed(x)] V food(y)
eats (Anil, Peanuts) Λ alive(Anil)
∀x ¬ eats(Anil, x) V eats(Harry, x)
∀x¬ [¬ killed(x) ] V alive(x)
∀x ¬ alive(x) V ¬ killed(x)
likes(John, Peanuts).
Move negation (¬)inwards and rewrite
∀x ¬ food(x) V likes(John, x)
food(Apple) Λ food(vegetables)
∀x ∀y ¬ eats(x, y) V killed(x) V food(y)
eats (Anil, Peanuts) Λ alive(Anil)
∀x ¬ eats(Anil, x) V eats(Harry, x)
∀x ¬killed(x) ] V alive(x)
∀x ¬ alive(x) V ¬ killed(x)
likes(John, Peanuts).
Rename variables or standardize variables
∀x ¬ food(x) V likes(John, x)
food(Apple) Λ food(vegetables)
∀y ∀z ¬ eats(y, z) V killed(y) V food(z)
eats (Anil, Peanuts) Λ alive(Anil)
∀w¬ eats(Anil, w) V eats(Harry, w)
∀g ¬killed(g) ] V alive(g)
∀k ¬ alive(k) V ¬ killed(k)
likes(John, Peanuts).
Eliminate existential instantiation quantifier by elimination.
In this step, we will eliminate existential quantifier ∃, and this process is known as Skolemization. But in this example problem since there is no existential quantifier so all the statements will remain same in this step.
Drop Universal quantifiers.
In this step we will drop all universal quantifier since all the statements are not implicitly quantified so we don't need it.
¬ food(x) V likes(John, x)
food(Apple)
food(vegetables)
¬ eats(y, z) V killed(y) V food(z)
eats (Anil, Peanuts)
alive(Anil)
¬ eats(Anil, w) V eats(Harry, w)
killed(g) V alive(g)
¬ alive(k) V ¬ killed(k)
likes(John, Peanuts).
Note: Statements "food(Apple) Λ food(vegetables)" and "eats (Anil, Peanuts) Λ alive(Anil)" can be written in two separate statements.
Distribute conjunction ∧ over disjunction ¬.
This step will not make any change in this problem.
Step-3: Negate the statement to be proved

In this statement, we will apply negation to the conclusion statements, which will be written as ¬likes(John, Peanuts)

Step-4: Draw Resolution graph:


Now in this step, we will solve the problem by resolution tree using substitution. For the above problem, it will be given as follows:

Resolution in FOL
Hence the negation of the conclusion has been proved as a complete contradiction with the given set of statements.

Explanation of Resolution graph:
In the first step of resolution graph, ¬likes(John, Peanuts) , and likes(John, x) get resolved(canceled) by substitution of {Peanuts/x}, and we are left with ¬ food(Peanuts)
In the second step of the resolution graph, ¬ food(Peanuts) , and food(z) get resolved (canceled) by substitution of { Peanuts/z}, and we are left with ¬ eats(y, Peanuts) V killed(y) .
In the third step of the resolution graph, ¬ eats(y, Peanuts) and eats (Anil, Peanuts) get resolved by substitution {Anil/y}, and we are left with Killed(Anil) .
In the fourth step of the resolution graph, Killed(Anil) and ¬ killed(k) get resolve by substitution {Anil/k}, and we are left with ¬ alive(Anil) .
In the last step of the resolution graph ¬ alive(Anil) and alive(Anil) get resolved.
                     
'''

import copy
import time


class Parameter:
    variable_count = 1

    def __init__(self, name=None):
        if name:
            self.type = "Constant"
            self.name = name
        else:
            self.type = "Variable"
            self.name = "v" + str(Parameter.variable_count)
            Parameter.variable_count += 1

    def isConstant(self):
        return self.type == "Constant"

    def unify(self, type_, name):
        self.type = type_
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name


class Predicate:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def __eq__(self, other):
        return self.name == other.name and all(a == b for a, b in zip(self.params, other.params))

    def __str__(self):
        return self.name + "(" + ",".join(str(x) for x in self.params) + ")"

    def getNegatedPredicate(self):
        return Predicate(negatePredicate(self.name), self.params)


class Sentence:
    sentence_count = 0

    def __init__(self, string):
        self.sentence_index = Sentence.sentence_count
        Sentence.sentence_count += 1
        self.predicates = []
        self.variable_map = {}
        local = {}

        for predicate in string.split("|"):
            name = predicate[:predicate.find("(")]
            params = []

            for param in predicate[predicate.find("(") + 1: predicate.find(")")].split(","):
                if param[0].islower():
                    if param not in local:  # Variable
                        local[param] = Parameter()
                        self.variable_map[local[param].name] = local[param]
                    new_param = local[param]
                else:
                    new_param = Parameter(param)
                    self.variable_map[param] = new_param

                params.append(new_param)

            self.predicates.append(Predicate(name, params))

    def getPredicates(self):
        return [predicate.name for predicate in self.predicates]

    def findPredicates(self, name):
        return [predicate for predicate in self.predicates if predicate.name == name]

    def removePredicate(self, predicate):
        self.predicates.remove(predicate)
        for key, val in self.variable_map.items():
            if not val:
                self.variable_map.pop(key)

    def containsVariable(self):
        return any(not param.isConstant() for param in self.variable_map.values())

    def __eq__(self, other):
        if len(self.predicates) == 1 and self.predicates[0] == other:
            return True
        return False

    def __str__(self):
        return "".join([str(predicate) for predicate in self.predicates])


class KB:
    def __init__(self, inputSentences):
        self.inputSentences = [x.replace(" ", "") for x in inputSentences]
        self.sentences = []
        self.sentence_map = {}

    def prepareKB(self):
        self.convertSentencesToCNF()
        for sentence_string in self.inputSentences:
            sentence = Sentence(sentence_string)
            for predicate in sentence.getPredicates():
                self.sentence_map[predicate] = self.sentence_map.get(
                    predicate, []) + [sentence]

    def convertSentencesToCNF(self):
        for sentenceIdx in range(len(self.inputSentences)):
            # Do negation of the Premise and add them as literal
            if "=>" in self.inputSentences[sentenceIdx]:
                self.inputSentences[sentenceIdx] = negateAntecedent(
                    self.inputSentences[sentenceIdx])

    def askQueries(self, queryList):
        results = []

        for query in queryList:
            negatedQuery = Sentence(negatePredicate(query.replace(" ", "")))
            negatedPredicate = negatedQuery.predicates[0]
            prev_sentence_map = copy.deepcopy(self.sentence_map)
            self.sentence_map[negatedPredicate.name] = self.sentence_map.get(
                negatedPredicate.name, []) + [negatedQuery]
            self.timeLimit = time.time() + 40

            try:
                result = self.resolve([negatedPredicate], [
                                      False]*(len(self.inputSentences) + 1))
            except:
                result = False

            self.sentence_map = prev_sentence_map

            if result:
                results.append("TRUE")
            else:
                results.append("FALSE")

        return results

    def resolve(self, queryStack, visited, depth=0):
        if time.time() > self.timeLimit:
            raise Exception
        if queryStack:
            query = queryStack.pop(-1)
            negatedQuery = query.getNegatedPredicate()
            queryPredicateName = negatedQuery.name
            if queryPredicateName not in self.sentence_map:
                return False
            else:
                queryPredicate = negatedQuery
                for kb_sentence in self.sentence_map[queryPredicateName]:
                    if not visited[kb_sentence.sentence_index]:
                        for kbPredicate in kb_sentence.findPredicates(queryPredicateName):

                            canUnify, substitution = performUnification(
                                copy.deepcopy(queryPredicate), copy.deepcopy(kbPredicate))

                            if canUnify:
                                newSentence = copy.deepcopy(kb_sentence)
                                newSentence.removePredicate(kbPredicate)
                                newQueryStack = copy.deepcopy(queryStack)

                                if substitution:
                                    for old, new in substitution.items():
                                        if old in newSentence.variable_map:
                                            parameter = newSentence.variable_map[old]
                                            newSentence.variable_map.pop(old)
                                            parameter.unify(
                                                "Variable" if new[0].islower() else "Constant", new)
                                            newSentence.variable_map[new] = parameter

                                    for predicate in newQueryStack:
                                        for index, param in enumerate(predicate.params):
                                            if param.name in substitution:
                                                new = substitution[param.name]
                                                predicate.params[index].unify(
                                                    "Variable" if new[0].islower() else "Constant", new)

                                for predicate in newSentence.predicates:
                                    newQueryStack.append(predicate)

                                new_visited = copy.deepcopy(visited)
                                if kb_sentence.containsVariable() and len(kb_sentence.predicates) > 1:
                                    new_visited[kb_sentence.sentence_index] = True

                                if self.resolve(newQueryStack, new_visited, depth + 1):
                                    return True
                return False
        return True


def performUnification(queryPredicate, kbPredicate):
    substitution = {}
    if queryPredicate == kbPredicate:
        return True, {}
    else:
        for query, kb in zip(queryPredicate.params, kbPredicate.params):
            if query == kb:
                continue
            if kb.isConstant():
                if not query.isConstant():
                    if query.name not in substitution:
                        substitution[query.name] = kb.name
                    elif substitution[query.name] != kb.name:
                        return False, {}
                    query.unify("Constant", kb.name)
                else:
                    return False, {}
            else:
                if not query.isConstant():
                    if kb.name not in substitution:
                        substitution[kb.name] = query.name
                    elif substitution[kb.name] != query.name:
                        return False, {}
                    kb.unify("Variable", query.name)
                else:
                    if kb.name not in substitution:
                        substitution[kb.name] = query.name
                    elif substitution[kb.name] != query.name:
                        return False, {}
    return True, substitution


def negatePredicate(predicate):
    return predicate[1:] if predicate[0] == "~" else "~" + predicate


def negateAntecedent(sentence):
    antecedent = sentence[:sentence.find("=>")]
    premise = []

    for predicate in antecedent.split("&"):
        premise.append(negatePredicate(predicate))

    premise.append(sentence[sentence.find("=>") + 2:])
    return "|".join(premise)


def getInput(filename):
    with open(filename, "r") as file:
        noOfQueries = int(file.readline().strip())
        inputQueries = [file.readline().strip() for _ in range(noOfQueries)]
        noOfSentences = int(file.readline().strip())
        inputSentences = [file.readline().strip()
                          for _ in range(noOfSentences)]
        return inputQueries, inputSentences


def printOutput(filename, results):
    print(results)
    with open(filename, "w") as file:
        for line in results:
            file.write(line)
            file.write("\n")
    file.close()


if __name__ == '__main__':
    inputQueries_, inputSentences_ = getInput('input.txt')
    knowledgeBase = KB(inputSentences_)
    knowledgeBase.prepareKB()
    results_ = knowledgeBase.askQueries(inputQueries_)
    printOutput("output.txt", results_)
    
 

#-----------------------------------------------------------------------------------------

  
'''
EXP8 Learning Algorithm
Output :- 

 initial stack  [['B'], ['A', 'D'], ['C']]
goal stack  ['B', 'C', 'A', 'D']
Step 1 [['B'], ['A'], ['D'], ['C']]
step 2  [['B', 'C'], [' '], ['D'], ['C']]
step 3  [['B', 'C', 'A'], [' '], [' '], ['C']]
step 4  [['B', 'C', 'A', 'D'], [' '], [' '], [' ']]
                     
'''

'''
 Aim:- Implementation of Learning Algorithm (Linear regression)

WORKING PRINCIPLE:-
Linear regression shows the linear relationship between the
independent variable (X-axis) and the dependent variable (Y-axis).To
calculate best-fit line linear regression uses a traditional slope-intercept
form. A regression line can be a Positive Linear Relationship or a
Negative Linear Relationship.The goal of the linear regression algorithm
is to get the best values for a0 and a1 to find the
best fit line and the best fit line should have the least error. In Linear
Regression, Mean Squared Error (MSE) cost function is used, which helps
to figure out the best possible values for a0 and a1, which provides the
best fit line for the data points. Using the MSE function, we will change
the values of a0 and a1 such that the MSE value settles at the minima.
Gradient descent is a method of updating a0 and a1 to minimize the cost
function(MSE).
                     
'''
stac = [['B'],['A','D'],['C']]
finalstac = ['B','C','A','D']
print("initial stack ",stac)
print("goal stack ",finalstac)
nu = 4
intermediate = []
def matrixIndex(st, arg):
    for i in range(len(st)):
        for j in range(len(st[i])):
            if(st[i][j]==arg):
                return(i,j)
for i in stac:
    for j in i:
        j = list(j)
        intermediate.append(j)
print("Step 1", intermediate)
for i in range (0,len(finalstac)):
    t = finalstac[i]
    if(i==0):
        flagm,flagn = matrixIndex(intermediate,t)
        continue
    else:
        intermediate[flagm].append(t)
        indm,indn = matrixIndex(intermediate,t)
        intermediate[indn][indm]=" "
        print("step "+str(i+1)+" ",intermediate)
        
    
#-----------------------------------------------------------------------------------------

'''
EXP9 NLP Program
Aim:- Implementation of NLP for executing cleaning of text and lemmitization of text
provided

(To be executed in Jupyter Notebook or Google Colab)

WORKING PRINCIPLE:-

In natural language processing, human language is separated into
fragments so that the grammatical structure of sentences and the
meaning of words can be analyzed and understood in context.
Let’s see the various different steps that are followed while preprocessing the data
also used for dimensionality reduction.

1. Tokenization
2. Lower casing
3. Stop words removal
4. Stemming
5. Lemmatization

Each term is the axis in the vector space model. In muti-dimensional space, the text
or document are constituted as vectors. The number of different words represents
the number of dimensions.

The python library that is used to do the preprocessing tasks in nlp is nltk. You can
install the nltk package using “pip install nltk”.

1. Tokenization:

It is a method in which sentences are converted into words.
import nltk
from nltk.tokenize import word_tokenize
token = word_tokenize("My Email address is: taneshbalodi8@gmail.com")
token

Tokenization

(Read also: Sentiment Analysis of YouTube Comments)

2. Lowercasing:

the tokenized words into lower case format. (NLU -> nlu). Words having the same
meaning like nlp and NLP if they are not converted into lowercase then these both
will constitute as non-identical words in the vector space model.
Lowercase = []
for lowercase in token:
Lowercase.append(lowercase.lower())
Lowercase

Lowercasing

3. Stop words removal:

These are the most often used that do not have any significance while determining
the two different documents like (a, an, the, etc.) so they are to be removed. Check
the below image wherefrom the sentence “Introduction to Natural Language
Processing” the “to” word is removed.
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from string import punctuation
punct = list(punctuation)
print(dataset[1]['quote'])
tokens = word_tokenize(dataset[1]['quote'])
len(tokens)

Without removing Stopwords

We got to see 50 tokens without removing stopwords, Now we shall remove
stopwords.

cleaned_tokens = [token for token in tokens if token not in stop_words
and token not in punctuation]
len(cleaned_tokens)

By cleaning the stopwords we got the length of the dataset as 24.
(Referred blog: What is SqueezeBERT in NLP?)
4. Stemming:

It is the process in which the words are converted to its base from. Check the below
code implementation where the words of the sentence are converted to the base
form.
from nltk.stem import PorterStemmer
ps = PorterStemmer()
print(ps.stem('jumping'))
print(ps.stem('lately'))
print(ps.stem('assess'))
print(ps.stem('ran'))

Stemming

5. Lemmatization:

Different from stemming, lemmatization lowers the words to word in the present
language for example check the below image where word has and is are changed to
ha and be respectively.
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('ran', 'v'))
print(lemmatizer.lemmatize('better',
                     
'''

import nltk 


raw_docs = ["Here are some very simple basic sentences.",
"They won't be very interesting, I'm afraid.",
"The point of these examples is to _learn how basic text cleaning works_ on *very simple* data."]

# Tokenizing text into bags of words
from nltk.tokenize import word_tokenize
tokenized_docs = [word_tokenize(doc) for doc in raw_docs]
print(tokenized_docs)

'''
[['Here', 'are', 'some', 'very', 'simple', 'basic', 'sentences', '.'], ['They', 'wo', "n't", 'be', 'very', 'interesting', ',', 'I', "'m", 'afraid', '.'], ['The', 'point', 'of', 'these', 'examples', 'is', 'to', '_learn', 'how', 'basic', 'text', 'cleaning', 'works_', 'on', '*', 'very', 'simple', '*', 'data', '.']]
'''

# Removing punctuation
import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

tokenized_docs_no_punctuation = []

for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    
    tokenized_docs_no_punctuation.append(new_review)
    
print(tokenized_docs_no_punctuation)

'''
[['Here', 'are', 'some', 'very', 'simple', 'basic', 'sentences'], ['They', 'wo', 'nt', 'be', 'very', 'interesting', 'I', 'm', 'afraid'], ['The', 'point', 'of', 'these', 'examples', 'is', 'to', 'learn', 'how', 'basic', 'text', 'cleaning', 'works', 'on', 'very', 'simple', 'data']]
'''

nltk.download('stopwords')

# Cleaning text of stopwords
from nltk.corpus import stopwords

tokenized_docs_no_stopwords = []

for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    
    tokenized_docs_no_stopwords.append(new_term_vector)

print(tokenized_docs_no_stopwords)

'''
[['Here', 'simple', 'basic', 'sentences'], ['They', 'wo', 'nt', 'interesting', 'I', 'afraid'], ['The', 'point', 'examples', 'learn', 'basic', 'text', 'cleaning', 'works', 'simple', 'data']]
'''

# Stemming and Lemmatizing
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        final_doc.append(porter.stem(word))
        #final_doc.append(snowball.stem(word))
        #final_doc.append(wordnet.lemmatize(word))
    
    preprocessed_docs.append(final_doc)

print(preprocessed_docs)

'''
[['here', 'simpl', 'basic', 'sentenc'], ['they', 'wo', 'nt', 'interest', 'I', 'afraid'], ['the', 'point', 'exampl', 'learn', 'basic', 'text', 'clean', 'work', 'simpl', 'data']]
'''

     
#---------------------------------------------------------------------------------------------------------------------

'''
EXP10 DeepLearning Algorithms
Output :- 

24/24 [==============================] - 0s 791us/step - loss: 0.4761 - accuracy: 0.7734
Accuracy: 77.34
[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0] => 1 (expected 1)
[1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0] => 0 (expected 0)
[8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0] => 1 (expected 1)
[1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0] => 0 (expected 0)
[0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0] => 1 (expected 1)

Aim:- Use Deep learning for predicting diabetes chances in different age groups in India

(Download Pima-India-Diabetes-Database from Kaggle and then upload in Jupyter Notebook or Google Colab)

Working Principle:
Keras is a deep learning algorithm toll that wraps the efficient numerical computation libraries
Theano and TensorFlow and allows you to define and train neural network models in just a few lines
of code. The steps to be followed are:
1. Load Data.
2. Define Keras Model.
3. Compile Keras Model.
4. Fit Keras Model.
5. Evaluate Keras Model.
6. Tie It All Together.
7. Make Predictions
8.Derive the results
                     
'''


# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10,verbose=0)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
    
    
    

#----------------------------------------------------------------END-----------------------------------------------------