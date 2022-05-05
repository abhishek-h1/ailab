graph = {
  '0' : [ '1','3'],
  '1' : ['3','2','5','6'],
  '2' : ['1','3','4','5'],
  '3' : ['0','1','2','4'],
  '4' : ['3','2','6'],
  '5' : ['1','2'],
  '6' : ['1','4']
}

visited_bfs = []
queue = []

def bfs(visited_bfs, graph, node):
  visited_bfs.append(node)
  queue.append(node)

  while queue:
    s = queue.pop(0) 
    print (s, end = " ") 

    for neighbour in graph[s]:
      if neighbour not in visited_bfs:
        visited_bfs.append(neighbour)
        queue.append(neighbour)

visited = set()

def dfs(visited, graph, node):
    if node not in visited:
        print (node, end=" ")
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

print("BFS:" , end =" ")
bfs(visited_bfs, graph, '0')
print('\n')
print("DFS:" , end =" ")
dfs(visited, graph, '0')
print('\n')