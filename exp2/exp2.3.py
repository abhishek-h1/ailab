colors = ['Red', 'Blue', 'Green', 'Yellow', 'Black']
states = ['Thiruchendur', 'Thirupparakunram', 'Pazhamudircholai', 'Palani','Swamimalai','Thirutani']
neighbors = {}
neighbors['Palani'] = ['Thirupparakunram', 'Pazhamudircholai']
neighbors['Thirutani'] = ['Swamimalai']
neighbors['Thirupparakunram'] = ['Thiruchendur']
neighbors['Swamimalai'] = ['Pazhamudircholai']
neighbors['Pazhamudircholai'] = ['Thirupparakunram','Palani']
neighbors['Thiruchendur'] = ['Thirupparakunram']
colors_of_states = {}
def promising(state, color):
    for neighbor in neighbors.get(state):
        color_of_neighbor = colors_of_states.get(neighbor)
        if color_of_neighbor == color:
            return False
    return True
def get_color_for_state(state):
    for color in colors:
        if promising(state, color):
            return color
            
if __name__=="__main__":
    for state in states:
        colors_of_states[state] = get_color_for_state(state)
    print (colors_of_states)

