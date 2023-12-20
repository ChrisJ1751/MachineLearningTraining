# Importing Libraries 
import numpy as np

# Setting parameters 'Gamma' and 'Alpha' for Q-Learning
gamma = 0.75 #Discount Factor
alpha = 0.9 #Learning Rate

# Define the Environment

# States
location_to_state = {'A': 0, 
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# Actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# Rewards
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

# Building the AI Solution with Q-Learning over many iterations
# Initializing Q-Values
Q = np.array(np.zeros([12,12]))

# For Loop: Implementing the Q-Learning Process iterating 1000 times
#for i in range(1000):
#    current_state = np.random.randint(0,12) # Upper bound is excluded
#    playable_actions = []
#    for j in range(12):
#        if R[current_state, j] > 0:
#            playable_actions.append(j)
#    next_state = np.random.choice(playable_actions)
#    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
#    Q[current_state, next_state] += alpha * TD 
    
# Going Into Production
# Making a mapping from states to locations
state_to_location = {state: location for location, state in location_to_state.items()}

#def route(starting_location, ending_location):
#    R_new = np.copy(R)
#    ending_state = location_to_state[ending_location]
#    R_new[ending_state, ending_state] = 1000
#    
#    route = [starting_location]
#    next_location = starting_location
#    while (next_location != ending_location):
#        starting_state = location_to_state[starting_location]
#        next_state = np.argmax(Q[starting_state,])
#        next_location = state_to_location[next_state]
#        route.append(next_location)
#        starting_location = next_location
#    return route

# Printing final route
#print('Route:')
#route('E', 'G')

#Final Function
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    for i in range(1000):
        current_state = np.random.randint(0,12) # Upper bound is excluded
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD 
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

print('Route:')
route('E', 'G')
