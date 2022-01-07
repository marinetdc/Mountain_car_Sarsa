import gym
import numpy as np
import matplotlib.pyplot as plt 
from collections import defaultdict
import random

env = gym.make("MountainCar-v0")

env.viewer = False

#Environment values
print(env.observation_space.high)	#[0.6  0.07]
print(env.observation_space.low)	#[-1.2  -0.07]
print(env.action_space.n)			#3

ALPHA = 0.1    # learning rate
GAMMA = 0.9    # importance of next action
EPSILON = 0.05  # exploration chance

EPISODE_DISPLAY = 200

EPISODES = 30000

DISCRETE_SIZE = 40

def discretised_state(env, discrete_size):
    return np.linspace(env.observation_space.low, env.observation_space.high, discrete_size)

DISCRATE_STATE = discretised_state(env, DISCRETE_SIZE)

Q = defaultdict(lambda: np.zeros(len(DISCRATE_STATE)))

def get_utility(q, state, action):
    return q.get((state, action), 0.0)  

def discretize_state(state, discrete_states):
    #print(state[1])
    #print((state[0] < discrete_states[1:,0]) )
    return ( 
        np.where((discrete_states[:-1,0] <= state[0]) * (state[0] < discrete_states[1:,0]))[0][0], 
        np.where((discrete_states[:-1,1] <= state[1]) * (state[1] < discrete_states[1:,1]))[0][0]
    )
#discretize_state((0.6,0.01039), DISCRATE_STATE)
# For stats
ep_rewards = []
ep_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}
lenght_ep = []
all_ep = []


for episode in range(EPISODES) : 

    episode_reward = 0
    done = False 

    current_state = discretize_state(env.reset(), DISCRATE_STATE) # starting at starting_position  A ENLEVER 

    if random.random() < EPSILON :  # exploration 
        action = env.action_space.sample()
    else:

        q = [get_utility(Q,current_state, act) for act in range(env.action_space.n)]
        max_utility = max(q)

        # In case there're several state-action max values
        # we select a random one among them
        if q.count(max_utility) > 1:
            best_actions = [i for i in range(env.action_space.n) if q[i] == max_utility]
            action = random.choice(best_actions)

        else:
            action = q.index(max_utility) 
    
    i=0
    #while not done:	##laaaa
    for iter in range(200) : 
        

        if not done : # ajouté à l'instant 

            new_state, reward, done, _ = env.step(action)
            new_state = discretize_state(new_state, DISCRATE_STATE)

            if random.random() < EPSILON :  # exploration 
                new_action = env.action_space.sample()

            else:

                q = [get_utility(Q, new_state, act) for act in range(env.action_space.n)]
                max_utility = max(q)

                # In case there're several state-action max values
                # we select a random one among them
                if q.count(max_utility) > 1:
                    best_actions = [i for i in range(env.action_space.n) if q[i] == max_utility]
                    new_action = random.choice(best_actions)

                else:
                    new_action = q.index(max_utility) 

            if not done: ###LAAAAA
                
                old_utility = Q.get((current_state, action), None)

                if old_utility is None:
                    Q[(current_state, action)] = reward
                else:
                    next_max_utility = get_utility(Q, new_state, new_action) 
                    #print('next', next_max_utility)
                    #print(Q)
                    Q[(current_state, action)] = old_utility + ALPHA * (reward + GAMMA * next_max_utility - old_utility)

            elif new_state[0] >= env.goal_position:
                Q[(current_state,action)] = 0
                    
            current_state = new_state
            action = new_action

            episode_reward += reward
            #print(reward)
        
        else : 
            if i == 0 : 
                i = iter 

    #lenght_ep.append(i)
    #all_ep.append(episode)
    ep_rewards.append(episode_reward)
    
    #print(Q)
    
    if not episode % EPISODE_DISPLAY:

        avg_reward = sum(ep_rewards[-EPISODE_DISPLAY:])/len(ep_rewards[-EPISODE_DISPLAY:])
        ep_rewards_table['ep'].append(episode)
        ep_rewards_table['avg'].append(avg_reward)
        ep_rewards_table['min'].append(min(ep_rewards[-EPISODE_DISPLAY:]))
        ep_rewards_table['max'].append(max(ep_rewards[-EPISODE_DISPLAY:]))
        lenght_ep.append(i)
        
    
        print(f"Episode:{episode} avg:{avg_reward} min:{min(ep_rewards[-EPISODE_DISPLAY:])} max:{max(ep_rewards[-EPISODE_DISPLAY:])}")

env.close()

        
plt.plot(ep_rewards_table['ep'], ep_rewards_table['avg'], label="avg")
plt.plot(ep_rewards_table['ep'], ep_rewards_table['min'], label="min")
plt.plot(ep_rewards_table['ep'], ep_rewards_table['max'], label="max")
plt.legend(loc=4) #bottom right
plt.title('Mountain Car SARSA')
plt.ylabel('Average reward/Episode')
plt.xlabel('Episodes')
plt.show()

#print(lenght_ep)

plt.plot(ep_rewards_table['ep'], lenght_ep, label='all')
plt.title('Longuer des episodes')
plt.show()

