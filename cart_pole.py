import neuralfit as nf
import numpy as np
import gym

# Initialize the environment
env = gym.make('CartPole-v1')

# Initialize and compile the model
model = nf.Model(4,1)
model.compile(monitors=['size'])

# Define evaluation functions
def evaluate (genomes):
    losses = np.zeros(len(genomes))
    random_seed = np.random.randint(0,1000) 
    for i in range(len(genomes)):
        observation, _ = env.reset(seed=random_seed)
        for t in range(1000):
            observation = np.reshape(observation, (1,4))
            action = int(np.clip(genomes[i].predict(observation),0,1)[0][0])
            observation, reward, done, info, _ = env.step(action)
            losses[i] -= reward
            if done:
                break

    return losses

# Start evolving
model.func_evolve(evaluate, epochs=50)

# Enable rendering, and visualize the result
env = gym.make('CartPole-v1', render_mode='human')
evaluate([model])
