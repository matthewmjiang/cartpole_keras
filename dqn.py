import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # deque = list-like container with fast appends and pops on both ends
        self.memory = deque(maxlen=2000) # deque of previous experiences and observations
        self.discount_rate = 0.95 # also known as gamma
        self.exploration_rate = 1.0 # also known as epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.buildModel()

    def buildModel(self):
        # make a Sequential model, which is a linear stack of layers.
        model = Sequential()
        # make an input layer that has state size 4 (so it expects 4 input variables) and a hidden
        # layer with 24 nodes that uses the ReLU activation function
        
        # Activation function: takes the summed weighted input and creates an output. These
        # are mathematical equations

        # ReLU: rectified linear activation function; outputs the given input directly
        # or 0.0 if input < 0.
       
        # the Dense class just makes a fully connected layer
        input_layer = Dense(24, input_dim = self.state_size, activation='relu')
        model.add(input_layer)
       
        # make another hidden layer with 24 nodes and the reLU activation function
        hidden_layer = Dense(24, activation='relu')
        model.add(hidden_layer)
       
        # make the output layer with 2 nodes (action: left or right). Has linear activation
        # function, meaning it just outputs the input directly.
        output_layer = Dense(self.action_size, activation='linear')
        model.add(output_layer)

        # MAKE THE MODEL
        # The loss function evaluates how well our dataset is modeled by our algorithm. The lower
        # the number, the better our model predicts.

        # We use the Mean Squared Error loss function (MSE): which takes the difference between
        # a predicted value and the real value, squares it, sums it with the other squared values,
        # then divides it by the size of the dataset.

        # We will also use an optimization algorithm to update the weights of the network iteratively.
        # The Adam algorithm takes the benefits of the Adaptive Gradient Algorithm (AdaGrad) and 
        # the Root Mean Square Propagation algorithm (RMSProp) and combines them, creating a 
        # computationally and memory efficient optimizer.
        # We will use a learning rate of 0.001 as suggested by the creator of the algorithm. The
        # learning rate determines the proportion that weights are updated (so larger values
        # will result in faster initial learning).
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))


        return model
       
    # this function will store states, actions and resulting rewards to our "memory" so that we can
    # use them to retrain the model everytime new experiences overwrite the previous experiences.
    def memorize(self, state, action, reward, next_state, done):
        # Save the current state, action, reward, and next state to our memory.
        # Done just indicates if the current state is the final state.
        self.memory.append((state, action, reward, next_state, done))
    

    # Here we train the neural network using experiences from our memory.
    def replay(self, batch_size):
        # make a batch of randomly sampled elements from memory
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            # if we reached the final state, simply make the reward our target since there is no
            # more future reward
            target = reward

            if not done:
                # predict the future discounted reward
                # The predict function will predict the reward of the current state based on the
                # input and old trained data.
                # Here, we add the current reward to our future reward, which will be the 
                # predicted next best action (we use numPy's amax to get this value) and multiplied
                # by the discount rate so that the future reward is less valuable.
                target = reward + self.discount_rate * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # The fit function feeds input and output to the model and makes the model train on that
            # data to better approximate the output based on the input. 1 epoch is simply one iteration
            # over the entire dataset provided (state and target_f).
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Here we try to decay the exploration rate over time so that the agent starts predicting more
        # as more iterations occur.
        if self.exploration_rate > self.epsilon_min:
            self.exploration_rate *= self.epsilon_decay
        """
        if (self.exploration_rate > self.epsilon_min) and (len(self.memory) == 2000):
            self.exploration_rate *= self.epsilon_decay
        """
    # This function will tell the agent what action to pick.
    def act(self, state):
        # The exploration rate will have the agent randomly pick an action sometimes instead of trying
        # to predict.
        if np.random.rand() <= self.exploration_rate:
            # return a random action
            return env.action_space.sample()

        # Other times the agent will try to predict the action that will give the highest reward.
        
        # Here we try to predict the reward value based on the current state
        action_values = self.model.predict(state)
       
        # And pick the action based on the predicted reward. The argmax function will return the 
        # index with the highest value.
        return np.argmax(action_values[0])


if __name__ == "__main__":
    
    # make the environment
    env = gym.make('CartPole-v1')
    
    # make our agent
    # make state_size
    state_size = env.observation_space.shape[0]
    # make action_size = number of actions we can perform
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)

    # the game is "solved" if we get a score over 195 for 100 consecutive episodes
    solved_counter = 0
    # determine the number of times we wish to replay the game
    episodes = 1000

    # iterate the game 
    for episode in range(episodes):
        if solved_counter == 100:
            print("The agent has solved the game in {} episodes.".format(episode))
            break
        # reset the state at the beginning of each episode
        state = env.reset()
        # give a new shape to the state array -> 1 row and 4 columns
        state = np.reshape(state, [1, 4])

        # time_step represents each frame of the game
        # Our goal is to keep the pole upright untill we reach the score of 500
        # the higher time_step is, the higher the score.
        for time_step in range(500):
            # render the game
            env.render()

            # pick an action to perform (left or right) based on the current state
            action = agent.act(state)

            # apply that action to the environment and save the current info from the timestep
            next_state, reward, done, info = env.step(action)

            # make the reward highly unattractive if the episode ends so that the agent will try
            # to make the game last longer. Essentially, the agent will avoid ending the game and
            # therefore increase learning performance.
            if done:
                reward = -10

            # reshape the next_state array
            next_state = np.reshape(next_state, [1, 4])

            # store the current state, action, reward, and done to memory
            agent.memorize(state, action, reward, next_state, done)
            
            # go to the next state for the next time_step
            state = next_state

            # print the score and go to the next episode when the current episode ends
            if done:
                if time_step >= 195:
                    solved_counter += 1
                else:
                    solved_counter = 0
                
                print("episode: {}/{}, score: {}, memory length: {}, solved progress: {}".format(episode, episodes, time_step, len(agent.memory), solved_counter)) 
                """
                print("episode: {}/{}, score: {}".format(episode, episodes, time_step))
                """
                break
            """
            # train the agent with 32 random episodes after amassing enough samples
            # do it after each timestep so that the agent is trained more frequently.
            # i.e if we train after each episode, the agent is only updated 1000 times.
            if len(agent.memory) > 32:
                agent.replay(32)
            """
        if len(agent.memory) > 32:
            agent.replay(32)

    
    
