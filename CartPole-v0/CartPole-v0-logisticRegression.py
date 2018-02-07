import gym
import random
import numpy as np
from statistics import mean
from sklearn.linear_model import LogisticRegression
import pickle
import os

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200
score_requirement = 100
initial_games = 10000


def play(clf):
    scores = []
    for episode in range(100):
        score = 0
        env.reset()
        prev_observation = []
        for t in range(goal_steps):
            # env.render()
            if t < 1:
                action = env.action_space.sample()
            else:
                action = clf.predict(prev_observation)[0]

            observation, reward, done, info = env.step(action)
            score += reward
            prev_observation = observation.reshape(1, -1)
            if done:
                break
        scores.append(score)
        print('Average reward: ', mean(scores))
        if mean(scores) > 195:
            print('Goal achieved at episode: ', episode)
            break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append(data)

        env.reset()
        scores.append(score)

    return np.array(training_data)


def train_classifier():
    training_data = initial_population()

    x = []
    y = []
    for sample in training_data:
        x.append(sample[0])
        y.append(sample[1])

    out_clf = LogisticRegression()
    out_clf.fit(x, y)

    with open(model_path, 'wb') as f:
        pickle.dump(out_clf, f)

    return out_clf


model_path = 'logistic_regression.model.pickle'
model_exists = os.path.isfile(model_path)

if model_exists:
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
else:
    clf = train_classifier()

play(clf)
