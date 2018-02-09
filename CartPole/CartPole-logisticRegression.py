import gym
import random
import numpy as np
from statistics import mean
from sklearn.linear_model import LogisticRegression
import pickle
import os

env = gym.make('CartPole-v1')
score_requirement = 100
initial_games = 10000


def play(clf):
    scores = []
    for episode in range(5):
        score = 0
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = clf.predict(observation.reshape(1, -1))[0]
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
        print('Average reward: ', mean(scores))
        if mean(scores) >= 475:
            print('Goal achieved at episode: ', episode)
            break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
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
