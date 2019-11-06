import random
import numpy as np
import graph


def gen(n: int):
    boxs = ['1', '2', '3']
    balls = [
        ['red', 'red', 'red', 'blue', 'yellow'],
        ['yellow', 'yellow', 'yellow', 'red', 'blue'],
        ['blue', 'blue', 'blue', 'red', 'yellow']
    ]
    states = []
    observations = []
    for i in range(n):
        box_index = random.randint(0, 2)
        states.append(boxs[box_index])
        ball = random.choice(balls[box_index])
        observations.append(ball)
    return np.array(states), np.array(observations)


def test_hmm(train_size: int, test_size: int):
    # Test hidden Markov model
    # Train model
    model = graph.HMM()
    train_states, train_observations = gen(train_size)
    model.fit_supervised(train_observations, train_states)
    # Test model
    test_states, test_observations = gen(test_size)
    print(model.forward(test_observations), model.backward(test_observations))
    print(model.decode(test_observations))
    print(test_states)


if __name__ == '__main__':
    test_hmm(1000, 20)
