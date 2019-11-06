import numpy as np


class HMM(object):
    """Hidden Markov model."""

    def __init__(self):
        # Model parameters
        self._state_dict = None
        self._observation_dict = None
        self._pi = None  # Initial probability
        self._A = None  # Transition matrix
        self._B = None  # Observation matrix

    def fit_supervised(self,
                       observation_sequence: np.ndarray,
                       state_sequence: np.ndarray,
                       state_set: set = None,
                       observation_set: set = None,
                       initial_prob: dict = None) -> None:
        # Use state sequence and observation sequence to build a model
        # Observations
        if observation_set is None:
            observation_list = list(np.unique(observation_sequence))
        else:
            observation_list = list(observation_set)
        self._observation_dict = {observation: index for index, observation in
                                  enumerate(observation_list)}
        # States
        assert observation_sequence.shape[0] == state_sequence.shape[0]
        if state_set is None:
            state_list = list(np.unique(state_sequence))
        else:
            state_list = list(state_set)
        self._state_dict = {state: index for index, state in
                            enumerate(state_list)}
        num_state = len(state_list)
        num_observation = len(observation_list)
        t = len(state_sequence)
        # Initial probabilities
        if initial_prob is None:
            self._pi = {state: 1. / num_state for state in state_list}
        else:
            self._pi = initial_prob
        # Estimate transition matrix
        transition = np.zeros((num_state, num_state))
        for index in range(t - 1):
            source = self._state_dict[state_sequence[index]]
            destination = self._state_dict[state_sequence[index + 1]]
            transition[source][destination] += 1
        for source in range(num_state):
            transition[source] /= np.sum(transition[source])
        self._A = transition
        # Estimate observation matrix
        observation = np.zeros((num_state, num_observation))
        for index in range(t):
            state_index = self._state_dict[state_sequence[index]]
            observe_index = self._observation_dict[observation_sequence[index]]
            observation[state_index][observe_index] += 1
        for state in range(num_state):
            observation[state] /= np.sum(observation[state])
        self._B = observation

    def fit_unsupervised(self,
                         observation_sequence: np.ndarray,
                         num_state: int) -> None:
        # Baum-Welch algorithm
        # Use observation sequence only to build a model
        # Observations
        observation_list = list(np.unique(observation_sequence))
        self._observation_dict = {observation: index for index, observation in
                                  enumerate(observation_list)}
        # States
        state_list = list(range(num_state))
        self._state_dict = {state: index for index, state in
                            enumerate(state_list)}
        num_observation = len(observation_list)
        t = len(observation_sequence)
        # Use EM algorithm to estimate A and B
        # Initialize parameters to estimete
        self._A = np.ones((num_state, num_state)) / num_state
        self._B = np.ones((num_state, num_observation)) / num_observation
        self._pi = {state: 1. / num_state for state in state_list}
        # Iteratively update estimations
        while True:
            forward_state_probs = self.forward(observation_sequence,
                                               return_detail=True)
            backward_state_probs = self.backward(observation_sequence,
                                                 return_detail=True)
            # Calculate gamma & epsilon sequence for each time
            gamma, epsilon = [], []
            for ti in range(t):
                # Calculate forward probability sequence and
                # backward probability sequence.
                alpha_dict = forward_state_probs[ti]
                beta_dict = backward_state_probs[ti]
                next_beta_dict = backward_state_probs[ti + 1]
                # Gamma_ti(state_i)
                total = 0.0
                gamma_ti = [0.0 for _ in range(num_state)]
                for state, i in self._state_dict.items():
                    total += alpha_dict[state] * beta_dict[state]
                    gamma_ti[i] = alpha_dict[state] * beta_dict[state]
                gamma_ti = np.array(gamma_ti) / total
                gamma.append(gamma_ti)
                # Epsilon_ti(state_i, state_j)
                if ti == t - 1:
                    break
                total = 0.0
                epsilon_ti = [[0.0 for _ in range(num_state)] for _ in
                              range(num_state)]
                for state_i, i in self._state_dict.items():
                    for state_j, j in self._state_dict.items():
                        frac = alpha_dict[state_i] * self._A[i][j]
                        frac *= self._B[j][observation_sequence[ti + 1]]
                        frac *= next_beta_dict[state_j]
                        epsilon_ti[i][j] = frac
                        total += frac
                epsilon_ti = np.array(epsilon_ti) / total
                epsilon.append(epsilon_ti)
            # Arrayfy gamm and epsilon
            gamma, epsilon = np.array(gamma), np.array(epsilon)
            # Use gamma and epsilon to estimate A, B and pi
            for state_i, i in self._state_dict.items():
                self._pi[i] = gamma[0][i]
                for state_j, j in self._state_dict.items():
                    self._A[i][j] = epsilon[:, i, j].sum()
                    self._A[i][j] /= gamma[:t - 1, i].sum()
                for ti in range(t):
                    ot = self._observation_dict[observation_sequence[ti]]
                    self._B[i] += gamma[ti][ot]
                self._B[i] /= gamma.sum()

    def forward(self, observation_sequence: np.ndarray,
                return_detail: bool = False) -> float or list:
        # Calculate probability P(O|lambda).
        t = len(observation_sequence)
        prob_states = {}
        details = [prob_states]
        # Initialize probability of observations for the first state
        observe_index = self._observation_dict[observation_sequence[0]]
        for state, index in self._state_dict.items():
            prob_states[state] = self._pi[state] * self._B[index][observe_index]
        # Calculate iteratively forward
        for step in range(1, t):
            observe_index = self._observation_dict[observation_sequence[step]]
            tmp_prob = {}
            for curr_state, curr_index in self._state_dict.items():
                prob = 0.
                # Sum all transition probabilities
                for prev_state, prev_index in self._state_dict.items():
                    prob += prob_states[prev_state] * self._A[prev_index][
                        curr_index]
                prob *= self._B[curr_index][observe_index]
                tmp_prob[curr_state] = prob
            # Update probability
            prob_states = tmp_prob
            details.append(prob_states)
        if return_detail:
            # Return probability for all states in all T steps
            return details
        else:
            # P(O|lambda) = sum of all probabilities
            total = sum(prob_states.values())
            return total

    def backward(self, observation_sequence: np.ndarray,
                 return_detail: bool = False) -> float or dict:
        # Calculate probability P(O|lambda).
        t = len(observation_sequence)
        prob_states = {}
        details = [prob_states]
        # Initialize probability of observations for the last state
        for state in self._state_dict:
            prob_states[state] = 1.
        # Calculate iteratively backward
        for step in range(t - 1, 0, -1):
            observe_index = self._observation_dict[observation_sequence[step]]
            tmp_prob = {}
            for prev_state, prev_index in self._state_dict.items():
                prob = 0.
                # Sum all transition probabilities
                for curr_state, curr_index in self._state_dict.items():
                    prob += prob_states[curr_state] * self._A[prev_index][
                        curr_index] * self._B[curr_index][observe_index]
                tmp_prob[prev_state] = prob
            # Update probability
            prob_states = tmp_prob
            details.append(prob_states)
        # Calculate initial probability
        init_prob = {}
        observe_index = self._observation_dict[observation_sequence[0]]
        for init_state, init_index in self._state_dict.items():
            init_prob[init_state] = self._pi[init_state] * self._B[init_index][
                observe_index] * prob_states[init_state]
        if return_detail:
            # Return probability for all states in all T steps
            # Reverse probabilities: 1, 2, 3, ..., T
            details = details[::-1]
            return details
        else:
            # P(O|lambda) = sum of all probabilities
            total = sum(init_prob.values())
            return total

    def decode(self, obervation_sequence: np.ndarray) -> np.ndarray:
        # Predict most probabal hidden states using Viterbi algorithm.
        n = len(obervation_sequence)
        num_state = len(self._state_dict)
        prev_states = [[None for _ in range(num_state)] for _ in range(n)]
        probs = [[1. for _ in range(num_state)] for _ in range(n)]
        # Iteratively search for path with maximum probability
        for step in range(1, n):
            observe_index = self._observation_dict[obervation_sequence[step]]
            for state, curr_index in self._state_dict.items():
                # Find maximum probability in all previous states
                max_prob, max_prev_state = 0., 0
                for prev_state, prev_index in self._state_dict.items():
                    # Calculate probability of prev_state -> curr_state
                    prob = probs[step - 1][prev_index]
                    prob *= self._A[prev_index][curr_index]
                    prob *= self._B[curr_index][observe_index]
                    if max_prob < prob:
                        max_prob, max_prev_state = prob, prev_state
                prev_states[step][curr_index] = max_prev_state
                probs[step][curr_index] = max_prob
        # Find from tail to front
        # Index -> State
        state_list = list(self._state_dict.keys())
        # Find the state whose probability is maximum at last
        last_state = state_list[int(np.argmax(probs[-1]))]
        path = [last_state]
        for step in range(n - 1, 0, -1):
            state_index = self._state_dict[last_state]
            prev_state = prev_states[step][state_index]
            path.append(prev_state)
            last_state = prev_state
        path = path[::-1]
        return np.array(path)
