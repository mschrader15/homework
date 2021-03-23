import numpy as np
from copy import copy, deepcopy
from random import random, choice, randint, uniform

ACTIONS = {1: 'STICK', 0: 'HIT'}


class Cards:
    def __init__(self):
        self.distribution_range = (1, 10)

    def get_card(self, first=False):
        return randint(*self.distribution_range) * 1 if (uniform(0, 1) < 2 / 3) or first \
            else -1 * randint(*self.distribution_range)


class Player:
    def __init__(self):
        self._cards = []
        self._sum = 0

    def check_bust(self):
        if self._sum < 1 or self._sum > 21:
            return True
        return False

    def clear_cards(self):
        self._cards = []
        self._sum = 0

    def add_card(self, card):
        self._cards.append(card)
        self._sum = sum(self._cards)

    def get_sum(self):
        return self._sum

    def get_card(self, index):
        return self._cards[index]


class Dealer(Player):
    STICK_THRESHOLD = 17

    def __init__(self):
        super(Dealer, self).__init__()

    def play_strategy(self, card):
        self.add_card(card)
        if not self.check_bust():
            if self.get_sum() < self.STICK_THRESHOLD:
                return True
        return False


class State:
    def __init__(self, player_sum, dealer_first):
        self.dealer_first = dealer_first
        self.player_sum = player_sum
        self.terminal = False

    def copy(self):
        return copy(self)


class Easy21:

    def __init__(self, ):
        self.dealer = Dealer()
        self.player = Player()
        self.cards = Cards()
        self.states = [range(1, 11), range(1, 22)]
        self.actions = ACTIONS
        self.actions_short = list(range(len(self.actions.keys())))

    def reset(self):
        self.__init__()

    def initialize_game(self):
        self.dealer.clear_cards()
        self.player.clear_cards()
        self.dealer.add_card(self.cards.get_card(first=True))
        self.player.add_card(self.cards.get_card(first=True))
        s = State(self.player.get_sum(), self.dealer.get_card(0))
        return s

    def calc_reward(self):
        if self.dealer.check_bust() or (self.player.get_sum() > self.dealer.get_sum()):
            return 1
        elif self.player.get_sum() == self.dealer.get_sum():
            return 0
        return -1

    def step(self, action, state):
        state_1 = state.copy()
        if action == 1:
            while self.dealer.play_strategy(self.cards.get_card()):
                pass
            r = self.calc_reward()
            state_1.terminal = True
        else:
            card = self.cards.get_card()
            self.player.add_card(card)
            if self.player.check_bust():
                state_1.terminal = True
                r = -1
            else:
                state_1.player_sum = self.player.get_sum()
                r = 0
        return state_1, r


class MonteCarloAgent:

    def __init__(self, gym: Easy21):
        self.gym = gym()
        self.Q = np.zeros((len(self.gym.states[1]), len(self.gym.states[0]), len(self.gym.actions_short)))
        self.N = deepcopy(self.Q)
        self.N0 = 100
        self.discount_factor = 1

    def calc_e(self, state: State) -> float:
        return self.N0 / (self.N0 + self.N[state.player_sum - 1, state.dealer_first - 1].sum() * 1.)

    def get_best_action(self, state):
        rewards = self.Q[state.player_sum - 1][state.dealer_first - 1]
        max_reward = max(rewards)
        return choice([self.gym.actions_short[i] for i, reward in enumerate(rewards) if reward >= max_reward])

    def e_greedy(self, state):
        e = self.calc_e(state)
        if random() < e:
            return choice(self.gym.actions_short)
        else:
            return self.get_best_action(state)

    def update_q(self, history):
        for i, (s_k, a_k, r_k) in enumerate(history):
            p_i = s_k.player_sum - 1
            d_i = s_k.dealer_first - 1
            G_t = sum([r_j * (self.discount_factor ** j) for j, (_, _, r_j) in enumerate(history[i:])])
            self.N[p_i, d_i, a_k] += 1
            alpha = 1.0 / self.N[p_i, d_i, a_k]
            self.Q[p_i, d_i, a_k] += alpha * (G_t - self.Q[p_i, d_i, a_k])

    def _train(self, ):
        self.gym.reset()
        s_t = self.gym.initialize_game()
        history = []
        while not s_t.terminal:
            a_t = self.e_greedy(s_t)
            s_t_1, r_t = self.gym.step(a_t, s_t)
            history.append([s_t, a_t, r_t])
            s_t = s_t_1
        self.update_q(history)

    def run(self, iterations, ):
        for _ in range(int(iterations)):
            self._train()

    def get_V_star(self, ):
        player_sum = list(self.gym.states[1])
        dealer_showing = list(self.gym.states[0])
        V_star = [[1 - actions.index(max(actions)) for actions in dealer] for dealer in self.Q]
        return player_sum, dealer_showing, V_star


class SARSAAgent(MonteCarloAgent):
    def __init__(self, gym):
        super(SARSAAgent, self).__init__(gym)
        self.E = deepcopy(self.Q)

    def _sarsa(self, _lambda):
        self.gym.reset()
        s_t = self.gym.initialize_game()
        a_t = self.e_greedy(s_t)
        self.N[s_t.player_sum - 1][s_t.dealer_first - 1][a_t] += 1
        a_t_1 = a_t
        while not s_t.terminal:
            s_t_1, r_t = self.gym.step(a_t, s_t)
            idx = (s_t.player_sum - 1, s_t.dealer_first - 1, a_t)
            Q_t = self.Q[idx]
            if not s_t_1.terminal:
                a_t_1 = self.e_greedy(s_t_1)
                idx_1 = (s_t_1.player_sum - 1, s_t_1.dealer_first - 1, a_t_1)
                self.N[idx_1] += 1
                Q_t_1 = self.Q[idx_1]
            else:
                Q_t_1 = 0
            d = r_t + (Q_t_1 - Q_t) * _lambda
            self.E[idx] += 1
            a = 1.0 / self.N[idx]
            self.Q += a * d * self.E
            self.E *= self.discount_factor * _lambda
            s_t = s_t_1
            a_t = a_t_1

    def run(self, iterations, _lambda):
        for _ in range(iterations):
            self._sarsa(_lambda)


class LinearFunctionApproximation(MonteCarloAgent):

    def __init__(self, gym, Q_star=None):
        super(LinearFunctionApproximation, self).__init__(gym)

        self._e = 0.05
        self._lambda = 1
        self._alpha = 0.01
        self.Q_star = Q_star
        self._dealer_c = [[1, 4], [4, 7], [7, 10]]
        self._player_c = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
        self._action_c = [0, 1]
        # parameters are initialize randomly
        self._shape = list(map(len, [self._dealer_c, self._player_c, self._action_c]))
        self._theta = self.calc_theta()
        self._shaped_zeros = np.zeros(self._shape,)
        self._E = copy(self._shaped_zeros)
        self._param_num = np.prod(self._shape)

    def calc_theta(self):
        return np.random.randn(*self._shape) * 0.1

    def phi(self, s, a):
        d_sum, p_sum = s.dealer_first, s.player_sum
        features = copy(self._shaped_zeros)
        d_features = [(i, x[0] <= d_sum <= x[1]) for i, x in enumerate(self._dealer_c)]
        p_features = [(i, x[0] <= p_sum <= x[1]) for i, x in enumerate(self._player_c)]
        for i, d_inside in d_features:
            if d_inside:
                for j, p_inside in p_features:
                    if p_inside:
                        features[i, j, a] = 1
        return features

    def get_best_action(self, state):
        rewards = [np.dot(self.phi(state, a).flatten(), self._theta.flatten()) for a in self._action_c]
        max_reward = max(rewards)
        return choice([self.gym.actions_short[i] for i, reward in enumerate(rewards) if reward >= max_reward])

    def calc_e(self, state=None) -> float:
        return self._e

    def compose_Q(self):
        Q = np.zeros((len(self.gym.states[0]), len(self.gym.states[1]), len(self.gym.actions_short)))
        for i in self.gym.states[0]:
            for j in self.gym.states[1]:
                for a in self.gym.actions_short:
                    Q[i, j, a.value] = np.dot(self.phi(State(player_sum=j, dealer_first=i), a_t), self._theta)
        return Q

    def _step(self, log_error):
        self.gym.reset()
        s_t = self.gym.initialize_game()
        a_t = self.e_greedy(s_t)
        self._E = copy(self._shaped_zeros)
        # self._theta = self.calc_theta()
        a_t_1 = a_t
        while not s_t.terminal:
            s_t_1, r_t = self.gym.step(a_t, s_t)
            phi_t = self.phi(s_t, a_t)
            q = np.dot(phi_t.flatten(), self._theta.flatten())
            if not s_t_1.terminal:
                a_t_1 = self.e_greedy(s_t_1)
                phi_t_1 = self.phi(s_t_1, a_t_1)
                q_t_1 = np.dot(phi_t_1.flatten(), self._theta.flatten())
                d = r_t + q_t_1 - q
            else:
                d = r_t - q
            self._E += phi_t
            self._theta += self._alpha * d * self._E
            self._E *= self.discount_factor * self._lambda
            s_t = s_t_1
            a_t = a_t_1
        if log_error:
            return np.sum(np.square(self.Q_star - self.compose_Q()))
        return None

    def run(self, iterations, log_error=False):
        error = []
        for i in range(iterations):
            if log_error:
                error.append((i, self._step(log_error)))
            self._step(log_error)
        if log_error:
            return error
        self.Q = self.compose_Q()


if __name__ == "__main__":

    mc_agent = LinearFunctionApproximation(gym=Easy21, )

    mc_agent._lambda = 0.1

    mc_agent.run(iterations=int(1000),)

    print(mc_agent.Q)





