from enum import Enum
from random import uniform, randint, choice, random
from copy import copy


def check_bust(score):
    if (score < 1) or (score > 21):
        return True
    return False

class Action(Enum):
    STICK = 0
    HIT = 1


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
        self.reset_deck()

    def reset_deck(self):
        self._cards = []
        self._sum = 0

    def set_first(self, card):
        self.reset_deck()
        self.play_hand(card)

    def _add_to_hand(self, card):
        self._cards.append(card)

    def _add_cards(self, ):
        return sum(self._cards)

    def play_hand(self, card):
        self._add_to_hand(card)
        self._sum = self._add_cards()

    def _my_game_plan(self, ):
        if not check_bust(self._sum):
            return Action.HIT
        return Action.STICK

    def get_sum(self):
        return self._sum


class Dealer(Player):
    STICK_THRESHOLD = 17

    def __init__(self, card_obj):
        super().__init__()
        self._card_deck = card_obj

    def draw_card(self, first=False):
        return self._card_deck.get_card(first=first)

    def _my_game_plan(self, ):
        current_sum = self.get_sum()
        if check_bust(current_sum):
            return False
        if current_sum < Dealer.STICK_THRESHOLD:
            return True
        return False

    def take_turns(self, ):
        while self._my_game_plan():
            self.play_hand(self.draw_card())
            self.play_hand(self.draw_card())


#         return self._sum()


class State:
    def __init__(self, dealer_first_card, ):
        self.dealer_card = dealer_first_card
        self.player_sum = 0
        self.dealer_action = None

    def update_sum(self, player_sum):
        self.player_sum = player_sum

    def update_action(self, dealer_action):
        self.dealer_action = dealer_action


class Easy21:
    REWARD = {'bust': -1, 'win': 1, 'draw': 0}

    def __init__(self):
        self.dealer = Dealer(Cards())
        self.player = None
        self.state = None
        self.state_space = [[(j + 1, i + 1) for i in range(10)] for j in range(21)]
        self.action_space = [Action.HIT, Action.STICK]

    def get_reward(self, player_sum):
        if check_bust(self.dealer.get_sum()) or (player_sum > self.dealer.get_sum()):
            return 1
        return 0 if player_sum == self.dealer.get_sum() else -1

    def add_player(self, player):
        self.player = player

    def step(self, current_state, player_action):
        terminal = True
        if player_action == Action.STICK:
            self.dealer.take_turns()
            r = self.get_reward(current_state.player_sum)
        else:
            next_card = self.dealer.draw_card()
            self.player.play_hand(next_card)
            if not check_bust(self.player.get_sum()):
                current_state.update_sum(self.player.get_sum())
                r = 0
                terminal = False
            else:
                # current_state.player_sum = max()
                r = -1
        return current_state, r, terminal

    def start_game(self, ):
        s = State(self.dealer.draw_card(first=True), )
        self.dealer.set_first(s.dealer_card)
        self.player.set_first(self.dealer.draw_card(first=True))
        s.player_sum = self.player.get_sum()
        return s


class MonteCarloAgent(Player):
    def __init__(self, gym: Easy21, ):
        super().__init__()
        self.game = gym
        self.Q = [[[0 for _ in range(len(self.game.action_space))]
                   for _ in range(len(space))] for space in self.game.state_space]
        self.G = [[0 for _ in range(len(space))] for space in self.game.state_space]
        self.N = [[{'state': 0, 'action': [0, 0]} for _ in range(len(space))] for space in self.game.state_space]
        self.N_0 = 10000
        self.discount_factor = 1

    def _get_optimal_action(self, state):
        rewards = self.Q[state.player_sum - 1][state.dealer_card - 1]
        max_reward = max(rewards)
        return choice([i for i, reward in enumerate(rewards) if reward >= max_reward])

    def e_greedy(self, e, state):
        if random() < e:
            return randint(0, 1)  # only two choices
        return self._get_optimal_action(state)

    def _calc_e(self, state):
        return self.N_0 / (self.N_0 + self.N[state.player_sum - 1][state.dealer_card - 1]['state'])

    def _update_Q(self, history):
        for i, (s_t, a_t, r_t) in enumerate(history):
            player_index = s_t.player_sum - 1
            dealer_index = s_t.dealer_card - 1
            # G_t = sum([r_k * (self.discount_factor ** i_i) for i_i, (_, _, r_k) in enumerate(history[i:])])
            # self.G[player_index][dealer_index] += G_t
            self.Q[player_index][dealer_index][a_t] += (1 / self.N[player_index][dealer_index]['action'][a_t]) * \
                                                       (r_t - self.Q[player_index][dealer_index][a_t])

    def _run(self, ):
        t = 0
        history = []
        s_t = self.game.start_game()
        while True:
            a_t = self.e_greedy(self._calc_e(s_t), s_t)
            # s_t, r_t, terminal = self.game.step(s_t, self.game.action_space[a_t])
            self.N[s_t.player_sum - 1][s_t.dealer_card - 1]['state'] += 1
            self.N[s_t.player_sum - 1][s_t.dealer_card - 1]['action'][a_t] += 1
            s_t_1, r_t, terminal = self.game.step(copy(s_t), self.game.action_space[a_t])
            history.append([s_t, a_t, r_t])
            s_t = s_t_1
            # self.Q[s_t.player_sum - 1][s_t.dealer_card - 1][a_t] += \
            #     (1 / self.N[s_t.player_sum - 1][s_t.dealer_card - 1]['action'][a_t]) * \
            #     (r_t_1 - self.Q[s_t.player_sum - 1][s_t.dealer_card - 1][a_t])
            if terminal:
                break
            t += 1
        self._update_Q(history)
        return t

    def optimize(self, iterations):
        self.game.add_player(self)
        for iteration in range(iterations):
            self._run()

    def get_V_star(self, ):
        player_sum = [row[0][0] for row in self.game.state_space]
        dealer_showing = [item[1] for item in self.game.state_space[0]]
        V_star = [[1 - actions.index(max(actions)) for actions in dealer] for dealer in self.Q]
        return player_sum, dealer_showing, V_star


if __name__ == "__main__":
    mc_agent = MonteCarloAgent(gym=Easy21(), )

    mc_agent.optimize(iterations=int(10000))

    print(mc_agent.Q)
