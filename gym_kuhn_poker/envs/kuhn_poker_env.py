from copy import copy
from itertools import dropwhile
from typing import List
import enum
import numpy as np
import gym
from gym.spaces import Discrete, Tuple
from .one_hot_space import OneHotEncoding


class ActionType(enum.Enum):
    PASS = 0
    BET  = 1


class KuhnPokerEnv(gym.Env):
    '''
    Implementation of Kuhn's poker in accordance to OpenAI gym environment interface.
    '''

    def __init__(self, number_of_players=2, deck_size=3, betting_rounds=2, ante=1):
        '''
        :param number_of_players: Number of players (Default 2).
        :param deck_size: Size of the deck from which cards will be drawn, one for each player (Default 3).
        :param betting_rounds: Number of times that (Default: 2).
        :param ante: Amount of utility that all players must pay at the beginning of an episode (Default 1).
        '''
        assert number_of_players >= 2, "Game must be played with at least 2 players"
        assert deck_size >= number_of_players, "The deck of cards must contain at least one card per player"
        assert betting_rounds >= 2, "The deck of cards must contain at least one card per player"
        assert ante >= 1, "Minimun ante must be one"

        self.done = False
        self.number_of_players = number_of_players
        self.deck_size = deck_size
        self.betting_rounds = betting_rounds
        self.ante = ante

        self.action_space = Discrete(len(ActionType))
        self.action_space_size = len(ActionType)

        single_observation_space = self.calculate_observation_space()
        self.observation_space = Tuple([single_observation_space
                                       for _ in range(self.number_of_players)])
        self.state_space_size = None # TODO len(self.random_initial_state_vector())

        self.betting_history_index = (self.number_of_players +
                                      self.number_of_players * self.deck_size)
        self.betting_round_offset = self.number_of_players * (1 + len(ActionType))

    def reset(self):
        self.done = False
        self.current_player = 0
        self.history: List[ActionType] = []
        self.state = self.random_initial_state_vector()
        self.first_to_bet, self.winner = None, None
        # Players that will face off in card comparison after betting ends
        self.elegible_players = [i for i in range(self.number_of_players)]
        return [self.observation_from_state(player_id=i)
                for i in range(self.number_of_players)]

    def step(self, action):
        assert isinstance(action, int) and 0 <= action <= len(ActionType), \
               f"Action outside of valid range: [0,{len(ActionType)}]"
        assert not self.done, "Episode is over"
        move = ActionType(action)

        self.state = self.update_state(self.current_player, move)

        if self.done: reward_vector = self.reward_vector_for_winner(self.winner)
        else: reward_vector = [0] * self.number_of_players

        info = {}
        return [self.observation_from_state(i) for i in range(self.number_of_players)], \
               reward_vector, self.done, info

    def calculate_reward(self, state):
        if False: # If is terminal
            pass
        return [0] * self.number_of_players

    def update_state(self, player_id, move):
        self.history.append(move)

        if self.first_to_bet is None: player_move_index = self.betting_history_index
        else: player_move_index = self.betting_history_index + self.betting_round_offset
        player_move_index += self.current_player * (1 + len(ActionType))
        if move == ActionType.PASS:
            self.state[player_move_index] = 0
            self.state[player_move_index + 1] = 1
        if move == ActionType.BET:
            self.state[player_move_index] = 0
            self.state[player_move_index + 2] = 1
            if self.first_to_bet is None:
                self.first_to_bet = player_id
                self.elegible_players = []
            self.elegible_players += [player_id]
            self.state[-self.number_of_players + player_id] += 1

        # Update current player
        self.state[self.current_player] = 0
        self.current_player = (self.current_player + 1) % self.number_of_players
        self.state[self.current_player] = 1

        if self.is_state_terminal():
            self.done = True
            self.winner = self.get_winner()

        return self.state

    def get_winner(self):
        base_i = self.number_of_players
        player_hand = lambda p_i: slice(base_i + p_i * self.deck_size,
                                        base_i + p_i * self.deck_size + self.deck_size)
        player_hands = [self.state[player_hand(i)] for i in self.elegible_players]
        card_values = [hand.index(1) for hand in player_hands]
        max_card = max(card_values)
        winner = self.elegible_players[card_values.index(max_card)]
        return winner

    def get_total_pot(self):
        return sum(self.state[-self.number_of_players:])

    def get_player_pot(self, player_id):
        assert 0 <= player_id < self.number_of_players, \
               f"Player id must be in [0, {self.number_of_players}]"
        return self.state[-self.number_of_players + player_id]

    def reward_vector_for_winner(self, winner: int):
        assert 0 <= winner < self.number_of_players, \
               f"Player id must be in [0, {self.number_of_players}]"
        reward_vector = [- self.get_player_pot(i)
                         for i in range(self.number_of_players)]
        reward_vector[winner] += self.get_total_pot()
        return reward_vector

    def is_state_terminal(self):
        return self.all_players_passed() or self.betting_is_over()

    def all_players_passed(self):
        all_players_acted = len(self.history) == self.number_of_players
        return all_players_acted and all(map(lambda m: m == ActionType.PASS,
                                             self.history))

    def betting_is_over(self):
        after_bet_moves = dropwhile(lambda m: m != ActionType.BET,
                                    self.history)
        return len(list(after_bet_moves)) == self.number_of_players

    def random_initial_state_vector(self):
        # Player 1 always begins
        player_turn = [1, 0]
        # Deal 1 card to each player
        dealt_cards_per_player = np.random.choice(range(self.deck_size),
                                                  size=self.number_of_players,
                                                  replace=False)
        player_hands = self.vector_from_dealt_hands(dealt_cards_per_player)

        betting_history_vector = []
        for _ in range(self.betting_rounds):
            for _ in range(self.number_of_players):
                betting_history_vector += [1] + ([0] * len(ActionType))

        pot_contributions = [self.ante] * self.number_of_players
        return player_turn + player_hands + \
               betting_history_vector + \
               pot_contributions

    def vector_from_dealt_hands(self, dealt_cards_per_player):
        player_hands = [[0] * self.deck_size
                        for _ in range(self.number_of_players)]
        player_hands = []
        for dealt_card in dealt_cards_per_player:
            hand = [0] * self.deck_size
            hand[dealt_card] = 1
            player_hands += hand
        return player_hands

    def observation_from_state(self, player_id: int):
        '''
        Returns the observation vector for :param player_id:. It is strictly
        a sub-vector of the real state.

        Contains:
            - Player id
            - Card randomly dealt to :param player_id:
            - Betting history
            - Amount of utility on pot by each player

        :param player_id: Index from [0, self.number_of_players] of the player
                          whose observation vector is being generated
        :returns: Observation vector for :param player_id:
        '''
        encoded_id = [0] * self.number_of_players
        encoded_id[player_id] = 1

        dealt_cards_start_index = self.number_of_players
        player_card_index = dealt_cards_start_index + (player_id * self.deck_size)
        player_card = self.state[player_card_index: player_card_index + self.deck_size]

        betting_history_start_index = dealt_cards_start_index + (self.number_of_players * self.deck_size)
        betting_history_and_pot = self.state[betting_history_start_index:]

        return encoded_id + player_card + betting_history_and_pot

    def calculate_observation_space(self):
        player_id  = OneHotEncoding(self.number_of_players)
        dealt_card = OneHotEncoding(self.deck_size)

        betting_states = [OneHotEncoding(len(ActionType) + 1)
                          for _ in range(self.number_of_players)
                          for _ in range(self.betting_rounds)]
        pot_contributions = Tuple([Discrete(3) for _ in range(self.number_of_players)])
        return Tuple([player_id, dealt_card, *betting_states, pot_contributions])

    def render(self, mode='human', close=False):
        raise NotImplementedError('Rendering has not been coded yet')
