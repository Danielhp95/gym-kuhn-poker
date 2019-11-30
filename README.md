# gym-kuhn-poker

![Kuhn's poker tree](https://raw.githubusercontent.com/Danielhp95/gym-kuhn-poker/master/Kuhn_poker_tree.png)

Kuhn poker implemented in accordance to OpenAI gym interface.

## Game description

Kuhn poker is an extremely simplified form of poker developed by Harold W. Kuhn
as a simple model **zero-sum**, **two-player**, **imperfect-information**, **sequential** game,
amenable to a complete game-theoretic analysis. In Kuhn poker, the deck includes
only three playing cards, for example a King, Queen, and Jack.
One card is dealt to each player, which may place bets similarly
to a standard poker. If both players bet or both players pass,
the player with the higher card wins, otherwise, the betting player wins.

[Wikipedia Explanation of Kuhn's poker](https://www.wikiwand.com/en/Kuhn_poker)

## Usage & Configuration

`gym` must be installed. An Kuhn's poker environment can be created via running inside a `python` interpreter:

```python
>>> import gym
>>> import gym_kuhn_poker
>>> env = gym.make('KuhnPoker-v0', **dict()) # optional secondary argument
```

The `dict()` in the expression above includes keyword arguments for the underlying environment:
+ `number_of_players`: Number of players (Default 2).
+ `deck_size`: Size of the deck from which cards will be drawn, one for each player (Default 3).
+ `betting_rounds`: Number of times that (Default: 2).
+ `ante`: Amount of utility that all players must pay at the beginning of an episode (Default 1).

**NOTE:** Environment has only been tested with default values,
feel free to open an issue if they don't work for other values!

## Installation

### Installing via pip

This package is available in PyPi as `gym-kuhn-poker`

```bash
pip install gym-kuhn-poker
```

### Installing via cloning this repository

```bash
git clone https://www.github.com/Danielhp95/gym-kuhn-poker
cd gym-kuhn-poker
pip install -e .
```
