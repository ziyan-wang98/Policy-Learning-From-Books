import random
import math

class Player:
    def __init__(self, name, elo=500):
        self.name = name
        self.elo = elo

def expected_score(player_a, player_b):
    return 1 / (1 + math.exp((player_b.elo - player_a.elo) / 100))

def update_elo(player, opponent, score, k=64):
    expected = expected_score(player, opponent)
    elo_diff = opponent.elo - player.elo
    adjustment = k * (score - expected) * (1 + abs(elo_diff) / 1000)
    player.elo += adjustment

def play_game(player1, player2):
    #TODO
    return random.choice([1, 0, 0.5])

players = [
    Player("LLM_Aget"),
    Player("LLM_RAG"),
    Player("URI_(Our)"),
    Player("Random"),
    Player("Minmax_(Oracle)")
]

num_games = 100

for _ in range(num_games):
    for i in range(len(players)):
        for j in range(i+1, len(players)):
            result = play_game(players[i], players[j])
            update_elo(players[i], players[j], result)
            update_elo(players[j], players[i], 1 - result)


for player in sorted(players, key=lambda x: x.elo, reverse=True):
    print(f"{player.name}: {player.elo:.2f}")