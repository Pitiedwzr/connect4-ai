import numpy as np
import torch
import math
from game import create_board, get_valid_locations, drop_piece, get_next_open_row, winning_move, minimax
from agent import DQNAgent, ReplayMemory


EPISODES = 10000
BATCH_SIZE = 64
AI_PIECE = 1
OPPONENT_PIECE = 2
OPPONENT_TYPE = "Minimax" # Random or Minimax
MINIMAX_DEPTH = 1

agent = DQNAgent()
memory = ReplayMemory(capacity=10000)

print('Starting training...')
for episode in range(EPISODES):
    board = create_board()
    game_over = False
    turn = 0
    while not game_over:
        valid_moves = get_valid_locations(board)
        if turn == 0:
            state = board.copy()
            action = agent.act(state, valid_moves)
            row = get_next_open_row(board, action)
            drop_piece(board, row, action, AI_PIECE)
            reward = 0

            if winning_move(board, AI_PIECE):
                reward = 1
                game_over = True
            elif len(get_valid_locations(board)) == 0:
                reward = 0
                game_over = True

            next_state = board.copy()
            memory.push(state, action, reward, next_state, game_over)
            agent.learn(memory, BATCH_SIZE)
        else:
            if OPPONENT_TYPE == "Random":
                action = np.random.choice(valid_moves)
            else:
                action, _ = minimax(board, MINIMAX_DEPTH, -math.inf, math.inf, True)
            row = get_next_open_row(board, action)
            drop_piece(board, row, action, OPPONENT_PIECE)

            if winning_move(board, OPPONENT_PIECE):
                reward = -1
                game_over = True
            elif len(get_valid_locations(board)) == 0:
                reward = 0
                game_over = True

        turn = (turn + 1) % 2

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if episode % 100 == 0:
        print(f"Episode: {episode}, Epsilon: {agent.epsilon:.3f}, Memory Size: {len(memory)}")
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

print("Training finished!")
model_path = "model/connect4_model.pth"
torch.save(agent.policy_net.state_dict(), model_path)
print(f"Model saved to {model_path}!")