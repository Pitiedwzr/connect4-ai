import math
import random

import numpy as np
import torch

from agent import DQNAgent, ReplayMemory
from game import create_board, get_valid_locations, drop_piece, get_next_open_row, winning_move, minimax, get_state_tensor

EPISODES = 200000
BATCH_SIZE = 64
AI_PIECE = 1
OPPONENT_PIECE = 2
OPPONENT_TYPE = "Mix" # Random or Minimax or Mix
MIX_THRESHOLD = 0.5 # 0 - 1, Higher - More Minimax
MINIMAX_DEPTH = 2

agent = DQNAgent()
memory = ReplayMemory(capacity=50000)
win_count = 0
lose_count = 0
draw_count = 0

print('Starting training...')
for episode in range(EPISODES):
    board = create_board()
    game_over = False
    turn = 0
    episode_reward = 0
    ai_state = None
    ai_action = None

    while not game_over:
        valid_moves = get_valid_locations(board)
        if turn == 0: # AI Turn
            ai_state = get_state_tensor(board, AI_PIECE, OPPONENT_PIECE)
            ai_action = agent.act(ai_state, valid_moves)
            row = get_next_open_row(board, ai_action)
            drop_piece(board, row, ai_action, AI_PIECE)

            if winning_move(board, AI_PIECE):
                win_count += 1
                game_over = True
                memory.push(ai_state, ai_action, 1.0, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), True)
                agent.learn(memory, BATCH_SIZE)
                episode_reward += 1.0

            elif len(get_valid_locations(board)) == 0:
                draw_count += 1
                game_over = True
                memory.push(ai_state, ai_action, 0.0, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), True)
                agent.learn(memory, BATCH_SIZE)

        else: # Opponent Turn
            if OPPONENT_TYPE == "Random":
                op_action = np.random.choice(valid_moves)
            elif OPPONENT_TYPE == "Minimax":
                op_action, _ = minimax(board, MINIMAX_DEPTH, -math.inf, math.inf, True)
            elif OPPONENT_TYPE == "Mix":
                if random.randint(0, 1) >= MIX_THRESHOLD:
                    op_action = np.random.choice(valid_moves)
                else:
                    op_action, _ = minimax(board, MINIMAX_DEPTH, -math.inf, math.inf, True)

            row = get_next_open_row(board, op_action)
            drop_piece(board, row, op_action, OPPONENT_PIECE)
            reward = 0.0
            if winning_move(board, OPPONENT_PIECE):
                reward = -1.0
                lose_count += 1
                game_over = True
            elif len(get_valid_locations(board)) == 0:
                reward = 0.0
                draw_count += 1
                game_over = True

            memory.push(ai_state, ai_action, reward, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), game_over)
            agent.learn(memory, BATCH_SIZE)
            episode_reward += reward

        turn = (turn + 1) % 2

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if episode % 100 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        winning_rate = win_count / (win_count + lose_count + draw_count)
        mean_reward = episode_reward / 100
        print(f"Episode: {episode} | WinRate: {winning_rate:.2%} | AvgReward: {mean_reward:.4f} | Epsilon: {agent.epsilon:.3f}")
        win_count, lose_count, draw_count, episode_reward = 0, 0, 0, 0

print("Training finished!")
model_path = "model/connect4_model.pth"
torch.save(agent.policy_net.state_dict(), model_path)
print(f"Model saved to {model_path}!")