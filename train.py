import math
import random
import os
import numpy as np
import torch

from agent import DQNAgent, ReplayMemory
from game import (
    create_board, get_valid_locations, drop_piece, get_next_open_row,
    winning_move, minimax, get_state_tensor, get_immediate_winning_move
)

EPISODES = 10000
BATCH_SIZE = 64
WARMUP_STEPS = 1000
AI_PIECE = 1
OPPONENT_PIECE = 2
OPPONENT_TYPE = "Mix" # Random or Minimax or Mix
MIX_THRESHOLD = 0.5 # 0 - 1, Higher - More Minimax
MINIMAX_DEPTH = 2
LEARN_FREQUENCY = 2

def train():
    agent = DQNAgent()
    memory = ReplayMemory(capacity=50000)
    win_count = 0
    lose_count = 0
    draw_count = 0
    recent_rewards = 0
    total_steps = 0

    print('Starting training...')
    for episode in range(EPISODES):
        board = create_board()
        game_over = False
        turn = random.choice([0, 1]) # 50% AI goes first, 50% Opponent goes first
        episode_reward = 0
        ai_state = None
        ai_action = None

        while not game_over:
            valid_moves = get_valid_locations(board)
            if turn == 0: # AI Turn
                ai_state = get_state_tensor(board, AI_PIECE, OPPONENT_PIECE)
                ai_action = agent.act(ai_state, valid_moves, board=board, my_piece=AI_PIECE, opp_piece=OPPONENT_PIECE)
                row = get_next_open_row(board, ai_action)
                drop_piece(board, row, ai_action, AI_PIECE)
                total_steps += 1

                if winning_move(board, AI_PIECE):
                    win_count += 1
                    game_over = True
                    memory.push_with_symmetry(ai_state, ai_action, 1.0, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), True)
                    if len(memory) >= WARMUP_STEPS and total_steps % LEARN_FREQUENCY == 0:
                        agent.learn(memory, BATCH_SIZE)
                    episode_reward += 1.0

                elif len(get_valid_locations(board)) == 0:
                    draw_count += 1
                    game_over = True
                    memory.push_with_symmetry(ai_state, ai_action, 0.0, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), True)
                    if len(memory) >= WARMUP_STEPS and total_steps % LEARN_FREQUENCY == 0:
                        agent.learn(memory, BATCH_SIZE)

            else: # Opponent Turn
                # Tactical check for opponent
                op_win = get_immediate_winning_move(board, OPPONENT_PIECE)
                op_block = get_immediate_winning_move(board, AI_PIECE)

                if OPPONENT_TYPE == "Random":
                    if op_win is not None:
                        op_action = op_win
                    elif op_block is not None and random.random() < 0.7:
                        op_action = op_block
                    else:
                        op_action = np.random.choice(valid_moves)
                elif OPPONENT_TYPE == "Minimax":
                    op_action, _ = minimax(board, MINIMAX_DEPTH, -math.inf, math.inf, True)
                elif OPPONENT_TYPE == "Mix":
                    if random.random() >= MIX_THRESHOLD:
                        if op_win is not None:
                            op_action = op_win
                        elif op_block is not None and random.random() < 0.7:
                            op_action = op_block
                        else:
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

                # Guard: Only push transition when AI has already acted in this episode
                if ai_state is not None:
                    memory.push_with_symmetry(ai_state, ai_action, reward, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), game_over)
                    if len(memory) >= WARMUP_STEPS and total_steps % LEARN_FREQUENCY == 0:
                        agent.learn(memory, BATCH_SIZE)
                    episode_reward += reward

            turn = (turn + 1) % 2

        recent_rewards += episode_reward

        # Decay epsilon only after warmup steps
        if len(memory) >= WARMUP_STEPS and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode > 0 and episode % 100 == 0:
            agent.update_target_network()
            total_games = win_count + lose_count + draw_count
            winning_rate = win_count / total_games if total_games > 0 else 0.0
            mean_reward = recent_rewards / 100
            print(f"Episode: {episode} | WinRate: {winning_rate:.2%} | AvgReward: {mean_reward:.4f} | Epsilon: {agent.epsilon:.3f}")
            win_count, lose_count, draw_count, recent_rewards = 0, 0, 0, 0

    print("Training finished!")
    os.makedirs("model", exist_ok=True)
    model_path = "model/connect4_model.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}!")

if __name__ == "__main__":
    train()
