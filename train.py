import math
import random
import os
import numpy as np
import torch

from agent import DQNAgent, ReplayMemory
from game import (
    create_board, get_valid_locations, drop_piece, get_next_open_row,
    winning_move, minimax, get_state_tensor, get_immediate_winning_move,
    is_suicide_move
)

EPISODES = 50000
BATCH_SIZE = 64
WARMUP_STEPS = 1000
AI_PIECE = 1
OPPONENT_PIECE = 2
OPPONENT_TYPE = "SelfPlay" # Random or Minimax or SelfPlay
SELF_PLAY_UPDATE_FREQ = 300
MINIMAX_DEPTH = [2]
LEARN_FREQUENCY = 2

def get_opponent_action(board, valid_moves, opp_type, opponent_net=None, device="cpu"):
    op_win = get_immediate_winning_move(board, OPPONENT_PIECE)
    op_block = get_immediate_winning_move(board, AI_PIECE)

    if op_win is not None:
        return op_win
    if op_block is not None and random.random() < 0.8:
        return op_block

    rand_val = random.random()

    if opp_type == "SelfPlay" and opponent_net is not None and rand_val < 0.70:
        opp_state = get_state_tensor(board, OPPONENT_PIECE, AI_PIECE).to(device)
        with torch.no_grad():
            q_values = opponent_net(opp_state)[0].cpu().numpy()

        safe_moves = [c for c in valid_moves if not is_suicide_move(board, c, OPPONENT_PIECE, AI_PIECE)]
        candidate_moves = safe_moves if len(safe_moves) > 0 else valid_moves

        best_c = candidate_moves[0]
        max_q = -math.inf
        for c in candidate_moves:
            if q_values[c] > max_q:
                max_q = q_values[c]
                best_c = c
        return best_c

    elif (opp_type == "SelfPlay" and rand_val < 0.90) or opp_type == "Minimax":
        action, _ = minimax(
            board, random.choice(MINIMAX_DEPTH), -math.inf, math.inf, True,
            ai_piece=OPPONENT_PIECE, player_piece=AI_PIECE
        )
        return action if action in valid_moves else random.choice(valid_moves)

    else: # Random
        safe_moves = [c for c in valid_moves if not is_suicide_move(board, c, OPPONENT_PIECE, AI_PIECE)]
        choices = safe_moves if len(safe_moves) > 0 else valid_moves
        return random.choice(choices)

def train():
    agent = DQNAgent()
    memory = ReplayMemory(capacity=50000)

    from agent import DuelingConnect4Net
    opponent_net = DuelingConnect4Net().to(agent.device)
    opponent_net.load_state_dict(agent.policy_net.state_dict())
    opponent_net.eval()

    win_count = 0
    lose_count = 0
    draw_count = 0
    recent_rewards = 0
    total_steps = 0

    print(f'Starting training on {agent.device}...')
    for episode in range(1, EPISODES + 1):
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
                ai_action = agent.act(ai_state, valid_moves, board=board, my_piece=AI_PIECE, opp_piece=OPPONENT_PIECE, eval_mode=False)
                row = get_next_open_row(board, ai_action)
                drop_piece(board, row, ai_action, AI_PIECE)
                total_steps += 1

                if winning_move(board, AI_PIECE):
                    win_count += 1
                    game_over = True
                    memory.push_with_symmetry(ai_state, ai_action, 1.0, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), True)
                    episode_reward += 1.0

                elif len(get_valid_locations(board)) == 0:
                    draw_count += 1
                    game_over = True
                    memory.push_with_symmetry(ai_state, ai_action, 0.0, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), True)

                if len(memory) >= WARMUP_STEPS and total_steps % LEARN_FREQUENCY == 0:
                    agent.learn(memory, BATCH_SIZE)
                    agent.update_target_network(tau=0.005)

            else: # Opponent Turn
                op_action = get_opponent_action(
                    board, valid_moves, OPPONENT_TYPE,
                    opponent_net=opponent_net, device=agent.device
                )
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
                    next_state = get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE)
                    memory.push_with_symmetry(ai_state, ai_action, reward, next_state, game_over)
                    episode_reward += reward

                    if len(memory) >= WARMUP_STEPS and total_steps % LEARN_FREQUENCY == 0:
                        agent.learn(memory, BATCH_SIZE)
                        agent.update_target_network(tau=0.005)

            turn = (turn + 1) % 2

        recent_rewards += episode_reward

        # Decay epsilon only after warmup steps
        if len(memory) >= WARMUP_STEPS and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode % SELF_PLAY_UPDATE_FREQ == 0:
            opponent_net.load_state_dict(agent.policy_net.state_dict())
            print(f">>> [Self-Play] Opponent updated to policy at Episode {episode}.")

        if episode % 100 == 0:
            agent.update_target_network()
            total_games = win_count + lose_count + draw_count
            winning_rate = win_count / total_games if total_games > 0 else 0.0
            mean_reward = recent_rewards / 100
            print(f"Episode: {episode} | WinRate: {winning_rate:.2%} | AvgReward: {mean_reward:.4f} | Epsilon: {agent.epsilon:.3f}")
            win_count, lose_count, draw_count, recent_rewards = 0, 0, 0, 0

    print("Training finished!")
    os.makedirs("model", exist_ok=True)
    model_path = "model/connect4_model_selfplay.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}!")

if __name__ == "__main__":
    train()
