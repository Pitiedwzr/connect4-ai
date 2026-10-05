import math
import random
import os
import numpy as np
import torch
from accelerate import Accelerator

from agent import DQNAgent, ReplayMemory, DuelingConnect4Net
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
    # 1. Initialize Hugging Face Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Set unique seed per GPU process for diverse environment exploration
    base_seed = 42 + accelerator.process_index
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    agent = DQNAgent(accelerator=accelerator)
    agent.device = device
    agent.policy_net = agent.policy_net.to(device)
    agent.target_net = agent.target_net.to(device)

    # 2. Prepare policy network and optimizer with Accelerator (DDP wrapper)
    agent.policy_net, agent.optimizer = accelerator.prepare(agent.policy_net, agent.optimizer)

    # Helper function to update target network cleanly without DDP 'module.' prefix issues
    def update_target_network_ddp(tau=1.0):
        unwrapped_policy = accelerator.unwrap_model(agent.policy_net)
        if tau == 1.0:
            agent.target_net.load_state_dict(unwrapped_policy.state_dict())
        else:
            for target_param, policy_param in zip(agent.target_net.parameters(), unwrapped_policy.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    agent.update_target_network = update_target_network_ddp

    memory = ReplayMemory(capacity=50000)

    # 3. Setup self-play opponent network
    opponent_net = DuelingConnect4Net().to(device)
    unwrapped_policy = accelerator.unwrap_model(agent.policy_net)
    opponent_net.load_state_dict(unwrapped_policy.state_dict())
    opponent_net.eval()

    win_count = 0
    lose_count = 0
    draw_count = 0
    recent_rewards = 0.0

    # Distribute total episodes across available cards
    num_processes = accelerator.num_processes
    episodes_per_proc = EPISODES // num_processes
    log_freq = max(1, 100 // num_processes)
    self_play_freq = max(1, SELF_PLAY_UPDATE_FREQ // num_processes)

    if accelerator.is_main_process:
        print(f"Starting training across {num_processes} GPUs. Total episodes: {EPISODES} ({episodes_per_proc} per GPU)...")

    for episode in range(1, episodes_per_proc + 1):
        board = create_board()
        game_over = False
        turn = random.choice([0, 1]) # 50% AI goes first, 50% Opponent goes first
        episode_reward = 0.0
        ai_state = None
        ai_action = None
        episode_steps = 0

        while not game_over:
            valid_moves = get_valid_locations(board)
            if turn == 0: # AI Turn
                ai_state = get_state_tensor(board, AI_PIECE, OPPONENT_PIECE)
                ai_action = agent.act(ai_state, valid_moves, board=board, my_piece=AI_PIECE, opp_piece=OPPONENT_PIECE, eval_mode=False)
                row = get_next_open_row(board, ai_action)
                drop_piece(board, row, ai_action, AI_PIECE)
                episode_steps += 1

                if winning_move(board, AI_PIECE):
                    win_count += 1
                    game_over = True
                    memory.push_with_symmetry(ai_state, ai_action, 1.0, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), True)
                    episode_reward += 1.0

                elif len(get_valid_locations(board)) == 0:
                    draw_count += 1
                    game_over = True
                    memory.push_with_symmetry(ai_state, ai_action, 0.0, get_state_tensor(board.copy(), AI_PIECE, OPPONENT_PIECE), True)

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

            turn = (turn + 1) % 2

        recent_rewards += episode_reward

        # 4. Synchronized training step to prevent DDP all-reduce deadlocks
        accelerator.wait_for_everyone()

        # Check if all GPUs have enough samples to train
        ready_flag = torch.tensor([1.0 if len(memory) >= WARMUP_STEPS else 0.0], device=device)
        is_ready = accelerator.reduce(ready_flag, reduction="min").item() >= 1.0

        # Synchronize number of gradient steps across all cards
        steps_tensor = torch.tensor([float(episode_steps)], device=device)
        avg_steps = accelerator.reduce(steps_tensor, reduction="mean").item()
        learn_steps = max(1, int(round(avg_steps / LEARN_FREQUENCY)))

        if is_ready:
            for _ in range(learn_steps):
                agent.learn(memory, BATCH_SIZE)
                agent.update_target_network(tau=0.005)

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

        # Update opponent network periodically for self-play
        if episode % self_play_freq == 0:
            unwrapped = accelerator.unwrap_model(agent.policy_net)
            opponent_net.load_state_dict(unwrapped.state_dict())
            if accelerator.is_main_process:
                print(f">>> [Self-Play] Opponent updated to policy at Episode {episode * num_processes}.")

        # 5. Aggregate metrics across both GPUs and log from Rank 0
        if episode % log_freq == 0:
            agent.update_target_network(tau=1.0)

            metrics = torch.tensor(
                [win_count, lose_count, draw_count, recent_rewards],
                dtype=torch.float32,
                device=device
            )
            reduced_metrics = accelerator.reduce(metrics, reduction="sum")

            if accelerator.is_main_process:
                total_win = reduced_metrics[0].item()
                total_lose = reduced_metrics[1].item()
                total_draw = reduced_metrics[2].item()
                total_reward = reduced_metrics[3].item()
                total_games = total_win + total_lose + total_draw

                winning_rate = total_win / total_games if total_games > 0 else 0.0
                mean_reward = total_reward / (log_freq * num_processes)
                global_episode = episode * num_processes
                print(f"Episode: {global_episode}/{EPISODES} | WinRate: {winning_rate:.2%} | AvgReward: {mean_reward:.4f} | Epsilon: {agent.epsilon:.3f}")

            win_count, lose_count, draw_count, recent_rewards = 0, 0, 0, 0.0

    # 6. Save model on main process
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Training finished!")
        os.makedirs("model", exist_ok=True)
        model_path = "model/connect4_model_selfplay.pth"
        unwrapped_policy = accelerator.unwrap_model(agent.policy_net)
        torch.save(unwrapped_policy.state_dict(), model_path)
        print(f"Model saved to {model_path}!")

if __name__ == "__main__":
    train()