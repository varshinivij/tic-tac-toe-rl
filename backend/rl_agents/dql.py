"""
*Q-value* is the expected total reward you’ll get if you take a particular action in a given state and then follow your strategy afterwards.

Input: state of the board [1, 0, 0 ... -1] (9 values, 1 dimensional) 
Output: Q-value per position, unavailable positions masked
Input (9) → Hidden (32, ReLU) → Output (9, raw Q-values)

Relu Activation Function: f(x)=max(0,x) | non-linear activation + vanishing gradient problem {pass gradient if positive --> larger derivatives --> learning does not stop at the earlier layers}
MSE loss = (predicted_Q - target_Q)^2 

**** BELLMAN EQUATION ***
target=immediate reward + (discount factor * best possible future reward) 
target = r + gamma * max(next_Q) 

ε-greedy policy:
probability ε → pick random move (exploration)
probability 1-ε → pick best predicted move (argmax Q(s, :))
""" 


import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from rl_agents.q_learning import Board

# -------------------------------
# Neural Network for Tic-Tac-Toe
# -------------------------------
class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 9 (board) -> hidden: 12 -> output: 9 (Q-values per move)
        self.fc1 = nn.Linear(9, 12)  
        self.fc2 = nn.Linear(12, 9)  

    # mask illegal moves by setting Q-values to -inf
    def mask(self, tensor, board:Board):
        tensor = tensor.clone()  # avoid modifying original tensor
        for i in range(9):
            if not board.is_valid_move(i):
                tensor[0, i] = -float('inf')
        return tensor

    def forward(self, board:Board):
        state_tensor = torch.tensor(board.board, dtype=torch.float).unsqueeze(0)  # shape [1,9]
        x = torch.relu(self.fc1(state_tensor))
        x = self.fc2(x)
        x = self.mask(x, board)  
        return x  


# -------------------------------
# DQN wrapper / training logic
# -------------------------------
class TicTacToeDQ():
    def __init__(self, model:BasicNN, board:Board, player:int=1):
        self.model = model
        self.board = board
        self.player = player

    def reward(self):
        winner = self.board.winner()
        if winner == self.player:
            return 1 
        elif winner == -self.player:
            return -1 
        else:
            return 0 #draw or ongoing game 

    # calculate target Q-value with Bellman equation
    def target(self, next_board:Board, gamma:float):
        imm_reward = self.reward()
        if next_board.game_over():
            max_next_Q = 0
        else:
            with torch.no_grad():  # don't track gradients for next state --> here, we simply need the max_nextQ as a constant, its calculation should not update the weights of the layer
                max_next_Q = self.model(next_board).max().item()
        return imm_reward + gamma * max_next_Q

    def train_step(self, action:int, gamma:float=0.9):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        next_board = self.board.board.copy()
        next_board[action] = self.player
        next_board = Board(next_board)

        # current Q-values
        pred_Q = self.model(self.board)
        pred_Q_a = pred_Q[0, action]

        # compute target
        target_Q = torch.tensor(self.target(next_board, gamma), dtype=torch.float)

        # compute loss
        loss = criterion(pred_Q_a, target_Q)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.board = next_board #update the version of the board 

    # user move: plays opposite to DQN
    def user_move(self, idx:int=-1):
        if not self.board.valid_moves():  
            return
        elif idx == -1:
            idx = random.choice(self.board.valid_moves())  
        elif not self.board.is_valid_move(idx):  
            raise Exception("Invalid move allowed - error in front end")
        self.board.board[idx] = -self.player 

    # select DQN move with epsilon-greedy policy
    def select_move(self, epsilon:float=0.1):
        if random.random() < epsilon:
            return random.choice(self.board.valid_moves())
        else:
            q_values = self.model(self.board)
            return int(torch.argmax(q_values).item())


