import random

class Board:
    def __init__(self, board:list=None):
        if not board or type(board) != list:
            self.board = [0 for _ in range(9)]
        else:
            self.board = board 
    
    def clear(self):
        self.board = [0 for _ in range(9)]
    
    def is_full(self) -> bool:
        for i in range(len(self.board)):
            if self.board[i] == 0:
                return False
        return True 
    
    def values_match(self, idx1, idx2, idx3):
        if self.board[idx1] == self.board[idx2] and self.board[idx2] == self.board[idx3] and self.board[idx1] != 0:
            return True
        return False

    def horizontal_match(self):
        for i in range(0, 7, 3):
            if self.values_match(i, i+1, i+2):
                return self.board[i]
        return None

    def vertical_match(self):
        for i in range(3):
            if self.values_match(i, i+3, i+6):
                return self.board[i]
        return None
    
    def diagonal_match(self):
        if self.values_match(0, 4, 8):
            return self.board[4]
        elif self.values_match(2, 4, 6):
            return self.board[4]
        return None
    
    def __str__(self):
        return "".join(str(x) for x in self.board)
    
    def valid_moves(self):
        moves = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                moves.append(i)
        return moves 
    
    def is_valid_move(self, idx):
        return self.board[idx] == 0  
        
    def winner(self):
        winner = self.horizontal_match() or self.vertical_match() or self.diagonal_match() 
        return winner if winner else 0
    
    def game_over(self):
        return self.is_full() or self.winner()!=0


class TicTacToeGame:
    all_game_history = {}
    
    def __init__(self, board: Board, epsilon=0.1, learning_rate=0.15, player=1):
        self.board = board
        self.lr = learning_rate
        self.epsilon = epsilon
        self.player = player  # 1 or -1
        self.history = []

    def store_user_move(self, idx=-1):
        if not self.board.valid_moves():  
            return
        elif idx == -1:
            idx = random.choice(self.board.valid_moves())  
        elif not self.board.is_valid_move(idx):  
            raise Exception("Invalid move allowed - error in front end")
        
        if (str(self.board), idx) not in self.__class__.all_game_history:
            self.__class__.all_game_history[(str(self.board), idx)] = 0
        
        self.history.append((str(self.board), idx, -self.player))
        self.board.board[idx] = -self.player 

    def reward(self):
        winner = self.board.winner()
        if winner == self.player:
            return 1 
        elif winner == -self.player:
            return -1 
        else:
            return 0 #draw or ongoing game 

    def make_move(self):
        if not self.board.valid_moves():
            return
        state = str(self.board)
        moves = self.board.valid_moves()

        # Greedy: pick the known action with the highest Q-value for this state
        known = {m: self.__class__.all_game_history[(state, m)]
                 for m in moves if (state, m) in self.__class__.all_game_history}
        found = bool(known)

        if random.random() < self.epsilon or not found:
            idx = random.choice(moves)
        else:
            idx = max(known, key=known.get)

        if (state, idx) not in self.__class__.all_game_history:
            self.__class__.all_game_history[(state, idx)] = 0

        self.history.append((state, idx, self.player))
        self.board.board[idx] = self.player

    
    def select_move(self, epsilon: float = 0.0):
        moves = self.board.valid_moves()
        if not moves:
            return None
        if random.random() < epsilon:
            return random.choice(moves)
        state = str(self.board)
        best_move = max(moves, key=lambda m: self.__class__.all_game_history.get((state, m), 0.0))
        return best_move

    def update_Q(self, winner):
        self_reward = self.reward()
        other_reward = -self_reward

        for i in range(len(self.history)):
            state, move, player = self.history[i]
            reward = self_reward if player == self.player else other_reward
            self.__class__.all_game_history[(state, move)] = \
                self.__class__.all_game_history[(state, move)] + self.lr * \
                (reward - self.__class__.all_game_history[(state, move)])



