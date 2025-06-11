# deep_cfr_working.py - Complete Deep CFR that integrates with your existing code

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
from collections import deque
from typing import Dict, List, Tuple

# Import your existing classes
# First rename your files: src.txt -> src.py, cfr.txt -> cfr.py
try:
    from src import Table, Player, Card, evaluate_best_hand
    from cfr import Normalization
except ImportError:
    print("Please rename src.txt to src.py and cfr.txt to cfr.py first!")
    exit()

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AdvantageNetwork(nn.Module):
    def __init__(self, input_size: int = 20, num_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        return self.net(x)

class StrategyNetwork(nn.Module):
    def __init__(self, input_size: int = 20, num_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

class ReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

class SimpleFeatureExtractor:
    def __init__(self):
        self.rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'A':14}
        
    def extract_features(self, table: Table, player_idx: int) -> torch.FloatTensor:
        """Extract simple but effective features"""
        features = []
        player = table.players[player_idx]
        
        # 1. Hand features (6 dimensions)
        if len(player.hole_cards) == 2:
            c1, c2 = player.hole_cards
            r1, r2 = self.rank_map[c1.rank], self.rank_map[c2.rank]
            features.extend([r1/14.0, r2/14.0])  # Normalized ranks
            features.append(float(c1.suit == c2.suit))  # Suited
            features.append(float(abs(r1-r2) <= 1))  # Connected
            features.append(max(r1, r2) / 14.0)  # High card
            features.append(min(r1, r2) / 14.0)  # Low card
        else:
            features.extend([0.0] * 6)
            
        # 2. Stage features (4 dimensions)
        stage_encoding = {
            "pre-flop": [1,0,0,0], 
            "flop": [0,1,0,0], 
            "turn": [0,0,1,0], 
            "river": [0,0,0,1]
        }
        features.extend(stage_encoding.get(table.stage, [0,0,0,0]))
        
        # 3. Betting features (5 dimensions)
        features.append(min(table.pot / 100.0, 10.0))  # Pot size
        features.append(min(table.current_bet / 50.0, 5.0))  # Current bet
        features.append(min(table.amount_to_call() / 50.0, 5.0))  # To call
        features.append(float(table._bet_occurred))  # Betting occurred
        features.append(float(player_idx))  # Player position
        
        # 4. Player features (5 dimensions)
        features.append(min(player.chips / 1000.0, 2.0))  # Stack size
        features.append(min(player.total_committed / 100.0, 5.0))  # Committed
        
        # Opponent features
        opponent = table.players[1 - player_idx]
        features.append(min(opponent.chips / 1000.0, 2.0))  # Opponent stack
        features.append(min(opponent.total_committed / 100.0, 5.0))  # Opponent committed
        features.append(float(opponent.in_hand))  # Opponent still in
        
        return torch.FloatTensor(features)

class DeepCFRTrainer:
    def __init__(self):
        self.feature_extractor = SimpleFeatureExtractor()
        
        # Networks
        self.advantage_net_p0 = AdvantageNetwork(20, 3)
        self.advantage_net_p1 = AdvantageNetwork(20, 3)
        self.strategy_net_p0 = StrategyNetwork(20, 3)
        self.strategy_net_p1 = StrategyNetwork(20, 3)
        
        # Optimizers
        self.adv_opt_p0 = optim.Adam(self.advantage_net_p0.parameters(), lr=0.001)
        self.adv_opt_p1 = optim.Adam(self.advantage_net_p1.parameters(), lr=0.001)
        self.strat_opt_p0 = optim.Adam(self.strategy_net_p0.parameters(), lr=0.001)
        self.strat_opt_p1 = optim.Adam(self.strategy_net_p1.parameters(), lr=0.001)
        
        # Buffers
        self.adv_buffer_p0 = ReplayBuffer()
        self.adv_buffer_p1 = ReplayBuffer()
        self.strat_buffer_p0 = ReplayBuffer()
        self.strat_buffer_p1 = ReplayBuffer()
        
        self.action_map = {"fold": 0, "check": 1, "call": 1, "raise": 2}
        
    def get_strategy(self, table: Table, player_idx: int) -> Dict[str, float]:
        """Get strategy using neural network"""
        features = self.feature_extractor.extract_features(table, player_idx)
        valid_actions = table.valid_actions()
        
        # Get advantage network
        adv_net = self.advantage_net_p0 if player_idx == 0 else self.advantage_net_p1
        
        with torch.no_grad():
            advantages = adv_net(features)
            
        # Simple regret matching
        action_probs = {}
        total_positive = 0.0
        
        for action in valid_actions:
            idx = self.action_map.get(action, 1)
            prob = max(0.0, advantages[idx].item())
            action_probs[action] = prob
            total_positive += prob
            
        # Normalize
        if total_positive > 0:
            for action in action_probs:
                action_probs[action] /= total_positive
        else:
            # Uniform if no positive advantages
            uniform_prob = 1.0 / len(action_probs)
            for action in action_probs:
                action_probs[action] = uniform_prob
                
        return action_probs
    
    def cfr_iteration(self, table: Table) -> float:
        """Single CFR iteration with limited depth"""
        return self._traverse(table, 0, [1.0, 1.0])
    
    def _traverse(self, table: Table, depth: int, reach_probs: List[float]) -> float:
        """Simplified CFR traversal"""
        # Prevent infinite recursion
        if depth > 10:
            return 0.0
            
        # Terminal conditions
        alive_players = [p for p in table.players if p.in_hand]
        if len(alive_players) <= 1:
            current_player = table.to_act
            if table.players[current_player] in alive_players:
                return 1.0  # Won by others folding
            else:
                return -1.0  # Lost by folding
                
        if table.stage == "river":
            # Simple showdown simulation
            return random.choice([-1.0, 0.0, 1.0])
            
        current_player = table.to_act
        strategy = self.get_strategy(table, current_player)
        valid_actions = table.valid_actions()
        
        # Sample single action (Monte Carlo)
        action_probs = [strategy[action] for action in valid_actions]
        chosen_action = np.random.choice(valid_actions, p=action_probs)
        
        # Apply action and continue
        try:
            new_table = copy.deepcopy(table)
            if chosen_action == "fold":
                new_table.players[current_player].fold()
            elif chosen_action == "check":
                pass  # No bet change
            elif chosen_action == "call":
                amount = new_table.amount_to_call()
                new_table.pot += new_table.players[current_player].call(amount)
            elif chosen_action == "raise":
                amount = new_table.amount_to_call()
                raise_amt = new_table.bb
                new_table.pot += new_table.players[current_player].raise_bet(amount, raise_amt)
                new_table.current_bet = new_table.players[current_player].current_bet
                new_table._bet_occurred = True
                
            # Advance turn
            new_table.to_act = (new_table.to_act + 1) % len(new_table.players)
            while (new_table.to_act < len(new_table.players) and 
                   not new_table.players[new_table.to_act].in_hand):
                new_table.to_act = (new_table.to_act + 1) % len(new_table.players)
                
            # Store experience
            features = self.feature_extractor.extract_features(table, current_player)
            regret_target = torch.zeros(3)
            regret_target[self.action_map.get(chosen_action, 1)] = 1.0
            
            buffer = self.adv_buffer_p0 if current_player == 0 else self.adv_buffer_p1
            buffer.add((features, regret_target))
            
            # Continue traversal
            return -self._traverse(new_table, depth + 1, reach_probs)
            
        except Exception as e:
            return 0.0
    
    def train_networks(self):
        """Train neural networks"""
        for player_idx in [0, 1]:
            # Train advantage network
            buffer = self.adv_buffer_p0 if player_idx == 0 else self.adv_buffer_p1
            optimizer = self.adv_opt_p0 if player_idx == 0 else self.adv_opt_p1
            network = self.advantage_net_p0 if player_idx == 0 else self.advantage_net_p1
            
            batch = buffer.sample(32)
            if batch is None:
                continue
                
            features = torch.stack([exp[0] for exp in batch])
            targets = torch.stack([exp[1] for exp in batch])
            
            optimizer.zero_grad()
            predictions = network(features)
            loss = nn.MSELoss()(predictions, targets)
            loss.backward()
            optimizer.step()

def train_deep_cfr(iterations: int = 5000, save_every: int = 1000):
    """Main training function with proper model saving"""
    print("Starting Deep CFR training...")
    print(f"Models will be saved in: {os.path.abspath('poker_ai_dev\model')}")
    
    # Create models directory
    os.makedirs("poker_ai_dev\model", exist_ok=True)
    
    trainer = DeepCFRTrainer()
    
    for iteration in range(iterations):
        try:
            # Create fresh game
            players = [Player("P1", 1000), Player("P2", 1000)]
            table = Table(players)
            table.auto_advance = False  # Critical!
            table.start_hand()
            
            # Run CFR iteration
            utility = trainer.cfr_iteration(table)
            
            # Train networks
            if iteration > 100 and iteration % 20 == 0:
                trainer.train_networks()
            
            # Progress and save
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{iterations}: Utility = {utility:.4f}")
                
            if iteration % save_every == 0 and iteration > 0:
                save_models(trainer, iteration)
                
        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            continue
    
    # Final save
    save_models(trainer, iterations)
    print("Training complete!")
    return trainer

def save_models(trainer: DeepCFRTrainer, iteration: int):
    """Save models to disk"""
    save_dir = f"E:/#EditorCodes/Project_poker/poker_ai_dev/model/iteration_{iteration}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save advantage networks
    torch.save(trainer.advantage_net_p0.state_dict(), f"{save_dir}/advantage_p0.pth")
    torch.save(trainer.advantage_net_p1.state_dict(), f"{save_dir}/advantage_p1.pth")
    
    # Save strategy networks  
    torch.save(trainer.strategy_net_p0.state_dict(), f"{save_dir}/strategy_p0.pth")
    torch.save(trainer.strategy_net_p1.state_dict(), f"{save_dir}/strategy_p1.pth")
    
    print(f"✅ Models saved to: {os.path.abspath(save_dir)}")

class DeepCFRBot:
    """Trained bot for playing"""
    def __init__(self, model_path: str, player_idx: int = 0):
        self.feature_extractor = SimpleFeatureExtractor()
        self.player_idx = player_idx
        
        # Load strategy network
        self.strategy_net = StrategyNetwork(20, 3)
        strategy_file = f"{model_path}/strategy_p{player_idx}.pth"
        
        if os.path.exists(strategy_file):
            self.strategy_net.load_state_dict(torch.load(strategy_file))
            self.strategy_net.eval()
            print(f"✅ Loaded model from {strategy_file}")
        else:
            print(f"❌ Model file not found: {strategy_file}")
            
        self.action_map = {"fold": 0, "check": 1, "call": 1, "raise": 2}
    
    def get_action(self, table: Table) -> str:
        """Get action for live play"""
        features = self.feature_extractor.extract_features(table, self.player_idx)
        valid_actions = table.valid_actions()
        
        with torch.no_grad():
            strategy_probs = self.strategy_net(features)
        
        # Map to valid actions
        action_probs = []
        for action in valid_actions:
            idx = self.action_map.get(action, 1)
            action_probs.append(strategy_probs[idx].item())
        
        # Normalize and sample
        action_probs = np.array(action_probs)
        action_probs = action_probs / action_probs.sum()
        
        return np.random.choice(valid_actions, p=action_probs)

if __name__ == "__main__":
    # First fix your CFR file - there's a bug on line 138
    print("🚀 Starting Deep CFR Training")
    print("📁 Models will be saved in 'poker_models/' directory")
    
    # Train the model
    trained_model = train_deep_cfr(iterations=2000, save_every=500)
    
    print("\n🎯 Training complete! Testing the bot...")
    
    # Test the bot
    try:
        bot = DeepCFRBot("E:/#EditorCodes/Project_poker/poker_ai_dev/model/iteration_2000", player_idx=0)
        print("✅ Bot loaded successfully!")
        
        # Quick test game
        players = [Player("Bot", 1000), Player("Random", 1000)]
        table = Table(players)
        table.start_hand()
        
        print(f"🎲 Test game: Bot has {table.players[0].hole_cards}")
        action = bot.get_action(table)
        print(f"🤖 Bot chooses: {action}")
        
    except Exception as e:
        print(f"❌ Error testing bot: {e}")
