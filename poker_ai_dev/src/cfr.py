import random
import itertools
from collections import Counter
from src import evaluate_best_hand,Card
#from src import _score_5,evaluate_best_hand
#from src import Table as table
import sys


import copy

# A global dictionary to hold all encountered InfoSets:
INFOSETS = {}  # key (string) → InfoSet object

_RANK_TO_INT = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "10": 10,
        "J": 11, "Q": 12, "K": 13, "A": 14
    }
_INT_TO_RANK = {v: k for k, v in _RANK_TO_INT.items()}

PREFLOP_EQ_TABLE = {}

_crand = random.SystemRandom()

def monte_carlo_equity(table, hole_cards, community_cards, trials):
    wins = 0
    ties = 0

    # 1) Build the "remaining deck" by removing Hero's and board cards from table.deck
    known = set(hole_cards) | set(community_cards)
    # We do NOT modify table.deck in place (so we copy it)
    full_deck = [card for card in table.deck if card not in known]
    

    # Decide how often to update (every N steps)
    update_every = 1000

    # Length of the bar itself (number of characters)
    bar_length = 30


    print("calculating equity")
    for i in range(trials):
        # ─── 2A) Sample 2 random cards for Villain from full_deck ───
        villian_hole = _crand.sample(full_deck, 2)

        # ─── 2B) Remove villain_hole from a fresh copy of full_deck ───
        # We need a new "deck minus villain" for dealing remaining board cards.
        remaining = [c for c in full_deck if c not in villian_hole]

        # ─── 2C) "Complete" the board if fewer than 5 cards are known ───
        needed = 5 - len(community_cards)
        if needed > 0:
            board_extra =  _crand.sample(remaining, needed)
            final_board = list(community_cards) + board_extra
        else:
            final_board = list(community_cards)

        # ─── 2D) Build Hero's full 7 cards and Villain's full 7 cards ───
        hero_7 = list(hole_cards) + final_board
        villain_7 = list(villian_hole) + final_board

        # ─── 2E) Evaluate both hands using your engine's evaluator ───
        hero_score = evaluate_best_hand(hero_7)
        villain_score = evaluate_best_hand(villain_7)

        # ─── 2F) Increment wins/ties appropriately ───
        if hero_score > villain_score:
            wins += 1
        elif hero_score == villain_score:
            ties += 1
        # else: villain wins → do nothing

        ##_____PROGRESS BAR_____
        if (i % update_every == 0) or (i == trials - 1):
            # percent complete (use i+1 so that at the final iteration it shows 100%)
            percent = 100 * (i + 1) / trials

            # how many “filled” characters in the bar?
            filled_len = int(bar_length * (i + 1) // trials)
            bar = "█" * filled_len + "-" * (bar_length - filled_len)

            # \r returns cursor to the start of the line so we overwrite it
            sys.stdout.write(f"\r|{bar}| {percent:6.2f}%")
            sys.stdout.flush()

    print()

    # ─── 3) Final equity estimate ───
    return ((wins + 0.5 * ties) / trials)


class Normalization:
    


    #This function does **not** look at the board or community cards—its only job
    #is to canonicalize any two hole cards into the standard “preflop bucket”
    def normalize_hand_suits(self,hole_cards):
        if len(hole_cards) != 2:
            raise ValueError("normalize_hand_suits: expected exactly two hole cards, got "f"{len(hole_cards)} cards")
        

        c1, c2 = hole_cards
        r1 = _RANK_TO_INT[c1.rank]
        r2 = _RANK_TO_INT[c2.rank]

        if r1 > r2:
            rank_high, rank_low = r1, r2
        elif r2 > r1:
            rank_high, rank_low = r2, r1
        else:
            rank_high = rank_low = r1

        # 5) Determine if the two cards are suited
        suited_flag = (c1.suit == c2.suit)

        # 6) Return the canonical 3‐tuple
        return (rank_high, rank_low, suited_flag)
    
    

    def get_preflop_equity(self,table, hole_cards, trials=2000):
        # 1) Canonicalize to one of 169 keys
        key = self.normalize_hand_suits(hole_cards)

        if key in PREFLOP_EQ_TABLE:
            return PREFLOP_EQ_TABLE[key]
        
        equity = monte_carlo_equity(table, hole_cards, community_cards=(), trials=trials)

        # 4) Store in cache and return
        PREFLOP_EQ_TABLE[key] = equity
        return equity
    
    def build_full_preflop_table(table, trials=2000):
        #Enumerate all 169 (rank_high,rank_low,suited) combinations, run Monte Carlo
        #once for each

        ranks_int = list(range(2, 15))
        for r_high in ranks_int[::-1]:
            for r_low in ranks_int[::-1]:
                if r_low > r_high:
                    continue


        # Case 1: Pair (r_high==r_low)
            if r_high == r_low:
                key = (r_high, r_low, False)
                if key not in PREFLOP_EQ_TABLE:
                    # Create any two cards of that rank; suit choice doesn’t matter for a pair
                    c1 = Card(_INT_TO_RANK[r_high], "Spades")
                    c2 = Card(_INT_TO_RANK[r_low],  "Hearts")
                    eq = monte_carlo_equity(table, (c1, c2), community_cards=(), trials=trials)
                    PREFLOP_EQ_TABLE[key] = eq

        # Case 2: Suited (r_high > r_low, suited=True)
            else:
                key_s = (r_high, r_low, True)
                if key_s not in PREFLOP_EQ_TABLE:
                    c1 = Card(_INT_TO_RANK[r_high], "Spades")
                    c2 = Card(_INT_TO_RANK[r_low],  "Spades")
                    eq_s = monte_carlo_equity(table, (c1, c2), community_cards=(), trials=trials)
                    PREFLOP_EQ_TABLE[key_s] = eq_s


                key_o = (r_high, r_low, False)
                if key_o not in PREFLOP_EQ_TABLE:
                    # Offsuit: pick two different suits arbitrarily
                    c1 = Card(_INT_TO_RANK[r_high], "Spades")
                    c2 = Card(_INT_TO_RANK[r_low],  "Hearts")
                    eq_o = monte_carlo_equity(table, (c1, c2), community_cards=(), trials=trials)
                    PREFLOP_EQ_TABLE[key_o] = eq_o

        print("Built PREFLOP_EQ_TABLE: total entries =", len(PREFLOP_EQ_TABLE))


    def postflop_equity_vs_random(table, hole_cards, community_cards, trials=2000):
        """
        Approximate the equity of `hole_cards` vs. a
        single random opponent, **given** the existing community_cards
        (which must be length 3 or 4).

        Returns a float in [0,1].

        - If len(community_cards)=3 (flop), we sample Villain’s 2 hole cards
        from "deck minus (hole_cards ∪ community_cards)" and finish with random
        turn+river.
        - If len(community_cards)=4 (turn), we sample Villain’s 2 hole cards
        and finish with random river.
        - If len(community_cards)=5 (river), we simply do one showdown (no randomness).
        """
        n_board = len(community_cards)
        if n_board > 5 or n_board < 3:
            raise ValueError("postflop_equity_vs_random requires 3≤len(board)≤5")

        wins = 0
        ties = 0

        # Build set of “known” cards: Hero’s hole + current board
        known = set(hole_cards) | set(community_cards)

        for _ in range(trials):
            # 1) Build “deck minus known”
            full_deck = [c for c in table.deck if c not in known]

            # 2) Sample Villain’s hole from that
            villain_hole = random.sample(full_deck, 2)

            # 3) Build a fresh “remaining deck” after giving villain hole
            rem = [c for c in full_deck if c not in villain_hole]

            # 4) Complete the board if needed:
            if n_board == 3:
                # on the flop: deal 2 more cards (turn+river)
                extra = random.sample(rem, 2)
                final_board = list(community_cards) + extra       # 5 cards
            elif n_board == 4:
                # on the turn: deal 1 more card (river)
                extra = random.sample(rem, 1)
                final_board = list(community_cards) + extra       # 5 cards
            else:
                # river: board is already length‐5
                final_board = list(community_cards)

            # 5) Build 7‐card hands
            hero_7    = list(hole_cards)    + final_board
            villain_7 = list(villain_hole)  + final_board

            # 6) Evaluate both
            h_score = evaluate_best_hand(hero_7)
            v_score = evaluate_best_hand(villain_7)

            if h_score > v_score:
                wins += 1
            elif h_score == v_score:
                ties += 1
            # else villain wins → nothing to add

        return (wins + 0.5 * ties) / trials


    def get_preflop_bucket(self,hole_cards, num_buckets=20):
        #Requires that PREFLOP_EQ_TABLE is already populated for this shape
        key = self.normalize_hand_suits(hole_cards)

        if key not in PREFLOP_EQ_TABLE:
            raise KeyError(
                f"Missing preflop equity for shape {key}. "
                "Did you call get_preflop_equity(...) or build_full_preflop_table(...) first?"
            )
        
        eq = PREFLOP_EQ_TABLE[key]  #gives float value in [0.0, 1.0]

        #Multiply by num_buckets and floor to get an integer index
        idx = int(eq * num_buckets)

        if idx >= num_buckets:
            idx = num_buckets - 1
        if idx < 0:
            idx = 0
    
        return idx
    

class InformationNode:
    def __init__(self,Norm : Normalization):
        self.norm = Norm
    def create_infoset_key(self,table, player_idx, num_buckets=20, postflop_trials=1000):
        player = table.players[player_idx]
        hole = player.hole_cards       # tuple of 2 Card objects
        board = list(table.community)  # list of 0..5 Card objects
        stage = table.stage            # one of "pre-flop", "flop", "turn", "river"

        if stage == "pre-flop":
            hole_bucket = self.norm.get_preflop_bucket(hole, num_buckets)

        elif stage in ("flop", "turn"):
            # Postflop: bucket by equity given board
            hole_bucket = self.norm.get_postflop_bucket(table, hole, board,
                                                num_buckets=num_buckets,
                                                trials=postflop_trials)
            
        else:
            hole_bucket = self.norm.get_postflop_bucket(table, hole, board,
                                                        num_buckets=num_buckets,
                                                        trials=postflop_trials // 2)
            

        #Build betting‐history string

        history = ""
        for p in table.players:
            if p.action_taken:
                history += p.action_taken[0]
        
        to_call = table.amount_to_call()
        if(to_call <0):
            to_call = 0

        #creating a return keyed structure
        key = f"stage={stage}|hb={hole_bucket}|hist={history}|tocall={to_call}"
        return key
    
class InfoSet:
    #it is a container of the node properties 
    def __init__(self,key,legal_actions):
        self.key = key
        self.legal_actions = legal_actions[:]#safed the legal actions in the local class leagal_actions(self)
        #setting up strategy and regret sum values for each node container
        self.regret_sum = {action: 0.0 for action in legal_actions}
        self.strategy_sum = {action: 0.0 for action in legal_actions}

    def get_strategy(self,realization_weight):
        """
        Computes a mixed strategy from current regret sums via regret‐matching,
        then accumulates (strategy × realization_weight) into strategy_sum.

        Input:
          - realization_weight: probability weight reaching this node for THIS player
        Returns:
          - strategy (dict action→probability)
        """
        strategy = {}
        Z = 0.0

        # 1) For each action: 
        #    if regret_sum[action] > 0 → strategy[action] = regret_sum[action];
        #    else strategy[action] = 0.0
        for action in self.legal_actions:
            r = self.regret_sum[action]
            strategy[action] = r if r > 0 else 0.0
            Z += strategy[action]

        # 2) If all regrets ≤ 0, use uniform strategy
        if Z <= 0:
            for action in self.legal_actions:
                strategy[action] = 1.0 / len(self.legal_actions)
        else:
            # Normalize positive regrets to probabilities
            for action in self.legal_actions:
                strategy[action] /= Z

        # 3) Accumulate into strategy_sum for average later
        for action in self.legal_actions:
            self.strategy_sum[action] += realization_weight * strategy[action]

        return strategy

    def get_average_strategy(self):
        """
        After many CFR iterations, the average strategy at this infoset is:
          avg_strat[action] = strategy_sum[action] / (sum of all strategy_sum entries)

        Returns:
          - avg_strategy (dict action→float)
        """
        avg_strategy = {}
        total = sum(self.strategy_sum.values())
        if total > 0:
            for action in self.legal_actions:
                avg_strategy[action] = self.strategy_sum[action] / total
        else:
            # If we never visited this infoset, return uniform
            for action in self.legal_actions:
                avg_strategy[action] = 1.0 / len(self.legal_actions)
        return avg_strategy
    



class CFRGameState:
    """
    A thin wrapper around your Table instance to allow safe copying
    for recursion. Each CFR node gets its own CFRGameState, which
    holds a deep copy of the entire Table and players.

    We assume:
      - table.to_act tells us whose turn it is (0 or 1)
      - table.stage in {"pre-flop", "flop", "turn", "river", "showdown"}
      - table._settle_showdown() will split the pot (but in CFR, we want
        payoff without modifying the original, so we’ll handle payoff manually)
    """

    def __init__(self, table):
        # Make a deep copy of the entire Table (including Player states)
        self.table = copy.deepcopy(table)

    def is_terminal(self):
        """
        Returns True if this is a terminal node (someone won, or showdown).
        Two cases:
         1) Only one player remains in_hand (everyone else folded).
         2) stage == "river" and the betting round is over → showdown.
        """
        alive = [p for p in self.table.players if p.in_hand]
        if len(alive) == 1:
            return True  # folded-to-one‐player
        if self.table.stage == "river":
            # If no more actions possible on river, it is showdown.
            # We can check if betting round is over:
            #   a) Everyone who can match has matched, or
            #   b) Everyone checked.
            # Reuse your existing is_betting_round_over logic if you wrote it.
            # For simplicity, we assume the CFR driver only calls is_terminal()
            # once the betting round on river is complete.
            return True
        return False

    def get_payoff(self, player_idx):
        """
        Returns the terminal *utility* for player_idx (zero‐sum, normalized to +1/–1).
        Two cases:
          1) One player folded: that player wins the pot. We return +1 for the winner,
             −1 for loser.
          2) River showdown: compute best 7‐card hands for both, compare:
             If hero > villain → +1, equal → 0, hero < villain → −1.

        For simplicity, we ignore the actual pot size and just give +1 or −1.
        """
        # 1) Fold‐to‐one‐player
        alive = [p for p in self.table.players if p.in_hand]
        if len(alive) == 1:
            winner = alive[0]
            if self.table.players[player_idx] == winner:
                return +1.0
            else:
                return -1.0

        # 2) Showdown (river)
        # Build each player’s 7‐card list:
        final_board = list(self.table.community)  # length == 5 on river
        hands = []
        for p in self.table.players:
            hole = list(p.hole_cards)
            full7 = hole + final_board
            score = evaluate_best_hand(full7)
            hands.append(score)

        # Compare
        if hands[player_idx] > hands[1 - player_idx]:
            return +1.0
        elif hands[player_idx] == hands[1 - player_idx]:
            return 0.0
        else:
            return -1.0
        

def cfr_ex(game_state, reach_probs):
    """
    Recursively runs one iteration of CFR on the given game_state.

    Args:
      - game_state  : an instance of CFRGameState (holds a copy of Table)
      - reach_probs : a tuple (p0, p1) of reach probabilities for players 0 and 1
                      i.e. probability under the current strategy that we arrived here
                      for each player.

    Returns:
      - utility to the player who is about to act at this node (float)
        (in a two-player zero-sum game, if it’s player i's turn, cfr() returns
         +u_i; and for the opponent it’s the negation in the recursion).
    """
    table = game_state.table
    player_idx = table.to_act  # whose turn it is: 0 or 1

    # 1) Terminal check
    if game_state.is_terminal():
        return game_state.get_payoff(player_idx)

    # 2) Build the infoset key string
    key = InformationNode.create_infoset_key(table, player_idx)
    # 3) Determine legal actions at this node
    legal = table.valid_actions()  # e.g. ["fold","call","raise"] or ["fold","check","raise"]

    # 4) If this is the first time we see key, create a new InfoSet
    if key not in INFOSETS:
        INFOSETS[key] = InfoSet(key, legal)
    infoset = INFOSETS[key]

    # 5) Get this player's current strategy at this infoset (regret-matching)
    p_reach = reach_probs[player_idx]
    strategy = infoset.get_strategy(p_reach)  # action → probability

    # 6) For each action a:
    node_util = 0.0
    action_utils = {}  # store utility for each action

    for a in legal:
        # 6A) Make a deep copy of the entire state
        next_state = CFRGameState(table)  # deep-copies table and players
        next_state.table.auto_advance = False  # ensure no auto-advance inside perform_action

        # 6B) Apply the chosen action a (with cfr_mode=True so no auto-advance)
        #     We assume perform_action(action, amount=0) modifies next_state.table
        #     NOTE: if 'raise' needs an amount, you might supply a default raise size for abstraction.
        if a == "fold":
            next_state.table.perform_action("fold")
        elif a == "check":
            next_state.table.perform_action("check")
        elif a == "call":
            next_state.table.perform_action("call")
        elif a == "raise":
            # In a full solver, we'd iterate over possible raise sizes.
            # For simplicity, assume a fixed “half‐pot” raise or a single abstract raise
            next_state.table.perform_action("raise", raise_by=  table.bb )
        else:
            raise ValueError(f"Unknown action '{a}' in CFR")

        # 6C) **Advance streets if the betting round ended** (since we disabled auto_advance):
        #      We need a helper is_betting_round_over(table):
        if table.is_betting_round_over(next_state.table):
            if next_state.table.stage == "pre-flop":
                next_state.table.deal_flop()
            elif next_state.table.stage == "flop":
                next_state.table.deal_turn()
            elif next_state.table.stage == "turn":
                next_state.table.deal_river()
            elif next_state.table.stage == "river":
                # We stay at river; is_terminal() will pick it up next call
                pass

        # 6D) Recursively call cfr() for the resulting node:
        #      - Update reach probabilities: if it's player_idx's turn, multiply that player's reach_prob by strategy[a].
        new_reach = list(reach_probs)
        new_reach[player_idx] *= strategy[a]

        # 6E) Recurse and get utility (for the **next** to‐act player)
        util_next = cfr_ex(next_state, tuple(new_reach))

        # 6F) Since zero-sum and we're passing utility to current player,
        #      utility to current player = -util_next (because roles switch)
        action_utils[a] = -util_next
        node_util += strategy[a] * action_utils[a]

    # 7) Update regret sums for each action
    #    Regret for a = action_utils[a] - node_util
    for a in legal:
        regret = action_utils[a] - node_util
        opp_idx = 1 - player_idx
        # Counterfactual reach prob for opponent = reach_probs[opp_idx]
        # Update regret_sum[a] by opponent’s reach prob * regret
        infoset.regret_sum[a] += reach_probs[opp_idx] * regret

    return node_util