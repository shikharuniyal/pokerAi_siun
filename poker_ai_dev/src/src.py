import random, secrets
from typing import List, Tuple
import sys
import itertools
from collections import Counter


# CARD prop
ranks = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
suits = ['Hearts','Diamonds','Clubs','Spades']

# mapping cards to int
_RANK_MAP = {r: i for i, r in enumerate(
    ['2','3','4','5','6','7','8','9','10','J','Q','K','A'], start=2)}

def _score_5(cards):
        
        ranks = sorted((_RANK_MAP[c.rank] for c in cards), reverse=True)
        suits = [c.suit for c in cards]
        cnt = Counter(ranks)
        counts = cnt.most_common()  #[('14(as A)', 3), ('5', 2), ('2', 2)...]
        counts.sort(key=lambda x: (x[1], x[0]), reverse=True)  # x[1], x[0] means count and then rank for count tie breaker
        group_ranks = [r for r,_ in counts]
        group_sizes = [s for _,s in counts]

        # flush check :

        is_flush = len(set(suits)) == 1

        #Check straight (including wheel)
        unique = sorted(set(ranks), reverse=True)
        straight_high = 0
        # 1) Look for any run of five: from each candidate high down to high == 4
        for r in unique:
            # e.g. if r==10, we check {10,9,8,7,6}
            needed = {r, r-1, r-2, r-3, r-4}
            if needed.issubset(unique):
                straight_high = r
                break

        #wheel straight A2345
        if straight_high == 0 and {14, 2, 3, 4, 5}.issubset(unique):
            straight_high = 5

        is_straight = (straight_high > 0)   

        if is_straight and is_flush:  #straight flush
            return (9, straight_high)
        if group_sizes[0] == 4:  # 4 of a kind
            kicker = next(r for r in ranks if r != group_ranks[0])
            return (8, group_ranks[0], kicker)
        if group_sizes[0] == 3 and group_sizes[1] >= 2:
            return (7, group_ranks[0], group_ranks[1])
        if is_flush:
            return (6, *ranks)
        if is_straight:
            return (5, straight_high)
        if group_sizes[0] == 3:
            kickers = [r for r in ranks if r != group_ranks[0]][:2]
            return (4, group_ranks[0], *kickers)
        if group_sizes[0] == 2 and group_sizes[1] == 2:
            kicker = next(r for r in ranks if r not in group_ranks[:2])
            return (3, group_ranks[0], group_ranks[1], kicker)
        if group_sizes[0] == 2:
            kickers = [r for r in ranks if r != group_ranks[0]][:3]
            return (2, group_ranks[0], *kickers)
        return (1, *ranks)


def evaluate_best_hand(seven_cards):
        best = (0,)
        for combo in itertools.combinations(seven_cards, 5):
            sc = _score_5(combo)
            if sc > best:
                best = sc
        return best

class Card:
    def __init__(self, rank: str, suit: str):
        self.rank, self.suit = rank, suit
    def __repr__(self):
        return f"{self.rank} of {self.suit}"

class Cards:
    @staticmethod
    def build_deck() -> List[Card]:
        return [Card(r, s) for s in suits for r in ranks]

    @staticmethod
    def shuffle(deck: List[Card]) -> None:
        for i in range(len(deck)-1, 0, -1):
            j = secrets.randbelow(i+1)
            deck[i], deck[j] = deck[j], deck[i]

    @staticmethod
    def deal_hole(deck: List[Card], n_players: int, hole_cards: int = 2
                 ) -> Tuple[Tuple[Card,...], ...]:
        hands = [[] for _ in range(n_players)]
        for _ in range(hole_cards):
            for p in range(n_players):
                hands[p].append(deck.pop(0))
        return tuple(tuple(h) for h in hands)

    @staticmethod
    def deal_community(deck: List[Card]) -> Tuple[Card,...]:
        deck.pop(0)
        flop = [deck.pop(0) for _ in range(3)]
        deck.pop(0)
        turn = deck.pop(0)
        deck.pop(0)
        river = deck.pop(0)
        return tuple(flop + [turn] + [river])
    

# PLayer Properties
class Player:
    def __init__(self, name: str, chips: int):
        self.name = name
        self.chips = chips
        self.hole_cards: Tuple[Card,...] = ()
        self.current_bet: int = 0 #current street
        self.total_committed: int = 0 
        self.in_hand: bool = True #to check for folded
        self.action_taken: str = None

    def reset_for_new_hand(self):
        self.current_bet = 0
        self.total_committed = 0
        self.in_hand = True
        self.hole_cards = ()

    def receive_holes(self, cards: Tuple[Card,...]):
        self.hole_cards = cards

    def bet(self, amt: int) -> int:
        amt = min(amt, self.chips)
        self.chips -= amt
        self.current_bet += amt
        self.total_committed += amt
        return amt

    def call(self, to_call: int) -> int:
        return self.bet(to_call)

    def raise_bet(self, to_call: int, raise_by: int) -> int:
        self.call(to_call)
        return self.bet(raise_by)

    def fold(self):
        self.in_hand = False

# ────── TABLE ──────
class Table:
    STAGES = ("pre-flop", "flop", "turn", "river", "showdown")
    def __init__(self, players: List[Player], sb: int = 5, bb: int = 10):
        self.players = players
        self.N = len(players)
        self.sb, self.bb = sb, bb
        self._bet_occurred = False
        self.auto_advance = False  #to toggle auto advance feature on/off

        self.deck: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.stage = None
        self.dealer = -1
        self.to_act = 0
        self.comm_buffer: Tuple[Card,...] = ()
        self.community: Tuple[Card,...] = ()

        self.last_raiser: int = None


    def start_hand(self):
        for p in self.players:
            p.reset_for_new_hand()
        self.pot = 0
        self.current_bet = 0
        self.stage = "pre-flop"
        self.community = ()
        self.comm_buffer = ()
        self._bet_occurred = False

        self.last_raiser = None

        #shuffling
        self.deck = Cards.build_deck()
        Cards.shuffle(self.deck)

        #dealer change
        self.dealer = (self.dealer + 1) % self.N
        sb_pos = (self.dealer + 1) % self.N
        bb_pos = (self.dealer + 2) % self.N

        #blinds small and big respectivesly
        self.pot += self.players[sb_pos].bet(self.sb)
        self.players[sb_pos].current_bet = self.sb
        self.players[sb_pos].action_taken = "raise"
        
        self.pot += self.players[bb_pos].bet(self.bb)
        self.players[bb_pos].current_bet = self.bb
        self.players[bb_pos].action_taken = "raise"

        self.current_bet = self.bb
        self.last_raiser = bb_pos

        # dealing hole cards to after the deck is processed and the players are in the game
        holes = Cards.deal_hole(self.deck, self.N, 2)
        for p, h in zip(self.players, holes):
            p.receive_holes(h)

        #to store community cards
        self.comm_buffer = Cards.deal_community(self.deck)

        if self.N == 2:#special case if only two players
            self.to_act = (self.dealer + 1) % self.N
        else:
            self.to_act = (self.dealer + 3) % self.N    

    def _reset_betting_round(self, start_after: int):
        for p in self.players:
            p.current_bet = 0
            p.action_taken = ""
        self.current_bet = 0
        self.to_act = ((self.to_act + 1) % self.N) 
        self._bet_occurred = False
        self.last_raiser = None

        self.to_act = (self.dealer + 1) % self.N
        while not self.players[self.to_act].in_hand or self.players[self.to_act].chips == 0:
            self.to_act = (self.to_act + 1) % self.N

    def amount_to_call(self) -> int:
        p = self.players[self.to_act]
        return max(0, self.current_bet - p.current_bet)

    def valid_actions(self) -> List[str]:
        to_call = self.amount_to_call()
        if to_call == 0:
            return ["fold", "check", "raise"]
        else:
            return ["fold", "call", "raise"]
        
    def is_betting_round_over(self,table):
        still = [p for p in table.players if p.in_hand]
        no_bets_and_checked = (
            not table._bet_occurred and
            all(p.action_taken == "check" for p in still)
        )
        bets_matched = (
            table._bet_occurred and
            all(
            (p.current_bet == table.current_bet) or
            (p.chips == 0) or
            (not p.in_hand)
            for p in table.players
            )
        )
        return no_bets_and_checked or bets_matched

    def perform_action(self, action: str, raise_by: int = 0):
        p = self.players[self.to_act]
        to_call = self.amount_to_call()
        
        if not p.in_hand:
            return
        
        if action == "fold":
            p.fold()                    
        elif action == "check":
            if to_call != 0:
                raise ValueError("Cannot check when there's a bet")
        elif action == "call":
            self.pot += p.call(to_call)
            self._bet_occurred = True
        elif action == "raise":
            self.pot += p.raise_bet(to_call, raise_by)
            self.current_bet = p.current_bet
            self._bet_occurred = True
            self.last_raiser = self.to_act
        else:
            raise ValueError("Unknown action")
        
        p.action_taken = action

        self._next_to_act()

        ## AUTO ADVANCE FEATURE check
        if not self.auto_advance:
            return
        #______________________________________________

        active = [q for q in self.players if q.in_hand and q.chips > 0]

        if len(active) == 1:
            winner = active[0]
            print(f"\nAll others folded. {winner.name} wins {self.pot} chips.")
            winner.chips += self.pot
            self.start_hand()#starts new hand when only one player is left
            return
        
        still_active = [q for q in self.players if q.in_hand]

        #to check if everyone checked
        no_bets_and_checked = (
            not self._bet_occurred and
            all(p.action_taken == "check" for p in still_active)
        )

        #to check if everyone has better for the new round
        #and everyone has matched the bet amt
        bets_matched = (
            self._bet_occurred
            and all(
                (q.current_bet == self.current_bet) or (q.chips == 0) or (not q.in_hand)
                for q in self.players
            )
        )

        if no_bets_and_checked or bets_matched:
            if self.stage == "pre-flop":
                self.deal_flop()
            elif self.stage == "flop":
                self.deal_turn()
            elif self.stage == "turn":
                self.deal_river()
            elif self.stage == "river":
                self.stage = "showdown"
                self._settle_showdown()
                if any(pl.chips < self.bb or pl.chips <= 0 for pl in self.players):
                    print("Game over: a player has insufficient chips to continue.")
                    return
                self.start_hand()
                return
                
        

        
    def _next_to_act(self):
        nxt = (self.to_act + 1) % self.N
        while True:
            playernxt = self.players[nxt]
            if playernxt.in_hand and playernxt.chips>0:#checking if the player is ALL IN or not to play
                self.to_act = nxt
                return
            nxt = (nxt + 1) % self.N
            
    def HandStrength(self, p: Player) -> Tuple:
        if not p.in_hand or len(p.hole_cards) != 2 or len(self.community) < 5:
            return (0,)#to return empty tup/least hand for  folded player or incomplete game
        all_cards = list(p.hole_cards) + list(self.community)
        return evaluate_best_hand(all_cards)
    
    def deal_flop(self):
        
        if any(pl.in_hand and pl.current_bet != self.current_bet for pl in self.players):
            raise RuntimeError("Bets not equalized")
        self.stage = "flop"
        self.community = self.comm_buffer[:3]
        self._reset_betting_round(start_after=self.dealer)

    def deal_turn(self):
        if self.stage != "flop":
            raise RuntimeError("Must deal flop first")
        if any(pl.in_hand and pl.current_bet != self.current_bet for pl in self.players):
            raise RuntimeError("Bets not equalized")
        self.stage = "turn"
        self.community += (self.comm_buffer[3],)
        self._reset_betting_round(start_after=self.dealer)

    def deal_river(self):
        if self.stage != "turn":
            raise RuntimeError("Must deal turn first")
        if any(pl.in_hand and pl.current_bet != self.current_bet for pl in self.players):
            raise RuntimeError("Bets not equalized")
        self.stage = "river"
        self.community += (self.comm_buffer[4],)
        self._reset_betting_round(start_after=self.dealer)

    
    def _settle_showdown(self):
        
        # IT splits the pot for side pots for all in's for player's total contributed to th epot
        
        commits = {pl: pl.total_committed for pl in self.players if pl.total_committed > 0}
        if not commits:
            return  # no one committed anything

        # Sort unique commitment levels
        unique_levels = sorted(set(commits.values()))
        prev_level = 0
        # its a list of (pot_size, [eligible_players_for_this_pot])
        side_pots: List[Tuple[int, List[Player]]] = []

        for lvl in unique_levels:
            chunk = lvl - prev_level
            #involved => everyone who contributed at least lvl
            involved = [pl for pl, amt in commits.items() if amt >= lvl]
            pot_amount = chunk * len(involved)
            side_pots.append((pot_amount, involved.copy()))
            prev_level = lvl

        #For each side‐pot, determine which players are still "in hand" (not folded)
        #Award that side‐pot to the best such hand.
        for pot_amount, involved in side_pots:
            eligible = [pl for pl in involved if pl.in_hand]
            if not eligible:
                continue
           
            best_score = max((self.HandStrength(pl) for pl in eligible))#get best hand strength
            winners = [pl for pl in eligible if self.HandStrength(pl) == best_score]
            share = pot_amount // len(winners)
            for w in winners:
                w.chips += share
            winners_names = ", ".join(w.name for w in winners)
            print(f"Showdown: pot of {pot_amount} split among {winners_names} (each gets {share}).")


# POker helper class
class Poker:
    def __init__(self, players: List[Player], sb=5, bb=10):
        self.table = Table(players, sb, bb)

    def start_hand(self): self.table.start_hand()
    def flop(self):       self.table.deal_flop()
    def turn(self):       self.table.deal_turn()
    def river(self):      self.table.deal_river()
    def actions(self):    return self.table.valid_actions()
    def to_call(self):    return self.table.amount_to_call()
    def do(self, act, amt=0): self.table.perform_action(act, amt)

    @property
    def community(self): return self.table.community
    @property
    def pot(self):       return self.table.pot
    @property
    def players_state(self):
        return [
            (p.name, p.chips, p.current_bet, p.total_committed, p.in_hand, p.hole_cards)
            for p in self.table.players
        ]
