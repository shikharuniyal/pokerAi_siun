import random, secrets
from typing import List, Tuple
import sys
# ────── CARD UTILS ──────
ranks = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
suits = ['Hearts','Diamonds','Clubs','Spades']

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
        # cryptographically secure shuffle
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
        # burn, flop(3), burn, turn, burn, river
        deck.pop(0)
        flop = [deck.pop(0) for _ in range(3)]
        deck.pop(0)
        turn = deck.pop(0)
        deck.pop(0)
        river = deck.pop(0)
        return tuple(flop + [turn] + [river])

# ────── PLAYER ──────
class Player:
    def __init__(self, name: str, chips: int):
        self.name = name
        self.chips = chips
        self.hole_cards: Tuple[Card,...] = ()
        self.current_bet: int = 0       # this street
        self.total_committed: int = 0   # this hand
        self.in_hand: bool = True       # folded?
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

        # runtime
        self.deck: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.stage = None
        self.dealer = -1
        self.to_act = 0
        self.comm_buffer: Tuple[Card,...] = ()
        self.community: Tuple[Card,...] = ()

    def start_hand(self):
        # reset players & table
        for p in self.players: p.reset_for_new_hand()
        self.pot = 0
        self.current_bet = 0
        self.stage = "pre-flop"

        # shuffle
        self.deck = Cards.build_deck()
        Cards.shuffle(self.deck)
        self._bet_occurred = False

        # rotate dealer & post blinds (rule 3)
        self.dealer = (self.dealer + 1) % self.N
        sb_pos = (self.dealer + 1) % self.N
        bb_pos = (self.dealer + 2) % self.N
        self.pot += self.players[sb_pos].bet(self.sb)
        self.pot += self.players[bb_pos].bet(self.bb)
        self.players[sb_pos].current_bet = self.sb
        self.players[bb_pos].current_bet = self.bb
        self.current_bet = self.bb
        self.players[sb_pos].action_taken = "raise"
        self.players[bb_pos].action_taken = "raise"

        # deal holes
        holes = Cards.deal_hole(self.deck, self.N, 2)
        for p, h in zip(self.players, holes):
            p.receive_holes(h)

        # prepare community buffer
        self.comm_buffer = Cards.deal_community(self.deck)

        # If 2 players, heads‑up: SB (dealer+1) acts first pre‑flop
        if self.N == 2:
            self.to_act = (self.dealer + 1) % self.N
        else:
            # UTG (first left of big blind)
            self.to_act = (self.dealer + 3) % self.N    

    def _reset_betting_round(self, start_after: int):
        # reset per‑round bets
        for p in self.players:
            p.current_bet = 0
            p.action_taken = ""
        self.current_bet = 0
        # that start_after has already acted (posted bb), so next
        self.to_act = ((self.to_act + 1) % self.N) 
        self._bet_occurred = False

    def amount_to_call(self) -> int:
        p = self.players[self.to_act]
        return max(0, self.current_bet - p.current_bet)

    def valid_actions(self) -> List[str]:
        to_call = self.amount_to_call()
        if to_call == 0:
            return ["fold", "check", "raise"]
        else:
            return ["fold", "call", "raise"]

    def perform_action(self, action: str, raise_by: int = 0):
        p = self.players[self.to_act]
        to_call = self.amount_to_call()
        
        if not p.in_hand:
            return
        
        if action == "fold":
            p.fold()                         # rule 5
        elif action == "check":
            if to_call != 0:
                raise ValueError("Cannot check when there's a bet")
        elif action == "call":
            self.pot += p.call(to_call)
            self._bet_occurred = True
        elif action == "raise":
            self.pot += p.raise_bet(to_call, raise_by)
            self.current_bet = p.current_bet  # rule 2
            
            self._bet_occurred = True
        else:
            raise ValueError("Unknown action")
        
        p.action_taken = action

        self._next_to_act()

        # @@@AUTO ADVANCE@@@ if end‑of‑round reached:
        # condition A: no one has bet, and everyone checked
        # condition B: bets occurred, and everyone matched current_bet
        #active = [pl for pl in self.players if pl.in_hand]
        #no_bets_and_checked = (not self._bet_occurred
        #    and all(pl.current_bet == 0 for pl in active))
        #bets_matched = (self._bet_occurred
        #    and all(pl.current_bet == self.current_bet for pl in active))

        active = [p for p in self.players if p.in_hand]

        # A) no bets, everyone checked
        no_bets_and_checked = (
            not self._bet_occurred and
            all(p.action_taken == "check" for p in active)
        )

        # B) bets occurred, everyone matched
        bets_matched = (
            self._bet_occurred and
            all(p.current_bet == self.current_bet for p in active)
        )

        if no_bets_and_checked or bets_matched:
            # auto‑advance street
            if self.stage == "pre-flop":
                self.deal_flop()
            elif self.stage == "flop":
                self.deal_turn()
            elif self.stage == "turn":
                self.deal_river()

        counter : int = 0
        for p in self.players:
            if p.in_hand == True:
                counter +=1
        if counter == 1:
            print("game is ended ALL FOLDED!!! player won")
            sys.exit(0)
        '''
        no_bets_and_checked: bool = False
        bets_matched: bool = False
        flag = 0
        flg = 0
        for i in self.players:
            if p.action_taken =="check":
                flag +=1
            if p.action_taken =="call":
                flg +=1
        if flag == self.N:
            no_bets_and_checked = True
        if flg == self.N:
            bets_matched = True

        if no_bets_and_checked or bets_matched:
            # move to next street automatically
            if self.stage == "pre-flop":
                self.deal_flop()
            elif self.stage == "flop":
                self.deal_turn()
            elif self.stage == "turn":
                self.deal_river()
            
        

        self._next_to_act()

        '''

    def _next_to_act(self):
        
        #start = self.to_act
        while True:
            self.to_act = (self.to_act + 1) % self.N
            if self.players[self.to_act].in_hand:
                # if all in‑hand have matched current_bett then street ends
                if all((not pl.in_hand) or pl.current_bet == self.current_bet
                       for pl in self.players):
                    break
                return

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

# ────── POKER FACADE ──────
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
