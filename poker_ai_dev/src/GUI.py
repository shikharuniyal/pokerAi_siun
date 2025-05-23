import tkinter as tk
from tkinter import ttk, messagebox
from srcrev2 import Poker, Player

class PokerGUI:
    def __init__(self, root):
        self.game = Poker([Player(f"P{i+1}", 1000) for i in range(2)])
        self.show_p1 = tk.BooleanVar(value=False)
        self.build_ui(root)

    def build_ui(self, root):
        # Community & pot
        fr = ttk.Frame(root); fr.pack(fill="x", pady=5)
        ttk.Label(fr, text="Community:").pack(side="left")
        self.comm_var = tk.StringVar()
        ttk.Label(fr, textvariable=self.comm_var).pack(side="left", padx=5)
        ttk.Label(fr, text="  Pot:").pack(side="left", padx=(20,0))
        self.pot_var = tk.StringVar()
        ttk.Label(fr, textvariable=self.pot_var).pack(side="left")

         # Players (with indicator)
        self.player_vars = []
        self.indicators = []
        fr2 = ttk.Frame(root)
        fr2.pack(fill="x", pady=5)

        for _ in self.game.table.players:
            row = ttk.Frame(fr2)
            row.pack(fill="x", pady=2)

            # 1) Indicator Label (initially blank, fixed width)
            dot = ttk.Label(row, text=" ", width=2, anchor="center")
            dot.pack(side="left")
            self.indicators.append(dot)

            # 2) Player info Label
            var = tk.StringVar()
            lbl = ttk.Label(row, textvariable=var)
            lbl.pack(side="left", anchor="w")
            self.player_vars.append(var)

        
        
        # Deal / Street Buttons
        cf = ttk.Frame(root); cf.pack(fill="x", pady=5)
        self.deal_btn = ttk.Button(cf, text="Deal Hand", command=self.on_deal)
        self.deal_btn.pack(side="left")
        self.flop_btn = ttk.Button(cf, text="Flop", command=self.on_flop, state="disabled")
        self.flop_btn.pack(side="left", padx=2)
        self.turn_btn = ttk.Button(cf, text="Turn", command=self.on_turn, state="disabled")
        self.turn_btn.pack(side="left", padx=2)
        self.river_btn = ttk.Button(cf, text="River", command=self.on_river, state="disabled")
        self.river_btn.pack(side="left", padx=2)

        # Action Controls
        af = ttk.Frame(root); af.pack(fill="x", pady=5)
        ttk.Label(af, text="Action:").pack(side="left")
        self.action_cb = ttk.Combobox(af, values=[], state="readonly", width=8)
        self.action_cb.pack(side="left", padx=2)
        self.amt_entry = ttk.Entry(af, width=5)
        self.amt_entry.pack(side="left")
        self.act_btn = ttk.Button(af, text="OK", state="disabled", command=self.on_action)
        self.act_btn.pack(side="left", padx=2)

        rf = ttk.Frame(root); rf.pack(fill="x", pady=5)
        ttk.Checkbutton(
            rf,
            text="Show Player 1 Cards",
            variable=self.show_p1,
            command=self.update_ui   # call update_ui whenever toggled
        ).pack(side="left")

    def reveal_p1(self):
        self.show_p1_cards = True
        self.update_ui()

    def on_deal(self):
        self.game.start_hand()
        self.deal_btn.config(state="disabled")
        self.refresh_controls()
        self.update_ui()

    def on_flop(self):
        try:
            self.game.flop()
        except RuntimeError as e:
            messagebox.showwarning("Cannot Flop", str(e))
            return
        self.refresh_controls()
        self.update_ui()

    def on_turn(self):
        try:
            self.game.turn()
        except RuntimeError as e:
            messagebox.showwarning("Cannot Turn", str(e))
            return
        self.refresh_controls()
        self.update_ui()

    def on_river(self):
        try:
            self.game.river()
        except RuntimeError as e:
            messagebox.showwarning("Cannot River", str(e))
            return
        self.refresh_controls()
        self.update_ui()

    def on_action(self):
        action = self.action_cb.get()
        amt = 0
        if action == "raise":
            try:
                amt = int(self.amt_entry.get())
            except ValueError:
                messagebox.showerror("Invalid Amount", "Enter a valid integer to raise.")
                return

        try:
            self.game.do(action, amt)
        except Exception as e:
            messagebox.showerror("Action Error", str(e))
        self.refresh_controls()
        self.update_ui()

    def refresh_controls(self):
        """Enable/disable buttons based on game state."""
        table = self.game.table

        # Enable action controls if it's an active player's turn
        if table.stage in ("pre-flop","flop","turn","river"):
            valid = self.game.actions()
            if valid:
                self.action_cb.config(values=valid, state="readonly")
                self.action_cb.current(0)
                self.act_btn.config(state="normal")
            else:
                self.action_cb.config(state="disabled")
                self.act_btn.config(state="disabled")
        else:
            self.action_cb.config(state="disabled")
            self.act_btn.config(state="disabled")

        # Street buttons only enabled when all active players have matched current_bet
        def bets_equalized():
            return all(
                (not p[4]) or (p[2] == table.current_bet)
                for p in self.game.players_state
            )

        if table.stage == "pre-flop" and bets_equalized():
            self.flop_btn.config(state="normal")
        else:
            self.flop_btn.config(state="disabled")

        if table.stage == "flop" and bets_equalized():
            self.turn_btn.config(state="normal")
        else:
            self.turn_btn.config(state="disabled")

        if table.stage == "turn" and bets_equalized():
            self.river_btn.config(state="normal")
        else:
            self.river_btn.config(state="disabled")

        # Deal button only at very start
        self.deal_btn.config(state="normal" if table.stage is None else "disabled")

    def update_ui(self):
        # Community & Pot
        self.comm_var.set(", ".join(map(str, self.game.community)))
        self.pot_var .set(str(self.game.pot))

        # Players: unpack (name, chips, current_bet, total_committed, in_hand, hole)
        for idx, (var, st) in enumerate(zip(self.player_vars, self.game.players_state)):
            name, chips, round_bet, total, in_hand, hole = st
            status = "IN" if in_hand else "FD"
            # hide Player 1s cards until reveal
            if idx == 0 and not self.show_p1.get():
                display_hole = ["?", "?"]
            else:
                display_hole = list(map(str, hole))

            var.set(f"{name}: Chips={chips}, RoundBet={round_bet}, Total={total}, {status} | {', '.join(display_hole)}")

             # Highlight whose turn it is with a red dot
            if idx == self.game.table.to_act and in_hand:
                self.indicators[idx].config(text="●", foreground="red")
            else:
                self.indicators[idx].config(text=" ", foreground="")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Poker GUI")
    PokerGUI(root)
    root.mainloop()
