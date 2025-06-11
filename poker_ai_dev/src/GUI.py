import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from src import Poker, Player

class PokerGUI:
    def __init__(self, n_players=2):
        # 1) Game logic
        self.game = Poker([Player(f"P{i+1}", 1000) for i in range(n_players)])
        self.show_p1 = tk.BooleanVar(value=False)

        # 2) Load card images named like "1c.png", "10h.png", "Js.png", etc.
        self.card_images = {}
        self._load_card_images()

        # 3) Create one window per player
        self.player_windows = []
        for idx in range(n_players):
            win = tk.Toplevel()
            win.title(f"Player {idx+1}")
            widgets = self.build_ui(win, idx)
            self.player_windows.append(widgets)

        # 4) Initial draw
        self.refresh_controls()
        self.update_ui()

    def _load_card_images(self):
        # Adjust this path to point at your card PNG folder:
        base = r"E:/#EditorCodes/Project_poker/poker_ai_dev/cards"
        for fname in os.listdir(base):
            name, ext = os.path.splitext(fname)
            if ext.lower() != ".png": 
                continue
            # e.g. name="10h" or "Js"
            rank = name[:-1]
            suit = name[-1].lower()  # c, d, h, or s
            key  = f"{rank}_{suit}"
            path = os.path.join(base, fname)
            img  = Image.open(path).resize((72, 96), resample=Image.LANCZOS)
            self.card_images[key] = ImageTk.PhotoImage(img)

    def build_ui(self, root, player_idx):
        # Community & pot
        fr = ttk.Frame(root); fr.pack(fill="x", pady=5)
        ttk.Label(fr, text="Community:").pack(side="left")
        comm_var = tk.StringVar(); ttk.Label(fr, textvariable=comm_var).pack(side="left", padx=5)
        ttk.Label(fr, text="Pot:").pack(side="left", padx=(20,0))
        pot_var = tk.StringVar();  ttk.Label(fr, textvariable=pot_var).pack(side="left")

                # — Community card images —
        comm_img_frame = ttk.Frame(root); comm_img_frame.pack(pady=5)
        comm_lbls = []
        for _ in range(5):               # max 5 community cards
            lbl = ttk.Label(comm_img_frame)
            lbl.pack(side="left", padx=2)
            comm_lbls.append(lbl)


        # Hole‐card images for this player
        hole_frame = ttk.Frame(root); hole_frame.pack(pady=5)
        lbl1 = ttk.Label(hole_frame); lbl1.pack(side="left", padx=2)
        lbl2 = ttk.Label(hole_frame); lbl2.pack(side="left", padx=2)

        # Player list + turn indicator
        player_vars = []; indicators = []
        fr2 = ttk.Frame(root); fr2.pack(fill="x", pady=5)
        for _ in self.game.table.players:
            row = ttk.Frame(fr2); row.pack(fill="x", pady=2)
            dot = ttk.Label(row, text= " ", width=2); dot.pack(side="left")
            indicators.append(dot)
            var = tk.StringVar()
            ttk.Label(row, textvariable=var).pack(side="left")
            player_vars.append(var)

        # Deal / street buttons
        cf = ttk.Frame(root); cf.pack(fill="x", pady=5)
        deal_btn  = ttk.Button(cf, text="Deal",  command=self.on_deal)
        flop_btn  = ttk.Button(cf, text="Flop",  command=self.on_flop,  state="disabled")
        turn_btn  = ttk.Button(cf, text="Turn",  command=self.on_turn,  state="disabled")
        river_btn = ttk.Button(cf, text="River", command=self.on_river, state="disabled")
        for b in (deal_btn, flop_btn, turn_btn, river_btn): b.pack(side="left", padx=2)

        # Action controls
        af = ttk.Frame(root); af.pack(fill="x", pady=5)
        ttk.Label(af, text="Action:").pack(side="left")
        action_cb = ttk.Combobox(af, state="readonly", width=8)
        action_cb.pack(side="left", padx=2)
        amt_entry = ttk.Entry(af, width=5); amt_entry.pack(side="left")
        act_btn = ttk.Button(af, text="OK", state="disabled", command=self.on_action)
        act_btn.pack(side="left", padx=2)

        # Show P1 cards toggle
        ttk.Checkbutton(
            root,
            text="Show P1 Cards",
            variable=self.show_p1,
            command=self.update_ui
        ).pack(anchor="w", padx=5, pady=5)

        return {
            "idx":         player_idx,
            "comm_var":   comm_var,
            "pot_var":    pot_var,
            "lbl1":       lbl1,
            "lbl2":       lbl2,
            "player_vars":player_vars,
            "indicators": indicators,
            "deal_btn":   deal_btn,
            "flop_btn":   flop_btn,
            "turn_btn":   turn_btn,
            "river_btn":  river_btn,
            "action_cb":  action_cb,
            "amt_entry":  amt_entry,
            "act_btn":    act_btn,
            "comm_lbls": comm_lbls,
        }

    def on_deal(self):
        self.game.start_hand()
        '''for w in self.player_windows:
            # clear community cards
            for lbl in w["comm_lbls"]:
                lbl.config(image="")
            # clear hole cards
            w["lbl1"].config(image="")
            w["lbl2"].config(image="")
        self.game.start_hand()'''
        self.refresh_controls()
        self.update_ui()

    def on_flop(self):
        try:    self.game.flop()
        except RuntimeError as e: messagebox.showwarning("Flop", str(e)); return
        self.refresh_controls(); self.update_ui()

    def on_turn(self):
        try:    self.game.turn()
        except RuntimeError as e: messagebox.showwarning("Turn", str(e)); return
        self.refresh_controls(); self.update_ui()

    def on_river(self):
        try:    self.game.river()
        except RuntimeError as e: messagebox.showwarning("River", str(e)); return
        self.refresh_controls(); self.update_ui()

    def on_action(self):
        idx = self.game.table.to_act
        w   = self.player_windows[idx]
        action = w["action_cb"].get()
        amt = 0
        if action == "raise":
            try:    amt = int(w["amt_entry"].get())
            except: messagebox.showerror("Amount","Enter a number"); return
        try:
            self.game.do(action, amt)
        except Exception as e:
            messagebox.showerror("Action", str(e))
        self.refresh_controls(); self.update_ui()

    def refresh_controls(self):
        t = self.game.table
        valid = self.game.actions() if t.stage in ("pre-flop","flop","turn","river") else []
        def bets_eq():
            return all((not p[4]) or (p[2]==t.current_bet) for p in self.game.players_state)

        for w in self.player_windows:
            pidx = w["idx"]
            active = (t.to_act==pidx)

            # action
            if t.stage in ("pre-flop","flop","turn","river") and active and valid:
                w["action_cb"].config(values=valid, state="readonly"); w["action_cb"].current(0)
                w["act_btn"].config(state="normal")
            else:
                w["action_cb"].config(state="disabled"); w["act_btn"].config(state="disabled")

            # street
            ready = active and bets_eq()
            w["flop_btn"].config( state="normal" if t.stage=="pre-flop" and ready else "disabled")
            w["turn_btn"].config( state="normal" if t.stage=="flop"    and ready else "disabled")
            w["river_btn"].config(state="normal" if t.stage=="turn"    and ready else "disabled")

            # deal
            w["deal_btn"].config(state="normal" if t.stage is None and active else "disabled")

    def update_ui(self):
        comm = ", ".join(map(str, self.game.community))
        pot  = str(self.game.pot)

        for w in self.player_windows:
            pidx = w["idx"]   # the player this window represents

            # 1) Community text and pot
            w["comm_var"].set(comm)
            w["pot_var"].set(pot)

            # 2) Community card *images*
            for ci, lbl in enumerate(w["comm_lbls"]):
                if ci < len(self.game.community):
                    card = self.game.community[ci]
                    key  = f"{card.rank}_{card.suit[0].lower()}"
                    img  = self.card_images.get(key)
                    if img:
                        lbl.config(image=img)
                        lbl.image = img
                    else:
                        lbl.config(image="")
                else:
                    lbl.config(image="")

            # 3) Hole-card images (only if dealt and allowed)
            _, _, _, _, alive, hole = self.game.players_state[pidx]
            if alive and len(hole) == 2 and (pidx != 0 or self.show_p1.get()):
                # build keys using single-letter suits
                r1, s1 = hole[0].rank, hole[0].suit[0].lower()
                r2, s2 = hole[1].rank, hole[1].suit[0].lower()
                key1, key2 = f"{r1}_{s1}", f"{r2}_{s2}"

                img1 = self.card_images.get(key1)
                img2 = self.card_images.get(key2)

                if img1:
                    w["lbl1"].config(image=img1)
                    w["lbl1"].image = img1
                if img2:
                    w["lbl2"].config(image=img2)
                    w["lbl2"].image = img2
            else:
                w["lbl1"].config(image="")
                w["lbl2"].config(image="")

            # 4) Player list + turn indicator
            for j, var in enumerate(w["player_vars"]):
                name, ch, rb, tot, inh, hole = self.game.players_state[j]
                status = "IN" if inh else "FD"
                disp   = ["?","?"] if (j == 0 and not self.show_p1.get()) else list(map(str, hole))
                var.set(f"{name}: Chips={ch}, Bet={rb}, Total={tot}, {status} | {', '.join(disp)}")

                dot = w["indicators"][j]
                if j == self.game.table.to_act and inh:
                    dot.config(text="●", foreground="red")
                else:
                    dot.config(text=" ", foreground="")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()                # hide the extra root window
    PokerGUI(n_players=2)          # or however many players you like
    root.mainloop()
