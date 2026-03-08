import tkinter as tk

from .constants import (
    CELL_HIT,
    CELL_UNKNOWN,
    CELL_MISS,
    CELL_HIT,
    CELL_SUNK,
    GRID_SIZE,
    HORIZONTAL,
    SHIP_SIZES,
    VERTICAL,
)

class GUIwindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Battleship")

        self.buttons = []
        self.board = None

        self.playing = True

        self.player_ships = []

        # Initialize visual (grid of buttons)
        for r in range(GRID_SIZE):
            row = []
            for c in range(GRID_SIZE):
                btn = tk.Button(self.root, width=3, height=1, bg="blue", state=tk.DISABLED)
                btn.grid(row=r, column=c)
                row.append(btn)
            self.buttons.append(row)

    def _ask_orientation(self, parent):
        result = {"value": None}
    
        popup = tk.Toplevel(parent)
        popup.title("Choose Orientation")
        popup.geometry("250x120")
    
        tk.Label(popup, text="Select ship orientation:").pack(pady=10)
    
        def choose_horizontal():
            result["value"] = "H"
            popup.destroy()
    
        def choose_vertical():
            result["value"] = "V"
            popup.destroy()
    
        tk.Button(popup, text="Horizontal", width=12, command=choose_horizontal).pack(pady=5)
        tk.Button(popup, text="Vertical", width=12, command=choose_vertical).pack(pady=5)
    
        popup.transient(parent)
        popup.grab_set()
        parent.wait_window(popup)
    
        return result["value"]
    
    def phase_placement(self, board):
        def on_click(r, c, length):
            for i in range(r, min(GRID_SIZE, r + length)):
                self.buttons[i][c].config(bg="snow")
                
            for i in range(c, min(GRID_SIZE, c + length)):
                self.buttons[r][i].config(bg="snow")
            orientation = HORIZONTAL if self._ask_orientation(self.root) == "H" else "V"

            placed = board.place_ship(length, r, c, orientation)
            if placed:
                self.root.quit()
            else:
                pass #err msg

        for length in SHIP_SIZES:
            self.root.title(f"Battleship: Place ship of size {length}")

            placement_board = board.placement_matrix()
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if placement_board[r][c]:
                        self.buttons[r][c].config(state=tk.DISABLED, bg="gray")
                    else:
                        self.buttons[r][c].config(state=tk.NORMAL, bg="blue", command=lambda r=r, c=c: on_click(r, c, length))

            self.root.mainloop()

        # Create board showing own ships below, once placement phase is done
        separator = []
        for i in range(GRID_SIZE):
            btn = tk.Button(self.root, width=3, height=1, bg="black", state=tk.DISABLED)
            btn.grid(row=GRID_SIZE, column=i)
            separator.append(btn)
        self.player_ships.append(separator)

        placement_board = board.placement_matrix()

        for r in range(GRID_SIZE+1, (GRID_SIZE * 2)+1):
            row = []
            for c in range(GRID_SIZE):
                if placement_board[r-(GRID_SIZE + 1)][c]:
                    btn = tk.Button(self.root, width=3, height=1, bg="gray", state=tk.DISABLED)
                else:
                    btn = tk.Button(self.root, width=3, height=1, bg="navy", state=tk.DISABLED)
                btn.grid(row=r, column=c)
                row.append(btn)
            self.player_ships.append(row)
        
        self.root.title("Battleship")

    def phase_shot(self, observation, agent_shots: list) -> int:
        # At start of shooting phase, update board where opponent shot
        for shot in agent_shots:
            if self.player_ships[shot[0]][shot[1]]["bg"] == "gray":
                self.player_ships[shot[0]][shot[1]].config(bg="red4")
            elif self.player_ships[shot[0]][shot[1]]["bg"] == "navy":
                self.player_ships[shot[0]][shot[1]].config(bg="sky blue")

        chosen_row = {"value": 0}
        chosen_col = {"value": 0}

        def take_shot(r, c):
            self.root.quit()
            chosen_row["value"], chosen_col["value"] = r, c

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if observation[r][c] == CELL_MISS:
                    self.buttons[r][c].config(state=tk.DISABLED, text="o", bg="sky blue")
                elif observation[r][c] == CELL_HIT:
                    self.buttons[r][c].config(state=tk.DISABLED, text="x", bg="red")
                elif observation[r][c] == CELL_SUNK:
                    self.buttons[r][c].config(state=tk.DISABLED, text="X", bg="red4")
                else:
                    self.buttons[r][c].config(state=tk.NORMAL, command=lambda r=r, c=c: take_shot(r, c), text="", bg="blue")
        
        self.root.mainloop()

        return chosen_row["value"] * GRID_SIZE + chosen_col["value"]