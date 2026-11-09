"""
name_states.py - suggest which of our states (0..3) is the classic microstate A, B, C or D.

Each class has a typical SHAPE across the scalp (see microstate_model.suggest_state_names):
   A  left-back  <->  right-front diagonal
   B  left-front <->  right-back  diagonal
   C  front <-> back axis
   D  centre-front peak vs the edges
Each of our maps is correlated with these idealised shapes (absolute correlation, because
microstates ignore polarity) and the best one-to-one assignment is chosen.

CAUTION: these are idealised shapes, not the published template maps. Treat the output as a
suggestion and confirm by eye against the figures of [9] and [10]. If two classes score almost
the same for a map, its name is uncertain (C and D are our least certain names).

Run:  eeg_env\\Scripts\\python.exe scripts\\name_states.py      (prints only)
"""

import microstate_model as mm

model = mm.load_model()
names, score, classes, margins = mm.suggest_state_names(model)

# Step 1: the full score table
print("Absolute correlation of each of OUR states with the idealised classes:\n")
print("state " + " ".join(f"{n:>6}" for n in classes))
for s in range(score.shape[0]):
    print(f"{s:>5} " + " ".join(f"{score[s, j]:6.2f}" for j in range(len(classes))))

# Step 2: the suggested names and how certain each one is
print("\nSuggested one-to-one naming:")
for s in sorted(names):
    note = "" if margins[s] > 0.1 else "   <-- UNCERTAIN (another class scores almost as high)"
    print(f"  state {s}  ->  {names[s]}   (score {score[s, classes.index(names[s])]:.2f}, "
          f"margin {margins[s]:.2f}){note}")
print(f"\nNames used in the project (microstate_model.STATE_NAMES): {mm.STATE_NAMES}")
