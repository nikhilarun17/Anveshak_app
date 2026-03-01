"""
Mars Rover Bio-Muscle Arm - Optimal Value Function via Dynamic Programming
State: (prev_action, fatigue, turns_left)
  - prev_action: 'safe', 'fast', or 'none' (initial state)
  - fatigue: 0..9 (fatigue >= 10 => failure/terminal)
  - turns_left: 0..5

Actions: 'safe', 'fast'

Reward(s, a):
  - +10 - c(a) if operation completes without failure  [c(safe)=5, c(fast)=2]
  - -100 if failure (fatigue >= 10 after increment)
  
Fatigue transitions:
  Safe action:
    +1 FU with prob 0.8
    +2 FU with prob 0.2
    (resets consecutive-fast penalty)

  Fast action (prev was safe or 'none'):
    +3 FU with prob 0.7
    +4 FU with prob 0.3

  Fast action (prev was fast) - consecutive:
    +5 FU with prob 0.6
    +7 FU with prob 0.4

  Micro-tear (only fast, fatigue AFTER base increment >= 8... 
  actually: if fatigue BEFORE action >= 8, after normal increment apply micro-tear):
  Wait - re-reading: "once fatigue reaches critical instability (fatigue >= 8),
  if fast action executed, AFTER normal fatigue increment is applied,
  with prob 0.2 additional +4 FU damage."
  
  Interpretation: check if current fatigue >= 8 AND action is fast
  => apply base increment, then apply micro-tear check.
  
  Actually let's re-read carefully: "once fatigue reaches a critical instability 
  regime (fatigue >= 8 FU), the bio-muscle becomes structurally fragile. 
  If a fast action is executed in this regime, a micro-tear may occur after 
  the normal fatigue increment is applied."
  
  So: if state fatigue >= 8 AND action == fast:
    1. Apply base fast increment (get new_fatigue_base)
    2. With prob 0.2, add +4 more FU (micro-tear)
    3. With prob 0.8, no additional damage
"""

from itertools import product

ACTIONS = ['safe', 'fast']
PREV_ACTIONS = ['none', 'safe', 'fast']
MAX_FATIGUE = 9   # states 0..9; failure if >= 10
MAX_TURNS = 5
FAILURE_THRESHOLD = 10

COST = {'safe': 5, 'fast': 2}


def get_fatigue_transitions(fatigue, prev_action, action):
    """
    Returns list of (new_fatigue, probability) after taking action.
    Handles base increments + micro-tear if applicable.
    Does NOT cap at 9 - caller checks for failure.
    """
    # Base fatigue increments
    if action == 'safe':
        base_increments = [(1, 0.8), (2, 0.2)]
    else:  # fast
        if prev_action == 'fast':
            # consecutive fast
            base_increments = [(5, 0.6), (7, 0.4)]
        else:
            # previous was safe or none
            base_increments = [(3, 0.7), (4, 0.3)]

    # Build outcomes after base increment
    outcomes = {}  # new_fatigue -> probability

    for delta, prob in base_increments:
        new_f = fatigue + delta

        if action == 'fast' and fatigue >= 8:
            # Micro-tear check (independent of base increment)
            # With prob 0.2: +4 more FU
            f_no_tear = new_f
            f_tear = new_f + 4

            p_no_tear = 0.8
            p_tear = 0.2

            outcomes[f_no_tear] = outcomes.get(f_no_tear, 0) + prob * p_no_tear
            outcomes[f_tear] = outcomes.get(f_tear, 0) + prob * p_tear
        else:
            outcomes[new_f] = outcomes.get(new_f, 0) + prob

    return list(outcomes.items())


def is_terminal(fatigue):
    return fatigue >= FAILURE_THRESHOLD


def compute_value_function():
    """
    V_t(prev_action, fatigue, turns_left) using backward DP.
    t goes from 0 (no turns left) to 5 (full mission).
    
    V[turns_left][prev_action][fatigue] = expected future reward
    """

    # Initialize: V at turns_left=0 => no more operations, value=0
    # We'll store V as dict: state -> value
    # state = (prev_action, fatigue, turns_left)

    V = {}

    # Base case: turns_left = 0, no more operations
    for pa in PREV_ACTIONS:
        for f in range(FAILURE_THRESHOLD):  # 0..9
            V[(pa, f, 0)] = 0.0

    # DP from turns_left=1 to 5
    for turns_left in range(1, MAX_TURNS + 1):
        for pa in PREV_ACTIONS:
            for f in range(FAILURE_THRESHOLD):
                state = (pa, f, turns_left)
                best_val = None

                for action in ACTIONS:
                    outcomes = get_fatigue_transitions(f, pa, action)
                    expected = 0.0

                    for new_f, prob in outcomes:
                        if new_f >= FAILURE_THRESHOLD:
                            # Failure
                            reward = -100
                            expected += prob * reward
                        else:
                            # Success
                            reward = 10 - COST[action]
                            next_pa = action
                            next_turns = turns_left - 1
                            future = V.get((next_pa, new_f, next_turns), 0.0)
                            expected += prob * (reward + future)

                    if best_val is None or expected > best_val:
                        best_val = expected

                V[state] = best_val

    return V


def compute_policy(V):
    """Extract optimal policy from value function."""
    policy = {}

    for turns_left in range(1, MAX_TURNS + 1):
        for pa in PREV_ACTIONS:
            for f in range(FAILURE_THRESHOLD):
                state = (pa, f, turns_left)
                best_val = None
                best_action = None

                for action in ACTIONS:
                    outcomes = get_fatigue_transitions(f, pa, action)
                    expected = 0.0

                    for new_f, prob in outcomes:
                        if new_f >= FAILURE_THRESHOLD:
                            reward = -100
                            expected += prob * reward
                        else:
                            reward = 10 - COST[action]
                            next_pa = action
                            next_turns = turns_left - 1
                            future = V.get((next_pa, new_f, next_turns), 0.0)
                            expected += prob * (reward + future)

                    if best_val is None or expected > best_val:
                        best_val = expected
                        best_action = action

                policy[state] = best_action

    return policy


def get_q_values(V, pa, f, tl):
    """Return Q-values for both actions at a given state."""
    q = {}
    for action in ACTIONS:
        outcomes = get_fatigue_transitions(f, pa, action)
        expected = 0.0
        for new_f, prob in outcomes:
            if new_f >= FAILURE_THRESHOLD:
                expected += prob * (-100)
            else:
                reward = 10 - COST[action]
                future = V.get((action, new_f, tl - 1), 0.0)
                expected += prob * (reward + future)
        q[action] = expected
    return q


def print_full_table(V, policy):
    print("=" * 85)
    print("FULL OPTIMAL VALUE FUNCTION, Q-VALUES & POLICY")
    print("State: (prev_action, fatigue, turns_left)")
    print("=" * 85)

    for turns_left in range(1, MAX_TURNS + 1):
        print(f"\n{'─'*85}")
        print(f"  turns_left = {turns_left}")
        print(f"{'─'*85}")
        print(f"{'prev_action':<12} {'fatigue':<9} {'Q(safe)':<14} {'Q(fast)':<14} {'V(s)':<14} {'OPTIMAL ACTION'}")
        print(f"{'─'*85}")
        for pa in PREV_ACTIONS:
            for f in range(FAILURE_THRESHOLD):
                state = (pa, f, turns_left)
                val = V.get(state, 0.0)
                act = policy.get(state, 'N/A')
                q = get_q_values(V, pa, f, turns_left)
                marker_s = " <--" if act == 'safe' else ""
                marker_f = " <--" if act == 'fast' else ""
                print(f"{pa:<12} {f:<9} {q['safe']:<10.4f}{marker_s:<4} {q['fast']:<10.4f}{marker_f:<4} {val:<14.4f} {act}")


if __name__ == "__main__":
    print("Computing optimal value function via Dynamic Programming...")
    print("State representation: (prev_action, fatigue, turns_left)")
    print("Reward: +10 - c(a) on success | -100 on failure")
    print("c(safe) = 5, c(fast) = 2\n")

    V = compute_value_function()
    policy = compute_policy(V)

    print_full_table(V, policy)

    # Print initial state value
    init_state = ('none', 0, 5)
    print(f"\n{'=' * 70}")
    print(f"Initial state {init_state}: V = {V[init_state]:.4f}")
    print(f"Optimal first action: {policy[init_state]}")
    print("=" * 70)


# ─────────────────────────────────────────────
# BONUS: Monte Carlo Simulation of 1000 missions
# ─────────────────────────────────────────────

import random

def simulate_mission(policy, seed_offset=0):
    """
    Simulate one mission following the optimal policy.
    Returns (total_reward, final_fatigue, failed: bool)
    """
    state = ('none', 0, 5)
    total_reward = 0

    while True:
        pa, f, tl = state

        if tl == 0:
            return total_reward, f, False   # completed successfully

        if f >= FAILURE_THRESHOLD:
            return total_reward, f, True    # already failed

        action = policy.get(state, 'safe')
        transitions = get_fatigue_transitions(f, pa, action)

        # Sample next fatigue
        r = random.random()
        cumulative = 0.0
        new_f = transitions[-1][0]  # fallback
        for nf, prob in transitions:
            cumulative += prob
            if r <= cumulative:
                new_f = nf
                break

        if new_f >= FAILURE_THRESHOLD:
            total_reward += -100
            return total_reward, new_f, True

        reward = 10 - COST[action]
        total_reward += reward
        state = (action, new_f, tl - 1)


def run_simulation(n=1000, seed=42):
    random.seed(seed)

    V = compute_value_function()
    pol = compute_policy(V)

    failures = 0
    total_rewards = []
    final_fatigues = []
    failure_steps = []

    for i in range(n):
        reward, fatigue, failed = simulate_mission(pol, seed_offset=i)
        total_rewards.append(reward)
        final_fatigues.append(fatigue)
        if failed:
            failures += 1

    prob_failure   = failures / n
    avg_reward     = sum(total_rewards) / n
    avg_fatigue    = sum(final_fatigues) / n

    # Distribution of total rewards
    reward_counts = {}
    for r in total_rewards:
        reward_counts[r] = reward_counts.get(r, 0) + 1

    print("\n" + "=" * 60)
    print(f"  MONTE CARLO SIMULATION  —  {n} missions")
    print("=" * 60)
    print(f"  Probability of actuator failure : {prob_failure:.4f}  ({failures}/{n})")
    print(f"  Average total reward            : {avg_reward:.4f}")
    print(f"  Average final fatigue           : {avg_fatigue:.4f}")
    print()
    print("  Reward distribution:")
    for rv in sorted(reward_counts.keys(), reverse=True):
        bar = "█" * int(reward_counts[rv] / n * 50)
        print(f"    Reward {rv:>5} : {reward_counts[rv]:>5} missions  {bar}")
    print()
    print("  Final fatigue distribution:")
    fatigue_counts = {}
    for fg in final_fatigues:
        fatigue_counts[fg] = fatigue_counts.get(fg, 0) + 1
    for fv in sorted(fatigue_counts.keys()):
        bar = "█" * int(fatigue_counts[fv] / n * 50)
        print(f"    Fatigue {fv:>2}   : {fatigue_counts[fv]:>5} missions  {bar}")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation(1000)