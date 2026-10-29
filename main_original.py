"""
PredictiveRoute: ML-Based Network Congestion Prediction and Prevention
======================================================================

IMPORTANT: Everything here is SYNTHETIC.
  * The training dataset is made up by a random number generator.
  * The routing results come from a small Python simulation, NOT a real network.
  * Do not treat these numbers as real-world performance.

What this script does (in order):
  1. Generates a synthetic network dataset.
  2. Trains a RandomForest model to predict congestion and prints its scores.
  3. Simulates two routes (Route A and Route B).
  4. Runs two routing strategies (static and predictive) under low/medium/high traffic.
  5. Prints a comparison table and saves plots into the "results" folder.

Run it with:   python main.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Save plots to files only (no pop-up windows, works everywhere)
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# SETTINGS (you can change these and re-run)
# ----------------------------------------------------------------------------

SEED = 42                 # Fixed seed so you get the same results every run
NUM_SAMPLES = 2000        # Number of rows in the synthetic dataset
NUM_STEPS = 300           # Simulation length: 300 time steps (think: 300 seconds)
NUM_RUNS = 10             # Each scenario is run 10 times with different random traffic, then averaged

# Folder where plots/CSV are saved (always next to main.py, wherever you run it from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# The 4 input columns the model uses (the 5th column, "congestion", is the answer)
FEATURES = ["bandwidth_utilization", "traffic_load", "latency", "queue_length"]

# One packet = 1500 bytes = 12000 bits. So 1 Megabit holds about 83 packets.
PACKETS_PER_MEGABIT = 1_000_000 / 12_000

# The two routes. Capacity is in Mbps (megabits per second).
# Each simulation step is 1 second, so "Mbps" also means "megabits per step".
ROUTE_SETTINGS = {
    "A": {"capacity": 100, "base_latency": 20, "background_load": 20, "buffer": 12},
    "B": {"capacity": 90,  "base_latency": 30, "background_load": 10, "buffer": 12},
}
# buffer = how many megabits the router queue can hold before it starts dropping packets.
# background_load = average traffic from OTHER users already on that route (Mbps).

# How much traffic OUR flow sends (average Mbps) in each scenario.
TRAFFIC_SCENARIOS = {"Low": 30, "Medium": 60, "High": 80}


# ----------------------------------------------------------------------------
# STEP 1: Generate a synthetic dataset
# ----------------------------------------------------------------------------

def generate_dataset(num_samples, rng):
    """Create fake network measurements and label each row as congested or not.

    We first pick how busy the link is (utilization), then build the other
    columns from it so they look realistic: busier link -> longer queue and
    higher latency. The label comes from a "congestion score" plus a little
    random noise, so the problem is not perfectly easy.
    """
    # How busy the link is, from 10% to 130% (above 100% means overloaded)
    busy = rng.uniform(10, 130, num_samples)

    # Feature 1: bandwidth utilization (%), cannot go above 100
    bandwidth_utilization = np.clip(busy + rng.normal(0, 3, num_samples), 0, 100)

    # Feature 2: traffic load in Mbps (reference link capacity = 100 Mbps)
    traffic_load = np.clip(busy + rng.normal(0, 5, num_samples), 5, None)

    # Feature 3: packet queue length (packets). Queue only builds up when busy > 70%.
    queue_length = np.where(busy > 70, (busy - 70) * 17, 0) + rng.normal(0, 30, num_samples)
    queue_length = np.clip(queue_length, 0, 1000)

    # Feature 4: latency (ms) = base delay + a bit for load + waiting time in the queue
    queue_delay = (queue_length / PACKETS_PER_MEGABIT) / 100 * 1000
    latency = 20 + 15 * (bandwidth_utilization / 100) ** 2 + queue_delay
    latency = latency + rng.normal(0, 4, num_samples)

    # Hidden congestion score: higher utilization, queue, and latency -> more congestion
    score = (
        0.4 * (bandwidth_utilization / 100)
        + 0.3 * (queue_length / 1000)
        + 0.3 * ((latency - 20) / 100)
        + rng.normal(0, 0.04, num_samples)  # noise
    )
    congestion = (score > 0.55).astype(int)  # 1 = congestion, 0 = no congestion

    return pd.DataFrame(
        {
            "bandwidth_utilization": bandwidth_utilization.round(2),
            "traffic_load": traffic_load.round(2),
            "latency": latency.round(2),
            "queue_length": queue_length.round(0),
            "congestion": congestion,
        }
    )


# ----------------------------------------------------------------------------
# STEP 2: Train and evaluate the model
# ----------------------------------------------------------------------------

def train_model(dataset):
    """Split the data, train a RandomForest, print scores, and return the model."""
    X = dataset[FEATURES]      # inputs
    y = dataset["congestion"]  # answer we want to predict

    # 80% of rows for training, 20% for testing (stratify keeps class balance equal)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)

    # Check how well it does on data it has never seen
    y_pred = model.predict(X_test)

    print("=" * 60)
    print("MODEL RESULTS (on the synthetic TEST set)")
    print("=" * 60)
    print(f"Training rows: {len(X_train)}   Testing rows: {len(X_test)}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 Pred 0   Pred 1")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Actual 0 (no)    {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"Actual 1 (yes)   {cm[1][0]:6d}   {cm[1][1]:6d}")
    print()
    return model


# ----------------------------------------------------------------------------
# STEP 3: Network simulation (two routes)
# ----------------------------------------------------------------------------

def make_route(name):
    """Create a route with an empty queue and an 'idle' starting state."""
    settings = ROUTE_SETTINGS[name]
    return {
        "capacity": settings["capacity"],
        "base_latency": settings["base_latency"],
        "buffer": settings["buffer"],
        "queue": 0.0,  # megabits currently waiting in the router queue
        # "state" = the last measurements of this route (what a monitor would see)
        "state": {"util": 0.0, "load": 0.0, "latency": settings["base_latency"],
                  "queue_packets": 0.0, "loss_fraction": 0.0},
    }


def update_route(route, load, rng):
    """Simulate ONE second on a route that receives 'load' Mbps of traffic."""
    # Traffic above capacity can't be sent, so it waits in the queue
    route["queue"] = max(0.0, route["queue"] + load - route["capacity"])

    # If the queue is full, the extra traffic is dropped (packet loss)
    lost = 0.0
    if route["queue"] > route["buffer"]:
        lost = route["queue"] - route["buffer"]
        route["queue"] = route["buffer"]

    utilization = min(load / route["capacity"], 1.0)

    # Latency = base delay + small load-based delay + waiting time in the queue + tiny jitter
    queue_delay_ms = route["queue"] / route["capacity"] * 1000
    latency = route["base_latency"] + 15 * utilization ** 2 + queue_delay_ms
    latency += rng.normal(0, 2)

    route["state"] = {
        "util": utilization * 100,                              # percent
        "load": load,                                           # Mbps
        "latency": max(latency, 1.0),                           # ms
        "queue_packets": route["queue"] * PACKETS_PER_MEGABIT,  # packets
        "loss_fraction": lost / load if load > 0 else 0.0,      # 0.0 to 1.0
    }


def make_series(num_steps, mean, noise, burst_size, rng):
    """Make a traffic pattern: an average level, random wiggle, and short bursts."""
    series = rng.normal(mean, noise, num_steps)
    in_burst = False
    for t in range(num_steps):
        # A burst starts with 5% chance, and continues with 70% chance each step
        in_burst = (rng.random() < 0.70) if in_burst else (rng.random() < 0.05)
        if in_burst:
            series[t] += burst_size
    return np.clip(series, 0, None)


def make_traffic(flow_mean, rng):
    """Create all the random traffic for one simulation run.

    Both strategies get the SAME traffic, so the comparison is fair.
    """
    return {
        "flow": make_series(NUM_STEPS, flow_mean, flow_mean * 0.10, flow_mean * 0.25, rng),
        "bg_A": make_series(NUM_STEPS, ROUTE_SETTINGS["A"]["background_load"], 5, 25, rng),
        "bg_B": make_series(NUM_STEPS, ROUTE_SETTINGS["B"]["background_load"], 3, 15, rng),
    }


# ----------------------------------------------------------------------------
# STEP 4: Routing strategies + simulation loop
# ----------------------------------------------------------------------------

def run_simulation(strategy, traffic, model, seed):
    """Run one simulation with 'static' or 'predictive' routing. Returns the metrics."""
    rng = np.random.default_rng(seed)  # only used for latency jitter
    routes = {"A": make_route("A"), "B": make_route("B")}
    current = "A"  # Both strategies start on Route A (the predefined route)

    route_changes = 0
    latencies = []
    total_lost = 0.0    # megabits of OUR traffic that was dropped
    total_sent = 0.0    # megabits of OUR traffic we tried to send

    for t in range(NUM_STEPS):
        flow = traffic["flow"][t]

        # ---- Decide which route to use this second ----
        if strategy == "predictive":
            seen = routes[current]["state"]  # latest measurements of the current route
            row = pd.DataFrame(
                [[seen["util"], seen["load"], seen["latency"], seen["queue_packets"]]],
                columns=FEATURES,
            )
            congestion_predicted = model.predict(row)[0] == 1

            if congestion_predicted:
                other = "B" if current == "A" else "A"
                other_route = routes[other]
                # Free capacity on the other route = its capacity - traffic already on it
                available = other_route["capacity"] - other_route["state"]["load"]
                # Only switch if the other route can really carry our flow
                if available >= flow:
                    current = other
                    route_changes += 1
        # strategy == "static": we never change 'current', it stays Route A

        # ---- Send traffic and update both routes ----
        for name, route in routes.items():
            load = traffic["bg_" + name][t] + (flow if name == current else 0.0)
            update_route(route, load, rng)

        # ---- Record what OUR flow experienced on the route it used ----
        used = routes[current]["state"]
        latencies.append(used["latency"])
        total_lost += flow * used["loss_fraction"]
        total_sent += flow

    return {
        "avg_latency_ms": float(np.mean(latencies)),
        "packet_loss_pct": 100.0 * total_lost / total_sent,
        "route_changes": route_changes,
    }


def compare_strategies(model):
    """Run both strategies for every traffic level and return a results table."""
    rng = np.random.default_rng(SEED + 1)
    rows = []

    for scenario, flow_mean in TRAFFIC_SCENARIOS.items():
        for run in range(NUM_RUNS):
            traffic = make_traffic(flow_mean, rng)      # same traffic for both strategies
            jitter_seed = int(rng.integers(1_000_000))  # same jitter for both strategies
            for strategy in ["static", "predictive"]:
                result = run_simulation(strategy, traffic, model, jitter_seed)
                result.update({"scenario": scenario, "strategy": strategy})
                rows.append(result)

    all_runs = pd.DataFrame(rows)

    # Average the NUM_RUNS runs for each scenario + strategy
    summary = (
        all_runs.groupby(["scenario", "strategy"], sort=False)[
            ["avg_latency_ms", "packet_loss_pct", "route_changes"]
        ]
        .mean()
        .round(2)
        .reset_index()
    )
    return summary


# ----------------------------------------------------------------------------
# STEP 5: Plots
# ----------------------------------------------------------------------------

def plot_metric(summary, column, title, ylabel, filename):
    """Draw a bar chart: Low/Medium/High on the x-axis, static vs predictive side by side."""
    scenarios = list(TRAFFIC_SCENARIOS.keys())
    x = np.arange(len(scenarios))
    width = 0.35

    static = [summary[(summary.scenario == s) & (summary.strategy == "static")][column].iloc[0]
              for s in scenarios]
    predictive = [summary[(summary.scenario == s) & (summary.strategy == "predictive")][column].iloc[0]
                  for s in scenarios]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, static, width, label="Static routing", color="#9e9e9e")
    bars2 = ax.bar(x + width / 2, predictive, width, label="Predictive routing", color="#1976d2")
    ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_xlabel("Traffic load")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.text(0.5, 0.005, "Synthetic simulation - not a real network", ha="center",
             fontsize=8, style="italic", color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN: run everything in order
# ----------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("NOTE: Dataset is SYNTHETIC and results come from a SIMULATION,")
    print("      not from a real physical network.\n")

    # 1) Dataset
    dataset = generate_dataset(NUM_SAMPLES, rng)
    dataset.to_csv(os.path.join(RESULTS_DIR, "synthetic_dataset.csv"), index=False)
    print(f"Created synthetic dataset: {len(dataset)} rows")
    print(f"Congestion rows: {dataset['congestion'].sum()}  "
          f"No-congestion rows: {(dataset['congestion'] == 0).sum()}\n")
    print(dataset.head())
    print()

    # 2) Model
    model = train_model(dataset)

    # 3 + 4) Simulation and strategy comparison
    print("=" * 60)
    print(f"SIMULATION RESULTS ({NUM_STEPS} steps per run, averaged over {NUM_RUNS} runs)")
    print("=" * 60)
    summary = compare_strategies(model)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(RESULTS_DIR, "comparison_results.csv"), index=False)

    # 5) Plots
    plot_metric(summary, "avg_latency_ms", "Average Latency (lower is better)",
                "Latency (ms)", "latency_comparison.png")
    plot_metric(summary, "packet_loss_pct", "Packet Loss (lower is better)",
                "Packet loss (%)", "packet_loss_comparison.png")
    plot_metric(summary, "route_changes", "Route Changes per Run",
                "Number of route changes", "route_changes_comparison.png")

    print(f"\nSaved plots and CSV files in: {RESULTS_DIR}")
    print("Reminder: these are simulated, synthetic results - not real-world performance.")


if __name__ == "__main__":
    main()
