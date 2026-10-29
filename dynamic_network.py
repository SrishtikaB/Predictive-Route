"""Synthetic PredictiveRoute topology: 10 routers and multiple paths."""
import networkx as nx

NODES = ["Source"] + [f"R{i}" for i in range(1, 11)] + ["Destination"]
DEFAULT_LINKS = [
    ("Source", "R1"), ("Source", "R2"),
    ("R1", "R3"), ("R1", "R4"), ("R2", "R4"), ("R2", "R5"),
    ("R3", "R6"), ("R3", "R7"), ("R4", "R6"), ("R4", "R8"),
    ("R5", "R8"), ("R5", "R9"), ("R6", "R7"), ("R6", "R10"),
    ("R7", "R9"), ("R7", "R10"), ("R8", "R9"), ("R8", "R10"),
    ("R9", "Destination"), ("R10", "Destination"),
    ("R3", "R4"), ("R6", "R8"), ("R7", "R8"), ("R9", "R10"),
]

def default_link_table():
    return [{"source": a, "target": b, "capacity_mbps": 100.0,
             "base_latency_ms": float(5 + (i % 5) * 2),
             "background_mbps": float(10 + (i % 4) * 5), "enabled": True}
            for i, (a, b) in enumerate(DEFAULT_LINKS)]

def create_network(link_rows, traffic_mbps=30.0):
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    for row in link_rows:
        a, b = str(row["source"]), str(row["target"])
        if not bool(row.get("enabled", True)) or a not in NODES or b not in NODES or a == b:
            continue
        cap, latency, background = float(row["capacity_mbps"]), float(row["base_latency_ms"]), float(row["background_mbps"])
        if cap <= 0 or latency < 0 or background < 0:
            raise ValueError("Capacity must be > 0; latency and background traffic must be >= 0.")
        load = background + float(traffic_mbps)
        util = 100 * load / cap
        congested = util >= 80
        weight = latency + max(0, util - 50) * 0.15
        if congested:
            weight += 35 + max(0, util - 80) * 0.5
        graph.add_edge(a, b, capacity_mbps=cap, base_latency_ms=latency,
                       background_mbps=background, total_load_mbps=load,
                       utilization_pct=util, congested=congested, weight=weight)
    return graph

def find_best_route(graph, source="Source", destination="Destination"):
    if source not in graph or destination not in graph or not nx.has_path(graph, source, destination):
        raise ValueError(f"No available route from {source} to {destination}. Enable links that connect both endpoints.")
    return nx.shortest_path(graph, source=source, target=destination, weight="weight")

def route_metrics(graph, route):
    edges = list(zip(route, route[1:]))
    if not edges:
        return {"route_cost": 0.0, "estimated_latency_ms": 0.0, "mean_utilization_pct": 0.0, "congested_links": 0, "estimated_loss_pct": 0.0}
    util = [graph[a][b]["utilization_pct"] for a, b in edges]
    return {
        "route_cost": round(sum(graph[a][b]["weight"] for a, b in edges), 2),
        "estimated_latency_ms": round(sum(graph[a][b]["base_latency_ms"] for a, b in edges), 2),
        "mean_utilization_pct": round(sum(util) / len(util), 1),
        "congested_links": sum(graph[a][b]["congested"] for a, b in edges),
        # Demo estimate only; not measured packet loss.
        "estimated_loss_pct": round(sum(max(0, u - 80) * 0.05 for u in util) / len(util), 2),
    }
