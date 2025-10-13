import torch
import networkx as nx
import matplotlib.pyplot as plt

# Load the saved graph
data = torch.load("global_graph.pt", weights_only=False)

def visualize_case_subgraph(data, fact_id=0, max_neighbors=5):
    """
    Visualize a small subgraph around one fact node:
    Fact → Statutes → Charges
    """
    # Fact → Statute edges
    fact_edges = data["fact", "mentions", "statute"].edge_index
    statutes = fact_edges[1][fact_edges[0] == fact_id].tolist()
    statutes = statutes[:max_neighbors]  # limit for clarity

    # Statute → Charge edges
    statute_charge_edges = data["statute", "implies", "charge"].edge_index
    charges = statute_charge_edges[1][
        [s in statutes for s in statute_charge_edges[0].tolist()]
    ].tolist()

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_node(f"Fact {fact_id}", color="blue")

    for sid in statutes:
        G.add_node(f"Statute {sid}", color="green")
        G.add_edge(f"Fact {fact_id}", f"Statute {sid}")

    for sid in statutes:
        for cid in charges:
            G.add_node(f"Charge {cid}", color="red")
            G.add_edge(f"Statute {sid}", f"Charge {cid}")

    # Draw graph
    colors = [G.nodes[n]["color"] for n in G.nodes]
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color=colors, node_size=1500, font_size=8, arrows=True)
    plt.show()

# Example: visualize fact node 0
visualize_case_subgraph(data, fact_id=0, max_neighbors=5)
