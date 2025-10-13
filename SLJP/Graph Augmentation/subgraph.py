import networkx as nx
import matplotlib.pyplot as plt
import random

# ======================
# 1. Sample Case Data
# ======================
cases = [
    {
        "case_id": "Case1",
        "facts": ["Murder with knife", "Occurred at night"],
        "statutes": ["IPC 302", "IPC 34"],
        "precedents": ["State vs Ram", "Queen vs Dudley"]
    },
    {
        "case_id": "Case2",
        "facts": ["Theft of gold chain", "No eyewitness"],
        "statutes": ["IPC 379"],
        "precedents": ["State vs Shyam"]
    },
    {
        "case_id": "Case3",
        "facts": ["Dowry harassment", "Suicide of victim"],
        "statutes": ["IPC 304B", "IPC 498A"],
        "precedents": ["State vs Rajesh"]
    },
    {
        "case_id": "Case4",
        "facts": ["Forgery of documents", "Fake property papers"],
        "statutes": ["IPC 465", "IPC 468"],
        "precedents": ["Lal vs State"]
    },
    {
        "case_id": "Case5",
        "facts": ["Bribery in office", "Trap case with police"],
        "statutes": ["Prevention of Corruption Act Sec 7"],
        "precedents": ["CBI vs Anil"]
    },
    {
        "case_id": "Case6",
        "facts": ["Rape allegation", "Victim testimony"],
        "statutes": ["IPC 376"],
        "precedents": ["State vs Mahesh"]
    },
    {
        "case_id": "Case7",
        "facts": ["Acid attack", "Permanent disfigurement"],
        "statutes": ["IPC 326A"],
        "precedents": ["State vs Kiran"]
    },
    {
        "case_id": "Case8",
        "facts": ["Kidnapping of minor", "Demand for ransom"],
        "statutes": ["IPC 364A"],
        "precedents": ["State vs Manoj"]
    },
    {
        "case_id": "Case9",
        "facts": ["Cyber fraud", "Fake bank website"],
        "statutes": ["IT Act Sec 66D"],
        "precedents": ["State vs Ravi"]
    },
    {
        "case_id": "Case10",
        "facts": ["Arson in marketplace", "Property damage"],
        "statutes": ["IPC 436"],
        "precedents": ["State vs Arjun"]
    },
]

# ======================
# 2. Create Graph
# ======================
G = nx.Graph()

for case in cases:
    case_node = case["case_id"]
    G.add_node(case_node, type="case")

    for f in case["facts"]:
        G.add_node(f, type="fact")
        G.add_edge(case_node, f)

    for s in case["statutes"]:
        G.add_node(s, type="statute")
        G.add_edge(case_node, s)

    for p in case["precedents"]:
        G.add_node(p, type="precedent")
        G.add_edge(case_node, p)

# ======================
# 3. Visualization
# ======================
pos = nx.spring_layout(G, seed=42, k=0.8)  # better spacing

# Color nodes by type
color_map = []
for node, data in G.nodes(data=True):
    if data["type"] == "case":
        color_map.append("lightblue")
    elif data["type"] == "fact":
        color_map.append("lightgreen")
    elif data["type"] == "statute":
        color_map.append("salmon")
    elif data["type"] == "precedent":
        color_map.append("violet")
    else:
        color_map.append("grey")

plt.figure(figsize=(14, 10))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=color_map,
    node_size=1200,
    font_size=9,
    font_weight="bold",
    edge_color="gray",
)
plt.title("Graph Representation of Facts, Statutes, and Precedents (10 Sample Cases)", fontsize=14)
plt.show()
