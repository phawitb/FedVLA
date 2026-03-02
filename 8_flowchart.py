import graphviz

# =========================================================
# Helpers
# =========================================================
def _common_graph_attrs(dot, rankdir="TB"):
    dot.attr(rankdir=rankdir, size="12,10")
    dot.attr("node", shape="box", style="filled", fontname="Helvetica", fontsize="11")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

def _token_node_label(mod, in_dim, n_tokens, d_model):
    return (
        f"{mod}\n"
        f"Linear({in_dim} → {d_model}×{n_tokens}) + LayerNorm\n"
        f"reshape → [B, {n_tokens}, {d_model}]"
    )

def _seq_label(nv, ns, nt, d_model):
    seq_len = nv + ns + nt
    return (
        "Token Concatenation\n"
        f"[B, {nv}+{ns}+{nt}={seq_len}, {d_model}] + pos_emb"
    )

def _maybe_text_block(dot, use_text: bool, text_dim=512, n_tokens_text=2, d_model=512):
    if use_text:
        dot.node(
            "tfeat",
            f"Text feat (CLIP text)\n[1×{text_dim}]",
            fillcolor="#C6F6D5",
        )
        dot.node(
            "txt_tok",
            _token_node_label("text_to_tokens", text_dim, n_tokens_text, d_model),
            fillcolor="#9AE6B4",
        )
        dot.edge("tfeat", "txt_tok", label="encode")
        return True
    else:
        # show as optional
        dot.node(
            "tfeat",
            "Text feat (optional)\n(not used in this checkpoint)",
            fillcolor="#E2E8F0",
        )
        return False


# =========================================================
# 1) Centralized Training Diagram
# =========================================================
def create_centralized_diagram(
    out_name="diagram_centralized",
    use_text=True,
    vision_dim=512,
    text_dim=512,
    state_dim=39,
    action_dim=4,
    d_model=512,
    n_tokens_vision=4,
    n_tokens_state=2,
    n_tokens_text=2,  # used only if use_text=True
    n_layers=6,
    n_heads=8,
):
    dot = graphviz.Digraph(comment="Centralized Training", format="png")
    _common_graph_attrs(dot, rankdir="TB")

    dot.attr(label="CENTRALIZED TRAINING (Single Machine)", labelloc="t", fontsize="16")

    # Data
    dot.node("data", "Prepared Dataset (.npz)\n(v_feats, t_feats, states, actions)\ntrain_idx/val_idx", shape="cylinder", fillcolor="#CBD5E0")

    # Inputs
    dot.node("vfeat", f"Vision feat (CLIP visual)\n[1×{vision_dim}]", fillcolor="#C6F6D5")
    dot.node("sfeat", f"State (proprioception)\n[1×{state_dim}]", fillcolor="#C6F6D5")

    dot.node("vis_tok", _token_node_label("vision_to_tokens", vision_dim, n_tokens_vision, d_model), fillcolor="#9AE6B4")
    dot.node("st_tok", _token_node_label("state_to_tokens", state_dim, n_tokens_state, d_model), fillcolor="#9AE6B4")

    has_text = _maybe_text_block(dot, use_text, text_dim=text_dim, n_tokens_text=n_tokens_text, d_model=d_model)

    dot.node("seq", _seq_label(n_tokens_vision, n_tokens_state, (n_tokens_text if has_text else 0), d_model), fillcolor="#81E6D9")

    dot.node(
        "trunk",
        f"Transformer Encoder (global)\n{n_layers} layers, {n_heads} heads\ninput: [B, seq_len, {d_model}]\noutput: [B, seq_len, {d_model}]",
        fillcolor="#90CDF4",
    )
    dot.node(
        "pool",
        "Mean Pool over tokens\n[B, seq_len, d_model] → [B, d_model]",
        fillcolor="#BEE3F8",
    )
    dot.node(
        "head",
        f"Action Head\nLinear → GELU → Linear → Tanh\n[B, {d_model}] → [B, {action_dim}]\n(action in [-1,1])",
        fillcolor="#FBB6CE",
    )
    dot.node(
        "loss",
        "Training Loss\nMSE(pred, action)\n(backprop update all params)",
        fillcolor="#F6E05E",
    )

    # edges
    dot.edge("data", "vfeat")
    dot.edge("data", "sfeat")
    dot.edge("vfeat", "vis_tok")
    dot.edge("sfeat", "st_tok")
    dot.edge("vis_tok", "seq")
    dot.edge("st_tok", "seq")
    if has_text:
        dot.edge("txt_tok", "seq")

    dot.edge("seq", "trunk")
    dot.edge("trunk", "pool")
    dot.edge("pool", "head")
    dot.edge("head", "loss")

    dot.render(out_name, cleanup=True)
    print(f"✅ Saved: {out_name}.png")


# =========================================================
# 2) FedAvg Training Diagram (HPTLikePolicy)
# =========================================================
def create_fedavg_diagram(
    out_name="diagram_fedavg",
    use_text=True,
    vision_dim=512,
    text_dim=512,
    state_dim=39,
    action_dim=4,
    d_model=512,
    n_tokens_vision=4,
    n_tokens_state=2,
    n_tokens_text=2,
    rounds=50,
    clients_per_round=3,
    local_epochs=1,
):
    dot = graphviz.Digraph(comment="FedAvg Training", format="png")
    _common_graph_attrs(dot, rankdir="TB")
    dot.attr(label="FEDAVG TRAINING (HPTLikePolicy)", labelloc="t", fontsize="16")

    # Server
    with dot.subgraph(name="cluster_server") as s:
        s.attr(label="Federated Server", style="dashed", color="blue", bgcolor="#F0F7FF")
        s.node("srv_w", "Global Model Weights Wᵍ\n(all params aggregated)\nHPTLikePolicy", fillcolor="#FBBC05")
        s.node("fedavg", "FedAvg Aggregation\nWᵍ ← Σ (n_k / Σn) · W_k\n(weights by #samples)", fillcolor="#4285F4", fontcolor="white")
        s.node("rounds", f"Rounds: {rounds}\nClients/round: {clients_per_round}", fillcolor="#E2E8F0")

    # Clients
    with dot.subgraph(name="cluster_clients") as c:
        c.attr(label="Clients (tasks as clients)", style="rounded", color="green", bgcolor="#F0FFF0")
        c.node("cdata", "Client dataset D_k\n(indices by task_id)", shape="cylinder", fillcolor="#C6F6D5")
        c.node("dl", f"Local Training\nepochs={local_epochs}\noptimizer=Adam\nloss=MSE", fillcolor="#F6E05E")
        c.node("wk", "Local Weights W_k\n(after local updates)", fillcolor="#90CDF4")

        # show token pipeline just once
        c.node("vfeat", f"v_feat [1×{vision_dim}]", fillcolor="#C6F6D5")
        c.node("sfeat", f"state [1×{state_dim}]", fillcolor="#C6F6D5")
        c.node("vis_tok", f"vision_to_tokens → [B,{n_tokens_vision},{d_model}]", fillcolor="#9AE6B4")
        c.node("st_tok", f"state_to_tokens → [B,{n_tokens_state},{d_model}]", fillcolor="#9AE6B4")

        has_text = _maybe_text_block(c, use_text, text_dim=text_dim, n_tokens_text=n_tokens_text, d_model=d_model)

        c.node("seq", f"concat tokens + pos_emb\n[B, seq_len, {d_model}]", fillcolor="#81E6D9")
        c.node("model", "HPTLikePolicy\nTransformer trunk + head", fillcolor="#BEE3F8")

        c.edge("cdata", "vfeat")
        c.edge("cdata", "sfeat")
        c.edge("vfeat", "vis_tok")
        c.edge("sfeat", "st_tok")
        c.edge("vis_tok", "seq")
        c.edge("st_tok", "seq")
        if has_text:
            c.edge("txt_tok", "seq")
        c.edge("seq", "model")
        c.edge("model", "dl")
        c.edge("dl", "wk")

    # Federated links
    dot.edge("srv_w", "dl", label="Broadcast Wᵍ (round start)", color="blue", style="bold")
    dot.edge("wk", "fedavg", label="Upload W_k", color="red", style="bold")
    dot.edge("fedavg", "srv_w", label="Update global Wᵍ", color="blue", style="bold")
    dot.edge("rounds", "fedavg", style="dotted")

    dot.render(out_name, cleanup=True)
    print(f"✅ Saved: {out_name}.png")


# =========================================================
# 3) FedVLA Training Diagram (DGMoE + EDA, trunk-only aggregate)
# =========================================================
def create_fedvla_diagram(
    out_name="diagram_fedvla",
    use_text=True,
    vision_dim=512,
    text_dim=512,
    state_dim=39,
    action_dim=4,
    d_model=512,
    n_tokens_vision=4,
    n_tokens_state=2,
    n_tokens_text=2,
    n_layers=6,
    n_experts=4,
    rounds=50,
    clients_per_round=3,
    local_epochs=3,
):
    dot = graphviz.Digraph(comment="FedVLA Training (DGMoE + EDA)", format="png")
    _common_graph_attrs(dot, rankdir="TB")
    dot.attr(label="FEDVLA TRAINING (DGMoE + EDA, trunk aggregated only)", labelloc="t", fontsize="16")

    # Server cluster
    with dot.subgraph(name="cluster_server") as s:
        s.attr(label="Federated Server", style="dashed", color="blue", bgcolor="#F0F7FF")
        s.node("gtrunk", "Global TRUNK params (aggregated)\ntrunk.* + trunk_norm.*", fillcolor="#FBBC05")
        s.node(
            "eda",
            "EDA Aggregation (layer-wise)\n1) collect selection counts sel_mat[l,k]\n2) cosine-sim across clients per layer\n3) compute weights w(l,i)\n4) aggregate trunk params by layer",
            fillcolor="#4285F4",
            fontcolor="white",
        )
        s.node("rounds", f"Rounds: {rounds}\nClients/round: {clients_per_round}", fillcolor="#E2E8F0")

    # Client cluster
    with dot.subgraph(name="cluster_clients") as c:
        c.attr(label="Clients (each task_id is a client)", style="rounded", color="green", bgcolor="#F0FFF0")

        c.node("pdata", "Client dataset D_k\n(non-IID by task)", shape="cylinder", fillcolor="#C6F6D5")

        c.node(
            "personal",
            "Personal params (kept per client)\nvision_to_tokens, state_to_tokens,\n(text_to_tokens if used), pos_emb, head",
            fillcolor="#FDE68A",
        )

        c.node(
            "trunk",
            f"Trunk blocks (GLOBAL shared init)\nL={n_layers}\nEach block: Self-Attn + DGMoE",
            fillcolor="#BEE3F8",
        )
        c.node(
            "dgmoe",
            f"DGMoE inside each layer\nToken-side gate (Wt + residual Wgt)\nExpert-side threshold gate (We)\nExperts={n_experts} (FFN)",
            fillcolor="#90CDF4",
        )

        c.node("sel", "Selection stats\nsel_mat: [L×K]\n(counts of active experts)", fillcolor="#CBD5E0")

        c.node("localtrain", f"Local training\nepochs={local_epochs}\nloss=Huber (SmoothL1)\nupdate: PERSONAL + TRUNK locally", fillcolor="#F6E05E")

        c.node("upload", "Upload to server:\n(TRUNK params only) + sel_mat", fillcolor="#FEB2B2")

        # Token pipeline
        c.node("vfeat", f"v_feat [1×{vision_dim}]", fillcolor="#C6F6D5")
        c.node("sfeat", f"state [1×{state_dim}]", fillcolor="#C6F6D5")
        c.node("vis_tok", f"vision_to_tokens → [B,{n_tokens_vision},{d_model}]", fillcolor="#9AE6B4")
        c.node("st_tok", f"state_to_tokens → [B,{n_tokens_state},{d_model}]", fillcolor="#9AE6B4")

        has_text = _maybe_text_block(c, use_text, text_dim=text_dim, n_tokens_text=n_tokens_text, d_model=d_model)
        seq_len = n_tokens_vision + n_tokens_state + (n_tokens_text if has_text else 0)
        c.node("seq", f"concat tokens + pos_emb\n[B,{seq_len},{d_model}]", fillcolor="#81E6D9")

        c.node("head", f"Head → action\n[B,{d_model}] → [B,{action_dim}]", fillcolor="#FBB6CE")

        # edges
        c.edge("pdata", "vfeat")
        c.edge("pdata", "sfeat")
        c.edge("vfeat", "vis_tok")
        c.edge("sfeat", "st_tok")
        c.edge("vis_tok", "seq")
        c.edge("st_tok", "seq")
        if has_text:
            c.edge("txt_tok", "seq")

        c.edge("seq", "trunk")
        c.edge("trunk", "dgmoe")
        c.edge("dgmoe", "sel", label="collect sel_mat")
        c.edge("trunk", "head")
        c.edge("head", "localtrain")
        c.edge("personal", "localtrain", style="dotted", label="personalized update")
        c.edge("localtrain", "upload")

    # Fed links
    dot.edge("gtrunk", "trunk", label="Broadcast global trunk", color="blue", style="bold")
    dot.edge("upload", "eda", label="send trunk+sel_mat", color="red", style="bold")
    dot.edge("eda", "gtrunk", label="Update global trunk", color="blue", style="bold")
    dot.edge("rounds", "eda", style="dotted")

    dot.render(out_name, cleanup=True)
    print(f"✅ Saved: {out_name}.png")


# =========================================================
# 4) Inference Diagram (used by your 7_run_sim_cal_success_rate.py)
# =========================================================
def create_inference_diagram(
    out_name="diagram_inference",
    use_text=True,
    vision_dim=512,
    text_dim=512,
    state_dim=39,
    action_dim=4,
    d_model=512,
    n_tokens_vision=4,
    n_tokens_state=2,
    n_tokens_text=2,
    n_layers=6,
    n_experts=4,
):
    dot = graphviz.Digraph(comment="Inference", format="png")
    _common_graph_attrs(dot, rankdir="LR")
    dot.attr(label="INFERENCE (MetaWorld rollout)", labelloc="t", fontsize="16")

    # Inputs
    with dot.subgraph(name="cluster_in") as i:
        i.attr(label="Live Inputs per step", color="gray", bgcolor="#F7FAFC")
        i.node("img", f"Rendered frame (RGB)\n480×480", fillcolor="#CBD5E0")
        i.node("vfeat", f"CLIP Visual encode\n→ v_feat [1×{vision_dim}]", fillcolor="#C6F6D5")
        if use_text:
            i.node("txt", "Instruction prompt\n(from text_inputs_json)", fillcolor="#CBD5E0")
            i.node("tfeat", f"CLIP Text encode\n→ t_feat [1×{text_dim}]", fillcolor="#C6F6D5")
        else:
            i.node("txt", "No text used\n(this checkpoint)", fillcolor="#E2E8F0")
        i.node("sfeat", f"Env observation/state\n[1×{state_dim}]", fillcolor="#C6F6D5")

    # Tokenization
    dot.node("vis_tok", f"vision_to_tokens\n→ [1,{n_tokens_vision},{d_model}]", fillcolor="#9AE6B4")
    dot.node("st_tok", f"state_to_tokens\n→ [1,{n_tokens_state},{d_model}]", fillcolor="#9AE6B4")
    if use_text:
        dot.node("txt_tok", f"text_to_tokens\n→ [1,{n_tokens_text},{d_model}]", fillcolor="#9AE6B4")

    seq_len = n_tokens_vision + n_tokens_state + (n_tokens_text if use_text else 0)
    dot.node("seq", f"concat tokens + pos_emb\n[1,{seq_len},{d_model}]", fillcolor="#81E6D9")

    # Core
    dot.node("trunk", f"Trunk\n{n_layers}×(Attn + DGMoE)\nExperts={n_experts}", fillcolor="#90CDF4")
    dot.node("pool", "mean pool → [1,d_model]", fillcolor="#BEE3F8")
    dot.node("head", f"Head → action\n[1,{action_dim}] in [-1,1]", fillcolor="#FBB6CE")

    # Output
    dot.node("step", "env.step(action)\n→ next obs, info(success)", fillcolor="#F6E05E")

    # edges
    dot.edge("img", "vfeat")
    dot.edge("vfeat", "vis_tok")
    dot.edge("sfeat", "st_tok")
    if use_text:
        dot.edge("txt", "tfeat")
        dot.edge("tfeat", "txt_tok")

    dot.edge("vis_tok", "seq")
    dot.edge("st_tok", "seq")
    if use_text:
        dot.edge("txt_tok", "seq")

    dot.edge("seq", "trunk")
    dot.edge("trunk", "pool")
    dot.edge("pool", "head")
    dot.edge("head", "step")

    dot.render(out_name, cleanup=True)
    print(f"✅ Saved: {out_name}.png")


# =========================================================
# Run all
# =========================================================
if __name__ == "__main__":
    # Set use_text=True if your checkpoint has text_to_tokens / uses CLIP text feature.
    USE_TEXT = True

    create_centralized_diagram(out_name="diagram_centralized", use_text=USE_TEXT)
    create_fedavg_diagram(out_name="diagram_fedavg", use_text=USE_TEXT)
    create_fedvla_diagram(out_name="diagram_fedvla", use_text=USE_TEXT)
    create_inference_diagram(out_name="diagram_inference", use_text=USE_TEXT)