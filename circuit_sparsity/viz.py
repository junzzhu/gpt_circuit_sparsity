import argparse
import concurrent.futures
import functools
import importlib
import json
import os
import pickle
import sys
import threading
import time
from collections import defaultdict
from itertools import islice

import blobfile as bf
import circuit_sparsity.registries
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import torch
from circuit_sparsity.inference.gpt import GPTConfig
from circuit_sparsity.single_tensor_pt_load_slice import read_tensor_slice_from_file
from idemlib import CacheHelper
from natsort import natsorted
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit_plotly_events import plotly_events
from tiktoken import get_encoding

from circuit_sparsity.registries import MODEL_BASE_DIR


BLUE = torch.tensor([0, 0, 255])  # base RGB for nodes / edges
SAMPLES_SHOW = 5

cmap = cm.get_cmap("coolwarm")


def _normalize_viz_dir(path: str) -> str:
    if "://" in path:
        return path
    return os.path.abspath(os.path.expanduser(path))


def _parse_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--viz-dir",
        action="append",
        default=[],
        help="Additional base directories (local or cloud) that contain viz data.",
    )
    args, _ = parser.parse_known_args()
    return [_normalize_viz_dir(p) for p in args.viz_dir]


EXTRA_VIZ_DIRS = _parse_cli_args()


def list_join(xss: list[list]) -> list:
    """monadic join for lists"""
    return [x for xs in xss for x in xs]


def get_highlighted_code(code_tokens, mask):
    # Build an HTML string with highlighted spans for masked tokens
    highlighted_code = ""
    for token, is_masked in zip(code_tokens, mask):
        # Convert literal '\n' tokens to real newlines in the final string
        # so the <pre> block can interpret them.
        if token == "\n":
            # Just append a real newline char
            token_content = "\n"
        else:
            token_content = token

        if is_masked:
            # Wrap token in a highlighted <span>
            highlighted_code += (
                f'<span style="background-color: yellow;">{token_content}</span>'
            )
        else:
            highlighted_code += token_content

    return highlighted_code


def get_display_snippet(highlighted_code):
    return f"""
        <pre style="flex-direction: column-reverse; font-family: initial; font-size: initial; line-height: 1; margin: 0; font-weight: initial; overflow: hidden;"><code class="python">{highlighted_code}
</code></pre>"""


def display_code_heatmap(
    tokens,
    mask,
    minmax=None,
    highlight_idx=None,
    highlight_idx2=None,
    center_zero=False,
):
    if minmax:
        norm = plt.Normalize(minmax[0], minmax[1])
    else:
        if center_zero:
            extreme = max(abs(min(mask)), abs(max(mask)))
            norm = plt.Normalize(-extreme, extreme)
        else:
            norm = plt.Normalize(min(mask), max(mask))
    colors = cmap(norm(mask.float().cpu()))
    if highlight_idx == "max":
        largest = mask.max()
        second_largest = torch.sort(mask).values[-2]
        second_smallest = torch.sort(mask).values[1]
        smallest = mask.min()
        stddev = mask.std()

        if largest - second_largest > stddev * 0.2:
            highlight_idx = np.argmax(mask)
        if second_smallest - smallest > stddev * 0.2:
            highlight_idx2 = np.argmin(mask)

    html = (
        '<div style="white-space: pre-wrap; color:black; '
        'background:white; font-family:monospace; border: 1px solid grey;">'
    )

    for i, (tok, rgba, maskval) in enumerate(zip(tokens, colors, mask, strict=True)):
        r, g, b, a = (
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
            rgba[3],
        )

        # Default background style
        style = f"background: rgba({r},{g},{b},{a:.4f});"

        # Add bounding box if this is the highlighted token
        if highlight_idx is not None and i == highlight_idx:
            if highlight_idx2 is None:
                style += " border:2px solid #000000; border-radius:3px; padding:1px;"
            else:
                style += " border:2px solid #ff0000; border-radius:3px; padding:1px;"

        if highlight_idx2 is not None and i == highlight_idx2:
            style += " border:2px solid #0000ff; border-radius:3px; padding:1px;"

        html += f'<span style="{style}" title="activation: {maskval:.4f}">{tok}</span>'

    html += "</div>"
    return html


def rebin_hist(
    counts: torch.Tensor,
    *,
    bin_start: float = -20.0,
    bin_width: float = 0.01,
    n_out: int = 50,
):
    """
    Re‑bin a 4000‑element histogram (bin_width = 0.01) into ≤ n_out bars
    after trimming empty tails.
    Returns (left_edges, heights, widths) ready for plt.bar(..., align='edge').
    """

    counts = counts.numpy()

    # 1. Trim zero tails ---------------------------------------------------
    nz = np.flatnonzero(counts)
    if nz.size == 0:  # completely empty
        return np.empty(0), np.empty(0), np.empty(0)

    lo, hi = nz[0], nz[-1] + 1  # hi is exclusive
    trimmed = counts[lo:hi]
    edges = bin_start + np.arange(lo, hi + 1) * bin_width  # len = trimmed.size + 1

    m = trimmed.size
    n_bins = min(n_out, m)  # never ask for more bars than we have bins

    # 2. Contiguous grouping with roughly equal #bins per group -------------
    split_idx = np.linspace(0, m, n_bins + 1, dtype=int)

    left_edges = np.empty(n_bins, dtype=float)
    heights = np.empty(n_bins, dtype=trimmed.dtype)
    widths = np.empty(n_bins, dtype=float)

    for k in range(n_bins):
        lo_k, hi_k = split_idx[k], split_idx[k + 1]  # [lo_k, hi_k)
        heights[k] = trimmed[lo_k:hi_k].sum()
        left_edges[k] = edges[lo_k]
        widths[k] = edges[hi_k] - edges[lo_k]

    left_edges = torch.from_numpy(left_edges)
    heights = torch.from_numpy(heights)
    widths = torch.from_numpy(widths)

    return left_edges, heights, widths


def load_graph_data(importances):
    layers = list(importances["ch_interv_losses"].values())  # [-4:-1]
    row_names = list(importances["ch_interv_losses"].keys())  # [-4:-1]
    pair_data = importances["pair_data"]
    pair_data_connections = importances["pair_data_connections"]
    locs = importances["ch_interv_losses"]

    # get deltas
    layer_imps = {
        k: x - importances["interv_loss"]  / 2  # every single stored task loss number everywhere is off by a factor of 2 for stupid reasons.
        for k, x in importances["layer_interv_losses"].items()
    }
    task_samples = importances["task_samples"]

    layers = [[y / 2.0 for y in x] for x in layers]  # every single stored task loss number everywhere is off by a factor of 2 for stupid reasons.

    # hotpatch for a bug in one very specific version of the importances data
    # where i forgot to list_join across ranks. should be able to remove eventually
    if not isinstance(pair_data[0][0], np.ndarray) and not isinstance(
        pair_data[0][0], torch.Tensor
    ):
        pair_data = list_join(pair_data)

    # edges are actually deltas, but node values aren't, so we need to subtract baselines from node values later in viz_3
    
    return (
        layers,
        row_names,
        pair_data,
        pair_data_connections,
        locs,
        layer_imps,
        task_samples,
    )


cache = CacheHelper(os.path.expanduser("~/data/dev/shm/cache"))
memcache = CacheHelper(None)


@st.cache_resource(show_spinner=False)  # avoids re‑loading on every rerun
# @cache("load_data_v5")
def load_data(
    viz_data_path,
):
    """Load the big blobs just once (cached)."""
    if viz_data_path.endswith(".pt"):
        with bf.BlobFile(viz_data_path, "rb") as f:
            viz_data = torch.load(f, weights_only=False, map_location="cpu")
    elif viz_data_path.endswith(".pkl"):
        with bf.BlobFile(viz_data_path, "rb") as f:
            viz_data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported viz data extension: {viz_data_path}")

    def _load_config(config_class, config_dict):
        import inspect

        param_names = inspect.signature(config_class).parameters.keys()
        return config_class(**{k: v for k, v in config_dict.items() if k in param_names})

    importances = viz_data.get("importances")
    if (
        isinstance(importances, dict)
        and isinstance(importances.get("beeg_model_config"), dict)
    ):
        importances["beeg_model_config"] = _load_config(
            GPTConfig, importances["beeg_model_config"]
        )

    return viz_data


def rgba_blue(strength: float, alpha: float = 1.0) -> str:
    """Map a scalar in [0,1] → rgba string, darker = weaker, bright = strong."""
    if strength == 0:
        return "rgba(0,0,0,0)"
    rgb = (BLUE * abs(strength)).to(torch.int)
    return f"rgba(0,0,255,{abs(strength):.4f})"  # scale channel intensities


def layout_rows(row_lengths, custom_layouts_dict=None):
    """Return x, y coordinates for each dot given variable row lengths."""
    xs, ys = [], []
    custom_layouts_dict = custom_layouts_dict or {}
    for y, n in enumerate(row_lengths):
        if y in custom_layouts_dict:
            custom_xs = custom_layouts_dict[y]
            xs.extend(custom_xs)
            ys.extend(np.full(n, y))
        else:
            xs.extend(np.arange(n))  # simple left→right layout
            ys.extend(np.full(n, y))
    return np.array(xs), np.array(ys)


def get_new_v(y, my_head_id, q_heads, v_heads): ...


def layout_rows_attn(row_lengths):
    xs, ys = [], []
    xs.extend(np.arange(row_lengths[0]))
    ys.extend(np.full(row_lengths[0], 0))

    z = sum(row_lengths[1:4])


mpl_lock = threading.Lock()


def matplotlib_lock(func):
    """Decorator to ensure that matplotlib plotting is thread-safe."""

    def wrapper(*args, **kwargs):
        with mpl_lock:
            return func(*args, **kwargs)

    return wrapper


def build_figure(
    row_strengths,
    row_names,
    edge_strengths,
    choice,
    point_id,
    d_head,
    n_head=None,
    node_indices=None,
    max_edge_val=1,
    baseline=0,
):
    """
    row_strengths : list[list[float]]  # per-node scalar ∈ [0,1]
    edge_strengths: list[np.ndarray]   # between row i & i+1 (shape n_i × n_{i+1})
    node_indices: dict[str, list[int]]
    choice: str (e.g 0.mlp)
    """

    abs_edge_strengths = True
    mlp_locs = [
        f"{choice}.act_in",
        f"{choice}.post_act",
        f"{choice}.resid_delta",
    ]
    attn_locs = [
        f"{choice}.act_in",
        f"{choice}.q",
        f"{choice}.k",
        f"{choice}.v",
        f"{choice}.resid_delta",
    ]
    attn_pairs = [
        (f"{choice}.act_in", f"{choice}.q"),
        (f"{choice}.act_in", f"{choice}.k"),
        (f"{choice}.act_in", f"{choice}.v"),
        (f"{choice}.v", f"{choice}.resid_delta"),
    ]


    node_names = {}
    for _, src_names, tgt_names, (loc_src, loc_tgt) in edge_strengths:
        if not loc_src.startswith(choice) or not loc_tgt.startswith(choice):
            continue
        if loc_src in node_names and len(src_names) > len(node_names[loc_src]):
            node_names[loc_src] = src_names
        elif loc_src not in node_names:
            node_names[loc_src] = src_names

        if loc_tgt in node_names and len(tgt_names) > len(node_names[loc_tgt]):
            node_names[loc_tgt] = tgt_names
        elif loc_tgt not in node_names:
            node_names[loc_tgt] = tgt_names

    if not node_names:
        st.code("some location here has no nodes! skipping")
        return None, None, None, None

    for loc, idxs in node_names.items():
        if len(idxs) == 0:
            continue
        if idxs and idxs[-1] == "bias":
            idxlen = len(idxs) - 1
        else:
            idxlen = len(idxs)
        locidx = (attn_locs if choice.split(".")[1] == "attn" else mlp_locs).index(loc)
        if locidx < 0:
            st.code(
                f"Location {loc} not found in {attn_locs if choice.split('.')[1] == 'attn' else mlp_locs}"
            )
        assert idxlen == len(row_strengths[locidx]), (
            f"{loc=} {len(idxs)=} {len(row_strengths[row_names.index(loc)])=}"
        )

    if choice.split(".")[1] == "mlp":
        row_strengths = [
            l
            for i, l in enumerate(row_strengths)
            if row_names[i]
            in [f"{choice}.act_in", f"{choice}.post_act", f"{choice}.resid_delta"]
        ]
        # st.code(f"{edge_strengths=}, {choice=}")
    
        assert all(isinstance(x[-1], tuple) for x in edge_strengths), f"{edge_strengths=}"

        edge_strengths = [
            x[0]
            for x in edge_strengths
            if x[-1]
            in [
                (f"{choice}.act_in", f"{choice}.post_act"),
                (f"{choice}.post_act", f"{choice}.resid_delta"),
            ]
        ]
        # st.code(f"{edge_strengths=}")

        row_lengths = [len(r) for r in row_strengths]
        X, Y = layout_rows(row_lengths)
        node_colors = [rgba_blue(s - baseline) for row in row_strengths for s in row]

        if abs_edge_strengths:
            edge_strengths = [np.abs(x[:, :]) for x in edge_strengths]
        disp_layouts_dict = {}
    elif choice.split(".")[1] == "attn":
        row_strengths = [
            row_strengths[row_names.index(row_name)] for row_name in attn_locs
        ]
        # st.code(f"{edge_strengths=}")
        edge_strengths_tgt_indexed = {
            x[-1][1]: x[0]
            for x in edge_strengths
            if x[-1]  # (src, tgt)
            in attn_pairs
        }

        def only_one(xs):
            assert len(xs) == 1, f"Expected one element, got {xs}"
            return xs[0]

        try:
            q_ids = node_names[
                only_one([k for k in node_names.keys() if k[-2:] == ".q"])
            ]
            k_ids = node_names[
                only_one([k for k in node_names.keys() if k[-2:] == ".k"])
            ]
            v_ids = node_names[
                only_one([k for k in node_names.keys() if k[-2:] == ".v"])
            ]
        except AssertionError:
            st.warning("No nodes found in this layer.")
            return None, None, None, None

        v_ids = [vid for vid in v_ids if vid != "bias"]

        head_offsets = []
        cur_offset = 0
        for headidx in range(n_head):
            head_offsets.append(cur_offset)
            delta = max(
                len([i for i, qid in enumerate(q_ids) if qid // d_head == headidx]),
                len([i for i, kid in enumerate(k_ids) if kid // d_head == headidx]),
                len([i for i, vid in enumerate(v_ids) if vid // d_head == headidx]),
            )

            cur_offset += delta
            if delta > 0:
                cur_offset += 2

        def ids_to_xpos(ids, padding=0):
            xpos = []
            i = 0
            numnonemptyheads = 0
            lasthead = -1
            for id in ids:
                headidx = id // d_head
                if headidx != lasthead:
                    lasthead = headidx
                    i = 0
                xp = head_offsets[headidx] + i
                xpos.append(xp)
                i += 1
            return xpos

        q_xpos = ids_to_xpos(q_ids)
        k_xpos = ids_to_xpos(k_ids)
        v_xpos = ids_to_xpos(v_ids)

        row_lengths = [len(r) for r in row_strengths]
        X, Y = layout_rows(row_lengths)
        disp_layouts_dict = {
            1: q_xpos,
            2: k_xpos,
            3: v_xpos,
        }
        node_colors = [rgba_blue(s - baseline) for row in row_strengths for s in row]

        if abs_edge_strengths:
            edge_strengths_tgt_indexed = {
                k: np.abs(x[:, :]) for k, x in edge_strengths_tgt_indexed.items()
            }


    # --- Nodes (Scattergl markers) ------------------------------------------

    # --- Edges (Scattergl lines) -------------------------------------------
    min_edge_val = 0  # TODO: implement this
    bins = np.linspace(min_edge_val, max_edge_val + 1e-4, 17)  # 8 bins → 8 traces
    fig = go.Figure()

    print("POINT ID", point_id)

    for ni, (x, y) in enumerate(zip(X, Y)):
        locs = mlp_locs if choice.split(".")[1] == "mlp" else attn_locs
        attn_info = ""
        if "attn" in choice and ("q" in locs[y] or "k" in locs[y] or "v" in locs[y]):
            idx = node_names[locs[y]][x]
            attn_info = f" (h{idx // d_head}.ch{idx % d_head})"
        fig.add_trace(
            go.Scatter(
                x=[disp_layouts_dict[y][x] if y in disp_layouts_dict else x],
                y=[y],
                mode="markers",
                marker=dict(
                    size=22,
                    color=node_colors[ni],
                    line=dict(
                        width=1,
                        color="black",  # if (x, y) != (point_id[1], point_id[0]) else "red"
                    ),
                ),
                customdata=[(ni, x, y)],  # 🆕 payload for click
                hovertemplate=(
                    f"idx{node_names[locs[y]][x]}{attn_info} ablation importance Δ {float(node_colors[ni].split(',')[-1][:-1]):.4f}"
                ),  # cut out the last value from this rgba(0,0,0,0.567) thing
                showlegend=False,
            )
        )

    annotations = []

    inconns = defaultdict(list)
    outconns = defaultdict(list)
    for k in range(len(bins) - 1):
        lo, hi = bins[k], bins[k + 1]
        x_seg, y_seg = [], []

        # collect edges whose strength is in this bin
        if choice.split(".")[1] == "mlp":
            for r in range(len(row_lengths) - 1):
                if r >= len(edge_strengths):
                    continue
                nonzero_ijs = edge_strengths[r].nonzero()
                for i, j in nonzero_ijs:
                    s = edge_strengths[r][i, j]
                    if s == 0:
                        continue
                    if lo <= s < hi:
                        x_seg += [i, j, None]
                        y_seg += [r, r + 1, None]
                        inconns[(locs[r + 1], node_names[locs[r + 1]][j])].append(
                            node_names[locs[r]][i]
                        )
                        outconns[(locs[r], node_names[locs[r]][i])].append(
                            node_names[locs[r + 1]][j]
                        )

        elif choice.split(".")[1] == "attn":
            for tgt_idx in range(1, len(row_lengths)):  # which target?
                target_name = row_names[tgt_idx]
                source_name = {y: x for x, y in attn_pairs}[target_name]
                source_idx = row_names.index(source_name)
                edge_strengths_tgt_indexed_shapes = {
                    k: v.shape for k, v in edge_strengths_tgt_indexed.items()
                }
                # st.code(f"{edge_strengths_tgt_indexed_shapes=}")
                nonzero_ijs = edge_strengths_tgt_indexed[target_name].nonzero()
                for i, j in nonzero_ijs:
                    if i == edge_strengths_tgt_indexed[target_name].shape[0]:
                        # bias has no in-connections
                        continue
                    if abs_edge_strengths:
                        s = abs(edge_strengths_tgt_indexed[target_name][i, j])
                    else:
                        s = edge_strengths_tgt_indexed[target_name][i, j]
                    if s == 0:
                        continue
                    if lo <= s < hi:
                        x_seg += [i, j, None]
                        y_seg += [source_idx, tgt_idx, None]
                        inconns[(source_name, node_names[source_name][i])].append(
                            node_names[target_name][j]
                        )
                        outconns[(target_name, node_names[target_name][j])].append(
                            node_names[source_name][i]
                        )

        if x_seg:  # skip empty bins
            for i in range(0, len(x_seg) - 2, 3):
                # add arrowheads

                annotations.append(
                    dict(
                        ax=disp_layouts_dict[y_seg[i]][x_seg[i]]
                        if y_seg[i] in disp_layouts_dict
                        else x_seg[i],
                        ay=y_seg[i],
                        axref="x",
                        ayref="y",
                        x=disp_layouts_dict[y_seg[i + 1]][x_seg[i + 1]]
                        if y_seg[i + 1] in disp_layouts_dict
                        else x_seg[i + 1],
                        y=y_seg[i + 1],
                        xref="x",
                        yref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor=rgba_blue((lo + hi) / 2 / max_edge_val, alpha=0.3),
                    )
                )

    fig.update_layout(
        xaxis=dict(visible=False),  # no axis line, ticks, or labels
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="closest",  # optional: remove extra padding
        annotations=annotations,
    )
    if choice.split(".")[1] == "mlp":
        fig.update_yaxes(
            tickmode="array",  # use the arrays below
            tickvals=[0, 1, 2],  # where the ticks sit
            ticktext=mlp_locs,  # what text to show
        )
    elif choice.split(".")[1] == "attn":
        fig.update_yaxes(
            tickmode="array",  # use the arrays below
            tickvals=[0, 1, 2, 3, 4],  # where the ticks sit
            ticktext=attn_locs,  # what text to show
        )
    fig.update_yaxes(
        showline=False, showgrid=False, zeroline=False, showticklabels=True
    )
    fig.update_yaxes(automargin=True)

    cur_selected = None

    def change_highlighted(point_id):
        nonlocal cur_selected

        # remove previously selected point
        if cur_selected is not None:
            fig.data.remove(cur_selected)

        newtrace = go.Scatter(
            x=[point_id[1]],
            y=[point_id[0]],
            mode="markers",
            marker=dict(
                size=22,
                color=node_colors[ni],
                line=dict(width=1, color="red"),
            ),
            customdata=[(ni, x, y)],  # 🆕 payload for click
            hovertemplate=(
                f"idx{node_names[locs[y]][x]}{attn_info} ablation importance {float(node_colors[ni].split(',')[-1][:-1]):.4f}"
            ),  # cut out the last value from this rgba(0,0,0,0.567) thing
            showlegend=False,
        )
        fig.add_trace(newtrace)
        cur_selected = newtrace

    return fig, change_highlighted, inconns, outconns


def jacob_viz(
    viz_data,
):
    retained_nodes = viz_data["circuit_data"]
    samples_dict = viz_data["samples"]
    importances = viz_data["importances"]

    model_config = importances["beeg_model_config"]

    enc = get_encoding(model_config.tokenizer_name)

    rows, row_names, edges, edges_weights, locs, layer_imps, task_data = (
        load_graph_data(importances)
    )

    tab_plot, tab_info = st.columns([1, 1])

    if "click_data" not in st.session_state:
        st.session_state.click_data = 0

    def _on_choice_change():
        st.session_state.click_data = 0

    with tab_plot:
        n_layer = importances["beeg_model_config"].n_layer
        layerz = [f"{i}.mlp" for i in range(n_layer)] + [
            f"{i}.attn" for i in range(n_layer)
        ]
        n_nodes = {
            l: sum(len(w) for k, w in retained_nodes.items() if k.startswith(l))
            for l in layerz
        }

        layer_options = [
            f"{l}: (ablation delta: {layer_imps[l]:.4f}; nodes {n_nodes[l]}) 🔵"
            if layer_imps[l] > 0.3
            else f"{l}: (ablation delta: {layer_imps[l]:.4f}; nodes {n_nodes[l]})"
            for l in layerz
        ]
        layer_options = natsorted(layer_options)
        layer_col, heatmap_toggle_col, conns_or_impacts_col = st.columns([3, 1, 1])
        idx = 0
        for opt in layer_options:
            if "🔵" in opt:
                idx = layer_options.index(opt)
                break
        with layer_col:
            choice = st.selectbox(
                label="layer",
                options=layer_options,
                index=idx,
                on_change=_on_choice_change,
            )
        #     )

        choice = choice.split(": ")[0]

        # edges format: [arr, retained_nodes[src].tolist(), retained_nodes[tgt].tolist(), (src, tgt)]
        if choice.split(".")[1] == "mlp":
            row_strengths = [
                l
                for i, l in enumerate(rows)
                if row_names[i]
                in [f"{choice}.act_in", f"{choice}.post_act", f"{choice}.resid_delta"]
            ]
            row_name_choices = [f"{choice}.act_in", f"{choice}.post_act", f"{choice}.resid_delta"]
            edge_strengths = [
                x[0]
                for x in edges
                if x[-1]
                in [
                    (f"{choice}.act_in", f"{choice}.post_act"),
                    (f"{choice}.post_act", f"{choice}.resid_delta"),
                ]
            ]
        elif choice.split(".")[1] == "attn":
            row_strengths = [
                l
                for i, l in enumerate(rows)
                if row_names[i]
                in [
                    f"{choice}.act_in",
                    f"{choice}.q",
                    f"{choice}.k",
                    f"{choice}.v",
                    f"{choice}.resid_delta",
                ]
            ]
            row_name_choices = [
                f"{choice}.act_in",
                f"{choice}.q",
                f"{choice}.k",
                f"{choice}.v",
                f"{choice}.resid_delta",
            ]

            edge_strengths = [
                x[0]
                for x in edges
                if x[-1]
                in [
                    (f"{choice}.act_in", f"{choice}.q"),
                    (f"{choice}.act_in", f"{choice}.k"),
                    (f"{choice}.act_in", f"{choice}.v"),
                    (f"{choice}.v", f"{choice}.resid_delta"),
                ]
            ]
        else:
            assert 0
        print(
            [x[0] for x in edges][0],
            "AAAAAASSEDFDSGDS",
            len(edges),
            len(edges[0]),
            len(edges[0][0]),
        )
        max_edge_val = max([(x[0].abs().max() if x[0].numel() > 0 else 0.0) for x in edges] + [0])
        try:
            edge_strengths = [x.unsqueeze(0) if x.numel() == 0 else x for x in edge_strengths]
            max_edge_val_this_layer = max(
                [(x[0].abs().max() if x[0].numel() > 0 else 0.0) for x in edge_strengths] + [0]
            )
        except ValueError:
            max_edge_val_this_layer = 0.0

        curve_dict = {}
        idx = 0
        for n in range(len(row_strengths)):
            for m in range(len(row_strengths[n])):
                curve_dict[idx] = [n, m]
                idx += 1

        # slider
        edgevar = edges  # if choice.split(".")[1] == "mlp" else edge_strengths


        thresh = st.slider(
            label="edge strength threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,  # 5 if show_impacts else 0.0,
            step=0.01,
            on_change=_on_choice_change,
        )

        def _recursive_map(f, x):
            if isinstance(x, dict):
                return {k: _recursive_map(f, v) for k, v in x.items()}
            elif isinstance(x, list):
                return [_recursive_map(f, v) for v in x]
            else:
                return f(x)

        edgevar = _recursive_map(
            lambda x: torch.where(x.abs() > thresh, x, 0.0) if isinstance(x, torch.Tensor) else x,
            edgevar,
        )

        def _display_loss(x):
            # all the stored loss numbers are off by a factor of 2 for stupid reasons. so we divide by 2
            x /= 2.0

            if x < 0.5:
                return f"<b>{x:.4f}</b>"
            else:
                return f"<b><span style='color: red;'>{x:.4f}</span> {warning_emoji}</b>"

        baseline_loss = importances["loss"]
        pruned_loss = importances["interv_loss"]
        warning_emoji = "⚠️"
        st.html(
            f"loss before pruning={_display_loss(baseline_loss)}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loss after pruning={_display_loss(pruned_loss)}"
        )
        fig, change_highlighted, inconn, outconn = build_figure(
            row_strengths,
            row_name_choices,
            edgevar,
            choice,
            None,  # curve_dict[st.session_state.click_data],
            max_edge_val=max_edge_val,
            d_head=importances["beeg_model_config"].d_head,
            n_head=importances["beeg_model_config"].n_head,
            baseline=pruned_loss,
        )
        if fig is None:
            return

        click_data = plotly_events(fig, click_event=True, hover_event=False)
        st.text(f"{max_edge_val=:.4f}   {max_edge_val_this_layer=:.4f}")

        # make heatmap of neuron indegrees and outdegrees
        degs = defaultdict(lambda: [None, None])  # indegree, outdegree
        for (neurloc, neuridx), connlist in inconn.items():
            if "mlp.post_act" not in neurloc:
                continue
            degs[neuridx][0] = len(connlist)
        for (neurloc, neuridx), connlist in outconn.items():
            if "mlp.post_act" not in neurloc:
                continue
            degs[neuridx][1] = len(connlist)

        st.code(f"{degs=}")

    with tab_info:
        if click_data:
            st.session_state.click_data = click_data[0]["curveNumber"]
        try:
            point_id = curve_dict[st.session_state.click_data]
        except KeyError:
            # force reload
            try:
                st.session_state.click_data = click_data[0]["curveNumber"]
            except IndexError:
                st.session_state.click_data = 0
            point_id = curve_dict[st.session_state.click_data]

        change_highlighted(point_id)

        composite_key = row_name_choices[point_id[0]]
        docs = samples_dict.get(composite_key)
        # HACK
        docs = {(k.item() if isinstance(k, torch.Tensor) else k): v for k, v in docs.items()}
        key3_selection = int(retained_nodes[composite_key][point_id[1]])
        print(docs.keys())
        print(retained_nodes[composite_key])
        print(key3_selection)
        print(len(docs[key3_selection]))
        key3_index = point_id[1]

        # composite_key example: 2.mlp.post_act
        # key3_selection: 407

        additional_info = ""
        if "attn.q" in composite_key or "attn.k" in composite_key or "attn.v" in composite_key:
            nheads = importances["beeg_model_config"].d_head
            additional_info = f" (h{key3_selection // nheads}.ch{key3_selection % nheads})"
        st.write(f"**{composite_key}.{key3_selection}{additional_info}**")
        st.text(
            f"input connections: {inconn[(composite_key, key3_selection)]} output connections: {outconn[(composite_key, key3_selection)]}"
        )

        col1, col2, col3 = st.columns([1, 1.25, 0.75])

        with col1:
            sample_choice = st.selectbox(
                label="sample distribution",
                options=["pretraining", "task distribution"],
                index=0,
            )

        if docs is None or key3_selection not in docs:
            st.info("No samples found for this key / channel.")
        else:
            # -----------------------------------------------------------------
            # Controls specific to the document viewer
            # -----------------------------------------------------------------
            dir_options = sorted(docs[key3_selection].keys())

            # Initialise Streamlit session state variables
            if "doc_direction" not in st.session_state:
                st.session_state.doc_direction = dir_options[0]

            if "doc_idx" not in st.session_state:
                st.session_state.doc_idx = 0

            if "sample_doc" not in st.session_state:
                st.session_state.sample_doc = 0

            print([len(k) for k in docs[key3_selection][st.session_state.doc_direction]])
            print("direction", st.session_state.doc_direction)
            print("doc idx", st.session_state.doc_idx)
            print("sample doc", st.session_state.sample_doc)
            print(
                "available",
                len(docs[key3_selection][st.session_state.doc_direction][st.session_state.doc_idx]),
            )

            sample_dict = {"0.1%": 0, "1%": 1, "10%": 2, "50%": 3}
            sample_dict = {0: "0.1%", 1: "1%", 2: "10%", 3: "50%"}

            fracs = [0.001, 0.01, 0.1, 0.5]

            if sample_choice == "pretraining":
                # docs[key3_selection][frac][0][:5] is a list; each element of that list is a 3-tuple (doc, act, ???)
                minmax_dat = torch.stack(
                    list_join(
                        [
                            [x[1] for x in docs[key3_selection][frac][top_or_bottom][:5]]
                            for frac in fracs
                            for top_or_bottom in [0, 1]
                        ]
                    )
                )
                minmax = (minmax_dat.min(), minmax_dat.max())
                st.text(f"color scale: min={minmax[0]:.4f}, max={minmax[1]:.4f}")

                new_col1, new_col2 = st.columns([1, 1])
                with new_col1:
                    for frac in fracs:
                        st.write(f"bottom {frac * 100}%")
                        for show_doc in docs[key3_selection][frac][0][:5]:
                            # The sample structure assumed as before
                            show_doc_tokens = [
                                decode_single_token(enc, t)
                                for t in show_doc[0].int()[
                                    max(0, show_doc[2] - 50) : show_doc[2] + 50
                                ]
                            ]
                            highlighted_code = display_code_heatmap(
                                show_doc_tokens,
                                show_doc[1][max(0, show_doc[2] - 50) : show_doc[2] + 50],
                                minmax=minmax,
                                highlight_idx=min(50, show_doc[2]),
                            )

                            st.components.v1.html(
                                get_display_snippet(highlighted_code), height=120, scrolling=True
                            )
                with new_col2:
                    for frac in fracs:
                        st.write(f"top {frac * 100}%")
                        for show_doc in docs[key3_selection][frac][1][:5]:

                            # The sample structure assumed as before
                            show_doc_tokens = [
                                decode_single_token(enc, t)
                                for t in show_doc[0].int()[
                                    max(0, show_doc[2] - 50) : show_doc[2] + 50
                                ]
                            ]
                            highlighted_code = display_code_heatmap(
                                show_doc_tokens,
                                show_doc[1][max(0, show_doc[2] - 50) : show_doc[2] + 50],
                                minmax=minmax,
                                highlight_idx=min(50, show_doc[2]),
                            )

                            st.components.v1.html(
                                get_display_snippet(highlighted_code), height=120, scrolling=True
                            )
            elif sample_choice == "task distribution":
                task_docs = task_data[0]
                task_acts = task_data[1][composite_key][key3_selection]

                def _center_zero_toggle(val=True, disabled=False):
                    return st.toggle(
                        label="zero centered color scale",
                        value=val,
                        on_change=_on_choice_change,
                        disabled=disabled,
                    )

                if "attn" in choice:
                    with col2:
                        attn_pattern_radio = st.radio(
                            label="token heatmap values",
                            options=["acts", "channel attn", "head attn"],
                            index=0,
                            on_change=_on_choice_change,
                            horizontal=True,
                        )

                    with col3:
                        center_zero_toggle = _center_zero_toggle(
                            disabled=attn_pattern_radio != "acts"
                        )  # ensure_k_positive and not attn_pattern_toggle)

                else:
                    attn_pattern_radio = "acts"
                    center_zero_toggle = _center_zero_toggle()

                task_acts = task_data[1][composite_key][key3_selection]

                ls = []
                for doc_idx in range(len(task_docs)):
                    # HACKY and slow, todo: replace
                    show_doc = task_docs[doc_idx]
                    l = min(
                        [i for i in range(len(show_doc)) if torch.all(show_doc.cpu()[i:] == 0)]
                        + [len(show_doc)]
                    )

                    ls.append(l)

                if attn_pattern_radio in ["head attn", "channel attn"]:
                    my_head_idx = key3_selection // importances["beeg_model_config"].d_head
                    st.code(f"{my_head_idx=}")

                    # task_data[1]: loc -> ch -> tensor(doc, ctx)
                    qs = task_data[1][choice + ".q"]
                    ks = task_data[1][choice + ".k"]

                    if qs.keys() != ks.keys():
                        st.code(
                            f"WARN: qs and ks should have the same keys! {qs.keys() - ks.keys()=}, {ks.keys() - qs.keys()=}"
                        )

                    keys = list(qs.keys() & ks.keys())
                    d_head = importances["beeg_model_config"].d_head
                    keys = [k for k in keys if k // d_head == my_head_idx]
                    if len(keys) == 0:
                        st.text("this head is pruned!")
                    else:
                        qs = torch.stack(
                            [qs[k].cpu().detach() for k in keys],
                            dim=-1,
                        ).to(torch.float32)
                        ks = torch.stack(
                            [ks[k].cpu().detach() for k in keys],
                            dim=-1,
                        ).to(torch.float32)


                        att = torch.einsum(
                            "ijk,iJk->ijJ",
                            qs,
                            ks,
                        )


                        # so many assumptions in this code!!! todo: refactor
                        attention_scale = 1 / np.sqrt(d_head)

                        n_tokens = att.shape[-1]
                        att = att * attention_scale
                        mask = torch.tril(torch.ones(n_tokens, n_tokens)).view(
                            1, n_tokens, n_tokens
                        )
                        att = att.masked_fill(mask == 0, float("-inf"))

                        att_log_denom = torch.logsumexp(att, dim=-1, keepdim=True)
                        attn_logit_minmax = (-att.max(), att.max())
                        att = (att - att_log_denom).exp()



                        if attn_pattern_radio == "channel attn":
                            channel = [
                                x
                                for x in retained_nodes[composite_key].tolist()
                                if d_head * my_head_idx <= x < d_head * (my_head_idx + 1)
                            ].index(key3_selection)

                            att = torch.einsum(
                                "ij,iJ->ijJ",
                                qs[:, :, channel],
                                ks[:, :, channel],
                            )

                            att = att * attention_scale
                            att = att.masked_fill(
                                mask == 0, float("-inf")
                            )  # mask out future tokens

                            # # intentionally use original denom here

                            # center att
                            att = att.exp()
                            st.code(f"{att.shape=}, {att.mean()=} {mask.shape=}")

                minmax = (
                    min(ac[:l].min() for ac, l in zip(task_acts, ls, strict=True)),
                    max(ac[:l].max() for ac, l in zip(task_acts, ls, strict=True)),
                )
                if center_zero_toggle or attn_pattern_radio != "acts":
                    minmax = (
                        -max(abs(minmax[0]), abs(minmax[1])),
                        max(abs(minmax[0]), abs(minmax[1])),
                    )
                if attn_pattern_radio == "channel attn":
                    # )  # attn_logit_minmax
                    minmax = None
                if minmax is not None:
                    st.text(
                        f"color scale: min={minmax[0]:.4f}, max={minmax[1]:.4f}. max/min highlighted if >0.2 std bigger than next largest/smallest"
                    )
                else:
                    st.text("color scale: differenet for each")

                new_col3, new_col4, new_col5 = st.columns(
                    [1, 1, 1 if attn_pattern_radio == "acts" else 0.0001]
                )
                if isinstance(task_acts, torch.Tensor):
                    task_acts = task_acts.detach().cpu()

                with new_col3:
                    st.text("class 1")
                    limit = None

                    for doc_idx in islice(range(len(task_docs) // 2), limit):
                        show_doc = task_docs[doc_idx]
                        if isinstance(show_doc, torch.Tensor):
                            show_doc = show_doc.cpu().detach()
                        l = ls[doc_idx]
                        show_doc = show_doc[:l]
                        show_doc_tokens = [decode_single_token(enc, t) for t in show_doc]

                        highlighted_code = display_code_heatmap(
                            show_doc_tokens,
                            task_acts[doc_idx][:l],
                            highlight_idx="max",
                            minmax=minmax,
                            center_zero=center_zero_toggle,
                        )
                        st.html(
                            get_display_snippet(highlighted_code),  # height=300, scrolling=True
                        )
                with new_col4:
                    st.text("class 2")
                    for doc_idx in islice(range(len(task_docs) // 2, len(task_docs)), limit):
                        show_doc = task_docs[doc_idx]
                        if isinstance(show_doc, torch.Tensor):
                            show_doc = show_doc.cpu().detach()

                        l = ls[doc_idx]
                        show_doc = show_doc[:l]
                        show_doc_tokens = [decode_single_token(enc, t) for t in show_doc[:l]]

                        highlighted_code = display_code_heatmap(
                            show_doc_tokens,
                            task_acts[doc_idx][:l],
                            highlight_idx="max",
                            minmax=minmax,
                            center_zero=center_zero_toggle,
                        )
                        st.html(
                            get_display_snippet(highlighted_code),  # height=300, scrolling=True
                        )
                if attn_pattern_radio == "acts":
                    with new_col5:
                        st.text("diff (always zero centered; diff cmap)")
                        for doc_idx in islice(range(len(task_docs) // 2, len(task_docs)), limit):
                            show_doc = task_docs[doc_idx]

                            l = ls[doc_idx]
                            show_doc = show_doc[:l]

                            show_doc_tokens = [decode_single_token(enc, t) for t in show_doc[:l]]

                            highlighted_code = display_code_heatmap(
                                show_doc_tokens,
                                task_acts[doc_idx][:l]
                                - task_acts[doc_idx - len(task_docs) // 2][
                                    : ls[doc_idx - len(task_docs) // 2]
                                ],
                                highlight_idx=None,
                                center_zero=True,
                            )
                            st.html(
                                get_display_snippet(highlighted_code),  # height=300, scrolling=True
                            )


def decode_single_token(enc, t):
    string_repr = enc.decode_single_token_bytes(t).decode("utf-8", errors="replace")
    assert isinstance(string_repr, str)
    return string_repr


@memcache("local_listdir_v0")
def local_listdir(path):
    # return list(os.listdir(path))
    return bf.listdir(path)


def _resolve_viz_file(base_dir):
    for ext in ("pt", "pkl"):
        cand = f"{base_dir}/viz_data.{ext}"
        try:
            if bf.exists(cand):
                return cand
        except FileNotFoundError:
            pass
    return None


def get_fatk(viz_data_path):
    ks = [int(x) for x in local_listdir(viz_data_path) if x.isdigit()]
    ks.sort()

    @cache("get_loss_for_k_v0")
    def _get_loss_for_k(viz_data_path, k):
        try:
            viz_file = _resolve_viz_file(viz_data_path + f"/{k}")
            if viz_file is None:
                return None
            dat = load_data(viz_file)
            return dat["all_loss"][-1][1]
        except FileNotFoundError:
            return None

    loss_at_k = {k: _get_loss_for_k(viz_data_path, k) for k in ks}
    loss_at_k = {k: v for k, v in loss_at_k.items() if v is not None}

    return ks, loss_at_k


def render_bridge_summary(viz_data):
    st.subheader("Bridge export summary")
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("Task accuracy", f"{viz_data.get('task_results', {}).get('accuracy', 0):.2%}")
    with meta_cols[1]:
        sparsity = viz_data.get("sparsity")
        if sparsity is not None:
            st.metric("Model sparsity", f"{sparsity:.2%}")
        else:
            st.metric("Model sparsity", "N/A")
    with meta_cols[2]:
        st.metric("k", viz_data.get("k", "N/A"))

    st.markdown(
        f"**Model:** {viz_data.get('model_name', 'unknown')} &nbsp;&nbsp; "
        f"**Task:** {viz_data.get('task', 'unknown')} &nbsp;&nbsp; "
        f"**Experiment:** {viz_data.get('experiment', 'unknown')}"
    )

    circuits = viz_data.get("circuits", {})
    top_components = circuits.get("top_k_components", [])
    if top_components:
        st.subheader("Top components")
        comp_df = pd.DataFrame(top_components, columns=["component", "score"])
        st.table(comp_df)

    layer_stats = circuits.get("all_importance")
    if layer_stats:
        st.subheader("Layer activation overview")
        sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1], reverse=True)
        layer_df = pd.DataFrame(sorted_layers, columns=["layer", "avg_activation"])
        st.bar_chart(layer_df.set_index("layer"))
        selected_layer = st.selectbox(
            "Inspect layer",
            options=[layer for layer, _ in sorted_layers],
        )
        st.metric(
            "Average activation magnitude",
            f"{layer_stats[selected_layer]:.4f}",
        )

    samples = (viz_data.get("task_results") or {}).get("samples", [])
    if samples:
        st.subheader("Sample predictions")
        max_rows = min(len(samples), 10)
        simple_rows = []
        for sample in samples[:max_rows]:
            simple_rows.append(
                {
                    k: v
                    for k, v in sample.items()
                    if k in {"input", "expected", "predicted", "correct"}
                }
            )
        st.table(pd.DataFrame(simple_rows))

    metadata = viz_data.get("metadata", {})
    if metadata:
        st.subheader("Metadata")
        st.json(metadata)

    prune_metrics = viz_data.get("prune_metrics")
    if prune_metrics and prune_metrics.get("datasets"):
        st.subheader("Prune loss trace")
        dataset_names = sorted(prune_metrics["datasets"].keys())
        default_dataset = dataset_names[0]
        select_key = f"prune_dataset_{viz_data.get('task', '')}_{viz_data.get('k', '')}"
        selected_dataset = st.selectbox(
            "Dataset",
            options=dataset_names,
            index=dataset_names.index(default_dataset),
            key=select_key,
        )
        trace_entries = prune_metrics["datasets"].get(selected_dataset, [])
        if trace_entries:
            trace_df = pd.DataFrame(trace_entries)
            if "step" in trace_df.columns:
                st.line_chart(trace_df.set_index("step")["loss"], height=300)
            st.table(trace_df[["stage", "loss", "perplexity"]])
        final_rows = []
        for dataset_name, rows in prune_metrics["datasets"].items():
            if not rows:
                continue
            final_rows.append(
                {
                    "dataset": dataset_name,
                    "loss": rows[-1]["loss"],
                    "perplexity": rows[-1]["perplexity"],
                }
            )
        if final_rows:
            st.subheader("Final metrics")
            st.table(pd.DataFrame(final_rows).set_index("dataset"))

    bridge_samples = viz_data.get("bridge_samples")
    selected_layer = None
    if bridge_samples:
        selected_layer = render_bridge_circuit_graph(viz_data)
        render_bridge_sample_distribution(bridge_samples, preferred_layer=selected_layer)
    elif not prune_metrics:
        st.info("No additional bridge metadata available for this run.")


def render_bridge_circuit_graph(viz_data):
    circuits = viz_data.get("circuits") or {}
    components = circuits.get("top_k_components")
    if not components:
        st.info("No circuit component data available.")
        return None

    layers = [name for name, _ in components]
    scores = [score for _, score in components]
    xs = list(range(len(layers)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=scores,
            mode="markers+lines",
            text=layers,
            hovertemplate="Layer: %{text}<br>Importance: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Bridge circuit components",
        xaxis_title="Component index",
        yaxis_title="Activation importance",
        template="plotly_white",
        height=350,
    )

    events = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        key=f"bridge_graph_{viz_data.get('task','')}_{viz_data.get('k','')}",
    )
    selected_layer = layers[0]
    if events:
        point_idx = events[0].get("pointNumber")
        if point_idx is not None and 0 <= point_idx < len(layers):
            selected_layer = layers[int(point_idx)]

    st.caption("Click a point to inspect its samples, or adjust the selector below.")
    return selected_layer


def render_bridge_sample_distribution(bridge_samples: dict, preferred_layer: str | None = None):
    st.subheader("Sample distribution (bridge)")
    layers = sorted(bridge_samples.keys())
    if not layers:
        st.info("No layer-level samples were captured.")
        return
    default_index = 0
    if preferred_layer in layers:
        default_index = layers.index(preferred_layer)
    layer_choice = st.selectbox(
        "Layer",
        options=layers,
        index=default_index,
        key=f"bridge_samples_{hash(tuple(layers))}",
    )
    layer_data = bridge_samples.get(layer_choice, {})
    top_entries = layer_data.get("top", [])
    bottom_entries = layer_data.get("bottom", [])

    def _render_table(entries, title):
        if not entries:
            st.info(f"No {title.lower()} entries for this layer.")
            return
        df = pd.DataFrame(entries)
        st.markdown(f"**{title}**")
        st.table(
            df[["input", "expected", "predicted", "activation"]]
        )

    cols = st.columns(2)
    with cols[0]:
        _render_table(top_entries, "Top activations")
    with cols[1]:
        _render_table(bottom_entries, "Lowest activations")


def resolve_embedding_assets(viz_data, model_name):
    embed_data = viz_data.get("embedding_data")
    if embed_data and embed_data.get("token_embeddings") is not None:
        return {
            "token": embed_data["token_embeddings"],
            "pos": embed_data.get("positional_embeddings"),
            "tokenizer_name": embed_data.get("tokenizer_name"),
            "source": "bridge",
        }

    if "importances" not in viz_data:
        return None

    model_path = get_model_path(model_name)
    try:
        token = get_embed_weights(model_path)
    except FileNotFoundError:
        return None

    try:
        pos = get_model_weights(model_path, lambda x: x["transformer.wpe.weight"]).half()
    except FileNotFoundError:
        pos = None

    tokenizer_name = viz_data["importances"]["beeg_model_config"].tokenizer_name

    return {
        "token": token,
        "pos": pos,
        "tokenizer_name": tokenizer_name,
        "model_path": model_path,
        "source": "registry",
    }


def render_embedding_tab(viz_data, model_name):
    embedding_info = resolve_embedding_assets(viz_data, model_name)
    if embedding_info is None or embedding_info.get("token") is None:
        st.info("No embedding weights available for this model/export.")
        return

    embed_weight = embedding_info["token"].to(torch.float32)
    st.caption(f"Embedding source: {embedding_info.get('source', 'unknown')}")

    chs_used_inds = torch.arange(embed_weight.shape[1])  # d_model
    tokens_used_mask = torch.ones(embed_weight.shape[0], dtype=torch.bool)

    cols = st.columns([1, 1])
    tokenizer_name = embedding_info.get("tokenizer_name")
    if tokenizer_name is None and "importances" in viz_data:
        tokenizer_name = viz_data["importances"]["beeg_model_config"].tokenizer_name

    with cols[0]:
        cols2 = st.columns([1, 2, 1])
        with cols2[0]:
            use_pca = st.toggle(
                "PCA components",
                value=False,
            )
        with cols2[1]:
            chidx = st.selectbox(
                "res channel index",
                options=[
                    f"{ch} (tokens writing: {(embed_weight[:, ch] != 0).sum():d})"
                    for ch in chs_used_inds.tolist()[: min(100, len(chs_used_inds))]
                ],
                index=0,
            )
            chidx = int(chidx.split(" ")[0])

        with cols2[2]:
            st.text(f"encname: {tokenizer_name or 'unknown'}")

        if use_pca:
            if (
                embedding_info.get("source") == "registry"
                and embedding_info.get("model_path") is not None
            ):
                U, S, V = get_embed_weights_pca(
                    embedding_info["model_path"], q=min(100, embed_weight.shape[1])
                )
            else:
                max_q = min(embed_weight.shape[0], embed_weight.shape[1])
                q = max(1, min(100, max_q - 1 if max_q > 1 else 1))
                U, S, V = torch.pca_lowrank(embed_weight, q=q)
            component_idx = min(chidx, U.shape[1] - 1)
            embsort = U[:, component_idx].cpu().sort(descending=True)
        else:
            embsort = embed_weight[:, chidx].sort(descending=True)

        def _filter_embsort(xs):
            return [
                x
                for x, tok in zip(xs, embsort.indices, strict=True)
                if tokens_used_mask[tok] and embed_weight[tok, chidx] != 0
            ]

        token_display_warning = False
        enc = None
        if tokenizer_name:
            try:
                enc = get_encoding(tokenizer_name)
            except (KeyError, ValueError):
                token_display_warning = True
        else:
            token_display_warning = True
        if token_display_warning:
            st.warning(
                "Tokenizer not available via tiktoken; token strings will be approximated by IDs."
            )

        cols2 = st.columns([1, 1])
        with cols2[0]:
            st.markdown(
                "**top 20 tokens by weight (filtered for 2048 most common tokens)**"
            )
            st.table(
                pd.DataFrame(
                    {
                        "tokid": _filter_embsort(embsort.indices.tolist())[:20],
                        "token": [
                            decode_single_token(enc, t).replace(" ", "␣")
                            if enc
                            else f"<tok {t}>"
                            for t in _filter_embsort(embsort.indices.tolist())[:20]
                        ],
                        "weight": _filter_embsort(embsort.values.tolist())[:20],
                    }
                ).set_index("token")
            )
        with cols2[1]:
            st.markdown(
                "**bottom 20 tokens by weight (filtered for 2048 most common tokens)**"
            )
            st.table(
                pd.DataFrame(
                    {
                        "tokid": _filter_embsort(embsort.indices.tolist())[-20:][::-1],
                        "token": [
                            decode_single_token(enc, t).replace(" ", "␣")
                            if enc
                            else f"<tok {t}>"
                            for t in _filter_embsort(embsort.indices.tolist())[-20:][
                                ::-1
                            ]
                        ],
                        "weight": _filter_embsort(embsort.values.tolist())[-20:][::-1],
                    }
                ).set_index("token")
            )

    with cols[1]:
        fig, ax = plt.subplots()
        sns.histplot(
            embsort.values.cpu().numpy(),
            ax=ax,
        )
        ax.set_yscale("log")
        st.pyplot(fig, use_container_width=False)

    wpe = embedding_info.get("pos")
    if wpe is None:
        st.info("Positional embedding weights not available for this model/export.")
        return

    wpe = wpe.to(torch.float32)
    seq_len = min(256, wpe.shape[0])
    hidden_dim = min(512, wpe.shape[1])
    if seq_len == 0 or hidden_dim == 0:
        st.warning("Positional embedding tensor is empty; skipping heatmap.")
        return

    wpe_crop = wpe[:seq_len, :hidden_dim]
    nonzero_cols = (wpe_crop != 0).any(dim=0)
    if nonzero_cols.any():
        wpe_crop = wpe_crop[:, nonzero_cols]

    fig, ax = plt.subplots()
    sns.heatmap(wpe_crop, ax=ax, center=0, cbar_kws={"label": "symlog"}, norm="symlog")
    ax.set_title("Positional embeddings (first 256 positions, 512 dims)")
    st.pyplot(fig, use_container_width=False)
    st.code(f"wpe.shape={wpe.shape}, crop={wpe_crop.shape}")


def faithfulness_at_k_plot(viz_data_path, viz_data):
    ks, loss_at_k = get_fatk(viz_data_path)
    if loss_at_k is None:
        return
    loss_at_k = {int(k): v for k, v in loss_at_k.items()}
    ys = [loss_at_k.get(k, None) for k in ks]

    num_total_nodes = viz_data["num_total_nodes"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=ys,
            mode="lines+markers",
            name="Faithfulness",
            line=dict(color="blue", width=2),
            marker=dict(size=8, color="blue", symbol="circle"),
        )
    )
    fig.update_layout(
        title=f"Loss at k  (num total nodes: {num_total_nodes})",
        xaxis_title="k",
        yaxis_title="Loss",
        template="plotly_white",
        width=800,
        height=400,
    )
    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=ks,
    )
    fig.update_yaxes(
        type="log",
    )
    # add black horizontal dotted line
    fig.add_hline(
        y=viz_data["importances"]["loss"],
        line=dict(color="black", width=2, dash="dot"),
    )

    fig.add_hline(
        y=0.01,
        line=dict(color="black", width=2, dash="dash"),
    )

    click_data = plotly_events(fig, click_event=True, hover_event=False)

    prune_config = viz_data["circuit_data"].get("prune_config", None)
    st.code(f"{json.dumps(prune_config)}")


def plot_all_pruning_losses(viz_data, k):
    all_loss: list[float] = viz_data.get("all_loss")

    if all_loss is None:
        return

    xs, ys = zip(*all_loss)
    xs = [(x + 1) * 8 for x in xs]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="Pruning loss",
            line=dict(color="blue", width=2),
            marker=dict(size=8, color="blue", symbol="circle"),
        )
    )
    fig.update_layout(
        title=f"Pruning loss @ k = {k}",
        xaxis_title="Pruning hpopt step",
        yaxis_title="Loss",
        template="plotly_white",
        width=800,
        height=400,
    )

    # set x-axis to log scale
    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=xs,
    )
    fig.update_yaxes(
        type="log",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.code(f"{viz_data['prune_config']=}")


def main():
    importlib.reload(circuit_sparsity.registries)

    # substantially reduces lag on reruns
    st.config.set_option("runner.postScriptGC", False)

    status_placeholder = st.empty()
    trace_mon_kill = install_trace_mon(status_placeholder)

    default_base = os.path.expanduser(f"{MODEL_BASE_DIR}/viz")
    ordered_paths = [default_base, *EXTRA_VIZ_DIRS]
    base_paths = []
    seen_paths = set()
    for path in ordered_paths:
        if path not in seen_paths:
            seen_paths.add(path)
            base_paths.append(path)
    modelnamecol, datasetcol, sweepnamecol, kcol, k_out_col = st.columns(
        [1.5, 1, 1, 1, 0.25]
    )
    tabs = st.tabs(["main viz", "wte/wpe viz"])
    with modelnamecol:
        model_options = []

        def get_models_for_base_path(base_path):
            try:
                return list(local_listdir(base_path))
            except FileNotFoundError:
                return []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(get_models_for_base_path, base_paths))
            for model_list in results:
                model_options.extend(model_list)

        model_options = sorted(set(model_options))

        if not model_options:
            st.error("No models found in the configured viz directories.")
            return

        preferred_models = ["csp_yolo1", "csp_yolo2"]
        default_index = 0
        for preferred in preferred_models:
            if preferred in model_options:
                default_index = model_options.index(preferred)
                break

        model_name = st.selectbox(
            "model",
            options=model_options,
            index=default_index,
        )
    with datasetcol:
        dataset_options = []

        def get_dataset_options_for_base_path(base_path):
            try:
                return list(local_listdir(os.path.join(base_path, model_name)))
            except FileNotFoundError:
                return []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(get_dataset_options_for_base_path, base_paths))
            for dataset_list in results:
                dataset_options.extend(dataset_list)
        dataset_options = list(set(dataset_options))
        dataset_name = st.selectbox(
            "dataset",
            options=dataset_options,
            index=dataset_options.index("single_double_quote")
            if "single_double_quote" in dataset_options
            else dataset_options.index("bracket_counting_beeg")
            if model_name == "csp_yolo2"
            else dataset_options.index("single_double_quote_beeg3")
            if "single_double_quote_beeg3" in dataset_options
            else 0,
        )
    with sweepnamecol:
        sweep_options = []

        def get_sweep_options_for_base_path(base_path):
            try:
                return list(local_listdir(f"{base_path}/{model_name}/{dataset_name}"))
            except FileNotFoundError:
                return []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(get_sweep_options_for_base_path, base_paths))
            for sweep_list in results:
                sweep_options.extend(sweep_list)

        sweep_options = list(set(sweep_options))

        sweep_name = st.selectbox("pruning method", options=sweep_options, index=0)

    with kcol:
        ks = []

        def get_ks_for_base_path(base_path):
            viz_data_path = f"{base_path}/{model_name}/{dataset_name}/{sweep_name}"
            try:
                candidate_xs = [
                    x
                    for x in local_listdir(viz_data_path)
                    if x.isdigit() or x == "k_optim"
                ]
            except FileNotFoundError:
                return []
            return [int(x) if x != "k_optim" else x for x in candidate_xs]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(get_ks_for_base_path, base_paths))
            for local_ks in results:
                ks.extend(local_ks)

        ks = list(set(ks))

        ks.sort(key=lambda x: int(x) if x != "k_optim" else -1)
        k = st.selectbox(
            "k",
            options=ks,
            index=ks.index(128) if 128 in ks else 0,
        )
        if isinstance(k, str):
            k = k.split(" ")[0]

    viz_data = None
    for base_path in base_paths:
        viz_data_path = f"{base_path}/{model_name}/{dataset_name}/{sweep_name}"
        print(f"{viz_data_path=}")
        viz_file = _resolve_viz_file(viz_data_path + f"/{k}")
        if viz_file is not None:
            viz_data = load_data(viz_file)
            break
    if viz_data is None:
        raise ValueError(f"No viz data found for any base path at {model_name}/{dataset_name}/{sweep_name}/{k}")

    is_bridge_data = "importances" not in viz_data

    if not is_bridge_data:
        mask_L0 = sum([v.numel() for v in viz_data["circuit_data"].values()])
    else:
        mask_L0 = viz_data.get("k", "N/A")

    with k_out_col:
        st.html(f"<br>k=<b>{mask_L0}</b>")

    train_metrics = [] if is_bridge_data else get_progress_data3(model_name)

    st.code(f"{viz_data_path=}")
    if is_bridge_data:
        with tabs[0]:
            render_bridge_summary(viz_data)
    else:
        with tabs[0]:
            jacob_viz(
                viz_data,
            )
            faithfulness_at_k_plot(viz_data_path, viz_data)
            cols = st.columns([2, 1, 1])

            with cols[0]:
                plot_all_pruning_losses(viz_data, k)

            with cols[1]:
                pass
            with cols[2]:
                st.html("<b>final metrics</b>")
                most_important_rows = [
                    "test_xent",
                    "num_alive_neurons/c_fc/layer_0",
                    "num_alive_neurons/c_fc/layer_1",
                    "num_alive_neurons/c_fc/layer_2",
                    "num_alive_neurons/c_fc/layer_3",
                    "step",
                    "L0",
                    "L0_non_embed",
                ]

                def _maybe_item(x):
                    if isinstance(x, torch.Tensor):
                        return x.item()
                    return x

                if len(train_metrics) > 0:
                    st.table(
                        {k: _maybe_item(train_metrics[-1][k]) for k in most_important_rows}
                        | {k: _maybe_item(train_metrics[-1][k]) for k in train_metrics[-1]}
                    )

    with tabs[1]:
        render_embedding_tab(viz_data, model_name)

    trace_mon_kill()
    status_placeholder.html("<pre>ready<br>&nbsp;</pre>")


@cache("get_embed_weights_pca_v1")
def get_embed_weights_pca(model_path, q):
    wte = get_embed_weights(model_path).float()
    return torch.pca_lowrank(wte, niter=10, q=q)


@cache("get_embed_weights_v1")
def get_embed_weights(model_path):
    return get_model_weights(model_path, lambda x: x["transformer.wte.weight"]).half()


def get_model_weights(model_path, fn=None):
    if fn is not None:
        with bf.BlobFile(model_path + "/final_model.pt", "rb") as f:
            return read_tensor_slice_from_file(f, fn, ())

    with bf.BlobFile(model_path + "/final_model.pt", "rb") as f:
        model = torch.load(f, map_location="cpu")

    return model


def get_model_path(model_name):
    return os.path.expanduser(
        f"{MODEL_BASE_DIR}/models/{model_name}"
    )


def get_train_curves_path(model_name):
    return os.path.expanduser(
        f"{MODEL_BASE_DIR}/train_curves/{model_name}"
    )


def get_progress_data3(model_name):
    logpath = get_train_curves_path(model_name)

    if logpath is None:
        return []
    try:
        with bf.BlobFile(os.path.join(logpath, "progress.json")) as fh:
            progress = [json.loads(x) for x in fh.readlines()]
    except FileNotFoundError:
        return []

    return progress


@functools.cache
def install_trace_mon(status_placeholder):
    # get worker thread
    my_ident = threading.get_ident()

    done = False
    i = 0

    def thread_main():
        nonlocal done
        nonlocal i
        spinner_segs = "|/-\\"
        try:
            while not done:
                time.sleep(0.01)
                frame = sys._current_frames()[my_ident]
                # get last frame in the stack
                last_frame = frame.f_back

                # get section of stack in circuit_sparsity project
                our_frames = []
                cur_frame = last_frame.f_back
                while cur_frame is not None:
                    if "circuit_sparsity" in cur_frame.f_code.co_filename:
                        our_frames.append(cur_frame)
                    cur_frame = cur_frame.f_back

                def _make_frame_html(last_frame):
                    return f"&gt; {last_frame.f_code.co_filename}:{last_frame.f_code.co_name}:{last_frame.f_lineno}"

                status_placeholder.html(
                    "<pre>"
                    + (
                        (_make_frame_html(our_frames[0]) + "<br />")
                        if len(our_frames) > 0
                        else "<br>"
                    )
                    + _make_frame_html(last_frame)
                    + "        "
                    + spinner_segs[(i // 10) % 4]
                    + "<br />"
                    + "</pre>"
                )
                i += 1
        except KeyError:
            pass
        finally:
            # Handle cleanup if needed
            print("Trace mon thread exiting")
            status_placeholder.html("<pre>ready<br>&nbsp;</pre>")

    thread = threading.Thread(target=thread_main, daemon=True)
    thread.start()
    add_script_run_ctx(thread, get_script_run_ctx())

    print("Trace mon started")

    def trace_mon_kill():
        nonlocal done
        done = True
        thread.join(timeout=0.1)

    return trace_mon_kill


def treemap(f, x):
    if isinstance(x, dict):
        return {k: treemap(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [treemap(f, v) for v in x]
    elif isinstance(x, tuple):
        return tuple(treemap(f, v) for v in x)
    else:
        return f(x)


if __name__ == "__main__":
    st.set_page_config(page_title="Circuit viz", layout="wide")

    main()
