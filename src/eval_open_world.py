import torch
import torch.nn.functional as F
import numpy as np
import os, random, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from protonet import ProtoNet
from tlu_dataset import TLUStatesDataset
from prototypical_loss import euclidean_dist

# ─────────────────────────────────────────────
# Episode sampling
# ─────────────────────────────────────────────

def _get_tensor(dataset, index):
    """Return a float tensor [C,H,W] for a single sample."""
    return dataset._get_pil(index).float()


def sample_episode(dataset, num_ways, num_shots, num_known_q,
                   num_unknown_ways, num_unknown_q):
    """
    Sample one Open-World episode.

    Returns
    -------
    support_data   : Tensor [n_sup, C, H, W]
    support_label  : Tensor [n_sup]          (0..num_ways-1)
    query_data     : Tensor [n_q, C, H, W]
    query_label    : Tensor [n_q]            (-1 = unknown)
    is_unknown     : bool list of length n_q
    """
    all_cls = dataset.full_class_list
    l2i = dataset.label2ind

    # Adaptive class selection
    actual_unknown_ways = min(num_unknown_ways, len(all_cls) - num_ways)
    
    if actual_unknown_ways <= 0:
        raise ValueError(
            f"Dataset only has {len(all_cls)} classes; need at least {num_ways + 1} "
            f"for Open World evaluation.")

    chosen = random.sample(all_cls, num_ways + actual_unknown_ways)
    known_cls   = chosen[:num_ways]
    unknown_cls = chosen[num_ways:]

    C, H, W = dataset.data_size
    n_sup   = num_ways * num_shots
    n_knq   = num_ways * num_known_q
    n_unq   = actual_unknown_ways * num_unknown_q
    n_q     = n_knq + n_unq

    sup_data  = torch.empty(n_sup, C, H, W)
    sup_label = torch.empty(n_sup, dtype=torch.long)
    q_data    = torch.empty(n_q,   C, H, W)
    q_label   = torch.full((n_q,), -1, dtype=torch.long)

    # Support + known queries
    for ci, cls_idx in enumerate(known_cls):
        pool = l2i[cls_idx]
        idx  = random.sample(pool, num_shots + num_known_q)
        for k, ii in enumerate(idx[:num_shots]):
            sup_data[ci*num_shots + k]  = _get_tensor(dataset, ii)
            sup_label[ci*num_shots + k] = ci
        for k, ii in enumerate(idx[num_shots:]):
            pos = ci*num_known_q + k
            q_data[pos]  = _get_tensor(dataset, ii)
            q_label[pos] = ci

    # Unknown queries
    for ui, cls_idx in enumerate(unknown_cls):
        pool = l2i[cls_idx]
        idx  = random.sample(pool, num_unknown_q)
        for k, ii in enumerate(idx):
            pos = n_knq + ui*num_unknown_q + k
            q_data[pos] = _get_tensor(dataset, ii)
            # q_label stays -1

    is_unknown = [False]*n_knq + [True]*n_unq
    return sup_data, sup_label, q_data, q_label, is_unknown


# ─────────────────────────────────────────────
# Single episode inference
# ─────────────────────────────────────────────

def run_episode(model, sup_data, sup_label, q_data, num_ways, device):
    """
    Run ProtoNet for one episode.

    Returns
    -------
    confidence : np.ndarray [total_queries]   max class score per query
    pred_class : np.ndarray [total_queries]   argmax class index
    """
    num_shots = sup_data.size(0) // num_ways
    
    model.eval()
    with torch.no_grad():
        sup_data = sup_data.to(device)
        q_data = q_data.to(device)
        
        # Get embeddings
        sup_embeddings = model(sup_data) # [n_sup, dim]
        q_embeddings = model(q_data)     # [n_q, dim]
        
        # Compute prototypes
        prototypes = []
        for i in range(num_ways):
            p = sup_embeddings[i*num_shots:(i+1)*num_shots].mean(0)
            prototypes.append(p)
        prototypes = torch.stack(prototypes) # [num_ways, dim]
        
        # Compute distances
        dists = euclidean_dist(q_embeddings, prototypes) # [n_q, num_ways]
        
        # Compute probabilities (as confidence)
        log_p_y = F.log_softmax(-dists, dim=1)
        probs = torch.exp(log_p_y).cpu().numpy() # [n_q, num_ways]

    confidence = probs.max(axis=1)   # [nq]
    pred_class = probs.argmax(axis=1)
    return confidence, pred_class


# ─────────────────────────────────────────────
# Compute S, U, HM at a given γ
# ─────────────────────────────────────────────

def compute_su(confidences, pred_classes, true_labels, is_unknown, gamma):
    confidences  = np.array(confidences)
    pred_classes = np.array(pred_classes)
    true_labels  = np.array(true_labels)
    is_unknown   = np.array(is_unknown)

    known_mask   = ~is_unknown
    unknown_mask =  is_unknown

    # Seen accuracy
    if known_mask.sum() == 0:
        S = 0.0
    else:
        not_rejected = confidences > gamma
        correct      = (pred_classes == true_labels)
        S = float((not_rejected & correct & known_mask).sum()) / known_mask.sum()

    # Unseen accuracy
    if unknown_mask.sum() == 0:
        U = 1.0
    else:
        rejected = confidences <= gamma
        U = float((rejected & unknown_mask).sum()) / unknown_mask.sum()

    HM = (2 * S * U / (S + U)) if (S + U) > 0 else 0.0
    return S, U, HM


# ─────────────────────────────────────────────
# Full evaluation
# ─────────────────────────────────────────────

def evaluate(model, dataset, args):
    """Run all episodes, sweep γ, compute metrics + plot."""
    num_ways         = args.num_ways
    num_shots        = args.num_shots
    num_known_q      = args.num_queries
    num_unknown_ways = args.num_unknown_ways
    num_unknown_q    = args.num_unknown_queries

    all_conf   = []
    all_pred   = []
    all_true   = []
    all_is_unk = []

    print(f"\n[Open World] Running {args.num_episodes} episodes ...")
    for ep in tqdm(range(args.num_episodes)):
        sup_d, sup_l, q_d, q_l, is_unk = sample_episode(
            dataset, num_ways, num_shots, num_known_q,
            num_unknown_ways, num_unknown_q)

        conf, pred = run_episode(model, sup_d, sup_l, q_d,
                                  num_ways, args.device)
        true = q_l.numpy()

        all_conf.extend(conf.tolist())
        all_pred.extend(pred.tolist())
        all_true.extend(true.tolist())
        all_is_unk.extend(is_unk)

    all_conf   = np.array(all_conf)
    all_pred   = np.array(all_pred)
    all_true   = np.array(all_true)
    all_is_unk = np.array(all_is_unk)

    # ── Sweep calibration factor γ ──────────────────────────────────────
    gammas    = np.linspace(0.0, 1.0, 201)
    S_list, U_list, HM_list = [], [], []

    for g in gammas:
        S, U, HM = compute_su(all_conf, all_pred, all_true, all_is_unk, g)
        S_list.append(S)
        U_list.append(U)
        HM_list.append(HM)

    S_arr  = np.array(S_list); U_arr  = np.array(U_list); HM_arr = np.array(HM_list)

    best_idx = HM_arr.argmax()
    opt_gamma = gammas[best_idx]
    best_S  = S_arr[best_idx]; best_U  = U_arr[best_idx]; best_HM = HM_arr[best_idx]

    # AUC
    sort_idx = np.argsort(S_arr)
    auc = float(np.trapz(U_arr[sort_idx], S_arr[sort_idx]))
    s_range = S_arr.max() - S_arr.min()
    auc_norm = (auc / s_range) if s_range > 1e-6 else 0.0

    # Top-1
    known_mask   = ~all_is_unk
    not_rejected = all_conf > opt_gamma
    top1 = float((known_mask & not_rejected & (all_pred == all_true)).sum()) / max(known_mask.sum(), 1)

    print("\n" + "="*52)
    print("  OPEN WORLD EVALUATION RESULTS")
    print(f"  Episodes : {args.num_episodes}  |  {num_ways}-way {num_shots}-shot")
    print(f"  Unknown  : {num_unknown_ways} class(es) × {num_unknown_q} queries/class")
    print("="*52)
    print(f"  Optimal γ (calibration factor) : {opt_gamma:.3f}")
    print(f"  Seen Accuracy        (S)       : {best_S*100:.2f}%")
    print(f"  Unseen Accuracy      (U)       : {best_U*100:.2f}%")
    print(f"  Harmonic Mean        (HM)      : {best_HM*100:.2f}%")
    print(f"  AUC (S-U curve, norm)          : {auc_norm:.4f}")
    print(f"  Top-1 Accuracy (known, @γ*)    : {top1*100:.2f}%")
    print("="*52)

    os.makedirs(args.save_dir, exist_ok=True)
    _plot_su_curve(S_arr, U_arr, best_S, best_U, opt_gamma, auc_norm, args.save_dir)
    _plot_conf_dist(all_conf, all_is_unk, opt_gamma, args.save_dir)

    return { 'S': best_S, 'U': best_U, 'HM': best_HM, 'AUC': auc_norm, 'Top1': top1, 'gamma': opt_gamma }


def _plot_su_curve(S, U, best_S, best_U, gamma, auc, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sort_idx = np.argsort(S)
    ax.plot(S[sort_idx], U[sort_idx], color='steelblue', lw=2, label=f'S-U curve (AUC={auc:.3f})')
    ax.scatter([best_S], [best_U], color='red', zorder=5, label=f'Optimal γ={gamma:.3f}')
    ax.set_xlabel('Seen Accuracy (S)'); ax.set_ylabel('Unseen Accuracy (U)')
    ax.set_title('Open World: Seen vs. Unseen Accuracy Curve')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'su_curve.png'), dpi=150); plt.close()


def _plot_conf_dist(confidences, is_unknown, gamma, save_dir):
    conf_known   = confidences[~is_unknown]
    conf_unknown = confidences[ is_unknown]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf_known,   bins=50, alpha=0.65, color='steelblue', label='Known')
    ax.hist(conf_unknown, bins=50, alpha=0.65, color='tomato',    label='Unknown')
    ax.axvline(gamma, color='black', linestyle='--', lw=1.5, label=f'γ* = {gamma:.3f}')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution: Known vs. Unknown')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_dist.png'), dpi=150); plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Open World Few-Shot Evaluation for ProtoNet')
    parser.add_argument('--checkpoint', type=str, required=True,  help='Checkpoint path')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--backbone',   type=str, default='cnn', choices=['cnn', 'resnet50'])
    parser.add_argument('--image_size', type=int, default=84)
    parser.add_argument('--device',     type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-nep', '--num_episodes', type=int, default=600,  help='Number of evaluation episodes')
    parser.add_argument('--num_ways',     type=int, default=5,    help='Known classes per episode')
    parser.add_argument('--num_shots',    type=int, default=5,    help='Support images per known class')
    parser.add_argument('--num_queries',  type=int, default=15,   help='Known queries per class')
    parser.add_argument('--num_unknown_ways',    type=int, default=5,  help='Unknown classes per episode')
    parser.add_argument('--num_unknown_queries', type=int, default=15, help='Unknown queries per class')
    parser.add_argument('--partition',  type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--save_dir',   type=str, default='./ow_results', help='Output directory')
    args = parser.parse_args()

    model = ProtoNet(backbone=args.backbone, x_dim=3).to(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    dataset = TLUStatesDataset(root=args.dataset_root, mode=args.partition, image_size=args.image_size)

    min_cls = args.num_ways + args.num_unknown_ways
    if len(dataset.full_class_list) < min_cls:
        raise RuntimeError(f"Partition '{args.partition}' only has {len(dataset.full_class_list)} classes.")

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    evaluate(model, dataset, args)


if __name__ == '__main__':
    main()
