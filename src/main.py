import argparse, os
import numpy as np
import networkx as nx

from utils import viz_utils
from generation.sis import DeterministicSIS

def get_sis_str(args):
    strs = []
    for arg, val in vars(args).items():
        strs.append(f"{arg}={val}")
    return "_".join(strs)


def main(args):
    ### Setup directories ###
    data_str = get_sis_str(args)
    figure_dir = os.path.join("../figures", data_str)
    os.makedirs(figure_dir, exist_ok=True)

    SIS = DeterministicSIS(n=args.n, p=args.p, gamma=args.gamma, d=args.d, tau=args.tau, delta=args.delta, 
        inf_alpha=args.inf_alpha, inf_beta=args.inf_beta, sus_alpha=args.sus_alpha, sus_beta=args.sus_beta, 
        rec_alpha=args.rec_alpha, rec_beta=args.rec_beta, int_alpha=args.int_alpha, int_beta=args.int_beta)
    
    # plot initial graph
    pos = nx.random_layout(SIS.G)
    for i in range(20):
        viz_utils.plot_SIS_graph(SIS, os.path.join(figure_dir, f"graph_t{i}.png"), pos=pos)
        viz_utils.plot_affinity_distribution(SIS, os.path.join(figure_dir, f"affinity_t{i}.png"))

        SIS.update()
        SIS.intervene(np.random.choice(SIS.G.nodes, size=round(args.n*0.1)))

    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--p", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--delta", type=float, default=0.55)
    parser.add_argument("--inf_alpha", type=float, default=1)
    parser.add_argument("--inf_beta", type=float, default=1)
    parser.add_argument("--sus_alpha", type=float, default=1)
    parser.add_argument("--sus_beta", type=float, default=1)
    parser.add_argument("--rec_alpha", type=float, default=1)
    parser.add_argument("--rec_beta", type=float, default=1)
    parser.add_argument("--int_alpha", type=float, default=1)
    parser.add_argument("--int_beta", type=float, default=1)
    args = parser.parse_args()

    main(args)