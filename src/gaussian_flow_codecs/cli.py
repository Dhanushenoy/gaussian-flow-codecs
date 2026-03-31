from __future__ import annotations

import argparse

from . import __version__
from .methods import run_adaptive, run_anisotropic, run_baseline, run_beta, run_multires


def add_shared_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vtk-file", required=True, help="Input structured-grid VTK snapshot.")
    parser.add_argument("--results-root", default="results", help="Output directory for artifacts.")
    parser.add_argument("--n-gaussians", type=int, default=4096, help="Kernel budget.")
    parser.add_argument(
        "--preset",
        choices=["fast", "default", "hq"],
        default="default",
        help="Training preset. 'fast' is quicker, 'hq' spends more optimization steps.",
    )
    parser.add_argument("--save-vtk-outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-fast-outputs", action=argparse.BooleanOptionalAction, default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gfc",
        description="CLI for gaussian-flow-codecs.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the current repository status.",
    )
    subparsers = parser.add_subparsers(dest="command")

    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit one of the Gaussian flow codecs to a single VTK snapshot.",
    )
    fit_subparsers = fit_parser.add_subparsers(dest="method")

    baseline_parser = fit_subparsers.add_parser("baseline", help="Diagonal Gaussian kernels on a regular grid.")
    add_shared_fit_arguments(baseline_parser)

    adaptive_parser = fit_subparsers.add_parser("adaptive", help="Error-targeted adaptive kernel placement.")
    add_shared_fit_arguments(adaptive_parser)

    anisotropic_parser = fit_subparsers.add_parser("anisotropic", help="Full-covariance anisotropic Gaussian kernels.")
    add_shared_fit_arguments(anisotropic_parser)

    multires_parser = fit_subparsers.add_parser("multires", help="Coarse/fine Gaussian kernel supports.")
    add_shared_fit_arguments(multires_parser)

    beta_parser = fit_subparsers.add_parser("beta", help="Compact-support beta-basis kernels.")
    add_shared_fit_arguments(beta_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.status:
        print("gaussian-flow-codecs is installed.")
        print("Available methods: baseline, adaptive, anisotropic, multires, beta")
        return

    if args.command == "fit":
        if args.method == "baseline":
            run_baseline(args)
            return
        if args.method == "adaptive":
            run_adaptive(args)
            return
        if args.method == "anisotropic":
            run_anisotropic(args)
            return
        if args.method == "multires":
            run_multires(args)
            return
        if args.method == "beta":
            run_beta(args)
            return
        fit_parser = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        fit_parser.choices["fit"].print_help()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
