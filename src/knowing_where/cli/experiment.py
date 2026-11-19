"""Train, evaluate, export, and predict with reproducible source-fusion systems."""

import argparse
from pathlib import Path

from ..experiments.data import download_dataset, load_dataset, load_queries
from ..experiments.demo import write_demo
from ..experiments.io import read_json, write_json
from ..experiments.runner import run_experiment
from ..experiments.system import FittedSystem
from ..routing import ModelProfile, RoutingPolicy


def _generation_options(parser):
    parser.add_argument('--max-calls', type=int, help='Hard cap on total generator attempts, including retries')
    parser.add_argument('--max-retries', type=int, default=0, help='Retries per path for timeout/connection failures')


def _check_destinations(outputs, inputs):
    outputs = [path.resolve() for path in outputs if path is not None]
    inputs = {path.resolve() for path in inputs if path is not None}
    if len(set(outputs)) != len(outputs) or any(path in inputs for path in outputs):
        raise ValueError('output paths must be distinct and cannot overwrite input files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Create synthetic data and run the complete offline pipeline')
    demo.add_argument('--output-dir', type=Path, default=Path('examples/experiments/demo'))
    demo.add_argument('--seed', type=int, default=42)
    demo.add_argument('--alpha', type=float, default=0.1)
    _generation_options(demo)
    run = commands.add_parser('run', help='Run a normalized JSONL dataset using recorded answer paths')
    run.add_argument('dataset', type=Path)
    run.add_argument('--models', type=Path, required=True, help='JSON list of model profiles')
    run.add_argument('--policy', type=Path, help='JSON routing policy')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--artifact-output', type=Path, help='Also save a fitted system for later prediction')
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--alpha', type=float, default=0.1)
    run.add_argument('--paths', type=int, default=10)
    run.add_argument('--joint-ghost', choices=['independent'], required=True,
                     help='Explicit corpus-absence independence approximation')
    _generation_options(run)
    export = commands.add_parser('export', help='Export a fitted system from a version 2 experiment report')
    export.add_argument('report', type=Path)
    export.add_argument('--output', type=Path, required=True)
    predict = commands.add_parser('predict', help='Predict label-free JSONL queries with a saved fitted system')
    predict.add_argument('artifact', type=Path)
    predict.add_argument('queries', type=Path)
    predict.add_argument('--output', type=Path, required=True)
    _generation_options(predict)
    fetch = commands.add_parser('download', help='Fetch a normalized JSONL dataset over HTTPS')
    fetch.add_argument('url')
    fetch.add_argument('--sha256', required=True)
    fetch.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'download':
            download_dataset(args.url, args.output, args.sha256)
            print(f'Validated dataset saved to {args.output}')
            return
        if args.command == 'export':
            _check_destinations([args.output], [args.report])
            FittedSystem.from_report(read_json(args.report)).save(args.output)
            print(f'Fitted system saved to {args.output}')
            return
        if args.command == 'predict':
            _check_destinations([args.output], [args.artifact, args.queries])
            result = FittedSystem.load(args.artifact).predict(load_queries(args.queries),
                        max_calls=args.max_calls, max_retries=args.max_retries)
            write_json(args.output, result)
            print(f"Saved {len(result['predictions'])} predictions to {args.output}")
            return
        if args.command == 'demo':
            dataset, model_path = write_demo(args.output_dir)
            output, paths, joint, policy = args.output_dir / 'report.json', 10, 'independent', RoutingPolicy()
            artifact_output = args.output_dir / 'system.json'
        else:
            dataset, model_path, output, paths, joint = args.dataset, args.models, args.output, args.paths, args.joint_ghost
            policy = RoutingPolicy(**read_json(args.policy)) if args.policy else RoutingPolicy()
            artifact_output = args.artifact_output
        _check_destinations([output, artifact_output], [dataset, model_path, getattr(args, 'policy', None)])
        models = [ModelProfile(**row) for row in read_json(model_path)]
        report = run_experiment(load_dataset(dataset), models, joint_ghost=joint,
                                seed=args.seed, alpha=args.alpha, paths=paths, policy=policy,
                                max_calls=args.max_calls, max_retries=args.max_retries)
        fitted = FittedSystem.from_report(report) if artifact_output else None
        write_json(output, report)
        if fitted:
            fitted.save(artifact_output)
            print(f'Fitted system saved to {artifact_output}')
        print(f'Experiment report saved to {output}')
        for method, result in report['comparisons'].items():
            metrics = result['overall']
            print(f"{method}: answer rate={metrics['answer_rate']:.1%}, "
                  f"mapped set coverage={metrics['mapped_set_coverage']:.1%}, "
                  f"mean set size={metrics['mean_set_size']:.2f}")
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f'experiment: {error}\n')


if __name__ == '__main__':
    main()
