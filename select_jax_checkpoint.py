"""Compare existing checkpoints on a fixed CPU validation suite."""
import argparse
import hashlib
import json
from pathlib import Path

from connect4_jax.cli import add_search_arguments, search_overrides


def select_checkpoints(paths, *, report_path, best_output=None, **settings):
    from connect4_jax.checkpoint import load_checkpoint, save_checkpoint
    from connect4_jax.evaluation import evaluate, evaluation_rank

    paths = [Path(path).resolve() for path in paths]
    report_path = Path(report_path).resolve()
    best_output = Path(best_output).resolve() if best_output else None
    protected = paths + ([Path(settings['opponent_model']).resolve()] if settings.get('opponent_model') else [])
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Supply distinct candidate checkpoint paths')
    if report_path in protected or best_output in protected or report_path == best_output:
        raise ValueError('Outputs must be separate from checkpoints and each other')
    if report_path.exists() or (best_output and best_output.exists()):
        raise FileExistsError('Selection outputs must be new files')
    if settings.get('opponent_model'):
        settings['opponent_model'] = str(Path(settings['opponent_model']).resolve())
        settings['opponent_sha256'] = hashlib.sha256(Path(settings['opponent_model']).read_bytes()).hexdigest()
    report = dict(selection_rule='suite', settings=settings, candidates=[], winner=None)
    winner_model = winner_metadata = None
    winner_rank = (-1, -1, -1)
    for path in paths:
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        model, metadata = load_checkpoint(path)
        evaluation = evaluate(model, selection='suite', **settings)
        if hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
            raise ValueError(f'Candidate checkpoint changed during evaluation: {path}')
        record = dict(path=str(path), sha256=fingerprint, iteration=metadata.get('iteration'), evaluation=evaluation)
        report['candidates'].append(record)
        rank = evaluation_rank(evaluation)
        if rank > winner_rank:
            winner_rank, winner_model, winner_metadata = rank, model, record
        print(json.dumps(dict(model=str(path), rank=rank)), flush=True)
    report['winner'] = {key: winner_metadata[key] for key in ('path', 'sha256', 'iteration')}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2))
    if best_output:
        save_checkpoint(best_output, winner_model, metadata=dict(
            iteration=winner_metadata['iteration'], search=winner_metadata['evaluation']['search'],
            evaluation=winner_metadata['evaluation'], source_checkpoint=winner_metadata['path'],
            source_sha256=winner_metadata['sha256']))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--report', required=True, help='New JSON report file')
    parser.add_argument('--best-output', help='Optional new inference checkpoint; source files are preserved')
    parser.add_argument('--games', type=int, default=3, help='Games per seat, opponent, seed and mode')
    parser.add_argument('--seeds', type=int, nargs='+', default=[12345, 12346, 12347])
    parser.add_argument('--depths', type=int, nargs='+', default=[4, 5])
    parser.add_argument('--opening-plies', type=int, default=4)
    parser.add_argument('--simulations', type=int, default=128)
    parser.add_argument('--search-policy', choices=('puct', 'gumbel'), default='gumbel')
    parser.add_argument('--c-puct', type=float, default=1.5)
    parser.add_argument('--opponent-model', help='Frozen reference included with equal weight in selection')
    add_search_arguments(parser, checkpoint_defaults=False)
    parser.set_defaults(proven_win_priority=True)
    args = parser.parse_args()
    select_checkpoints(args.models, report_path=args.report, best_output=args.best_output,
                       games=args.games, seeds=args.seeds, depths=args.depths, opening_plies=args.opening_plies,
                       simulations=args.simulations, search_policy=args.search_policy, c_puct=args.c_puct,
                       opponent_model=args.opponent_model, **search_overrides(args))


if __name__ == '__main__':
    main()
