"""Train VisTA-BAGEL: paired feasible / infeasible supervision on top of BAGEL's trainer.

Run it through torchrun (see launch.sh). The arguments of BAGEL's trainer are passed through unchanged;
the flags below are read here first.

  --hint {mixed_hint,decline_hint,none}   reminder in the training instruction (default mixed_hint:
                                          two of the four feasible and two of the four infeasible
                                          examples of every step carry the reminder)
  --milestones 16384                      comma-separated steps whose raw weights are saved
"""
import argparse
import os
import sys
from pathlib import Path

import vista_data
from runtime import fix_loss_logging, training_runtime


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--hint', default='mixed_hint', choices=('mixed_hint', 'decline_hint', 'none'))
    parser.add_argument('--milestones', default='16384')
    options, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    vista_data.register(options.hint)
    import train.pretrain_unified_navit as trainer
    fix_loss_logging(trainer)
    output = Path(sys.argv[sys.argv.index('--results_dir') + 1])
    milestones = [int(step) for step in options.milestones.split(',')]
    with training_runtime(trainer, output, milestones, model_only=os.environ.get('VISTA_FULL_STATE') != '1'):
        trainer.main()


if __name__ == '__main__':
    main()
