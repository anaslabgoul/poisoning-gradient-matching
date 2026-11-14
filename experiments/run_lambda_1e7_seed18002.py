"""One new training seed at lambda 1e-7, with its own undefended control."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.run_small_lambda import main


if __name__=='__main__':
    sys.exit(main(default_protocol='experiments/protocols/bilevel-1e-7-seed18002.json',
                  default_output='artifacts/bilevel-1e-7-seed18002'))
