"""Run only the requested 1e-7, 1e-6 and 1e-5 batch, with a cached baseline."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.run_small_lambda import main


if __name__=='__main__':
    sys.exit(main(default_protocol='experiments/protocols/bilevel-lambda-7-6-5.json',
                  default_output='artifacts/bilevel-lambda-7-6-5'))
