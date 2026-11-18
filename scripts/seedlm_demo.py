"""Ask a trained SeedLM for programs that turn the given arguments into the expected value.

Example:
  python scripts/seedlm_demo.py --model main-all --inputs "[[3, 1, 2], 2]" --expected "[1, 2]" \
      --describe "sort the list and keep the first n items"
"""
import argparse
import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.tasks import ROOT
from mylittlellm.seedlm import data, model as seedmodel, propose, synth


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="main-all", help="checkpoint name under outputs/seedlm/")
    parser.add_argument("--inputs", required=True, help="Python literal: the list of arguments")
    parser.add_argument("--expected", required=True, help="Python literal: the value the function must return")
    parser.add_argument("--describe", default="", help="task description; operators it names become hints")
    parser.add_argument("--name", default="solve", help="name of the emitted function")
    parser.add_argument("--width", type=int, default=32, help="beam width")
    parser.add_argument("--show", type=int, default=3, help="how many verified programs to print")
    args = parser.parse_args()

    checkpoint = torch.load(ROOT / "outputs/seedlm" / f"{args.model}.pt")
    vocabulary = data.Vocabulary(checkpoint["operators"])
    net = seedmodel.SeedLM(seedmodel.Config(**checkpoint["config"]))
    net.load_state_dict(checkpoint["state"])
    net.eval()
    inputs, expected = ast.literal_eval(args.inputs), ast.literal_eval(args.expected)
    if not isinstance(inputs, list):
        raise SystemExit("--inputs must be a list with one item per argument")
    named = [name for name, op in bi.OPERATORS.items() if op.named_in(args.describe.lower())]
    proposals = propose.propose(net, vocabulary, inputs, expected, named, args.width)
    if not proposals:
        # Nothing is printed unless it reproduces the example: an unverified guess is not an answer.
        print("No proposed program reproduces the example.")
        return
    for proposal in proposals[:args.show]:
        expr, helpers, _ = synth.to_expression(proposal["program"])
        print(f"# log-probability {proposal['log_prob']}: {' '.join(proposal['program'])}")
        print(bi.to_code(args.name, len(inputs), {"expr": expr, "helpers": helpers}))


if __name__ == "__main__":
    main()
