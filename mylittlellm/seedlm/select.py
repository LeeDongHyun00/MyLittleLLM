"""Choosing among the coder's verified programs by what they do on inputs nobody has answers for.

Every program in the beam that reproduces the visible example is equally consistent with it, and
the first one (the most probable) is often a coincidence: on held-out teacher tasks at beam width
128, the first verified program is correct for 57 tasks while some verified program is for 86.
The consensus round (mylittlellm/rulecoder/consensus.py) found that agreement on fresh inputs is
evidence of correctness. Here it is applied inside one coder's beam (execution-based minimum Bayes
risk): each verified program is run on probe inputs (type-preserving mutations of the example's
arguments), and the program whose behaviour carries the most probability mass of the beam wins.
"""
import copy
import math

from ..rulecoder import consensus
from ..rulecoder.limits import Timeout, time_limit
from . import pyx

SECONDS = 0.1


def behaviour(program, probes, literals=(), output=None):
    """[(status, canonical value)] of a prefix pyx program on each probe."""
    results = []
    for args in probes:
        try:
            with time_limit(SECONDS):
                value = pyx.execute(list(program), copy.deepcopy(args), literals, output)
            results.append(("ok", consensus.canonical(value)))
        except (Exception, Timeout) as error:
            results.append(("exc", type(error).__name__))
    return results


def mbr_pick(programs, log_probs, args, literals=(), output=None, count=24, minimum=3):
    """Index of the program whose behaviour on probes has the most beam probability behind it.

    `programs` are verified prefix programs in beam order with their log-probabilities. A program's
    support is its own probability plus that of every program that agrees with it on at least
    `minimum` probes and conflicts on none; ties keep beam order. With fewer than two programs,
    or no probes, the first program is kept.
    """
    if len(programs) < 2:
        return 0
    probes = consensus.probe_inputs(list(args), count)
    if not probes:
        return 0
    top = max(log_probs)
    weights = [math.exp(lp - top) for lp in log_probs]
    candidates = [{"results": behaviour(p, probes, literals, output), "weight": w}
                  for p, w in zip(programs, weights)]
    best, _ = consensus.mbr_select(candidates, minimum)
    return best
