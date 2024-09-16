import sys
import logging
from experiment import (
    SetDistanceExperiment,
    AdjectiveTypeWeightExperiment,
    SynPhraseDistanceExperiment,
    NonSubsectivityExperiment,
    exp_run,
    EXPERIMENTS
)
from encoding import ENCODERS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def main(argv):
    if ("all" in argv):
        logger.info("Running all...")
        for enc in ENCODERS:
                exp = SetDistanceExperiment(encoder=enc, vecop="mean", sent_enc=True, strict=True)
                exp_run(exp)
                exp = AdjectiveTypeWeightExperiment(encoder=enc, vecop="mean", sent_enc=True, strict=True)
                exp_run(exp)
                exp = SynPhraseDistanceExperiment(encoder=enc, vecop="mean", sent_enc=True, strict=True)
                exp_run(exp)
                exp = NonSubsectivityExperiment(encoder=enc, vecop="mean", sent_enc=True, strict=True)
                exp_run(exp)
    elif (len(argv) > 1 and argv[1] in EXPERIMENTS):
        exp = EXPERIMENTS[argv[1]]()
        exp_run(exp)
    else:
        logger.error("Experiment not recognized. Must be one of: " + " | ".join(EXPERIMENTS.keys()))


if __name__ == "__main__":
    main(sys.argv)