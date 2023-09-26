"""
“Going on a vacation” takes longer than “Going for a walk”:
A Study of Temporal Commonsense Understanding
https://arxiv.org/pdf/1909.03065.pdf

MC-TACO is a dataset of 13k question-answer pairs that require temporal commonsense
comprehension. The dataset contains five temporal properties, (1) duration (how long
an event takes), (2) temporal ordering (typical order of events), (3) typical time
(when an event occurs), (4) frequency (how often an event occurs), and (5) stationarity
(whether a state is maintained for a very long time or indefinitely).

WARNING: Running this task with a `--limit` arg will give misleading results! The
corresponding dataset is structured such that each multiple-choice-question gathered
by the authors is split into question-option pairs, where each such pair gets
siloed into an individual document for plausibility testing. Because the harness
shuffles these documents, setting `--limit` will likely "cut off" certain candidate
answers. This is a problem because the task's metrics require an exhaustive evaluation
of a question's options. See section 4 of the paper for details.

Homepage: https://leaderboard.allenai.org/mctaco/submissions/public
"""
from collections import defaultdict

import numpy as np
from efficiency_benchmark.dependencies.lm_eval.base import Task, rf

_CITATION = """
@inproceedings{ZKNR19,
    author = {Ben Zhou, Daniel Khashabi, Qiang Ning and Dan Roth},
    title = {“Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding },
    booktitle = {EMNLP},
    year = {2019},
}
"""


class MCTACO(Task):
    VERSION = 0
    DATASET_PATH = "mc_taco"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"{doc['sentence']}\nQuestion: {doc['question']}\n" f"Answer: {doc['answer']}\nPlausible:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["question"] + " " + doc["sentence"]

    def doc_to_target(self, doc):
        return " " + ["no", "yes"][doc["label"]]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        ll_no, _ = rf.loglikelihood(ctx, " no")
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        return ll_no, ll_yes

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        ll_no, ll_yes = results
        gold = doc["label"]
        pred = int(ll_yes > ll_no)
        question_id = self._question2id(doc)
        items = (gold, pred, question_id)
        return {"em": items, "f1": items}

    def _question2id(self, doc):
        """Returns an identifier for the question in the given document."""
        return " ".join([doc["sentence"], doc["question"]])

    def aggregation(self):
        return {
            "f1": f1,
            "em": exact_match,
        }

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }


def exact_match(items):
    """
    Counts a question as correct if the model accurately classifies the plausibility
    of an answer for all candidate answers. See section 4 "Evaluation Metrics" in the paper.
    """
    results = list(zip(*items))
    accuracies = defaultdict(list)
    for gold, pred, question in zip(results[0], results[1], results[2]):
        accuracies[question].append(pred == gold)
    return np.mean([int(all(accs)) for accs in accuracies.values()])


def f1(items):
    """See section 4 "Evaluation Metrics" in the paper about the F1 metric used."""
    results = list(zip(*items))
    # Group the positive ("yes" = 1) golds and predictions by question.
    gold_positives, pred_positives = defaultdict(list), defaultdict(list)
    for gold, pred, question in zip(results[0], results[1], results[2]):
        gold_positives[question].append(gold)
        pred_positives[question].append(pred)
    f1 = []
    for question in gold_positives.keys():
        gp, pp = sum(gold_positives[question]), sum(pred_positives[question])
        tp = sum(np.logical_and(gold_positives[question], pred_positives[question]))
        p = tp / pp if pp > 0.0 else 1.0
        r = tp / gp if gp > 0.0 else 1.0
        if p + r > 0.0:
            f1.append(2.0 * (p * r) / (p + r))
    return np.mean(f1)