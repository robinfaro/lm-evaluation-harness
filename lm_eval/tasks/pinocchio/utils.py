import string

import datasets
import numpy as np

def doc_to_text(doc) -> str:
    ctxs = 'You will be presented with a question. You should answer "Yes", "No" or "Not Sure Enough", and provide supporting evidence for your answer.'
    return "{}\nQuestion: {}\nAnswer:".format(
        ctxs,
        doc["claim"],
    )

def map_responses(response):
    mapping = {
        "Yes": 0,
        "No": 1,
        "Not Sure Enough": 2
    }
    return mapping[response]

def doc_to_targets_num(doc):
    return map_responses(doc["answer"])


def _process_results(doc, answer: int):
    # Accuracy if answer is one of the target labels
    acc = 1.0 if answer == doc_to_targets_num(doc) else 0.0

    # Find current category to give more granular bias scores.
    category = doc["domain"]

    # General metrics computed over all subsets
    metrics = {
        "acc": acc,
    }

    metrics.update(
        {
            "acc" + category: acc
        }
    )
    return metrics

def process_results_multiple_choice(doc, results):
    """
    Extracts LM answer as the choice with the highest likelihood.
    """

    lls, _ = zip(*results)

    answer = np.argmax(lls)

    # Further process results with extracted answer
    return _process_results(doc, answer)


def agg_accuracy(arr):
    acc_np = np.array(arr)
    return acc_np.mean()

def _clean_answer(answer: str):
    """Clean response to make comparing answers easier"""
    # Lower string and remove the word "the"
    answer = answer.lower().split(" ")
    answer = [w for w in answer if w not in ("the")]
    answer = " ".join(answer).strip()
    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    return answer

def process_results_generate_until(doc, results):
    """
    Extracts the answer given as one of the possible choices.
    If cannot determine answer, return -1 instead (wrong by default).
    """

    # Default answer is -1 (wrong)
    answer = -1
    for i, choice in enumerate(["Yes", "No", "Not Sure Enough"]):
        if _clean_answer(choice) in _clean_answer(results[0]).split(" "):
            answer = i
            break
        
    # Further process results with extracted answer
    return _process_results(doc, answer)


