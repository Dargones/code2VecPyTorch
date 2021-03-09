from vocabularies import Vocab
from catalyst.dl import BatchMetricCallback
import torch as tt

from log_tools import get_logger

logger = get_logger(__name__)


def get_metrics_dataset(prediction_generator, data_generator, vocab):
    """
    Get a generator returning batches of testing data and another generator returning predictions.
    Compute precision, recall, and f-score
    :param data_generator:          dataset-loader
    :param prediction_generator:    returned by catalyst.predict_loader
    :param vocab:                   maps indices to method names
    :return:                        (precision, recall, f-score)
    """
    tp = fp = fn = 0
    for p, d in zip(prediction_generator, data_generator):
        new_tp, new_fp, new_fn = get_tp_fp_fn_batch(p["logits"], d[1], vocab)
        tp += new_tp
        fp += new_fp
        fn += new_fn
    if tp == 0:
        return 0, 0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f_score


def get_tp_fp_fn_batch(predictions, targets, vocab):
    """
    Calculate sub-token true positives, false positives, and false negatives for a batch of example in the testing set.
    :param targets:          a tensor with target
    :param predictions:      a tensor with predictions returned by the model
    :param vocab:            maps indices to method names
    :return:                 (tp, fp, fn)
    """
    predictions = tt.argmax(predictions, dim=1)
    tp = fp = fn = 0
    for i in range(len(targets)):
        new_tp, new_fp, new_fn = get_tp_fp_fn(predictions[i].item(), targets[i].item(), vocab)
        tp += new_tp
        fp += new_fp
        fn += new_fn
    return tp, fp, fn


def get_f_score_batch(predictions, targets, vocab):
    tp, fp, fn = get_tp_fp_fn_batch(predictions, targets, vocab)
    if tp == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def get_tp_fp_fn(prediction, target, vocab):
    """
    Calculate the sub-token true positives, false positives, and false negatives for a single example in testing set.
    :param target:      an index into vocab specifying the true label for the example
    :param prediction:  an index into vocab specifying the predicted label for the example
    :param vocab:       maps indices to method names, which could be broken in subtokens if split with "|"
    :return:            (tp, fp, fn)
    """
    if target == vocab.get_ind(Vocab.OOV):
        logger.debug("target=%s, prediction=%s" % (vocab.get_key(target), vocab.get_key(prediction)))
        return 0, 0, 1  # From code2vec paper: "An unknown sub-token in the test label is counted as a false negative,
                        # therefore automatically hurting recall." TODO: double check that this is intended behavior
    target_tokens = set(vocab.get_key(target).split("|"))
    prediction_tokens = vocab.get_key(prediction).split("|")
    tp = len([x for x in prediction_tokens if x in target_tokens])
    fp = len(prediction_tokens) - tp
    fn = len(target_tokens) - tp
    logger.debug("target=%s, prediction=%s, tp=%d, fp=%d, fn=%d" % (vocab.get_key(target), vocab.get_key(prediction), tp, fp, fn))
    return tp, fp, fn


class SubtokenFScoreallback(BatchMetricCallback):

    def __init__(self, target_vocab, input_key="targets", output_key="logits", prefix="f-score", **kwargs):
        super().__init__(
            prefix=prefix,
            metric_fn=lambda x, y: get_f_score_batch(x, y, target_vocab),
            input_key=input_key,
            output_key=output_key,
            **kwargs,
        )
