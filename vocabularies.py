from os import path
from log_tools import get_logger
from config import Config

logger = get_logger(__name__)


class Vocab:

    OOV = "<OOV>"  # out of vocabulary
    VOCABS = {}  # vocabularies loaded into memory
    TOKEN_FREQS_FILE = ".token_freqs.tsv"
    PATH_FREQS_FILE = ".path_freqs.tsv"
    TARGET_FREQS_FILE = ".target_freqs.tsv"

    def __init__(self, filename, prune):
        """
        Read a frequency table from a file and create a vocabulary out of it potentially pruning the number of tokens.
        :param filename: the name of the file to read the frequency table from
        :param prune:    if prune=None, all tokens in the frequency table and the OOV token will be added to vocabulary.
                         if prune>1,    prune-1 most frequent tokens and the OOV token will be added to vocabulary.
                         if 0<prune<1,  n most frequent tokens and the OOV token will be added to vocabulary such that
                                        the sum of the frequencies of these n tokens is at least prune (fraction) of
                                        the total sum of all frequencies
        """
        freqs = Vocab.__read_frequencies(filename)
        self.prune = prune
        self.filename = filename
        self.keys = Vocab.__prune(freqs, prune)
        self.total_tokens = sum([freqs[key] for key in self.keys])
        self.keys += [Vocab.OOV]
        self.key_to_ind = {key: i for i, key in enumerate(self.keys)}
        Vocab.VOCABS[str(self)] = self

    def get_ind(self, key):
        if key not in self.key_to_ind:
            return self.key_to_ind[Vocab.OOV]
        return self.key_to_ind[key]

    def get_key(self, ind):
        return self.keys[ind]

    def __len__(self):
        return len(self.keys)

    def __str__(self):
        return self.filename + ":" + str(self.prune)

    @staticmethod
    def tokens(filename, prune):
        return Vocab.__load_or_create(filename + Vocab.TOKEN_FREQS_FILE, prune)

    @staticmethod
    def paths(filename, prune):
        return Vocab.__load_or_create(filename + Vocab.PATH_FREQS_FILE, prune)

    @staticmethod
    def targets(filename, prune):
        return Vocab.__load_or_create(filename + Vocab.TARGET_FREQS_FILE, prune)

    @staticmethod
    def prepare_for_file(filename, override=True):
        """
        Prepares frequency tables for a given dataset file.
        :param filename:  the name of the file.
        :param override:  if True, will create a new frequency table even if one already exists.
        :return:
        """
        if (not override) and (path.exists(filename + Vocab.TOKEN_FREQS_FILE)) \
                and (path.exists(filename + Vocab.PATH_FREQS_FILE)) \
                and (path.exists(filename + Vocab.TARGET_FREQS_FILE)):
            return
        path_freqs = {}
        token_freqs = {}
        target_freqs = {}
        with open(filename) as file:
            for line in file:
                contexts = [c for c in line.strip("\n").split(" ") if c != ""]
                target = contexts[0]
                target_freqs[target] = target_freqs.get(target, 0) + 1
                for context in contexts[1:]:
                    details = context.split(",")
                    token_freqs[details[0]] = token_freqs.get(details[0], 0) + 1
                    token_freqs[details[2]] = token_freqs.get(details[2], 0) + 1
                    path_freqs[details[1]] = path_freqs.get(details[1], 0) + 1
        with open(filename + Vocab.TOKEN_FREQS_FILE, "w") as file:
            for key in token_freqs.keys():
                file.write("%s\t%s\n" % (key, token_freqs[key]))
        with open(filename + Vocab.PATH_FREQS_FILE, "w") as file:
            for key in path_freqs.keys():
                file.write("%s\t%s\n" % (key, path_freqs[key]))
        with open(filename + Vocab.TARGET_FREQS_FILE, "w") as file:
            for key in target_freqs.keys():
                file.write("%s\t%s\n" % (key, target_freqs[key]))

    @staticmethod
    def __load_or_create(filename, prune):
        key = filename + ":" + str(prune)
        if key not in Vocab.VOCABS:
            Vocab(filename, prune)
        return Vocab.VOCABS[key]

    @staticmethod
    def __prune(freqs, prune):
        """Prune freqs as described in __init__ and return the list of remaining keys"""
        keys = sorted(freqs.keys(), key=lambda x: freqs[x], reverse=True)
        if not prune:
            return keys
        if prune > 1:
            keys = keys[:prune - 1]
            logger.info("Keeping %d most frequent unique tokens out of the total of %d (%.2f of all tokens)" %
                        (prune - 1, len(freqs.keys()), sum([freqs[key] for key in keys])/sum(freqs.values())))
            return keys
        i = 0
        curr_sum = 0
        total = sum(freqs.values())
        while curr_sum/total < prune:
            curr_sum += freqs[keys[i]]
            i += 1
        keys = keys[:i]
        logger.info("Keeping %d most frequent unique tokens out of the total of %d (%.2f of all tokens)" %
                    (i, len(freqs.keys()), sum([freqs[key] for key in keys]) / sum(freqs.values())))
        return keys

    @staticmethod
    def __read_frequencies(filename):
        """Read a frequency table from file"""
        logger.debug("Reading frequency table from %s" % filename)
        freqs = {}
        with open(filename) as file:
            for line in file:
                key, count = line.strip("\n").split("\t")
                freqs[key] = int(count)
        logger.debug("Read a frequency table for %d tokens. Total count is %d" %
                     (len(freqs.keys()), sum(freqs.values())))
        return freqs
