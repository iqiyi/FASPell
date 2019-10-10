from char_sim import CharFuncs
from masked_lm import MaskedLM
from bert_modified import modeling
import re
import json
import pickle
import argparse
import numpy
import logging
import plot
import tqdm
import time

####################################################################################################

__author__ = 'Yuzhong Hong <hongyuzhong@qiyi.com / eugene.h.git@gmail.com>'
__date__ = '10/09/2019'
__description__ = 'The main script for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker'


CONFIGS = json.loads(open('faspell_configs.json', 'r', encoding='utf-8').read())

WEIGHTS = (CONFIGS["general_configs"]["weights"]["visual"], CONFIGS["general_configs"]["weights"]["phonological"], 0.0)

CHAR = CharFuncs(CONFIGS["general_configs"]["char_meta"])


class LM_Config(object):
    max_seq_length = CONFIGS["general_configs"]["lm"]["max_seq"]
    vocab_file = CONFIGS["general_configs"]["lm"]["vocab"]
    bert_config_file = CONFIGS["general_configs"]["lm"]["bert_configs"]
    if CONFIGS["general_configs"]["lm"]["fine_tuning_is_on"]:
        init_checkpoint = CONFIGS["general_configs"]["lm"]["fine-tuned"]
    else:
        init_checkpoint = CONFIGS["general_configs"]["lm"]["pre-trained"]
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    topn = CONFIGS["general_configs"]["lm"]["top_n"]


class Filter(object):
    def __init__(self):
        self.curve_idx_sound = {0: {0: Curves.curve_null,  # 0 for non-difference
                              1: Curves.curve_null,
                              2: Curves.curve_null,
                              3: Curves.curve_null,
                              4: Curves.curve_null,
                              5: Curves.curve_null,
                              6: Curves.curve_null,
                              7: Curves.curve_null,
                              },
                          1: {0: Curves.curve_null,  # 1 for difference
                              1: Curves.curve_null,
                              2: Curves.curve_null,
                              3: Curves.curve_null,
                              4: Curves.curve_null,
                              5: Curves.curve_null,
                              6: Curves.curve_null,
                              7: Curves.curve_null,
                              }}

        self.curve_idx_shape = {0: {0: Curves.curve_null,  # 0 for non-difference
                                    1: Curves.curve_null,
                                    2: Curves.curve_null,
                                    3: Curves.curve_null,
                                    4: Curves.curve_null,
                                    5: Curves.curve_null,
                                    6: Curves.curve_null,
                                    7: Curves.curve_null,
                                    },
                                1: {0: Curves.curve_null,  # 1 for difference
                                    1: Curves.curve_null,
                                    2: Curves.curve_null,
                                    3: Curves.curve_null,
                                    4: Curves.curve_null,
                                    5: Curves.curve_null,
                                    6: Curves.curve_null,
                                    7: Curves.curve_null,
                                    }}

    def filter(self, rank, difference, error, filter_is_on=True, sim_type='shape'):
        if filter_is_on:
            if sim_type == 'sound':
                curve = self.curve_idx_sound[int(difference)][rank]
            else:
                # print(int(difference))
                curve = self.curve_idx_shape[int(difference)][rank]
        else:
            curve = Curves.curve_null

        if curve(error["confidence"], error["similarity"]) and self.special_filters(error):
            return True

        return False

    @staticmethod
    def special_filters(error):
        """
        Special filters for, essentially, grammatical errors. The following is some examples.
        """
        # if error["original"] in {'他': 0, '她': 0, '你': 0, '妳': 0}:
        #     if error["confidence"] < 0.95:
        #         return False
        #
        # if error["original"] in {'的': 0, '得': 0, '地': 0}:
        #     if error["confidence"] < 0.6:
        #         return False
        #
        # if error["original"] in {'在': 0, '再': 0}:
        #     if error["confidence"] < 0.6:
        #         return False

        return True


class Curves(object):
    def __init__(self):
        pass

    @staticmethod
    def curve_null(confidence, similarity):
        """This curve is used when no filter is applied"""
        return True

    @staticmethod
    def curve_full(confidence, similarity):
        """This curve is used to filter out everything"""
        return False

    @staticmethod
    def curve_01(confidence, similarity):
        """
        we provide an example of how to write a curve. Empirically, curves are all convex upwards.
        Thus we can approximate the filtering effect of a curve using its tangent lines.
        """
        flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
        flag2 = 0.1 * confidence + similarity - 0.6 > 0
        if flag1 or flag2:
            return True

        return False



class SpellChecker(object):
    def __init__(self):
        self.masked_lm = MaskedLM(LM_Config())
        self.filter = Filter()

    @staticmethod
    def pass_ad_hoc_filter(corrected_to, original):
        if corrected_to == '[UNK]':
            return False
        if '#' in corrected_to:
            return False
        if len(corrected_to) != len(original):
            return False
        if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', corrected_to):
            return False
        if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', original):
            return False
        return True

    def get_error(self, sentence, j, cand_tokens, rank=0, difference=True, filter_is_on=True, weights=WEIGHTS, sim_type='shape'):
        """
        PARAMS
        ------------------------------------------------
        sentence: sentence to be checked
        j: position of the character to be checked
        cand_tokens: all candidates
        rank: the rank of the candidate in question
        filters_on: only used in ablation experiment to remove CSD
        weights: weights for different types of similarity
        sim_type: type of similarity

        """

        cand_token, cand_token_prob = cand_tokens[rank]

        if cand_token != sentence[j]:
            error = {"error_position": j,
                     "original": sentence[j],
                     "corrected_to": cand_token,
                     "candidates": dict(cand_tokens),
                     "confidence": cand_token_prob,
                     "similarity": CHAR.similarity(sentence[j], cand_token, weights=weights),
                     "sentence_len": len(sentence)}

            if not self.pass_ad_hoc_filter(error["corrected_to"], error["original"]):
                logging.info(f'{error["original"]}'
                             f' --> <PASS-{error["corrected_to"]}>'
                             f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                return None

            else:
                if self.filter.filter(rank, difference, error, filter_is_on, sim_type=sim_type):
                    logging.info(f'{error["original"]}'
                                 f'--> {error["corrected_to"]}'
                                 f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                    return error

                logging.info(f'{error["original"]}'
                             f' --> <PASS-{error["corrected_to"]}>'
                             f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                return None

        logging.info(f'{sentence[j]}'
                     f' --> <PASS-{sentence[j]}>'
                     f' (con={cand_token_prob}, sim=null, on_top_difference={difference})')
        return None

    def make_corrections(self,
                         sentences,
                         tackle_n_gram_bias=CONFIGS["exp_configs"]["tackle_n_gram_bias"],
                         rank_in_question=CONFIGS["general_configs"]["rank"],
                         dump_candidates=CONFIGS["exp_configs"]["dump_candidates"],
                         read_from_dump=CONFIGS["exp_configs"]["read_from_dump"],
                         is_train=False,
                         train_on_difference=True,
                         filter_is_on=CONFIGS["exp_configs"]["filter_is_on"],
                         sim_union=CONFIGS["exp_configs"]["union_of_sims"]
                         ):
        """
        PARAMS:
        ------------------------------
        sentences: sentences with potential errors
        tackle_n_gram_bias: whether the hack to tackle n gram bias is used
        rank_in_question: rank of the group of candidates in question
        dump_candidates: whether save candidates to a specific path
        read_from_dump: read candidates from dump
        is_train: if the script is in the training mode
        train_on_difference: choose the between two sub groups
        filter_is_on: used in ablation experiments to decide whether to remove CSD
        sim_union: whether to take the union of the filtering results given by using two types of similarities

        RETURN:
        ------------------------------
        correction results of all sentences
        """

        processed_sentences = self.process_sentences(sentences)
        generation_time = 0
        if read_from_dump:
            assert dump_candidates
            topn_candidates = pickle.load(open(dump_candidates, 'rb'))
        else:
            start_generation = time.time()
            topn_candidates = self.masked_lm.find_topn_candidates(processed_sentences,
                                                                  batch_size=CONFIGS["general_configs"]["lm"][
                                                                      "batch_size"])
            end_generation = time.time()
            generation_time += end_generation - start_generation
            if dump_candidates:
                pickle.dump(topn_candidates, open(dump_candidates, 'wb'))

        # main workflow
        filter_time = 0
        skipped_count = 0
        results = []
        print('making corrections ...')
        if logging.getLogger().getEffectiveLevel() != logging.INFO:  # show progress bar if not in verbose mode
            progess_bar = tqdm.tqdm(enumerate(topn_candidates))
        else:
            progess_bar = enumerate(topn_candidates)

        for i, cand in progess_bar:
            logging.info("*" * 50)
            logging.info(f"spell checking sentence {sentences[i]}")
            sentence = ''
            res = []

            # can't cope with sentences containing Latin letters yet.
            if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', sentences[i]):
                skipped_count += 1
                results.append({"original_sentence": sentences[i],
                                "corrected_sentence": sentences[i],
                                "num_errors": 0,
                                "errors": []
                                })
                logging.info("containing Latin letters; pass current sentence.")

            else:

                # when testing on SIGHAN13,14,15, we recommend using `extension()` to solve
                # issues caused by full-width humbers;
                # when testing on OCR data, we recommend using `extended_cand = cand`
                extended_cand = extension(cand)
                # extended_cand = cand
                for j, cand_tokens in enumerate(extended_cand):  # spell check for each characters
                    if 0 < j < len(extended_cand) - 1:  # skip the head and the end placeholders -- `。`
                        # print(j)

                        char = sentences[i][j - 1]

                        # detect and correct errors
                        error = None

                        # spell check rank by rank
                        start_filter = time.time()
                        for rank in range(rank_in_question + 1):
                            logging.info(f"spell checking on rank={rank}")

                            if not sim_union:
                                if WEIGHTS[0] > WEIGHTS[1]:
                                    sim_type = 'shape'
                                else:
                                    sim_type = 'sound'
                                error = self.get_error(sentences[i],
                                                       j - 1,
                                                       cand_tokens,
                                                       rank=rank,
                                                       difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                       filter_is_on=filter_is_on, sim_type=sim_type)

                            else:

                                logging.info("using shape similarity:")
                                error_shape = self.get_error(sentences[i],
                                                             j - 1,
                                                             cand_tokens,
                                                             rank=rank,
                                                             difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                             filter_is_on=filter_is_on,
                                                             weights=(1, 0, 0), sim_type='shape')
                                logging.info("using sound similarity:")
                                error_sound = self.get_error(sentences[i],
                                                             j - 1,
                                                             cand_tokens,
                                                             rank=rank,
                                                             difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                             filter_is_on=filter_is_on,
                                                             weights=(0, 1, 0), sim_type='sound')
  

                                if error_shape:
                                    error = error_shape
                                    if is_train:
                                        error = None  # to train shape similarity, we do not want any error that has already detected by sound similarity
                                else:
                                    error = error_sound

                            if error:
                                if is_train:
                                    if rank != rank_in_question:  # not include candidate that has a predecessor already
                                        # taken as error
                                        error = None
                                        # break
                                    else:
                                        # do not include candidates produced by different candidate generation strategy
                                        if train_on_difference != (cand_tokens[0][0] != sentences[i][j - 1]):
                                            error = None
                                else:
                                    break

                        end_filter = time.time()
                        filter_time += end_filter - start_filter

                        if error:
                            res.append(error)
                            char = error["corrected_to"]
                            sentence += char
                            continue

                        sentence += char

                # a small hack: tackle the n-gram bias problem: when n adjacent characters are erroneous,
                # pick only the one with the greatest confidence.
                error_delete_positions = []
                if tackle_n_gram_bias:
                    error_delete_positions = []
                    for idx, error in enumerate(res):
                        delta = 1
                        n_gram_errors = [error]
                        try:
                            while res[idx + delta]["error_position"] == error["error_position"] + delta:
                                n_gram_errors.append(res[idx + delta])
                                delta += 1
                        except IndexError:
                            pass
                        n_gram_errors.sort(key=lambda e: e["confidence"], reverse=True)
                        error_delete_positions.extend([(e["error_position"], e["original"]) for e in n_gram_errors[1:]])

                    error_delete_positions = dict(error_delete_positions)

                    res = [e for e in res if e["error_position"] not in error_delete_positions]

                    def process(pos, c):
                        if pos not in error_delete_positions:
                            return c
                        else:
                            return error_delete_positions[pos]

                    sentence = ''.join([process(pos, c) for pos, c in enumerate(sentence)])

                # add the result for current sentence
                results.append({"original_sentence": sentences[i],
                                "corrected_sentence": sentence,
                                "num_errors": len(res),
                                "errors": res
                                })
                logging.info(f"current sentence is corrected to {sentence}")
                logging.info(f" {len(error_delete_positions)} errors are deleted to prevent n-gram bias problem")
                logging.info("*" * 50 + '\n')
        try:
            print(
                f"Elapsed time: {generation_time // 60} min {generation_time % 60} s in generating candidates for {len(sentences)} sentences;\n"
                f"              {filter_time} s in filtering candidates for {len(sentences) - skipped_count} sentences;\n"
                f"Speed: {generation_time / len(sentences) * 1000} ms/sentence in generating and {filter_time / (len(sentences) - skipped_count) * 1000} ms/sentence in filtering ")
        except ZeroDivisionError:
            print(
                f"Elapsed time: {generation_time // 60} min {generation_time % 60} s in generating candidates for {len(sentences)} sentences;\n"
                f"              {filter_time} s in filtering candidates for {len(sentences) - skipped_count} sentences;\n"
                f"Speed: {generation_time / len(sentences) * 1000} ms/sentence in generating and NaN ms/sentence in filtering ")
        return results

    def repeat_make_corrections(self, sentences, num=3, is_train=False, train_on_difference=True):
        all_results = []
        sentences_to_be_corrected = sentences

        for _ in range(num):
            results = self.make_corrections(sentences_to_be_corrected,
                                            is_train=is_train,
                                            train_on_difference=train_on_difference)
            sentences_to_be_corrected = [res["corrected_sentence"] for res in results]
            all_results.append(results)

        correction_history = []
        for i, sentence in enumerate(sentences):
            r = {"original_sentence": sentence, "correction_history": []}
            for item in all_results:
                r["correction_history"].append(item[i]["corrected_sentence"])
            correction_history.append(r)

        return all_results, correction_history

    @staticmethod
    def process_sentences(sentences):
        """Because masked language model is trained on concatenated sentences,
         the start and the end of a sentence in question is very likely to be
         corrected to the period symbol (。) of Chinese. Hence, we add two period
        symbols as placeholders to prevent this from harming FASPell's performance."""
        return ['。' + sent + '。' for sent in sentences]


def extension(candidates):
    """this function is to resolve the bug that when two adjacent full-width numbers/letters are fed to mlm,
       the output will be merged as one output, thus lead to wrong alignments."""
    new_candidates = []
    for j, cand_tokens in enumerate(candidates):
        real_cand_tokens = cand_tokens[0][0]
        if '##' in real_cand_tokens:  # sometimes the result contains '##', so we need to get rid of them first
            real_cand_tokens = real_cand_tokens[2:]

        if len(real_cand_tokens) == 2 and not re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', real_cand_tokens):
            a = []
            b = []
            for cand, score in cand_tokens:
                real_cand = cand
                if '##' in real_cand:
                    real_cand = real_cand[2:]
                a.append((real_cand[0], score))
                b.append((real_cand[-1], score))
            new_candidates.append(a)
            new_candidates.append(b)
            continue
        new_candidates.append(cand_tokens)

    return new_candidates


def repeat_test(test_path, spell_checker, repeat_num, is_train, train_on_difference=True):
    sentences = []
    for line in open(test_path, 'r', encoding='utf-8'):
        num, wrong, correct = line.strip().split('\t')
        sentences.append(wrong)

    all_results, correction_history = spell_checker.repeat_make_corrections(sentences, num=repeat_num,
                                                                            is_train=is_train,
                                                                            train_on_difference=train_on_difference)
    if is_train:
        for i, res in enumerate(all_results):
            print(f'performance of round {i}:')
            test_unit(res, test_path,
                      f'difference_{int(train_on_difference)}-rank_{CONFIGS["general_configs"]["rank"]}-results_{i}')
    else:
        for i, res in enumerate(all_results):
            print(f'performance of round {i}:')
            test_unit(res, test_path, f'test-results_{i}')

    w = open(f'history.json', 'w', encoding='utf-8')
    w.write(json.dumps(correction_history, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()


def repeat_non_test(sentences, spell_checker, repeat_num):
    all_results, correction_history = spell_checker.repeat_make_corrections(sentences, num=repeat_num,
                                                                            is_train=False,
                                                                            train_on_difference=True)

    w = open(f'history.json', 'w', encoding='utf-8')
    w.write(json.dumps(correction_history, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()
    for i, res in enumerate(all_results):
        w = open(f'results_{i}.json', 'w', encoding='utf-8')
        w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
        w.close()


def test_unit(res, test_path, out_name, strict=True):
    out = open(f'{out_name}.txt', 'w', encoding='utf-8')

    corrected_char = 0
    wrong_char = 0
    corrected_sent = 0
    wrong_sent = 0
    true_corrected_char = 0
    true_corrected_sent = 0
    true_detected_char = 0
    true_detected_sent = 0
    accurate_detected_sent = 0
    accurate_corrected_sent = 0
    all_sent = 0

    for idx, line in enumerate(open(test_path, 'r', encoding='utf-8')):
        all_sent += 1
        falsely_corrected_char_in_sentence = 0
        falsely_detected_char_in_sentence = 0
        true_corrected_char_in_sentence = 0

        num, wrong, correct = line.strip().split('\t')
        predict = res[idx]["corrected_sentence"]
        
        wrong_num = 0
        corrected_num = 0
        original_wrong_num = 0
        true_detected_char_in_sentence = 0

        for c, w, p in zip(correct, wrong, predict):
            if c != p:
                wrong_num += 1
            if w != p:
                corrected_num += 1
                if c == p:
                    true_corrected_char += 1
                if w != c:
                    true_detected_char += 1
                    true_detected_char_in_sentence += 1
            if c != w:
                original_wrong_num += 1

        out.write('\t'.join([str(original_wrong_num), wrong, correct, predict, str(wrong_num)]) + '\n')
        corrected_char += corrected_num
        wrong_char += original_wrong_num
        if original_wrong_num != 0:
            wrong_sent += 1
        if corrected_num != 0 and wrong_num == 0:
            true_corrected_sent += 1

        if corrected_num != 0:
            corrected_sent += 1

        if strict:
            true_detected_flag = (true_detected_char_in_sentence == original_wrong_num and original_wrong_num != 0 and corrected_num == true_detected_char_in_sentence)
        else:
            true_detected_flag = (corrected_num != 0 and original_wrong_num != 0)
        # if corrected_num != 0 and original_wrong_num != 0:
        if true_detected_flag:
            true_detected_sent += 1
        if correct == predict:
            accurate_corrected_sent += 1
        if correct == predict or true_detected_flag:
            accurate_detected_sent += 1

    print("corretion:")
    print(f'char_p={true_corrected_char}/{corrected_char}')
    print(f'char_r={true_corrected_char}/{wrong_char}')
    print(f'sent_p={true_corrected_sent}/{corrected_sent}')
    print(f'sent_r={true_corrected_sent}/{wrong_sent}')
    print(f'sent_a={accurate_corrected_sent}/{all_sent}')
    print("detection:")
    print(f'char_p={true_detected_char}/{corrected_char}')
    print(f'char_r={true_detected_char}/{wrong_char}')
    print(f'sent_p={true_detected_sent}/{corrected_sent}')
    print(f'sent_r={true_detected_sent}/{wrong_sent}')
    print(f'sent_a={accurate_detected_sent}/{all_sent}')

    w = open(f'{out_name}.json', 'w', encoding='utf-8')
    w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()


def parse_args():
    usage = '\n1. You can spell check several sentences by:\n' \
            'python faspell.py 扫吗关注么众号 受奇艺全网首播 -m s\n' \
            '\n' \
            '2. You can spell check a file by:\n' \
            'python faspell.py -m f -f /path/to/your/file\n' \
            '\n' \
            '3. If you want to do experiments, use mode e:\n' \
            ' (Note that experiments will be done as configured in faspell_configs.json)\n' \
            'python faspell.py -m e\n' \
            '\n' \
            '4. You can train filters under mode e by:\n' \
            'python faspell.py -m e -t\n' \
            '\n' \
            '5. to train filters on difference under mode e by:\n' \
            'python faspell.py -m e -t -d\n' \
            '\n'
    parser = argparse.ArgumentParser(
        description='A script for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker', usage=usage)

    parser.add_argument('multiargs', nargs='*', type=str, default=None,
                        help='sentences to be spell checked')
    parser.add_argument('--mode', '-m', type=str, choices=['s', 'f', 'e'], default='s',
                        help='select the mode of using FASPell:\n'
                             ' s for spell checking sentences as args in command line,\n'
                             ' f for spell checking sentences in a file,\n'
                             ' e for doing experiments on FASPell')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='under mode f, a file to be spell checked should be provided here.')
    parser.add_argument('--difference', '-d', action="store_true", default=False,
                        help='train on difference')
    parser.add_argument('--train', '-t', action="store_true", default=False,
                        help='True=to train FASPell with confidence-similarity graphs, etc.'
                             'False=to use FASPell in production')

    args = parser.parse_args()
    return args


def main():
    spell_checker = SpellChecker()
    args = parse_args()
    if args.mode == 's':  # command line mode
        try:

            assert args.multiargs is not None
            assert not args.train

            logging.basicConfig(level=logging.INFO)
            repeat_non_test(args.multiargs, spell_checker, CONFIGS["general_configs"]["round"])

        except AssertionError:
            print("Sentences to be spell checked cannot be none.")

    elif args.mode == 'f':  # file mode
        try:
            assert args.file is not None
            sentences = []
            for sentence in open(args.file, 'r', encoding='utf-8'):
                sentences.append(sentence.strip())
            repeat_non_test(sentences, spell_checker, CONFIGS["general_configs"]["round"])

        except AssertionError:
            print("Path to a txt file cannot be none.")

    elif args.mode == 'e':  # experiment mode
        
        if args.train:
            repeat_test(CONFIGS["exp_configs"]["training_set"], spell_checker, CONFIGS["general_configs"]["round"],
                    args.train, train_on_difference=args.difference)
            # assert not CONFIGS["exp_configs"]["union_of_sims"]  # union of sims is a strategy used only in testing
            name = f'difference_{int(args.difference)}-rank_{CONFIGS["general_configs"]["rank"]}-results_0'
            plot.plot(f'{name}.json',
                      f'{name}.txt',
                      store_plots=CONFIGS["exp_configs"]["store_plots"],
                      plots_to_latex=CONFIGS["exp_configs"]["store_latex"])
        else:
            repeat_test(CONFIGS["exp_configs"]["testing_set"], spell_checker, CONFIGS["general_configs"]["round"],
                    args.train, train_on_difference=args.difference)


if __name__ == '__main__':
    main()