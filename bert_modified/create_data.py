import numpy as np
import pickle
import os
import argparse

def cut_line(sentence):
    sent = ''
    delimiter = ['。', '；', '？', '！']
    for i, c in enumerate(sentence):
        sent += c
        if ((c in delimiter) and (sentence[min(len(sentence)-1, i + 1)] not in ['」', '”', '’'])) or i == len(sentence)-1:
            yield sent
            sent = ''

def cut_line2(sentence):
    sent = ''
    for i, c in enumerate(sentence):
        sent += c
        if c == '，':
            flag = True
            for j in range(i+1, min(len(sentence)-1, i+6)):
                if sentence[j] == '，' or j == len(sentence)-1:
                    flag = False

            if (flag and len(sent) > 20) or i == len(sentence)-1:
                yield sent[:-1] + '。'
                sent = ''


def make_docs(wrong, correct):
    w_res = []
    if ('。' in wrong[:-1]) or ('；' in wrong[:-1]) or ('？' in wrong[:-1]) or ('！' in wrong[:-1]):
        for w_sent in cut_line(wrong):
            w_res.append(w_sent + '\n')
            # wrong_file.write(w_sent + '\n')
    elif len(wrong) > 100:
        for w_sent in cut_line2(wrong):
            w_res.append(w_sent + '\n')
            # wrong_file.write(w_sent + '\n')
    else:
        w_res.append(wrong + '\n')
        # wrong_file.write(wrong + '\n')

    # wrong_file.write('\n')
    c_res = []
    if ('。' in correct[:-1]) or ('；' in correct[:-1]) or ('？' in correct[:-1]) or ('！' in correct[:-1]):
        for c_sent in cut_line(correct):
            c_res.append(c_sent + '\n')
            # correct_file.write(c_sent + '\n')
    elif len(wrong) > 100:
        for c_sent in cut_line2(correct):
            c_res.append(c_sent + '\n')
            # correct_file.write(c_sent + '\n')
    else:
        c_res.append(correct + '\n')
        # correct_file.write(correct + '\n')

    if len(w_res) != len(c_res):
        w_res = [wrong + '\n']
        c_res = [correct + '\n']

    for w_r, c_r in zip(w_res, c_res):
        if not len(w_r.strip()) == len(c_r.strip()):
            print(w_r)
            print(len(w_r.strip()))
            print(c_r)
            print(len(c_r.strip()))
            exit()

    for l in w_res:
        wrong_file.write(l)
    wrong_file.write('\n')

    for l in c_res:
        correct_file.write(l)
    correct_file.write('\n')


def main(fname, output_dir):
    confusions = {}

    for line in open(fname, 'r', encoding='utf-8'):
        num, wrong, correct = line.strip().split('\t')
        wrong = wrong.strip()
        correct = correct.strip()
        for w, c in zip(wrong, correct):
            if w!=c:
                if w + c not in confusions:
                    confusions[w + c] = 0
                confusions[w + c] += 1
        # if len(wrong) != len(correct):
        #     print(wrong)
        #     print(correct)
        #     exit()
        assert len(wrong) == len(correct)
        num = int(num)

        make_docs(wrong, correct)

        if wrong != correct:
            make_docs(correct, correct)

        poses = [pos for pos, (w, c) in enumerate(zip(wrong, correct)) if w != c]
        num = len(poses)

        if num >= 2:
            if len(poses) != num:
                print(wrong)
                print(correct)
                exit()
            assert len(poses) == num
            for i in range(1, num):
                selected_poses = [poses[k] for k in np.random.choice(num, i, replace=False)]
                fake_wrong = list(wrong)
                for p in selected_poses:
                    fake_wrong[p] = correct[p]

                fake_wrong = ''.join(fake_wrong)
                assert len(fake_wrong) == len(correct)
                assert fake_wrong != correct
                make_docs(fake_wrong, correct)

    # take the top frequency of confusions about the each character.
    top_confusions = {}
    for k in confusions:
        if k[0] not in top_confusions:
            top_confusions[k[0]] = confusions[k]
        elif top_confusions[k[0]] < confusions[k]:
            top_confusions[k[0]] = confusions[k]

    confusions_top = sorted(list(top_confusions.keys()), key=lambda x: top_confusions[x], reverse=True)

    correct_count = {}
    for line_c, line_w in zip(open(os.path.join(args.output, 'correct.txt'), 'r', encoding='utf-8'), open(os.path.join(args.output, 'wrong.txt'), 'r', encoding='utf-8')):
        if line_c.strip():
            wrong, correct = line_w.strip(), line_c.strip()
            wrong = wrong.strip()
            correct = correct.strip()
            for w, c in zip(wrong, correct):
                if w==c and w in top_confusions:
                    if w not in correct_count:
                        correct_count[w] = 0
                    correct_count[w] += 1
    proportions = {}
    for k in correct_count:
        assert correct_count[k] != 0
        proportions[k] = min(top_confusions[k] / correct_count[k], 1.0)

    print('confusion statistics:')

    for i in range(min(len(confusions_top), 20)):
        if confusions_top[i] in correct_count:
            correct_occurs = correct_count[confusions_top[i]]
            proportions_num = proportions[confusions_top[i]]
        else:
            correct_occurs = 0
            proportions_num = 'NaN'
        print(f'most frequent confusion pair for {confusions_top[i]} occurs {top_confusions[confusions_top[i]]} times,'
              f' correct ones occur {correct_occurs} times, mask probability should be {proportions_num}')

    pickle.dump(proportions, open(os.path.join(args.output, 'mask_probability.sav'), 'wb'))
    # print('top confusions:')
    # for i in range(20):
    #     print(f'{top_confusions[i]} occurs {confusions[confusions_top[i]]} times')

# main()
def parse_args():
    usage = '\n1. create wrong.txt, correct.txt and mask_probability.sav by:\n' \
            'python create_data.py -f /path/to/train.txt\n' \
            '\n' \
            '\n2. specify output dir by:\n' \
            'python create_data.py -f /path/to/train.txt -o /path/to/dir/\n' \
            '\n' 
    parser = argparse.ArgumentParser(
        description='A module for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker', usage=usage)

    parser.add_argument('--file', '-f', type=str, default=None,
                        help='original training data.')
    parser.add_argument('--output', '-o', type=str, default='',
                        help='output a file a dir; default is current directory.')
    # parser.add_argument('--verbose', '-v', action="store_true", default=False,
    #                     help='to show details of spell checking sentences under mode s')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    correct_file = open(os.path.join(args.output,'correct.txt'), 'w', encoding='utf-8')
    wrong_file = open(os.path.join(args.output,'wrong.txt'), 'w', encoding='utf-8')
    main(args.file, args.output)
