import sys
import random

# percent = 0.5
file_path = 'conll_03_vi.train_bioes'
percent = 0.8
file_path_out = 'conll_03_vietnamese.train_bioes'.format(percent)
file_out = open(file_path_out, 'w', encoding='utf-8')
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    all_samples = []
    sample = []
    for line in lines:
        if line == '\n':
            all_samples.append(sample)
            sample = []
            continue
        sample.append(line)
    # file_out.write('\n')

random.shuffle(all_samples)
train_sents = int(percent * len(all_samples))
dev_sents = int(0.9 * len(all_samples))
test_sents = len(all_samples) - dev_sents
train_samples = all_samples[:train_sents]
print(f'train: 0-{train_sents}, dev: {train_sents}-{dev_sents}, test: {train_sents}-{len(all_samples)}')
for sample in train_samples:
    for line in sample:
        file_out.write(line)
    file_out.write('\n')

dev_file = 'conll_03_vietnamese.testa_bioes'
dev_file = open(dev_file, 'w', encoding='utf-8')
for sample in all_samples[train_sents: dev_sents]:
    for line in sample:
        dev_file.write(line)
    dev_file.write('\n')

test_file = 'conll_03_vietnamese.testb_bioes'
test_file = open(test_file, 'w', encoding='utf-8')
for sample in all_samples[dev_sents:]:
    for line in sample:
        test_file.write(line)
    test_file.write('\n')


file_out.close()
test_file.close()
dev_file.close()