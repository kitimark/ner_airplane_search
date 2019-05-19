def load_data(part):
    sents, labels = [], []
    words, tags = [], []
    with open(part, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split()
                words.append(word)
                tags.append(tag)
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []
                
    return sents, labels