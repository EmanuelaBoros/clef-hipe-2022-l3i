def process_sentences_base(lines):
    #for the base format of datasets with "EndOfSentence" indicating the EOS
    sentences = []
    sentence = ""
    for line in lines[1:]:
        if line[0] == "#" or line[0] == "\n":
            continue
        fields = line.split("\t")
        #print(fields)
        word = fields[0]
        ne_c_l = fields[1]
        ne_f_l = fields[3]
        ne_n = fields[6]
        ne_l = fields[7]
        comment = fields[-1]

        sentence += word
        if "NoSpaceAfter" not in comment:
            sentence += " "
        if "EndOfSentence" in comment:
            sentence += "\n"
            sentences.append(sentence)
            sentence = ""

    return sentences

def process_sentences(lines):

    sentences = []
    sentence = ""
    for line in lines[1:]:
        if line[0] == "#":
            continue
        fields = line.split("\t")
        if len(fields) > 1:
            word = fields[0]
            ne_c_l = fields[1]
            ne_f_l = fields[3]
            ne_n = fields[6]
            ne_l = fields[7]
            comment = fields[-1]

            sentence += word
            if "NoSpaceAfter" not in comment:
                sentence += " "
        elif len(sentence) > 0:
            sentence += "\n"
            sentences.append(sentence)
            sentence = ""

    return sentences


def process_sentences_time(lines):

    sentences = []
    sentence = ""

    for line in lines[1:]:
        if "# hipe2022:date" in line:
            year = int(line.split(" = ")[-1].split("-")[0])
        if line[0] == "#":
            continue
        fields = line.split("\t")
        if len(fields) > 1:
            word = fields[0]
            ne_c_l = fields[1]
            ne_f_l = fields[3]
            ne_n = fields[6]
            ne_l = fields[7]
            comment = fields[-1]

            sentence += word
            if "NoSpaceAfter" not in comment:
                sentence += " "
        elif len(sentence) > 0:
            sentence += "\n"
            sentences.append([sentence, year])
            sentence = ""

    return sentences