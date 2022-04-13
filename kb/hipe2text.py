import csv
import os
import argparse

def parse_arguments():
    """Returns a command line parser

    Returns
    ----------
    argparse.Namespace

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--in_file",
                        dest="in_file",
                        help="""Path to tsv file.""",
                        type=str,
                        default="/home/cgonzale/HIPE-2022-data/data/v2.0/hipe2020/fr/HIPE-2022-v2.0-hipe2020-dev-fr.tsv")

    return parser.parse_args()


def process_phrases(lines):

    phrases = []
    phrase = ""
    for line in lines[1:]:
        if line[0] == "#" or line[0] == "\n":
            continue
        fields = line.split("\t")
        word = fields[0]
        comment = fields[-1]

        phrase += word
        if "NoSpaceAfter" not in comment:
            phrase += " "
        if "EndOfSentence" in comment:
            phrase += "\n"
            phrases.append(phrase)
            phrase = ""

    return phrases


def main():
    args = parse_arguments()

    with open(args.in_file, 'r') as f:
        lines = f.readlines()

    phrases = process_phrases(lines)

    out_file = args.in_file.replace(".tsv", ".txt")
    with open(out_file, "w") as f:
        f.writelines(phrases)

if __name__ == '__main__':
    """
    Starts the whole app from the command line
    """

    main()



