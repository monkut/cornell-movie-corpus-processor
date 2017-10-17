"""
A corpus parser for preparing data for a tensorflow chatbot
"""
import os
import random
from ast import literal_eval


DELIM = ' +++$+++ '


class CornellMovieCorpusProcessor:

    def __init__(self, lines='movie_lines.txt', conversations='movie_conversations.txt'):
        self.movie_lines_filepath = lines
        self.movie_conversations = conversations

    def get_id2line(self):
        """
        1. Read from 'movie-lines.txt'
        2. Create a dictionary with ( key = line_id, value = text )
        :return: (dict) {line-id: text, ...}
        """
        id2line = {}
        id_index = 0
        text_index = 4
        with open(self.movie_lines_filepath, 'r', encoding='iso-8859-1') as f:
            for line in f:
                items = line.split(DELIM)
                if len(items) == 5:
                    line_id = items[id_index]
                    dialog_text = items[text_index]
                    id2line[line_id] = dialog_text
        return id2line

    def get_conversations(self):
        """
        1. Read from 'movie_conversations.txt'
        2. Create a list of [list of line_id's]
        :return: [list of line_id's]
        """
        conversation_ids_index = -1
        conversations = []
        with open(self.movie_conversations, 'r', encoding='iso-8859-1') as f:
            for line in f:
                items = line.split(DELIM)
                conversation_ids_field = items[conversation_ids_index]
                conversation_ids = literal_eval(conversation_ids_field)  # evaluate as a python list
                conversations.append(conversation_ids)
        return conversations

    def get_question_answer_set(self, id2line, conversations):
        """
        Want to collect questions and answers
        (this current method is iffy... not sure how this accurately defines questions/answers...)

        :param conversations: (list) Collection line ids consisting of a single conversation
        :param id2line: (dict) mapping of line-ids to actual line text
        :return: (list) questions, (list) answers
        """
        questions = []
        answers = []

        # This uses a simple method in an attempt to gather question/answers
        for conversation in conversations:
            if len(conversation) % 2 != 0:
                conversation = conversation[:-1]  # remove last item

            for idx, line_id in enumerate(conversation):
                if idx % 2 == 0:
                    questions.append(id2line[line_id])
                else:
                    answers.append(id2line[line_id])

        return questions, answers


    def prepare_seq2seq_files(self, questions, answers, output_directory, test_set_size=30000):
        """
        Preparing training/test data for:
        https://github.com/llSourcell/tensorflow_chatbot

        :param questions: (list)
        :param answers: (list)
        :param output_directory: (str) Directory to write files
        :param test_set_size: (int) number of samples to use for test data set
        :return: train_enc_filepath, train_dec_filepath, test_enc_filepath, test_dec_filepath
        """

        # open files
        train_enc_filepath = os.path.join(output_directory, 'train.enc')
        train_dec_filepath = os.path.join(output_directory, 'train.dec')
        test_enc_filepath = os.path.join(output_directory, 'test.enc')
        test_dec_filepath = os.path.join(output_directory,'test.dec')

        train_enc = open(train_enc_filepath, 'w', encoding='utf8')
        train_dec = open(train_dec_filepath, 'w', encoding='utf8')
        test_enc = open(test_enc_filepath, 'w', encoding='utf8')
        test_dec = open(test_dec_filepath, 'w', encoding='utf8')

        # choose test_set_size number of items to put into testset
        test_ids = random.sample(range(len(questions)), test_set_size)

        for i in range(len(questions)):
            if i in test_ids:
                test_enc.write(questions[i] + '\n')
                test_dec.write(answers[i] + '\n')
            else:
                train_enc.write(questions[i] + '\n')
                train_dec.write(answers[i] + '\n')

        # close files
        train_enc.close()
        train_dec.close()
        test_enc.close()
        test_dec.close()
        return train_enc_filepath, train_dec_filepath, test_enc_filepath, test_dec_filepath


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    DEFAULT_OUTPUT_DIRECTORY = os.getcwd()
    parser.add_argument('-l', '--lines',
                        default='movie_lines.txt',
                        help='Path to Cornell Corpus, "movie_lines.txt"')
    parser.add_argument('-c', '--conversations',
                        default='movie_conversations.txt',
                        help='Path to Cornell Corpus, "movie_conversations.txt"')
    parser.add_argument('-o', '--output-directory',
                        dest='output_directory',
                        default=DEFAULT_OUTPUT_DIRECTORY,
                        help='Output directory for train/test data [DEFAULT={}]'.format(DEFAULT_OUTPUT_DIRECTORY))
    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        raise parser.error('"--output-directory" does not exist: {}'.format(args.output_directory))

    processor = CornellMovieCorpusProcessor(args.lines,
                                            args.conversations)
    print('Collection line-ids...')
    id2lines = processor.get_id2line()
    print('Collection conversations...')
    conversations = processor.get_conversations()
    print('Preparing question/answer sets...')
    questions, answers = processor.get_question_answer_set(id2lines, conversations)
    print('Outputting train/test enc/dec files for tensorflow chatbot...')
    result_filepaths = processor.prepare_seq2seq_files(questions, answers, args.output_directory)
    print('\n'.join(result_filepaths))


