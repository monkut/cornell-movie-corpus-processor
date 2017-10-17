# cornell-movie-corpus-processor
Intended to be a corpus processor for building a chatbot based on the cornell movie corpus.

This processes the [Cornell Movie Dialog Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) for use with training a chatbot.

My attempt to understand what this script is doing in it's preparation of data for building a tensorflow chatbot.

https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py


## CornellMovieCorpusProcessor class usage

This cleanup defines a single class, `CornellMovieCorpusProcessor` to process the dataset. 
(It could probably be better designed, but this is the initial cleanup for clarity....)

If you want to use the class directly you can do so like this:

```
from process import CornellMovieCorpusProcessor


processor = CornellMovieCorpusProcessor(movie_lines_filepath,
                                        movie_conversations_filepath)
id2lines = processor.get_id2line()
conversations = processor.get_conversations()
questions, answers = processor.get_question_answer_set(id2lines, conversations)
result_filepaths = processor.prepare_seq2seq_files(questions, answers, args.output_directory)
```
