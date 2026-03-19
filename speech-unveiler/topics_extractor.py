from keybert import KeyBERT
from bertopic import BERTopic
import nltk
try:
    nltk.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

class TopicExtractor:
    def remove_stopwords(self, text, language="english"):
        stopwords = nltk.corpus.stopwords.words(language)
        #stopwords = stopwords + ['any', 'word', 'you', 'want']
        return [word for word in text if word not in stopwords]

    def group_blocks(self, blocks):
        total_groups = len(blocks) // self.group_size
        for group in range(total_groups + 1):
            if group == 0:
                yield " ".join(blocks[group * self.group_size:(group + 1) * self.group_size])
            elif group < total_groups:
                yield " ".join(blocks[group * self.group_size - self.overlap:(group + 1) * self.group_size])
            else:
                yield " ".join(blocks[group * self.group_size - self.overlap:])

    def retrieve_text_chunks(self, file, language="english"):
        with open(file, "r") as f:
            text = f.read()
        words = self.remove_stopwords(text.split(), language)
        return list(self.group_blocks(words))

class KeyBertTopicExtractor(TopicExtractor):
    def __init__(self):
        self.model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

    def extract_topics(self, file, top_n=10, language="english"):
        with open(file, "r") as f:
            text = self.remove_stopwords(f.read().split(), language)
        return self.model.extract_keywords(" ".join(text), top_n=top_n)

    def topics_to_file(self, keywords, file):
        with open(file, 'w', encoding="utf-8") as f:
            lines = [f"{topic} -> {prob}" for topic, prob in keywords]
            f.write("\n".join(lines))

class BertTopicExtractor(TopicExtractor):
    def __init__(self):
        self.group_size = 100
        self.overlap = 10
        self.model = BERTopic()

    def topics_to_file(self, document_file, file):
        def get_topic_words_and_probabilities(index):
            words_and_probs = self.model.get_topic(index - 1)
            return ";".join([f"{word}:{prob}" for word, prob in words_and_probs]) if words_and_probs else ""
        with open(file, 'w', encoding="utf-8") as f:
            document_file.set_index('Topic')
            lines = [f"{row['Name']} -> {get_topic_words_and_probabilities(index)}" for index, row in document_file.iterrows()]
            f.write("\n".join(lines))

    def extract_topics(self, file, top_n=10, language="english"):
        self.model = BERTopic(language)
        grouped_blocks = self.retrieve_text_chunks(file, language)
        self.model.fit_transform(grouped_blocks)
        document_info = self.model.get_topic_info()
        return document_info[0:top_n]

# class Top2VecTopicExtractor(TopicExtractor):
#     def __init__(self):
#         self.group_size = 30
#         self.overlap = 5
#
#     def extract_topics(self, file, top_n=10, language="english"):
#         grouped_sentences = self.retrieve_text_chunks(file, language)
#         model = Top2Vec(grouped_sentences, embedding_model="distiluse-base-multilingual-cased")
#         max_num_topics = model.get_num_topics()
#         return model.get_topics(top_n if max_num_topics > top_n else max_num_topics)
#
#     def topics_to_file(self, topic_words, word_scores, topic_nums, file):
#         def get_topic_words_and_probabilities(index):
#             return ";".join([f"{topic_words[index][j]}:{word_scores[index][j]}" for j in range(len(topic_words[index]))])
#         with open(file, 'w', encoding="utf-8") as f:
#             lines = [f"{topic_nums[i]} -> {get_topic_words_and_probabilities(i)}" for i in range(len(topic_nums))]
#             f.write("\n".join(lines))