from dataclasses import dataclass
from .transcriber import Transcriber
from .summarizer import BuiltinLLMSummarizer, LocalLLMSummarizerAPI
from .topics_extractor import BertTopicExtractor, KeyBertTopicExtractor
from .converter import AudioExtractor
from tqdm import tqdm
from iso639 import Lang
import os
import re

@dataclass
class TranscriptionConfig:
    device: str = "cpu"
    batch_size: int = 16
    compute_type: str = "int8"
    whisperx_model: str = "small"

@dataclass
class SummarizerConfig:
    builtin_llm: bool = False
    endpoint: str = "localhost:8080"
    path: str = "/v1/chat/completions"
    max_words: int = 200,
    llm_model: str = None

@dataclass
class TopicExtractorConfig:
    top_n: int = 10
    generate_from_summary: bool = False,
    bertopic: bool = False

class LanguageFileManager:
    def save_language_to_file(self, file, language):
        iso_lang = Lang(language)
        with open(file + "-language.txt", "w") as f:
            f.write(iso_lang.name.lower())

    def retrieve_language_from_file(self, file):
        try:
            with open(file + "-language.txt", "r") as f:
                lines = f.readlines()
                return lines[0] if lines else "english"
        except FileNotFoundError:
            return "english"

class TranscriptionWorkflow(LanguageFileManager):
    def __init__(self,
                 convert_video_to_audio: bool,
                 transcription_config: TranscriptionConfig|None,
                 summarizer_config: SummarizerConfig|None,
                 topic_extractor_config: TopicExtractorConfig|None):
        self.convert_video_to_audio = convert_video_to_audio
        self.transcription_config = transcription_config
        self.summarizer_config = summarizer_config
        self.topic_extractor_config = topic_extractor_config

    def get_file_names_and_extensions(self, list_of_file_names, preserve_folder=False):
        for file in list_of_file_names:
            file_without_folder = re.split("[/\\\\]", file)[-1] if not preserve_folder else file
            file_name_bits = file_without_folder.rsplit(".", 1)
            extension = file_name_bits[-1]
            file_name_without_extension = "".join(file_name_bits[:-1])
            yield file_name_without_extension, extension

    def transcribe(self, list_of_files):
        transcriber = Transcriber(
            self.transcription_config.device,
            self.transcription_config.batch_size,
            self.transcription_config.compute_type,
            self.transcription_config.whisperx_model,
        )
        print("Transcribing audio files...")
        for file in tqdm(list_of_files):
            result, language = transcriber.transcribe(file + ".mp3")
            self.save_language_to_file(file, language)
            transcriber.to_result_file(result, file + ".txt")
            transcriber.to_subtitles_file(result, language, file + ".srt")

    def summarize(self, list_of_files):
        print("Summarizing transcriptions...")
        summarizer = BuiltinLLMSummarizer() if self.summarizer_config.builtin_llm \
            else LocalLLMSummarizerAPI(self.summarizer_config.endpoint, self.summarizer_config.path, self.summarizer_config.llm_model)
        for file in tqdm(list_of_files):
            language = self.retrieve_language_from_file(file)
            summary = summarizer.summarize(file + ".txt", self.summarizer_config.max_words, language)
            summarizer.summary_to_file(summary, file + "-summary.txt")

    def extract_topics(self, list_of_files):
        print("Extracting topics...")
        topic_extractor = BertTopicExtractor() if self.topic_extractor_config.bertopic else KeyBertTopicExtractor()
        for file in tqdm(list_of_files):
            language = self.retrieve_language_from_file(file)
            if self.topic_extractor_config.generate_from_summary:
                result = topic_extractor.extract_topics(file + "-summary.txt", self.topic_extractor_config.top_n, language)
            else:
                result = topic_extractor.extract_topics(file + ".txt", self.topic_extractor_config.top_n, language)
            topic_extractor.topics_to_file(result, file + "-topics.txt")
            # if self.topic_extractor_config.generate_from_summary:
            #     topic_words, word_scores, topic_nums = topic_extractor.extract_topics(file + "-summary.txt")
            # else:
            #     topic_words, word_scores, topic_nums = topic_extractor.extract_topics(file + ".txt", self.topic_extractor_config.top_n)

    def create_folders_per_file(self, list_of_files):
        for file in list_of_files:
            if not os.path.exists(file):
                os.makedirs(file)

    def run(self, list_of_file_names):
        files_without_extension = [file for file, _ in self.get_file_names_and_extensions(list_of_file_names)]
        files_with_folder = [f"{file}/{file}" for file in files_without_extension]
        self.create_folders_per_file(files_without_extension)
        if self.convert_video_to_audio:
            print("Extracting audio from video files...")
            for path, file_and_extension in tqdm(zip(list_of_file_names, self.get_file_names_and_extensions(list_of_file_names))):
                file, _ = file_and_extension
                AudioExtractor().convert(path, file + "/" + file + ".mp3")
        if self.transcription_config:
            if self.convert_video_to_audio:
                self.transcribe(files_with_folder)
            else:
                audio_files_without_extension = [file for file, _ in self.get_file_names_and_extensions(list_of_file_names, preserve_folder=True)]
                self.transcribe(audio_files_without_extension)
        if self.summarizer_config:
            self.summarize(files_with_folder)
        if self.topic_extractor_config:
            self.extract_topics(files_with_folder)

