import os.path
import click
from .transcription_workflow import TranscriptionWorkflow, TranscriptionConfig, SummarizerConfig, TopicExtractorConfig

@click.command()
@click.argument("file_or_folder", required=True)
@click.option("--audio", is_flag=True, default=False, help="The input is already an audio MP3 file so the extraction of audio from video files should be skipped.")
# Transcription options
@click.option("--skip-transcription", is_flag=True, default=False, show_default=True, help="Skips the transcription phase.")
@click.option("--device", default="cpu", show_default=True, help="Device to use for transcription. E.g.: cpu, cuda, etc.")
@click.option("--batch-size", default=16, show_default=True, help="Batch size for transcription.")
@click.option("--compute-type", default="int8", show_default=True, help="Compute type for transcription. Eg")
@click.option("--model", default="small", show_default=True, help="WhisperX model to use.")
# Summarizer options
@click.option("--skip-summarization", is_flag=True, default=False, show_default=True, help="Skips the summarization phase.")
@click.option("--builtin-llm", is_flag=True, default=False, show_default=True, help="Uses a builtin LLM, at the moment Qwen/Qwen3-0.6B. NOTE: as this model is not quantized you may expect higher hardware requirements.")
@click.option("--endpoint", default="localhost:8080", show_default=True, help="LLM API endpoint.")
@click.option("--path", default="/v1/chat/completions", show_default=True, help="LLM API path.")
@click.option("--max-words", default=200, show_default=True, help="Max words for summary.")
@click.option("--llm-model", default=None, help="The LLM to use. You can skip this if your system has a default one pre-defined.")
# Topic extractor options
@click.option("--skip-topic-extraction", is_flag=True, default=False, show_default=True, help="Skips the topic extraction phase.")
@click.option("--top-n", default=10, show_default=True, help="Number of topics to extract.")
@click.option("--from-summary", is_flag=True, default=False, help="Extract topics from summary instead of transcript.")
@click.option("--bertopic", is_flag=True, default=False, show_default=True, help="EXPERIMENTAL: Use BERTopic for the extraction of topics. It may fail on short transcripts.")
def cli(file_or_folder,
        audio,
        skip_transcription,
        device,
        batch_size,
        compute_type,
        model,
        skip_summarization,
        builtin_llm,
        endpoint,
        path,
        max_words,
        llm_model,
        skip_topic_extraction,
        top_n,
        from_summary,
        bertopic):
    transcription_config = TranscriptionConfig(device, batch_size, compute_type, model) if not skip_transcription else None
    summarizer_config = SummarizerConfig(builtin_llm, endpoint, path, max_words, llm_model) if not skip_summarization else None
    topic_extractor_config = TopicExtractorConfig(top_n, from_summary, bertopic) if not skip_topic_extraction else None
    workflow = TranscriptionWorkflow(
        convert_video_to_audio=not audio,
        transcription_config=transcription_config,
        summarizer_config=summarizer_config,
        topic_extractor_config=topic_extractor_config,
    )
    if os.path.isfile(file_or_folder):
        workflow.run([file_or_folder])
    elif os.path.isdir(file_or_folder):
        workflow.run(list(map(lambda x: x.path, os.scandir(file_or_folder))))
    else:
        raise Exception("The input is not a file or a directory")

if __name__ == "__main__":
    cli()