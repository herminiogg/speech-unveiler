import whisperx
from whisperx.SubtitlesProcessor import SubtitlesProcessor

class Transcriber:
    def __init__(self, device, batch_size, compute_type, whisperx_model):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.model = whisperx.load_model(whisperx_model, self.device, compute_type=self.compute_type)

    def transcribe(self, audio_file):
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        language = result["language"]
        model_a, metadata = whisperx.load_align_model(language_code=language, device=self.device)
        return whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False), language

    def to_result_file(self, result, result_file):
        with open(result_file, "w", encoding="utf-8") as f:
            f.write("\n".join([seg["text"] for seg in result["segments"]]))

    def to_subtitles_file(self, result, language, subtitles_file):
        SubtitlesProcessor(result["segments"], "en").save(filename=subtitles_file)