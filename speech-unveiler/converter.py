from moviepy import *

class AudioExtractor:

    def convert(self, input_video_file, output_audio_file):
        video = VideoFileClip(input_video_file)
        video.audio.write_audiofile(output_audio_file, logger=None)
