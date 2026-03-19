# speech-unveiler
speech-unveiler is an unopinionated CLI utility to transcribe, summarise and extract keywords
from video or audio files following a privacy-first and local approach.  It was born as a proof 
of concept for transcribing more than 1000 oral testimonies at [Kazerne Dossin](https://kazernedossin.eu/) 
helping to unveil (thus its name) the contents of this vast collection and facilitate its access and study.
It is designed, however, in a generic manner so you may use this tool to transcribe your meetings, create
a summary of them and extract the main topics as well.

## Approach
The tool follows the pipeline design principle by which:
1. Video inputs in MP4 format are transformed to MP3 audio files.
2. Audio files are transcribed using [WhisperX](https://github.com/m-bain/whisperx).
3. Transcriptions are summarised using LLMs.
4. Topics and keywords are extracted from the transcript or the summary.

Notwithstanding this workflow, you can always decide which steps to run and skip those in which you are not 
interested or that have been run in a previous execution. See [CLI](#cli) for more details.

## Hardware requirements and limitations
While this tool has been designed with the idea of enabling the execution of the full workflow on a conventional laptop 
(as the default CLI options demonstrate), for a faster processing a CUDA-capable GPU is strongly recommended. In this case,
the default options need to be overridden and the adequate values for your hardware need to be provided (see 
[examples](#examples)). 

## CLI
The CLI can be accessed through the following command:
```bash
python -m speech-unveiler [OPTIONS] FILE_OR_FOLDER
```
The tool accepts either a single file or a folder. When a folder is provided, all files within it are processed and should
be in the same file format.

The rest of the options are summarised in the CLI help section:
```
Usage: python -m speech-unveiler [OPTIONS] FILE_OR_FOLDER

Options:
  --audio                  The input is already an audio MP3 file so the
                           extraction of audio from video files should be
                           skipped.
  --skip-transcription     Skips the transcription phase.
  --device TEXT            Device to use for transcription. E.g.: cpu, cuda,
                           etc.  [default: cpu]
  --batch-size INTEGER     Batch size for transcription.  [default: 16]
  --compute-type TEXT      Compute type for transcription. Eg  [default: int8]
  --model TEXT             WhisperX model to use.  [default: small]
  --skip-summarization     Skips the summarization phase.
  --builtin-llm            Uses a builtin LLM, at the moment Qwen/Qwen3-0.6B.
                           NOTE: as this model is not quantized you may expect
                           higher hardware requirements.
  --endpoint TEXT          LLM API endpoint.  [default: localhost:8080]
  --path TEXT              LLM API path.  [default: /v1/chat/completions]
  --max-words INTEGER      Max words for summary.  [default: 200]
  --llm-model TEXT         The LLM to use. You can skip this if your system
                           has a default one pre-defined.
  --skip-topic-extraction  Skips the topic extraction phase.
  --top-n INTEGER          Number of topics to extract.  [default: 10]
  --from-summary           Extract topics from summary instead of transcript.
  --bertopic               EXPERIMENTAL: Use BERTopic for the extraction of
                           topics. It may fail on short transcripts.
  --help                   Show this message and exit.
```

### Options
#### Input
| Option | Default | Description |
|--------|---------|-------------|
| `FILE_OR_FOLDER` | *(required)* | Path to a single file or a directory of files to process. |
| `--audio` | `false` | Treat the input as an already-extracted MP3 audio file, skipping the video-to-audio conversion step. |

#### Transcription

These options control the WhisperX transcription engine. Use `--skip-transcription` if a transcript already exists.

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-transcription` | `false` | Skip the transcription phase entirely. |
| `--model` | `small` | WhisperX model size to use. Larger models are more accurate but slower and more resource-intensive. Common values: `tiny`, `base`, `small`, `medium`, `large`. |
| `--device` | `cpu` | Hardware device for inference. Use `cpu` for standard machines, `cuda` for NVIDIA GPUs. |
| `--compute-type` | `int8` | Numerical precision for inference. `int8` is fastest with lowest memory usage; `float16` or `float32` offer higher precision at greater cost. |
| `--batch-size` | `16` | Number of audio chunks processed simultaneously. Increase for faster throughput if memory allows; decrease if you encounter out-of-memory errors. |

**Choosing a model:** The `small` model is a good starting point for general use. For technical or domain-specific content, consider `medium` or `large` for improved accuracy. Note that larger models require significantly more memory and compute time.

---

### Summarisation

The summarisation phase sends the transcript to an LLM via a chat completions API. You can use a local server (e.g. llama.cpp, Ollama) or any compatible remote endpoint.

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-summarization` | `false` | Skip the summarisation phase entirely. |
| `--builtin-llm` | `false` | Use the built-in bundled LLM (`Qwen/Qwen3-0.6B`) instead of an external endpoint. Note: this model is not quantized and may require significant memory. |
| `--endpoint` | `localhost:8080` | Host and port of the LLM API server. |
| `--path` | `/v1/chat/completions` | API path on the endpoint. Compatible with OpenAI-style APIs. |
| `--llm-model` | `None` | The model name to pass to the API. Can be omitted if your server has a default model configured. |
| `--max-words` | `200` | Target maximum word count for the generated summary. |

**Using a remote LLM:** To use an external provider (e.g. a self-hosted server on another machine), set `--endpoint` to 
the appropriate host:port and `--path` to the API route. Ensure the endpoint is reachable from your machine. The `--llm-model`
option allows to provide the model in case the used server requires this attribute.

**Built-in LLM:** This option requires a GPU available and the memory usage increases the longer the input 
(i.e., the transcript) is. For long inputs or limited hardware the default option (a local LLM deployment) is strongly
recommended.

---

### Topic Extraction

Extracts the most prominent topics from either the full transcript or the generated summary.

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-topic-extraction` | `false` | Skip the topic extraction phase entirely. |
| `--top-n` | `10` | Maximum number of topics to extract and return. |
| `--from-summary` | `false` | Extract topics from the summary text rather than the full transcript. Useful for long recordings where the summary captures the key themes. |
| `--bertopic` | `false` | **Experimental.** Use BERTopic for topic modelling instead of the default method. Produces richer topic clusters but may fail on short transcripts due to insufficient data for clustering. |

**BERTopic:** It is possible to use BERTopic for performing topic modeling over the transcript, however it may fail on
short transcripts and/or provide inaccurate results. As such, this is still an experimental feature.

### Examples
Transcribing a set of videos:
```bash
python -m speech-unveiler videos
```

Transcribing a single MP3 file and using Ollama as LLM provider:
```bash
python -m speech-unveiler audio.mp3 --audio --endpoint localhost:11434 --llm-model qwen3-vl:2b-instruct-q4_K_M
```

Transcribing a single MP3 file and using Ollama as LLM provider:
```bash
python -m speech-unveiler audio.mp3 --audio --endpoint localhost:11434 --llm-model qwen3-vl:2b-instruct-q4_K_M
```

Skipping transcription (previously generated) and using the built-in LLM:
```bash
python -m speech-unveiler audio.mp3 --audio --skip-transcription --builtin-llm
```

### Running on Jupyter Notebooks
This [notebook](notebooks/running_on_google_colab.ipynb) exemplifies how to use this library within Jupyter Notebooks. 
If you need access to a GPU for a fast transcription you can this example within Google Colab which offers a free
GPU-tier allowing you to transcribe a handful of medium-size videos faster. For more intensive uses you may need to rely 
on dedicated GPU instances.