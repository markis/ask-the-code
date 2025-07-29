import os
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*deprecated.*")
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*__call__.*method.*faster.*")

from ask_the_code.cli import run

run()
