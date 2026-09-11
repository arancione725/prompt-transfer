"""Unified configuration for Qwen cross-model prompt transfer."""

# --- Model ---
MODEL_1_5B = "Qwen/Qwen2.5-1.5B"
MODEL_7B = "Qwen/Qwen2.5-7B"
HIDDEN_1_5B = 1536
HIDDEN_7B = 3584

# --- Prompt ---
PROMPT_LEN = 100
PROMPT_INIT_TOKEN = "<|endoftext|>"  # fallback init token
SUPERPOS_M = 128  # number of sampled token embeddings per SuperPos prompt position
SUPERPOS_TEMPERATURE = 0.5  # softmax temperature for convex-combination enforcement
DEFAULT_SEED = 42
PROMPT_INIT_TEXT = {
    "sst2": "Classify the sentiment of this movie review. Determine whether it is positive or negative. Analyze the tone and emotion expressed in the text. The overall feeling conveyed is",
    # Add more datasets here
}

SST2_SENTIMENT_SEEDS = [
    "positive", "negative", "good", "bad", "great", "terrible",
    "excellent", "awful", "wonderful", "horrible", "love", "hate",
    "liked", "disliked", "amazing", "boring", "fantastic", "poor",
    "best", "worst", "happy", "sad", "enjoy", "disappointed",
    "beautiful", "ugly", "fun", "dull", "brilliant",
    "nice", "nasty", "superb", "dreadful", "delightful", "miserable",
    "impressive", "pathetic", "lovely", "disgusting", "incredible",
    "lousy", "outstanding", "disappointing", "magnificent",
    "enjoyable", "tedious", "fabulous", "rotten", "splendid", "appalling",
    "charming", "vile", "pleasing", "atrocious", "awesome", "lame",
    "cool", "decent", "fine", "warm", "cold",
    "masterpiece", "garbage", "trash",
    "recommend", "avoid", "worth", "waste",
    # SST-2 label words
    "Positive", "Negative",
    "sentiment", "review", "movie", "film",
    # Common review modifiers
    "not", "very", "really", "so", "too", "quite", "pretty",
    "one", "of", "the", "most", "least", "ever", "never",
    "always", "sometimes", "rarely",
    # Punctuation / emphasis carriers
    "!", "?", "...",
]


# --- Dataset ---
DATASET = "sst2"
NUM_LABELS = 2
MAX_SEQ_LENGTH = 128

# --- Multi-GPU ---
# Set to 0 for auto-detect all available GPUs, 1 for single GPU
NUM_GPUS = 0

# --- Training (prompt tuning on source model) ---
PROMPT_LR = 1e-2
PROMPT_EPOCHS = 5
PROMPT_BATCH_SIZE = 16       # per-GPU batch size (effective batch = BATCH_SIZE × num_gpus)
PROMPT_WARMUP_RATIO = 0.06
PROMPT_WEIGHT_DECAY = 1e-4
PROMPT_ANCHOR_WEIGHT = 0.05  # strength of cosine-sim regularizer (keeps prompt near token manifold)

# --- Training (projector) ---
PROJECTOR_LR = 1e-3  # lower than paper's 1e-1 — 7B model gradients cause NaN at high LR
PROJECTOR_EPOCHS = 5
PROJECTOR_BATCH_SIZE = 16    # per-GPU batch size
PROJECTOR_WARMUP_RATIO = 0.0
PROJECTOR_WEIGHT_DECAY = 0.0

# --- Output ---
OUTPUT_DIR = "outputs_qwen"
