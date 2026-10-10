"""Defaults for the Qwen SuperPos + RFA refinement workflow."""

SOURCE_MODEL = "Qwen/Qwen2.5-1.5B"
TARGET_MODEL = "Qwen/Qwen2.5-7B"
DATASET = "sst2"
PROMPT_LENGTH = 100
BASIS_SIZE = 128
SUPERPOS_TEMPERATURE = 0.5
MAX_LENGTH = 128
DEFAULT_SEED = 42

TRAIN_LR = 1e-2
TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_WARMUP_RATIO = 0.06

REFINE_LR = 5e-3
REFINE_EPOCHS = 5
REFINE_BATCH_SIZE = 1
RFA_TEMPERATURE = 2.0
STAY_WEIGHT = 1.0

K_SWEEP_VALUES = [
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16,
    18, 20, 24, 28, 32, 40, 50, 64, 96, 128,
]
K_SWEEP_TRAIN_PROBES = 64
K_SWEEP_VALIDATION_PROBES = 128

SST2_SEED_WORDS = [
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
    "cool", "decent", "fine", "warm", "cold", "masterpiece", "garbage",
    "trash", "recommend", "avoid", "worth", "waste", "Positive", "Negative",
    "sentiment", "review", "movie", "film", "not", "very", "really", "so",
    "too", "quite", "pretty", "one", "of", "the", "most", "least", "ever",
    "never", "always", "sometimes", "rarely", "!", "?", "...",
]
