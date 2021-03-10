TRAIN_BS = 48
VALID_BS = 16
TEST_BS = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = "cuda"

EPOCHS = 50
LEARNING_RATE = 1e-4
FOLDS = 5

MODEL_PATH = 'models'
#MODEL_PATH = "https://melenoma-classifier-models.s3.amazonaws.com"

# IMAGE FILES
TRAINING_DATA_PATH = 'DATA/train224/'
TEST_DATA_PATH = 'DATA/test224/'

# DATA FILES
TRAIN_FOLDS = 'DATA/train_folds.csv'
TRAIN_DATASET = 'DATA/train.csv'
TEST_DATASET = 'DATA/test.csv'
SUBMISSION_DATASET = 'DATA/sample_submission.csv'