# copy into config.py

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CHECKPOINT_DIR = "./checkpoints"
SUMMARY_DIR = './summary'
SAMPLE_IMAGE_PATH = './image_example'
LOG_FILE = './logs/sem_segm.log'

MAX_STEPS = 2000
BATCH_SIZE = 20
ACCURACY_EVAL_TRAIN_PORTION = 0.1
ACCURACY_EVAL_TEST_PORTION = 0.25
DROP_PROB = 0.5
SAVE_AND_EVAL_EVERY = 100
SUMMARY_EVERY = 5

IM_PATH = "./PASCAL_2012/JPEGImages/"
SE_PATH = "./PASCAL_2012/SegmentationClass/"
TRAIN_VAL_PATH = "./PASCAL_2012/ImageSets/Segmentation/"
MAX_HEIGHT = 320
MAX_WIDTH = 320

