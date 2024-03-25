# local path
#XES_TRAIN_SRC = "hepsycode_models/Dataset-Benchmark/"
XES_TRAIN_SRC = "XES_original_train"

XES_TEST_SRC = "XES_original_test/"

XES_TRAIN_DST = "train_MG/"
DUMP_TRAIN = "preprocessed_train_data.pkl"
XES_TEST_DST = "test_MG/"

KB_TRAIN = "train.txt"
PARSED_DATA = "/test_xes_generated/"
REC_DST = "recommendations/"

# crossfold pahts
CROSS_ROOT_HEPSYCODE = "hepsycode_five_fold/"
CROSS_ROOT_STD = "xes_5_fold/"

XES_TRAIN_CROSS_SRC = '/train/'
XES_TEST_CROSS_SRC = '/test/'

CROSS_KB_SRC = '/train_MG/train.txt'

XES_TRAIN_CROSS_DST = '/train_MG/'
XES_TEST_CROSS_DST = '/test_MG/'
XES_GT_CROSS_DST = '/gt_MG/'

# eclipse MER paths
XES_SESSION_TRAIN_SRC = "XES_original_train/"
XES_SESSION_TRAIN_DST = "train_session_MG/"

# recommendation paramenters
CUT_OFF = 10
CONTEXT_RATIO = 0.8

