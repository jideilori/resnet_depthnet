# Load and evalute on Eigen's test data
from evaluate import load_test_data, evaluate
from model import Resnet_UNet
rgb, depth, crop = load_test_data()
evaluate(Resnet_UNet, rgb, depth, crop)