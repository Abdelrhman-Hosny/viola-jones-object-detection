from utils import compute_integral_image
from feature import Feature, Rectangle
from stage import Stage
from classifier import WeakClassifier

from xmlparser import parse_haar_cascade_xml

WINDOW_SIZE = (24, 24)

xml_path = r'./high-level/haar-cascades/haarcascade_frontalface_default.xml'

stages , features = parse_haar_cascade_xml(xml_path)

