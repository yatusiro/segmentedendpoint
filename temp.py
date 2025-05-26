from utils.cascade_classifier import CascadeClassifier

def initialize_classifier(app_config):
    """Initializes the classifier."""
    # Classifierの設定
    # cascaleファイルは deta/haarcascade/ にある。
    cascade_path = f"./data/haarcascade/{app_config['classifier']['cascade']}"
    min_size = app_config["classifier"]["min_size"]
    max_size = app_config["classifier"]["max_size"]
    classifier = CascadeClassifier(cascade_path, min_size=min_size, max_size=max_size)
    return classifier