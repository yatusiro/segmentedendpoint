import streamlit as st
import tomllib
from utils.image_edit import PerspectiveTransformer, SegmentExtractor
from utils.logger import logger
import numpy as np
from utils.camera_handler import AxisCamera
from utils.logger import logger
from utils.cascade_classifier import CascadeClassifier
from utils.probabilistic_hough import ProbabilisticHough


def initialize_session():
    """Initializes session-wide resources."""
    if "initialized" not in st.session_state:

        # 設定を読み込む
        st.session_state["app_config"] = load_app_config()

        # Camera handler を初期化
        st.session_state["camera_handler"] = initialize_camera_handler(
            st.session_state["app_config"]
        )

        # Perspective Transformer を初期化
        perspective_transformer = initialize_perspective_transformer(
            st.session_state["app_config"]
        )
        # Camera handlerと共通の名前を持つもののみを抽出
        camera_names = st.session_state["camera_handler"].keys()
        st.session_state["perspective_transformer"] = {
            name: perspective_transformer[name]
            for name in camera_names
            if name in perspective_transformer
        }

        # Segment Extractor を初期化
        segment_extractor = initialize_segment_extractor(st.session_state["app_config"])
        # Camera handlerと共通の名前を持つもののみを抽出
        st.session_state["segment_extractor"] = {
            name: segment_extractor[name]
            for name in camera_names
            if name in segment_extractor
        }

        # Streamling状態を初期化
        st.session_state["streaming"] = False
        st.session_state["current_frames"] = {}

        # Classifierを初期化
        st.session_state["classifier"] = initialize_classifier(
            st.session_state["app_config"]
        )

        st.session_state["kalman_filters"] = None

        # セッションの初期化フラグを設定
        st.session_state["initialized"] = True

        logger.info("Session initialized successfully.")