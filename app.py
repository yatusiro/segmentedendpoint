import streamlit as st
import cv2
import requests
import numpy as np
import time
from collections import deque
from utils.app_utils import log_and_display, initialize_session
from utils.logger import logger
from utils.kalman_filter import KalmanPointFilter
from datetime import datetime
import os

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


def main():
    logger.debug("Application started.")

    # セッションを初期化
    initialize_session()

    # 設定を取得
    app_config = st.session_state["app_config"]
    perspective_transformers = st.session_state["perspective_transformer"]
    segment_extractors = st.session_state["segment_extractor"]
    camera_handlers = st.session_state["camera_handler"]
    cascade_classifier = st.session_state["classifier"]
    hough_detector = st.session_state["hough_detector"]

    # Streamlit ページ設定
    st.set_page_config(
        page_title=app_config["page"]["main"]["title"],
        page_icon=app_config["page"]["main"]["icon"],
        layout=app_config["page"]["main"]["layout"],
        initial_sidebar_state=app_config["page"]["main"]["sidebar"],
    )
    st.title("Camera Stream")

    # 有効なカメラの数と名前を取得
    n_cameras = len(camera_handlers)
    camera_names = camera_handlers.keys()

    # コントロールボタン
    # ストリーミング状態に応じたボタン表示
    if not st.session_state["streaming"]:
        if st.button("▶️ Start process"):
            # カルマンフィルタインスタンスを作成
            kalman_filters = []

            # # dequeを使った時系列データ構造の作成
            # time_series_data = []

            # # 時系列データの最大長を設定
            # max_history = app_config.get("time_chart", {}).get("max_history", 100)

            for i, camera_name in enumerate(camera_names):
                number_of_segments = segment_extractors[camera_name].number

                # 各カメラの各セグメントにカルマンフィルタを作成
                kalman_filters.append([])
                # # 各カメラの時系列データ構造
                # time_series_data.append([])

                for _ in range(number_of_segments):
                    # カルマンフィルタの初期化
                    kalman_filters[i].append(
                        KalmanPointFilter(
                            process_noise=app_config["kalman_filter"]["process_noise"],
                            measurement_noise=app_config["kalman_filter"][
                                "measurement_noise"
                            ],
                        )
                    )

                    # # 各セグメントのdequeを初期化
                    # time_series_data[i].append(
                    #     {
                    #         "raw_x": deque(maxlen=max_history),
                    #         "raw_y": deque(maxlen=max_history),
                    #         "kalman_x": deque(maxlen=max_history),
                    #         "kalman_y": deque(maxlen=max_history),
                    #         "timestamps": deque(maxlen=max_history),
                    #     }
                    # )

            st.session_state["kalman_filters"] = kalman_filters
            # st.session_state["time_series_data"] = time_series_data
            # st.session_state["max_history"] = max_history
            # logger.info(
            #     f"Process started. Time series data limited to {max_history} points."
            # )
            st.session_state["streaming"] = True
            st.session_state["process_start_time"] = datetime.now()

            st.rerun()
    else:
        if st.button("⏸️ Stop process"):
            logger.info("Process stopped.")
            st.session_state["streaming"] = False
            st.rerun()

    # 残り時間を表示するためのプレースホルダー
    remaining_time = st.empty()

    # FPSに基づいてフレーム間隔を計算
    fps = app_config["camera"]["common"]["fps"]
    frame_interval = 1.0 / fps

    # プレースホルダー定義(2次元リスト)
    # placeholders = []
    placeholders_img = []
    placeholders_txt = []

    # 台形補正のプレースホルダー
    placeholders_perspective = []

    # raw imageのプレースホルダー
    placeholders_raw = []

    # レイアウト作成
    tab1, tab2, tab3 = st.tabs(["Segments", "Perspective", "Raw image"])
    # ストリーミングタブ
    with tab1:
        cols = []
        for i, camera_name in enumerate(camera_names):
            placeholders_img.append([])
            placeholders_txt.append([])
            st.write(f"{camera_name}")
            cols.append(st.columns(segment_extractors[camera_name].number))
            # プレースホルダーを作成
            for col in cols[i]:
                with col.container():
                    ph_img = st.empty()
                    ph_txt = st.empty()
                    placeholders_img[i].append(ph_img)
                    placeholders_txt[i].append(ph_txt)
                    

    with tab2:
        for i, camera_name in enumerate(camera_names):
            placeholders_perspective.append(st.empty())

    with tab3:
        for i, camera_name in enumerate(camera_names):
            placeholders_raw.append(st.empty())

    # Time Chart タブ
    # with tab2:
    #     fig = make_subplots(
    #         rows=2, cols=2, subplot_titles=["Raw X", "Raw Y", "Kalman X", "Kalman Y"]
    #     )
    #     for i, camera_name in enumerate(camera_names):
    #         for j in range(segment_extractors[camera_name].number):
    #             if "time_series_data" in st.session_state:
    #                 data = st.session_state["time_series_data"][i][j]
    #                 # dequeをリストに変換してプロット
    #                 if data["timestamps"]:
    #                     # 基準時間を設定（最初のタイムスタンプを0として相対時間に変換）
    #                     base_time = list(data["timestamps"])[0]
    #                     rel_times = [t - base_time for t in data["timestamps"]]

    #                     # dequeをリストに変換してプロット
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=rel_times,
    #                             y=list(data["raw_x"]),
    #                             mode="markers",
    #                             name="検出X",
    #                             marker=dict(color="red", size=6),
    #                         ),
    #                         row=1,
    #                         col=1,
    #                     )

    #                     # 他のプロットも同様に修正
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=rel_times,
    #                             y=list(data["raw_y"]),
    #                             mode="markers",
    #                             name="検出Y",
    #                             marker=dict(color="blue", size=6),
    #                         ),
    #                         row=1,
    #                         col=2,
    #                     )
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=rel_times,
    #                             y=list(data["kalman_x"]),
    #                             mode="lines",
    #                             name="カルマンX",
    #                             line=dict(color="green", width=2),
    #                         ),
    #                         row=2,
    #                         col=1,
    #                     )
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=rel_times,
    #                             y=list(data["kalman_y"]),
    #                             mode="lines",
    #                             name="カルマンY",
    #                             line=dict(color="orange", width=2),
    #                         ),
    #                         row=2,
    #                         col=2,
    #                     )
    #     fig.update_layout(height=600, width=800, title_text="Time Chart")
    #     st.plotly_chart(fig)

    # ストリーミングが停止中で、既存フレームがある場合は表示
    if not st.session_state["streaming"]:
        for i, camera_name in enumerate(camera_names):
            # current_frameがあるか確認する
            if camera_name in st.session_state["current_frames"]:
                # frameがある場合、処理を行う
                current_frame = st.session_state["current_frames"][camera_name]
                transformer = perspective_transformers[camera_name]
                extractor = segment_extractors[camera_name]

                # raw imageをプレースホルダーに表示
                placeholders_raw[i].image(
                    current_frame, channels="BGR", caption=camera_name
                )

                # 透視投影補正
                transformed_frame = transformer.transform(current_frame)
                # プレースホルダーに画像を表示
                placeholders_perspective[i].image(
                    transformed_frame, channels="BGR", caption=camera_name
                )

                # Segmentsを抽出
                segmented_images = extractor.extract(transformed_frame)
                seg_names = app_config["camera"]["individual"][i]["segment"]["names"]

                # プレースホルダーに画像を表示

                # for j, segment in enumerate(segmented_images):
                #     try:
                #         # 保存したプレースホルダー参照を使用して画像を表示
                #         placeholders_img[i][j].image(
                #             segment,
                #             channels="BGR",
                #         )
                #         caption = f"{j+1}\n(-- mm, -- mm)"
                #         placeholders_txt[i][j].markdown(caption, unsafe_allow_html=True)

                #     except Exception as e:
                #         logger.warning(f"Failed to display segment {j}: {e}")
                
                for j, segment in enumerate(segmented_images):
                    try:
                        # 保存したプレースホルダー参照を使用して画像を表示
                        placeholders_img[i][j].image(
                            segment,
                            channels="BGR",
                            use_container_width=True,
                        )
                        placeholders_txt[i][j].markdown(f"div style='text-align: center;'>{seg_names[j]}</br>(-- mm, -- mm)</div>", unsafe_allow_html=True)

                    except Exception as e:
                        logger.warning(f"Failed to display segment {j}: {e}")
                

    # ストリーミング処理
    while st.session_state["streaming"]:
        # 開始時間を記録
        start_time = time.time()

        # 各カメラの画像取得と処理
        for i, camera_name in enumerate(camera_names):
            handler = camera_handlers[camera_name]
            transformer = perspective_transformers[camera_name]
            extractor = segment_extractors[camera_name]

            snapshot = handler.get_snapshot()
            if snapshot.status_code == 200:
                # 画像をデコード
                frame = cv2.imdecode(
                    np.frombuffer(snapshot.content, dtype=np.uint8), cv2.IMREAD_COLOR
                )

                # raw imageをプレースホルダーに表示
                placeholders_raw[i].image(frame, channels="BGR", caption=camera_name)

                # 保存モードならフレームを保存
                if app_config["process"]["save_image"]:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(
                        "./data/image", camera_name, f"{timestamp}.jpg"
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, frame)

                # フレームをセッション状態に保存
                st.session_state["current_frames"][camera_name] = frame.copy()

                # 透視投影補正
                transformed_frame = transformer.transform(frame)
                # プレースホルダーに画像を表示
                placeholders_perspective[i].image(
                    transformed_frame, channels="BGR", caption=camera_name
                )

                # Segmentsを抽出
                segmented_images = extractor.extract(transformed_frame)
                seg_names = app_config["camera"]["individual"][i]["segment"]["names"]

                # プレースホルダーに画像を表示
                for j, segment in enumerate(segmented_images):
                    try:
                        # save_segment = trueの場合はセグメントを保存
                        if app_config["process"]["save_segment"]:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            dir = "workspace/data/image/train/no_anotation"
                            name = f"{camera_name}_{seg_names[j]}_{timestamp}.jpg"
                            save_path = os.path.join(dir, name)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            cv2.imwrite(save_path, segment)
                            
                        # # Probabilistic Houghを使用してエンドポイント検出
                        detection, point = hough_detector.detect(segment)
                        
                        if not detection:
                            # CascadeClassifierを使用して物体検出
                            color=(255,0,0)
                            detection, point = cascade_classifier.detect(segment)
                        else:
                            color=(0,0,255)

                        # カルマンフィルタを適用
                        # kalman_point = st.session_state["kalman_filters"][i][j].update(
                        #     point
                        # )

                        # # 時系列データの更新（dequeを使用）
                        # if "time_series_data" in st.session_state:
                        #     data = st.session_state["time_series_data"][i][j]
                        #     # 現在時刻を取得
                        #     current_time = time.time()

                        #     # dequeは自動的に最大長を管理するので、単純に追加するだけでよい
                        #     data["raw_x"].append(point[0] if detection else None)
                        #     data["raw_y"].append(point[1] if detection else None)
                        #     data["kalman_x"].append(kalman_point[0])
                        #     data["kalman_y"].append(kalman_point[1])
                        #     data["timestamps"].append(current_time)

                        # # 検出された場合,赤い円を描画
                        # if detection:
                        #     cv2.circle(
                        #         segment,
                        #         (point[0], point[1]),
                        #         3,
                        #         (0, 0, 255),
                        #         -1,
                        #     )

                        # カルマンフィルタの結果を円で描画
                        # 検出された場合は緑、検出されておらずFilteringによる推定の場合は緑
                        # if detection:
                        #     color = (0, 0, 255)  # 赤色
                        # else:
                        #     color = (0, 255, 0)  # 緑色
                        # if kalman_point[0] is not None and kalman_point[1] is not None:
                        #     # 検出された場合は赤、検出されておらずFilteringによる推定の場合は緑
                        #     cv2.circle(
                        #         segment,
                        #         kalman_point,
                        #         3,
                        #         color,  # 緑色
                        #         -1,
                        #     )




                        if detection:
                            kalman_point = st.session_state["kalman_filters"][i][j].update(
                                point
                            )
                            cv2.circle(
                                segment,
                                point,
                                3,
                                color,  # 赤色
                                -1,
                            )
                        else:
                            kalman_point = st.session_state["kalman_filters"][i][j].update(
                                (None, None)
                            )
                            if kalman_point[0] is not None and kalman_point[1] is not None:
                                cv2.circle(
                                    segment,
                                    kalman_point,
                                    3,
                                    (0, 255, 0),  # 緑色
                                    -1,
                                )
                        

                        # 保存したプレースホルダー参照を使用して画像を表示
                        placeholders_img[i][j].image(
                            segment,
                            channels="BGR",
                        )
                        if detection:
                            x_py, y_py = point
                            coord = f"({x_py:.1f} mm , {y_py:.1f} mm)"
                        else:
                            x_py, y_py = kalman_point
                            coord = f"({x_py:.1f} mm , {y_py:.1f} mm)"
                            
                        # caption = f"{j+1}\n{coord}"
                        # placeholders_txt[i][j].markdown(
                        #     caption, unsafe_allow_html=True
                        # )
                        placeholders_txt[i][j].markdown(
                            f"<div style='text-align: center;'>{seg_names[j]}</br>{coord}</div>",
                            unsafe_allow_html=True,
                        )

                    except Exception as e:
                        logger.warning(f"Failed to display segment {j}: {e}")
            else:
                # contentにはエラー画像が含まれているため、そのまま表示可能
                frame = cv2.imdecode(
                    np.frombuffer(snapshot.content, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                logger.error(
                    f"Failed to get snapshot from camera {camera_name}. Status code: {snapshot.status_code}"
                )

        # 処理時間を計算
        processing_time = time.time() - start_time
        # frame_intervalに基づいて待機時間を調整
        wait_time = max(0.1, frame_interval - processing_time)

        # auto stop が Trueの場合、一定時間後にストリーミングを停止
        if app_config["process"]["auto_stop"]:
            # 一定時間後にストリーミングを停止
            process_duration = (
                datetime.now() - st.session_state["process_start_time"]
            ).total_seconds() / 3600  # 時間単位に変換
            # 残り時間を表示
            remain_hour, remain_min = divmod(
                app_config["process"]["auto_stop_time"] - process_duration, 1
            )
            remain_min, remain_sec = divmod(remain_min * 60, 1)
            remain_sec = remain_sec * 60
            remaining_time.markdown(
                f"**Remaining time: {int(remain_hour)}:{int(remain_min)}:{int(remain_sec)}**"
            )
            if process_duration > app_config["process"]["auto_stop_time"]:
                logger.info("Process auto stopped.")
                st.session_state["streaming"] = False
                st.rerun()

        time.sleep(wait_time)


if __name__ == "__main__":
    main()
