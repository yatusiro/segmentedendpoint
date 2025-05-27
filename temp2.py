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