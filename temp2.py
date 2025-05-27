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

                        # CascadeClassifierを使用して物体検出
                        # detection, point = cascade_classifier.detect(segment)

                        # Probabilistic Houghを使用してエンドポイント検出
                        detection, point = hough_detector.detect(segment)
                        
                        # 検出結果をログに記録
                        print(point)

                        # カルマンフィルタを適用
                        kalman_point = st.session_state["kalman_filters"][i][j].update(
                            point
                        )

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
                        if detection:
                            color = (0, 0, 255)  # 赤色
                        else:
                            color = (0, 255, 0)  # 緑色
                        if kalman_point[0] is not None and kalman_point[1] is not None:
                            # 検出された場合は赤、検出されておらずFilteringによる推定の場合は緑
                            cv2.circle(
                                segment,
                                kalman_point,
                                3,
                                color,  # 緑色
                                -1,
                            )

                        # 保存したプレースホルダー参照を使用して画像を表示
                        placeholders[i][j].image(
                            segment,
                            channels="BGR",
                            caption=seg_names[j],
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