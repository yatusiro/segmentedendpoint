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