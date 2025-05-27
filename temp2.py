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
                        if j == 0:
                            # 画像sigmentの表示
                            placeholders[i][j].image(
                                segment, channels="BGR", caption=seg_names[j]
                            )

                        # # Probabilistic Houghを使用してエンドポイント検出
                        detection, point = hough_detector.detect(segment)
                        
                        if not detection:
                            # CascadeClassifierを使用して物体検出
                            detection, point = cascade_classifier.detect(segment)