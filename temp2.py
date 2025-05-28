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
                
