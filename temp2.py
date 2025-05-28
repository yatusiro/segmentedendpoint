# 1
placeholders = []
    # placeholders_img = []
    # placeholder_txt = []

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
            placeholders.append([])
            st.write(f"{camera_name}")
            cols.append(st.columns(segment_extractors[camera_name].number))
            # プレースホルダーを作成
            for col in cols[i]:
                with col:
                    placeholders[i].append(st.empty())

    with tab2:
        for i, camera_name in enumerate(camera_names):
            placeholders_perspective.append(st.empty())

    with tab3:
        for i, camera_name in enumerate(camera_names):
            placeholders_raw.append(st.empty())

#2
                # プレースホルダーに画像を表示

                for j, segment in enumerate(segmented_images):
                    try:
                        # 保存したプレースホルダー参照を使用して画像を表示
                        placeholders[i][j].image(
                            segment,
                            channels="BGR",
                            caption=seg_names[j],  # セグメント名
                        )
                    except Exception as e:
                        logger.warning(f"Failed to display segment {j}: {e}")


#3

                        

                        # 保存したプレースホルダー参照を使用して画像を表示
                        placeholders[i][j].image(
                            segment,
                            channels="BGR",
                            caption=seg_names[j], # セグメント名
                        )
                    except Exception as e:
                        logger.warning(f"Failed to display segment {j}: {e}")

