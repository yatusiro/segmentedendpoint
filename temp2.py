class KalmanPointFilter:
    """
    カルマンフィルタを使用して点の位置を推定するクラス
    """

    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        """
        Args:
            process_noise: プロセスノイズ（システムモデルの不確かさ）
            measurement_noise: 測定ノイズ（検出値の不確かさ）
        """
        # カルマンフィルタの初期化
        self.kalman = cv2.KalmanFilter(
            4, 2
        )  # 状態変数4次元(x,y,dx,dy), 測定値2次元(x,y)

        # 状態遷移行列 (A)
        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],  # x = x + dx
                [0, 1, 0, 1],  # y = y + dy
                [0, 0, 1, 0],  # dx = dx (速度は一定と仮定)
                [0, 0, 0, 1],  # dy = dy (速度は一定と仮定)
            ],
            np.float32,
        )

        # 測定行列 (H) - 状態から測定値への変換
        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],  # 測定値xは状態変数xに対応
                [0, 1, 0, 0],  # 測定値yは状態変数yに対応
            ],
            np.float32,
        )

        # プロセスノイズ共分散行列 (Q)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 測定ノイズ共分散行列 (R)
        self.kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * measurement_noise
        )

        # 事後誤差推定共分散行列 (P)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        # フィルタが初期化されたかどうか
        self.initialized = False

    def update(self, point):
        """
        測定値を使用してフィルタを更新し、推定位置を返す

        Args:
            point: 検出された点の(x, y)座標。検出されなかった場合は(None, None)

        Returns:
            推定された点の(x, y)座標
        """
        x, y = point

        # 検出点が無効な場合
        if x is None or y is None:
            if not self.initialized:
                # フィルタが初期化されていない場合は推定不能
                return (None, None)

            # 予測のみを実行して推定値を返す（測定値の更新なし）
            predicted = self.kalman.predict()
            return (int(predicted[0][0]), int(predicted[1][0]))

        # 検出点を測定値として設定
        measurement = np.array([[x], [y]], dtype=np.float32)

        if not self.initialized:
            # 初めての有効な検出点ならフィルタを初期化
            self.kalman.statePost = np.array(
                [[x], [y], [0], [0]],  # 初期状態: 位置=(x,y), 速度=(0,0)
                dtype=np.float32,
            )
            self.initialized = True
            return (x, y)

        # カルマンフィルタの予測ステップ
        predicted = self.kalman.predict()

        # カルマンフィルタの修正ステップ（測定値で更新）
        corrected = self.kalman.correct(measurement)

        # 補正された推定値を返す
        return (int(corrected[0][0]), int(corrected[1][0]))

    def reset(self):
        """フィルタをリセットする"""
        self.initialized = False
