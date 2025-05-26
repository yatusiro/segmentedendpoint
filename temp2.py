    def detect(self, image: Image.Image) -> Tuple[float, float]:
        processed = self.preprocess_image(image)
        return self.detect_endpoints(processed)
