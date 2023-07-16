from monai.transforms import RandomizableTransform


class RandShiftIntensity100(RandomizableTransform):
    def __init__(self, prob) -> None:
        super().__init__(prob)
        self._offset = None

    def randomize(self, data=None):
        super().randomize(None)
        self._offset = self.R.uniform(low=0, high=100)

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        return img + self._offset


transform = RandShiftIntensity100(1)
transform.set_random_state(seed=2)
print(transform(10))
