"""Transfer learning: a pretrained model corrected on a new machine.

Technical description
- The pretrained model stays fixed. A second model learns its log error on the
  new machine from the builds measured there:
  new performance = pretrained prediction * correction(build).
- The correction sees the build options, so single options can count differently
  on the new machine and the ranking of the builds can change.

Adjustable parameters
- The pretrained model and the correction model, e.g. no-hardware and sparse-bayes.
"""

from .common import configs


class ResidualCorrection:
    def __init__(self, pretrained, model_type, hardware):
        self.pretrained = pretrained
        self.model_type = model_type
        self.hardware = hardware

    def _source(self, builds):
        prediction, _ = self.pretrained.predict_performance(self.hardware, builds)
        return prediction

    def fit(self, table):
        """table: the runs measured on the new machine so far."""
        ratio = table["performance"].to_numpy() / self._source(configs(table))
        self.correction = self.model_type().fit(
            table.assign(performance=ratio, relative=ratio / ratio.max()))
        return self

    def predict_performance(self, hardware, builds):
        source = self._source(builds)
        factor, spread = self.correction.predict_performance(hardware, builds)
        return factor * source, spread * source
