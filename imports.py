# Generic imports
import pandas as pd
from typing import Tuple, cast
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

from thompson_sampling import PathwiseThompsonSampling
from plotting_helpers import *

# Ax & BoTorch imports
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax.utils.common.typeutils import not_none
from ax.core.types import TParameterization, TEvaluationOutcome, TModelPredict
from ax.core.observation import ObservationFeatures
from ax.modelbridge import TorchModelBridge
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.plot.contour import interact_contour_plotly
from ax.utils.notebook.plotting import render
from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics
from botorch.acquisition.analytic import (
    ProbabilityOfImprovement,
    ExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.models.gp_regression import SingleTaskGP
