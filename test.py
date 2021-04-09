
from optimizer import run

optimizer = ["GWO"]


objectivefunc = ["F3", "F4"]


NumOfRuns = 3

params = {"PopulationSize": 30, "Iterations": 50}

export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": True,
    "Export_boxplot": True,
}

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)
