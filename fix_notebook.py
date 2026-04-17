import json

with open("Walmart_Time_Series_Forecast.ipynb", "r") as f:
    data = json.load(f)

for cell in data["cells"]:
    if cell["cell_type"] == "code":
        sources = cell["source"]
        for i, line in enumerate(sources):
            if "from keras.wrappers.scikit_learn import KerasRegressor" in line:
                sources[i] = line.replace("from keras.wrappers.scikit_learn import KerasRegressor", "from scikeras.wrappers import KerasRegressor")
            if "data['Date'].dt.week" in line:
                sources[i] = line.replace("data['Date'].dt.week", "data['Date'].dt.isocalendar().week")
            if "LinearRegression(normalize=False)" in line:
                sources[i] = line.replace("LinearRegression(normalize=False)", "LinearRegression()")
            if "estimator_model = KerasRegressor(build_fn=create_model, verbose=1)" in line:
                sources[i] = line.replace("estimator_model = KerasRegressor(build_fn=create_model, verbose=1)", "estimator_model = KerasRegressor(model=create_model, verbose=1)")

with open("Walmart_Time_Series_Forecast.ipynb", "w") as f:
    json.dump(data, f, indent=2)
