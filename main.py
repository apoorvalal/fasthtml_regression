# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "python-fasthtml",
#     "scikit-learn",
# ]
# ///
from fasthtml.common import *
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.decomposition import PCA

app, rt = fast_app()

######################################################################

def generate_random_data(n, func, sigma, xmin, xmax):
    x = np.random.uniform(xmin, xmax, n)
    y = func(x) + np.random.normal(0, sigma, n)
    return "\n".join([f"{x[i]:.2f},{y[i]:.2f}" for i in range(n)])


def create_plot(data, regression_type=None):
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    if regression_type:
        X = np.array(x).reshape(-1, 1)
        Y = np.array(y)
        if regression_type == "linear":
            model = LinearRegression().fit(X, Y)
            y_pred = model.predict(X)
        elif regression_type == "pca":
            pca = PCA(n_components=1)
            pca.fit(np.c_[X, Y])
            y_pred = pca.mean_[1] + pca.components_[0, 1] / pca.components_[0, 0] * (
                X - pca.mean_[0]
            )
        elif regression_type == "quantile":
            model = QuantileRegressor(quantile=0.5, alpha=0).fit(X, Y)
            y_pred = model.predict(X)
        plt.plot(X, y_pred, color="red")
        plt.title(f"Scatter + {regression_type}")
    else:
        plt.title("Scatter")
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

######################################################################

@rt("/")
def get():
    return Titled(
        "Regression Visualizer",
        Form(
            Div(
                Label("f(x):", For="lambda-input"),
                Input(
                    type="text",
                    id="lambda-input",
                    name="lambda_input",
                    value="2*x+1",
                    placeholder="function",
                ),
                Label("n:", For="sample-size"),
                Input(
                    type="number",
                    id="sample-size",
                    name="sample_size",
                    value="20",
                    min="10",
                    max="2000",
                    step="1",
                ),
                Label("σ:", For="sigma"),
                Input(type="text", id="sigma", name="sigma", value="0.1"),
                style="display: flex; justify-content: space-between; gap: 5px;",
            ),
            Div(
                Label("xmin:", For="xmin"),
                Input(type="text", id="xmin", name="xmin", value="-5"),
                Label("xmax:", For="xmax"),
                Input(type="text", id="xmax", name="xmax", value="5"),
                Button("Generate", hx_post="/generate_data", hx_target="#data-input"),
                style="display: flex; justify-content: space-between; gap: 3px; margin-top: 10px;",
            ),
            Textarea(id="data-input", name="data_input", rows=5),
            Div(
                Select(
                    Option("", value="", selected=True),
                    Option("Linear Regression", value="linear"),
                    Option("First Principal Component", value="pca"),
                    Option("Quantile Regression (Median)", value="quantile"),
                    id="regression-type",
                    name="regression_type",
                ),
                Button("Plot", type="submit"),
                style="display: flex; justify-content: space-between; gap: 10px; margin-top: 10px;",
            ),
            hx_post="/update_plot",
            hx_target="#plot-container",
        ),
        Div(id="plot-container", style="text-align: center;"),
        Style(
            """
            form {
                max-width: 600px;
                margin: 0 auto;
            }
            #plot-container img {
                max-width: 900px;
                height: auto;
                margin: 0 auto;
            }
            input, select {
                flex-grow: 1;
            }
            label {
                flex-basis: 50px;
                text-align: right;
                padding-right: 10px;
            }
        """
        ),
    )


@rt("/generate_data")
def post(lambda_input: str, sample_size: int, sigma: str, xmin: str, xmax: str):
    try:
        func = eval("lambda x:" + lambda_input)
        data = generate_random_data(
            sample_size, func, float(sigma), float(xmin), float(xmax)
        )
        return data
    except Exception as e:
        return f"Error: {str(e)}"


@rt("/update_plot")
def post(data_input: str, regression_type: str):
    try:
        data = [
            list(map(float, line.split(","))) for line in data_input.strip().split("\n")
        ]
        img_base64 = create_plot(
            data, regression_type if regression_type != "" else None
        )
        return Img(src=f"data:image/png;base64,{img_base64}")
    except Exception as e:
        return f"Error: {str(e)}"

######################################################################
serve()
