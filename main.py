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

hdrs = (MarkdownJS(), HighlightJS(langs=["python"]), KatexMarkdownJS())

app, rt = fast_app(hdrs=hdrs)

######################################################################


def generate_random_data(n, func, sigma, xmin, xmax):
    x = np.random.uniform(xmin, xmax, n)
    y = func(x) + np.random.normal(0, sigma, n)
    return "\n".join([f"{x[i]:.2f},{y[i]:.2f}, 0" for i in range(n)])


def create_plot(data, regression_type=None):
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    manual = [point[2] for point in data]
    plt.style.use("dark_background")
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=manual, cmap="jet")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    if regression_type:
        X, Y = np.array(x).reshape(-1, 1), np.array(y)
        manual = np.array(manual)
        if regression_type == "linear":
            model = LinearRegression().fit(X, Y)
            x, y = X, model.predict(X)
        elif regression_type == "pca":
            pca = PCA(n_components=1)
            d = np.c_[X, Y]
            fitted = pca.inverse_transform(pca.fit_transform(d))
            x, y = fitted[:, 0], fitted[:, 1]
        elif regression_type == "quantile":
            model = QuantileRegressor(quantile=0.5, alpha=0).fit(X, Y)
            x, y = X, model.predict(X)
        plt.plot(x, y, color="green")
        # compute RMSE
        rmse = np.sqrt(np.mean((Y - y) ** 2))
        # rmse with manual points removed
        rmse_manual = np.sqrt(np.mean((Y[manual == 0] - y[manual == 0]) ** 2))
        plt.title(
            f"Scatter + {regression_type} \n RMSE: {rmse:.2f}\n RMSE(subset): {rmse_manual:.2f}"
        )
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
        Div(
            Div(
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
                            value="50",
                            min="10",
                            max="2000",
                            step="1",
                        ),
                        Label("σ:", For="sigma"),
                        Input(type="text", id="sigma", name="sigma", value="0.5"),
                        style="display: flex; justify-content: space-between; gap: 5px;",
                    ),
                    Div(
                        Label("xmin:", For="xmin"),
                        Input(type="text", id="xmin", name="xmin", value="-5"),
                        Label("xmax:", For="xmax"),
                        Input(type="text", id="xmax", name="xmax", value="5"),
                        Button(
                            "Generate",
                            hx_post="/generate_data",
                            hx_target="#data-input",
                        ),
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
                Div(id="plot-container", style="text-align: center; clear: both;"),
                Style(
                    """
            form {
                max-width: 100%;
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
                style="width: 70%; float: left;",
            ),
            Div(
                P(
                    "1. We generate random data points based on the function f(x) you provide."
                ),
                P(
                    "2. You can manually enter new points in the textarea. Each line should be in the format: x,y,0 or x,y,1 (where 1 indicates an outlier)."
                ),
                P(
                    "3. Choose a regression type and click 'Plot' to visualize the data and the regression line. Blank regression type will only plot the data."
                ),
                P(
                    "4. The plot shows both the generated and manually added points, along with the chosen regression line."
                ),
                P(
                    "5. The plot also reports RMSE with and without outliers. This is meant to demonstrate the sensitivity of regression fits to high leverage points. For OLS, this is $h_{ii} = x_i' (X'X)^{-1} x_i$, where $x_i$ is the ith row of X. This value is high for points far from the mean X.",
                    cls="marked",
                ),
                P(
                    "6. To see this for yourself, add an outlier Y value at the middle of the support of X. Next, change the X axis value to be way out in the tails. Look at the differences in the fitted line and corresponding RMSE."
                ),
                A(
                    "Apoorva Lal",
                    href="https://apoorvalal.github.io/",
                ),
                style="width: 28%; float: right; padding-left: 20px;",
            ),
            style="display: flex; justify-content: space-between;",
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
