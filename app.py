import sys
import os

from flask import Flask, request, render_template

from src.pipeline.prediction_pipeline import PredictPipeline, CustomData


application = Flask(__name__)
app = application


@app.route("/")
def home():
    return render_template("index.html")




@app.route("/return", methods=["GET", "POST"])
def return_prediction():

    if request.method == "GET":
        return render_template("return_prediction.html")

    else:

        data = CustomData(

            Quantity=float(request.form["Quantity"]),
            Price=float(request.form["Price"]),
            Order_Value=float(request.form["Order_Value"]),
            Invoice_Month=int(request.form["Invoice_Month"]),
            Invoice_Day=int(request.form["Invoice_Day"]),
            Country=request.form["Country"]

        )

        pred_df = data.get_return_dataframe()

        pipeline = PredictPipeline()

        prediction = pipeline.predict_return(pred_df)[0]

        if prediction == 1:
            result = "Order Likely to be Returned"
        else:
            result = "Order Not Likely to be Returned"

        return render_template("result.html", result=result)




@app.route("/high_value", methods=["GET", "POST"])
def high_value_prediction():

    if request.method == "GET":
        return render_template("high_value_prediction.html")

    else:

        data = CustomData(

            Quantity=0,
            Price=0,
            Order_Value=0,
            Invoice_Month=0,
            Invoice_Day=0,
            Country="",

            Recency=float(request.form["Recency"]),
            Frequency=float(request.form["Frequency"]),
            Monetary=float(request.form["Monetary"])

        )

        rfm_df = data.get_rfm_dataframe()

        pipeline = PredictPipeline()

        prediction = pipeline.predict_high_value(rfm_df)[0]

        if prediction == 1:
            result = "High Value Customer"
        else:
            result = "Regular Customer"

        return render_template("result.html", result=result)




@app.route("/segment", methods=["GET", "POST"])
def segmentation():

    if request.method == "GET":
        return render_template("segmentation.html")

    else:

        data = CustomData(

            Quantity=0,
            Price=0,
            Order_Value=0,
            Invoice_Month=0,
            Invoice_Day=0,
            Country="",

            Recency=float(request.form["Recency"]),
            Frequency=float(request.form["Frequency"]),
            Monetary=float(request.form["Monetary"])

        )

        rfm_df = data.get_rfm_dataframe()

        pipeline = PredictPipeline()

        cluster = pipeline.predict_customer_segment(rfm_df)[0]
        segment_map = {
        0: "VIP Customers",
        1: "Regular Customers",
        2: "At Risk Customers"
        }

        result = f"Customer Segment: {segment_map.get(cluster, 'Unknown')}"

        return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)